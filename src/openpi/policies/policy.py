from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _pad_or_trim_force_history(history: np.ndarray, current_force: np.ndarray, target_len: int) -> np.ndarray:
    history = np.asarray(history)
    if history.ndim == 1:
        history = history[None, :]
    if history.shape[0] == 0:
        history = current_force[None, :]
    if history.shape[0] < target_len:
        pad = np.repeat(history[:1], target_len - history.shape[0], axis=0)
        history = np.concatenate([pad, history], axis=0)
    elif history.shape[0] > target_len:
        history = history[-target_len:]
    return history


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = dict(sample_kwargs or {})
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._force_model_enabled = bool(getattr(model, "force_guidance", False))
        self._fra_enabled = bool(getattr(model, "fra_enabled", False))
        self._force_history_global = None
        self._force_history_local = None
        model_config = getattr(model, "config", None)
        self._force_history_global_len = int(getattr(model_config, "force_history_global_len", 64))
        self._force_history_local_len = int(getattr(model_config, "force_history_local_len", 16))
        if hasattr(model, "force_history_global_len"):
            self._force_history_global_len = int(model.force_history_global_len)
        if hasattr(model, "force_history_local_len"):
            self._force_history_local_len = int(model.force_history_local_len)
        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._predict_reflex = None
        else:
            # JAX model setup
            model.eval()
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._predict_reflex = (
                nnx_utils.module_jit(model.predict_reflex)
                if self._fra_enabled and hasattr(model, "predict_reflex")
                else None
            )
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        reflex_mode = bool(np.asarray(obs.get("fra_mode", False)))
        if bool(np.asarray(obs.get("reset", False))):
            self._force_history_global = None
            self._force_history_local = None

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if self._force_model_enabled and not self._is_pytorch_model and "force" in inputs:
            current_force = np.asarray(inputs["force"])
            provided_force_history_global = inputs.get("force_history_global")
            provided_force_history_local = inputs.get("force_history_local", inputs.get("force_history"))

            if provided_force_history_global is not None:
                self._force_history_global = _pad_or_trim_force_history(
                    provided_force_history_global, current_force, self._force_history_global_len
                )
            elif self._force_history_global is None:
                self._force_history_global = np.repeat(current_force[None, :], self._force_history_global_len, axis=0)
            else:
                self._force_history_global = np.concatenate(
                    [self._force_history_global[1:], current_force[None, :]], axis=0
                )

            if provided_force_history_local is not None:
                self._force_history_local = _pad_or_trim_force_history(
                    provided_force_history_local, current_force, self._force_history_local_len
                )
            elif self._force_history_local is None:
                self._force_history_local = np.repeat(current_force[None, :], self._force_history_local_len, axis=0)
            else:
                self._force_history_local = np.concatenate(
                    [self._force_history_local[1:], current_force[None, :]], axis=0
                )
            inputs["force_history_global"] = self._force_history_global
            inputs["force_history_local"] = self._force_history_local
            inputs["force_history"] = self._force_history_local
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        if reflex_mode:
            if self._predict_reflex is None:
                raise ValueError("fra_mode=True requires a JAX checkpoint trained with fra_enabled=True.")
            reflex_outputs = self._predict_reflex(observation)
            outputs = {
                "state": inputs["state"],
                "actions": reflex_outputs["corrected_action"][:, None, :],
                "fra_gate": reflex_outputs["gate"],
            }
        else:
            outputs = {
                "state": inputs["state"],
                "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
            }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        reflex_gate = np.asarray(outputs.pop("fra_gate")) if reflex_mode else None
        outputs = self._output_transform(outputs)
        if reflex_mode:
            nominal_action = np.asarray(obs["fra_nominal_action"])
            corrected_action = np.asarray(outputs["actions"])[0]
            assert reflex_gate is not None
            gate = reflex_gate
            arm_dim = int(getattr(self._model, "fra_arm_dim", gate.shape[-1]))
            outputs["fra"] = {
                "residual": corrected_action[:arm_dim] - nominal_action[:arm_dim],
                "gate": gate[:arm_dim],
            }
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
