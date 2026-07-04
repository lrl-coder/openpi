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
        force_guidance_from_residual = self._sample_kwargs.pop("force_guidance_from_residual", None)
        force_guidance_from_cst = self._sample_kwargs.pop("force_guidance_from_cst", False)
        self._force_guidance_enabled = bool(
            force_guidance_from_cst if force_guidance_from_residual is None else force_guidance_from_residual
        )
        force_residual_window = self._sample_kwargs.pop("force_residual_window", 1)
        self._force_residual_window = max(1, int(force_residual_window))
        self._force_model_enabled = bool(getattr(model, "force_guidance", False))
        self._prev_force_mu = None
        self._prev_force_log_sigma = None
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
            self._predict_force = None
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._predict_force = (
                nnx_utils.module_jit(model.predict_force)
                if self._force_guidance_enabled and hasattr(model, "predict_force")
                else None
            )
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        if bool(np.asarray(obs.get("reset", False))):
            self._prev_force_mu = None
            self._prev_force_log_sigma = None
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
                self._force_history_global = np.repeat(
                    current_force[None, :], self._force_history_global_len, axis=0
                )
            else:
                self._force_history_global = np.concatenate(
                    [self._force_history_global[1:], current_force[None, :]], axis=0
                )

            if provided_force_history_local is not None:
                self._force_history_local = _pad_or_trim_force_history(
                    provided_force_history_local, current_force, self._force_history_local_len
                )
            elif self._force_history_local is None:
                self._force_history_local = np.repeat(
                    current_force[None, :], self._force_history_local_len, axis=0
                )
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

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if self._force_guidance_enabled and not self._is_pytorch_model and self._prev_force_mu is not None:
            force = inputs.get("force")
            if force is not None:
                force = jnp.asarray(force)
                prev_mu = jnp.asarray(self._prev_force_mu)
                prev_log_sigma = jnp.asarray(self._prev_force_log_sigma)
                if prev_mu.ndim == 3:
                    force_history = inputs.get("force_history_local")
                    if force_history is not None:
                        window = min(self._force_residual_window, prev_mu.shape[1], force_history.shape[1])
                        observed_force = force_history[:, -window:, :]
                        pred_mu = prev_mu[:, :window, :]
                        pred_log_sigma = prev_log_sigma[:, :window, :]
                    else:
                        observed_force = force[:, None, :]
                        pred_mu = prev_mu[:, :1, :]
                        pred_log_sigma = prev_log_sigma[:, :1, :]
                else:
                    observed_force = force[:, None, :]
                    pred_mu = prev_mu[:, None, :]
                    pred_log_sigma = prev_log_sigma[:, None, :]

                residual = observed_force - pred_mu
                inv_var = jnp.exp(-2.0 * pred_log_sigma)
                per_step_error = jnp.sum(jnp.square(residual) * inv_var, axis=-1)
                sample_kwargs["force_prediction_error"] = jnp.mean(per_step_error, axis=-1)

        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        if self._force_guidance_enabled and self._predict_force is not None:
            force_mu, force_log_sigma = self._predict_force(observation, outputs["actions"])
            self._prev_force_mu = force_mu
            self._prev_force_log_sigma = force_log_sigma
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
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
