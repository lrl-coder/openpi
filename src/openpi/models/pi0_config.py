import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    # Force-aware pi0 extensions. Disabled by default to keep the released pi0 behavior unchanged.
    force_guidance: bool = False
    force_dim: int = 6
    # Global force history is a longer contact context for the VLM-level semantic target.
    force_history_global_len: int = 64
    # Legacy local-history shape retained for data/checkpoint compatibility. The action
    # expert consumes only `force`, projected by one linear layer.
    force_history_local_len: int = 16
    force_global_patch_size: int = 8
    force_semantic_feature_dim: int = 256
    # Width of the lightweight two-layer MLP that predicts absolute future
    # force from the concatenated action hidden state and clean-action estimate.
    force_predictor_hidden_dim: int = 256
    force_loss_weight: float = 0.0
    # Weight for contact-dynamics distillation.
    force_target_loss_weight: float = 0.0
    # Weight for the force encoder's physical summary prediction anchor.
    force_physical_loss_weight: float = 0.0

    # Force Reflex Adapter. Stage one leaves this disabled and trains the
    # force-guided pi0 normally. Stage two enables both flags below, loads the
    # stage-one checkpoint, and freezes every parameter except the adapter.
    fra_enabled: bool = False
    fra_training_only: bool = False
    fra_arm_dim: int = 7
    fra_hidden_dim: int = 256
    fra_max_chunk_age: int = 49
    fra_action_scale: float = 1.0
    fra_loss_weight: float = 1.0
    fra_magnitude_loss_weight: float = 1e-3
    fra_smoothness_loss_weight: float = 1e-2
    fra_huber_delta: float = 1.0

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if self.fra_enabled and not self.force_guidance:
            raise ValueError("fra_enabled=True requires force_guidance=True to reuse the Contact Dynamics Token.")
        if self.force_predictor_hidden_dim <= 0:
            raise ValueError("force_predictor_hidden_dim must be positive.")
        if self.fra_training_only and not self.fra_enabled:
            raise ValueError("fra_training_only=True requires fra_enabled=True.")
        if not 0 < self.fra_arm_dim <= self.action_dim:
            raise ValueError("fra_arm_dim must be in [1, action_dim].")
        if self.fra_hidden_dim <= 0:
            raise ValueError("fra_hidden_dim must be positive.")
        if self.fra_max_chunk_age < 0:
            raise ValueError("fra_max_chunk_age must be non-negative.")
        if self.fra_action_scale <= 0:
            raise ValueError("fra_action_scale must be positive.")
        if self.fra_huber_delta <= 0:
            raise ValueError("fra_huber_delta must be positive.")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                force=(
                    jax.ShapeDtypeStruct([batch_size, self.force_dim], jnp.float32) if self.force_guidance else None
                ),
                force_history_global=(
                    jax.ShapeDtypeStruct([batch_size, self.force_history_global_len, self.force_dim], jnp.float32)
                    if self.force_guidance
                    else None
                ),
                force_history_local=(
                    jax.ShapeDtypeStruct([batch_size, self.force_history_local_len, self.force_dim], jnp.float32)
                    if self.force_guidance
                    else None
                ),
                force_history=None,
                force_targets=(
                    jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.force_dim], jnp.float32)
                    if self.force_guidance
                    else None
                ),
                force_task_target=(
                    jax.ShapeDtypeStruct([batch_size, self.force_dim], jnp.float32) if self.force_guidance else None
                ),
                fra_joint_states=(
                    jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.fra_arm_dim], jnp.float32)
                    if self.fra_training_only
                    else None
                ),
                fra_force_history=(
                    jax.ShapeDtypeStruct(
                        [
                            batch_size,
                            self.action_horizon,
                            self.force_history_global_len,
                            self.force_dim,
                        ],
                        jnp.float32,
                    )
                    if self.fra_training_only
                    else None
                ),
                fra_nominal_action=(
                    jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32)
                    if self.fra_enabled and not self.fra_training_only
                    else None
                ),
                fra_previous_nominal_action=(
                    jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32)
                    if self.fra_enabled and not self.fra_training_only
                    else None
                ),
                fra_current_joint_state=(
                    jax.ShapeDtypeStruct([batch_size, self.fra_arm_dim], jnp.float32)
                    if self.fra_enabled and not self.fra_training_only
                    else None
                ),
                fra_chunk_progress=(
                    jax.ShapeDtypeStruct([batch_size, 1], jnp.float32)
                    if self.fra_enabled and not self.fra_training_only
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

    def get_fra_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze the complete VLA and Contact Dynamics Token, leaving only FRA trainable."""
        if not self.fra_enabled:
            raise ValueError("get_fra_freeze_filter requires fra_enabled=True.")
        return nnx.Not(nnx_utils.PathRegex(".*force_reflex_adapter.*"))
