import logging
import math

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

_LOG_2PI = math.log(2.0 * math.pi)


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.force_guidance = config.force_guidance
        self.force_dim = config.force_dim
        self.force_history_local_len = config.force_history_local_len
        self.force_loss_weight = config.force_loss_weight
        self.force_target_loss_weight = config.force_target_loss_weight
        self.force_guidance_lambda_max = config.force_guidance_lambda_max
        self.force_guidance_k = config.force_guidance_k
        self.force_guidance_tau0 = config.force_guidance_tau0
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        if self.force_guidance:
            # Deprecated compatibility layer: older force-guided checkpoints contain this
            # parameter. The active F_phi below decodes directly from action-expert hidden states.
            force_predictor_input_dim = (
                config.action_dim + config.action_dim + config.force_dim + paligemma_config.width
            )
            self.force_predictor_in = nnx.Linear(force_predictor_input_dim, action_expert_config.width, rngs=rngs)
            self.force_predictor_out = nnx.Linear(action_expert_config.width, 2 * config.force_dim, rngs=rngs)
            self.force_local_conv1 = nnx.Conv(
                config.force_dim, 64, kernel_size=3, padding="CAUSAL", rngs=rngs
            )
            self.force_local_norm1 = nnx.LayerNorm(64, rngs=rngs)
            self.force_local_conv2 = nnx.Conv(
                64, 128, kernel_size=3, padding="CAUSAL", kernel_dilation=2, rngs=rngs
            )
            self.force_local_norm2 = nnx.LayerNorm(128, rngs=rngs)
            self.force_local_conv3 = nnx.Conv(
                128,
                action_expert_config.width,
                kernel_size=3,
                padding="CAUSAL",
                kernel_dilation=4,
                rngs=rngs,
            )
            self.force_local_norm3 = nnx.LayerNorm(action_expert_config.width, rngs=rngs)
            self.force_local_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.force_target_in = nnx.Linear(paligemma_config.width, action_expert_config.width, rngs=rngs)
            self.force_target_out = nnx.Linear(action_expert_config.width, 2 * config.force_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _pool_prefix(self, prefix_out, prefix_mask):
        mask = prefix_mask.astype(prefix_out.dtype)[..., None]
        denom = jnp.maximum(jnp.sum(mask, axis=1), 1.0)
        return jnp.sum(prefix_out * mask, axis=1) / denom

    def _force_history_or_current(self, obs: _model.Observation, dtype, *, scope: str):
        batch_size = obs.state.shape[0]
        if scope == "local":
            target_len = self.force_history_local_len
            preferred_history = obs.force_history_local
        else:
            raise ValueError(f"Unknown force history scope: {scope}")

        if preferred_history is not None:
            force_history = preferred_history.astype(dtype)
        elif obs.force_history is not None:
            force_history = obs.force_history.astype(dtype)
        elif obs.force is not None:
            force_history = einops.repeat(obs.force.astype(dtype), "b f -> b h f", h=target_len)
        else:
            force_history = jnp.zeros((batch_size, target_len, self.force_dim), dtype=dtype)

        history_len = force_history.shape[1]
        if history_len < target_len:
            pad = jnp.repeat(force_history[:, :1, :], target_len - history_len, axis=1)
            force_history = jnp.concatenate([pad, force_history], axis=1)
        elif history_len > target_len:
            force_history = force_history[:, -target_len:, :]
        return force_history

    def _embed_force_local_token(self, obs: _model.Observation, dtype):
        force_history = self._force_history_or_current(obs, dtype, scope="local")
        hidden = self.force_local_conv1(force_history)
        hidden = self.force_local_norm1(hidden)
        hidden = nnx.swish(hidden)
        hidden = self.force_local_conv2(hidden)
        hidden = self.force_local_norm2(hidden)
        hidden = nnx.swish(hidden)
        hidden = self.force_local_conv3(hidden)
        hidden = self.force_local_norm3(hidden)
        hidden = nnx.swish(hidden)
        pooled = jnp.mean(hidden, axis=1)
        pooled = self.force_local_out(pooled)
        pooled = nnx.swish(pooled)
        return pooled[:, None, :]

    def _split_force_distribution(self, x):
        mu, log_sigma = jnp.split(x.astype(jnp.float32), 2, axis=-1)
        return mu, jnp.clip(log_sigma, -5.0, 3.0)

    def _predict_force_target(self, prefix_pooled):
        hidden = self.force_target_in(prefix_pooled)
        hidden = nnx.swish(hidden)
        return self._split_force_distribution(self.force_target_out(hidden))

    def _decode_force_from_action_features(self, action_features):
        return self._split_force_distribution(self.force_predictor_out(action_features))

    def _action_features_from_prefix_cache(
        self,
        obs: _model.Observation,
        actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        prefix_mask,
        kv_cache,
    ):
        batch_size = actions.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(obs, actions, timestep)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        assert full_attn_mask.shape == (
            batch_size,
            suffix_tokens.shape[1],
            prefix_mask.shape[1] + suffix_tokens.shape[1],
        )
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (step_prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert step_prefix_out is None
        return suffix_out[:, -self.action_horizon :]

    def _predict_force_from_action_head(
        self,
        obs: _model.Observation,
        actions: _model.Actions,
        prefix_mask,
        kv_cache,
        timestep: at.Float[at.Array, " b"] | None = None,
    ):
        if timestep is None:
            timestep = jnp.zeros((actions.shape[0],), dtype=actions.dtype)
        action_features = self._action_features_from_prefix_cache(obs, actions, timestep, prefix_mask, kv_cache)
        return self._decode_force_from_action_features(action_features)

    def _diag_gaussian_nll(self, target, mu, log_sigma):
        log_sigma = jnp.clip(log_sigma, -5.0, 3.0)
        inv_var = jnp.exp(-2.0 * log_sigma)
        nll = 0.5 * (jnp.square(target.astype(jnp.float32) - mu) * inv_var + 2.0 * log_sigma + _LOG_2PI)
        return jnp.mean(nll, axis=-1)

    def _force_distribution_info(self, prefix, target, mu, log_sigma, nll):
        target = jax.lax.stop_gradient(target.astype(jnp.float32))
        mu = jax.lax.stop_gradient(mu)
        log_sigma = jax.lax.stop_gradient(log_sigma)
        nll = jax.lax.stop_gradient(nll)
        residual = target - mu
        pred_sigma = jnp.exp(log_sigma)
        pred_var = jnp.square(pred_sigma)
        reduce_axes = tuple(range(target.ndim - 1))

        pred_sigma_axis = jnp.mean(pred_sigma, axis=reduce_axes)
        pred_var_axis = jnp.mean(pred_var, axis=reduce_axes)
        true_var_axis = jnp.var(target, axis=reduce_axes)
        residual_mse_axis = jnp.mean(jnp.square(residual), axis=reduce_axes)
        var_gap_axis = pred_var_axis - true_var_axis
        true_std_axis = jnp.sqrt(true_var_axis + 1e-8)
        residual_rmse_axis = jnp.sqrt(residual_mse_axis + 1e-8)

        info = {
            f"{prefix}_nll": jnp.mean(nll),
            f"{prefix}_nll_negative_frac": jnp.mean((nll < 0.0).astype(jnp.float32)),
            f"{prefix}_pred_log_sigma_mean": jnp.mean(log_sigma),
            f"{prefix}_pred_log_sigma_min": jnp.min(log_sigma),
            f"{prefix}_pred_log_sigma_max": jnp.max(log_sigma),
            f"{prefix}_pred_log_sigma_min_clip_frac": jnp.mean((log_sigma <= -4.99).astype(jnp.float32)),
            f"{prefix}_pred_log_sigma_max_clip_frac": jnp.mean((log_sigma >= 2.99).astype(jnp.float32)),
            f"{prefix}_pred_sigma_mean": jnp.mean(pred_sigma),
            f"{prefix}_pred_sigma_min": jnp.min(pred_sigma),
            f"{prefix}_pred_sigma_max": jnp.max(pred_sigma),
            f"{prefix}_pred_var_mean": jnp.mean(pred_var),
            f"{prefix}_true_var_mean": jnp.mean(true_var_axis),
            f"{prefix}_pred_var_minus_true_var_mean": jnp.mean(var_gap_axis),
            f"{prefix}_pred_var_abs_gap_mean": jnp.mean(jnp.abs(var_gap_axis)),
            f"{prefix}_residual_mse_mean": jnp.mean(residual_mse_axis),
            f"{prefix}_pred_sigma_to_true_std_ratio_mean": jnp.mean(pred_sigma_axis / (true_std_axis + 1e-6)),
            f"{prefix}_pred_var_to_residual_mse_ratio_mean": jnp.mean(pred_var_axis / (residual_mse_axis + 1e-6)),
            f"{prefix}_residual_rmse_to_pred_sigma_ratio_mean": jnp.mean(residual_rmse_axis / (pred_sigma_axis + 1e-6)),
        }
        for axis in range(self.force_dim):
            info[f"{prefix}_pred_var_axis_{axis}"] = pred_var_axis[axis]
            info[f"{prefix}_true_var_axis_{axis}"] = true_var_axis[axis]
            info[f"{prefix}_pred_var_minus_true_var_axis_{axis}"] = var_gap_axis[axis]
            info[f"{prefix}_residual_mse_axis_{axis}"] = residual_mse_axis[axis]
        if target.ndim == 3:
            within_horizon_var_axis = jnp.mean(jnp.var(target, axis=1), axis=0)
            within_horizon_gap_axis = pred_var_axis - within_horizon_var_axis
            info[f"{prefix}_true_var_within_horizon_mean"] = jnp.mean(within_horizon_var_axis)
            info[f"{prefix}_pred_var_minus_true_var_within_horizon_mean"] = jnp.mean(within_horizon_gap_axis)
            info[f"{prefix}_pred_var_abs_gap_within_horizon_mean"] = jnp.mean(jnp.abs(within_horizon_gap_axis))
            for axis in range(self.force_dim):
                info[f"{prefix}_true_var_within_horizon_axis_{axis}"] = within_horizon_var_axis[axis]
                info[f"{prefix}_pred_var_minus_true_var_within_horizon_axis_{axis}"] = within_horizon_gap_axis[axis]
        return info

    def predict_force(
        self,
        observation: _model.Observation,
        actions: _model.Actions,
    ) -> tuple[at.Float[at.Array, "b ah f"], at.Float[at.Array, "b ah f"]]:
        if not self.force_guidance:
            raise ValueError("predict_force is only available when force_guidance=True.")

        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (_, _), kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        return self._predict_force_from_action_head(observation, actions, prefix_mask, kv_cache)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if self.force_guidance:
            force_token = self._embed_force_local_token(obs, noisy_actions.dtype)
            tokens.append(force_token)
            input_mask.append(jnp.ones(force_token.shape[:2], dtype=jnp.bool_))
            # Local force history is a causal condition for state/action suffix tokens.
            ar_mask += [True]

        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        loss, _ = self._compute_loss_and_info(rng, observation, actions, train=train, return_info=False)
        return loss

    def compute_loss_info(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ):
        return self._compute_loss_and_info(rng, observation, actions, train=train, return_info=True)

    def _compute_loss_and_info(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        return_info: bool = False,
    ):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Fill the prefix KV cache once. The action head and F_phi both decode from suffix hidden features.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        action_features = self._action_features_from_prefix_cache(observation, x_t, time, prefix_mask, kv_cache)
        v_t = self.action_out_proj(action_features)

        fm_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        loss = fm_loss
        info = {"loss_fm": jnp.mean(fm_loss)} if return_info else {}

        if not self.force_guidance:
            return loss, info

        prefix_pooled = self._pool_prefix(prefix_out, prefix_mask)

        if self.force_loss_weight > 0.0 and observation.force_targets is not None:
            force_mu, force_log_sigma = self._predict_force_from_action_head(
                observation, actions, prefix_mask, kv_cache
            )
            force_loss = self._diag_gaussian_nll(observation.force_targets, force_mu, force_log_sigma)
            loss = loss + self.force_loss_weight * force_loss
            if return_info:
                info["loss_force_nll"] = jnp.mean(force_loss)
                info["loss_force_weighted"] = self.force_loss_weight * jnp.mean(force_loss)
                info.update(
                    self._force_distribution_info(
                        "force",
                        observation.force_targets,
                        force_mu,
                        force_log_sigma,
                        force_loss,
                    )
                )

        if self.force_target_loss_weight > 0.0 and (
            observation.force_targets is not None or observation.force_task_target is not None
        ):
            target_mu, target_log_sigma = self._predict_force_target(prefix_pooled)
            if observation.force_targets is not None:
                target_labels = observation.force_targets
                target_loss = self._diag_gaussian_nll(
                    target_labels,
                    target_mu[:, None, :],
                    target_log_sigma[:, None, :],
                )
                target_loss_for_chunk = target_loss
            else:
                target_labels = observation.force_task_target
                target_loss = self._diag_gaussian_nll(target_labels, target_mu, target_log_sigma)
                target_loss_for_chunk = target_loss[:, None]

            loss = loss + self.force_target_loss_weight * target_loss_for_chunk
            if return_info:
                info["loss_force_target_nll"] = jnp.mean(target_loss)
                info["loss_force_target_weighted"] = self.force_target_loss_weight * jnp.mean(target_loss)
                info.update(
                    self._force_distribution_info(
                        "force_target",
                        target_labels,
                        target_mu[:, None, :] if target_labels.ndim == 3 else target_mu,
                        target_log_sigma[:, None, :] if target_labels.ndim == 3 else target_log_sigma,
                        target_loss,
                    )
                )

        return loss, info

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        cst_tau: at.Float[at.Array, "b"] | at.Float[at.Array, ""] | None = None,
        force_guidance_lambda: at.Float[at.Array, "b"] | at.Float[at.Array, ""] | None = None,
        force_target_mu: at.Float[at.Array, "b f"] | None = None,
        force_target_log_sigma: at.Float[at.Array, "b f"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (cached_prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        enable_force_guidance = self.force_guidance and (
            cst_tau is not None or force_guidance_lambda is not None
        )
        if enable_force_guidance:
            prefix_pooled = self._pool_prefix(cached_prefix_out, prefix_mask)
            if force_target_mu is None or force_target_log_sigma is None:
                predicted_target_mu, predicted_target_log_sigma = self._predict_force_target(prefix_pooled)
                if force_target_mu is None:
                    force_target_mu = predicted_target_mu
                if force_target_log_sigma is None:
                    force_target_log_sigma = predicted_target_log_sigma
        else:
            prefix_pooled = None

        if enable_force_guidance:
            if force_guidance_lambda is None:
                cst_tau = jnp.asarray(cst_tau, dtype=jnp.float32)
                if cst_tau.ndim == 0:
                    cst_tau = jnp.broadcast_to(cst_tau, (batch_size,))
                force_guidance_lambda = self.force_guidance_lambda_max * jax.nn.sigmoid(
                    self.force_guidance_k * (cst_tau - self.force_guidance_tau0)
                )
            else:
                force_guidance_lambda = jnp.asarray(force_guidance_lambda, dtype=jnp.float32)
                if force_guidance_lambda.ndim == 0:
                    force_guidance_lambda = jnp.broadcast_to(force_guidance_lambda, (batch_size,))

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (step_prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert step_prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            if enable_force_guidance:
                assert prefix_pooled is not None
                assert force_target_mu is not None
                assert force_target_log_sigma is not None
                base_v_t = jax.lax.stop_gradient(v_t)

                def guidance_nll(action_sample):
                    clean_action_estimate = action_sample - time * base_v_t
                    force_mu, _ = self._predict_force_from_action_head(
                        observation,
                        clean_action_estimate,
                        prefix_mask,
                        kv_cache,
                        jnp.zeros((batch_size,), dtype=clean_action_estimate.dtype),
                    )
                    nll = self._diag_gaussian_nll(
                        force_target_mu[:, None, :],
                        force_mu,
                        force_target_log_sigma[:, None, :],
                    )
                    return jnp.sum(nll * force_guidance_lambda[:, None])

                guidance_grad = jax.grad(guidance_nll)(x_t)
                # The sampler integrates from t=1 to t=0 with negative dt, while v_t predicts noise - action.
                # Adding grad(NLL) to this velocity moves the clean-action estimate down the NLL gradient.
                v_t = v_t + guidance_grad

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
