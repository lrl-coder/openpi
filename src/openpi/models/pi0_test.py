import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_force_guided_head_has_current_force_projection_and_point_prediction():
    config = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=8,
        force_guidance=True,
    )
    model = nnx.eval_shape(config.create, jax.random.key(0))

    assert model.force_current_proj.in_features == config.force_dim
    assert model.force_current_proj.out_features == 64
    assert model.force_predictor_out.in_features == 64
    assert model.force_predictor_out.out_features == config.force_dim
    assert not hasattr(model, "force_local_conv1")
    assert not hasattr(model, "force_predictor_in")


class _CurrentForceHead(_pi0.Pi0):
    """Minimal module exposing the force-head helpers without constructing SigLIP."""

    def __init__(self):
        self.force_dim = 6
        self.force_current_proj = nnx.Linear(6, 8, rngs=nnx.Rngs(0))
        self.action_in_proj = nnx.Linear(7, 8, rngs=nnx.Rngs(1))
        self.force_predictor_out = nnx.Linear(8, 6, rngs=nnx.Rngs(2))


def test_current_force_token_ignores_force_history():
    model = _CurrentForceHead()
    current_force = jnp.arange(12, dtype=jnp.float32).reshape(2, 6)
    observation = _model.Observation(
        images={},
        image_masks={},
        state=jnp.ones((2, 7), dtype=jnp.float32),
        force=current_force,
        force_history_global=jnp.zeros((2, 64, 6), dtype=jnp.float32),
        force_history_local=jnp.zeros((2, 16, 6), dtype=jnp.float32),
    )

    token = model._embed_current_force_token(observation, jnp.float32)  # noqa: SLF001
    changed_histories = observation.replace(
        force_history_global=jnp.full((2, 64, 6), 1000.0, dtype=jnp.float32),
        force_history_local=jnp.full((2, 16, 6), -1000.0, dtype=jnp.float32),
    )
    token_with_changed_histories = model._embed_current_force_token(  # noqa: SLF001
        changed_histories, jnp.float32
    )
    expected = model.force_current_proj(current_force)[:, None, :]

    np.testing.assert_allclose(token, expected)
    np.testing.assert_allclose(token_with_changed_histories, expected)


def test_force_predictor_uses_l2_and_outputs_one_force_vector_per_action():
    model = _CurrentForceHead()
    action_features = jnp.ones((2, 4, 8), dtype=jnp.float32)
    action_hat_0 = jnp.ones((2, 4, 7), dtype=jnp.float32)
    prediction = model._decode_force_from_action_features(action_features, action_hat_0)  # noqa: SLF001

    target = jnp.array([[[1.0, 3.0], [2.0, 4.0]]], dtype=jnp.float32)
    zeros = jnp.zeros_like(target)
    l2_loss = model._force_l2_loss(target, zeros)  # noqa: SLF001

    assert prediction.shape == (2, 4, 6)
    np.testing.assert_allclose(l2_loss, np.array([[5.0, 10.0]], dtype=np.float32))


def test_force_aware_model_samples_actions_and_predicts_aligned_force():
    key = jax.random.key(0)
    config = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=8,
        force_guidance=True,
        force_loss_weight=0.05,
    )
    model = config.create(key)
    observation = config.fake_obs(batch_size=1)
    target_actions = config.fake_act(batch_size=1)

    loss, info = model.compute_loss_info(key, observation, target_actions)
    actions = model.sample_actions(key, observation, num_steps=2)
    force_prediction = model.predict_force(observation, actions)

    assert loss.shape == (1, config.action_horizon)
    assert actions.shape == (1, config.action_horizon, config.action_dim)
    assert force_prediction.shape == (1, config.action_horizon, config.force_dim)
    assert "loss_force_l2" in info
    assert np.all(np.isfinite(loss))
    assert np.all(np.isfinite(actions))
    assert np.all(np.isfinite(force_prediction))
