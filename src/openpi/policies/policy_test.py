from types import SimpleNamespace

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.policies import libero_policy
from openpi.policies import policy as _policy


class _ForceReflexModel(nnx.Module):
    force_guidance = True
    fra_enabled = True
    fra_arm_dim = 2

    def __init__(self):
        self.config = SimpleNamespace(force_history_global_len=4, force_history_local_len=2)

    def sample_actions(self, rng, observation, **kwargs):
        del rng, kwargs
        return jnp.ones((observation.state.shape[0], 3, 7), dtype=jnp.float32)

    def predict_reflex(self, observation):
        corrected = observation.fra_nominal_action.at[:, :2].add(0.5)
        return {
            "corrected_action": corrected,
            "residual": jnp.full((corrected.shape[0], 2), 0.5),
            "gate": jnp.full((corrected.shape[0], 2), 0.25),
        }


def test_policy_runs_reflex_without_sampling_a_new_chunk(monkeypatch):
    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda fn: fn)
    policy = _policy.Policy(
        _ForceReflexModel(),
        output_transforms=[
            libero_policy.LiberoOutputs(),
        ],
    )

    result = policy.infer(
        {
            "image": {},
            "image_mask": {},
            "state": np.zeros(7, dtype=np.float32),
            "force": np.zeros(6, dtype=np.float32),
            "force_history_global": np.zeros((4, 6), dtype=np.float32),
            "fra_mode": True,
            "fra_nominal_action": np.arange(7, dtype=np.float32),
            "fra_previous_nominal_action": np.arange(7, dtype=np.float32),
            "fra_current_joint_state": np.zeros(2, dtype=np.float32),
            "fra_chunk_progress": np.array([0.5], dtype=np.float32),
        }
    )

    assert result["actions"].shape == (1, 7)
    np.testing.assert_allclose(result["actions"][0, :2], [0.5, 1.5])
    np.testing.assert_allclose(result["fra"]["residual"], [0.5, 0.5])
    np.testing.assert_allclose(result["fra"]["gate"], [0.25, 0.25])


@pytest.mark.manual
def test_infer():
    from openpi.policies import aloha_policy
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    from openpi_client import action_chunk_broker

    from openpi.policies import aloha_policy
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
