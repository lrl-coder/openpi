from types import SimpleNamespace

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi import transforms
from openpi.policies import aloha_policy
from openpi.policies import libero_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class _ForcePredictionModel(nnx.Module):
    force_guidance = True

    def __init__(self):
        self.config = SimpleNamespace(force_history_global_len=4, force_history_local_len=2)

    def sample_actions(self, rng, observation, **kwargs):
        del rng, kwargs
        return jnp.ones((observation.state.shape[0], 3, 7), dtype=jnp.float32)

    def predict_force(self, observation, actions):
        del observation
        return jnp.zeros((actions.shape[0], actions.shape[1], 6), dtype=jnp.float32)


def test_policy_returns_unnormalized_force_prediction_for_corace():
    force_stats = transforms.NormStats(
        mean=np.arange(6, dtype=np.float32),
        std=np.full(6, 2.0, dtype=np.float32),
    )
    policy = _policy.Policy(
        _ForcePredictionModel(),
        sample_kwargs={"return_force_prediction": True},
        output_transforms=[
            transforms.Unnormalize({"force_prediction": force_stats}, strict=False),
            libero_policy.LiberoOutputs(),
        ],
    )

    result = policy.infer(
        {
            "image": {},
            "image_mask": {},
            "state": np.zeros(7, dtype=np.float32),
            "force": np.zeros(6, dtype=np.float32),
        }
    )

    assert result["actions"].shape == (3, 7)
    assert result["force_prediction"].shape == (3, 6)
    np.testing.assert_allclose(result["force_prediction"], np.broadcast_to(np.arange(6), (3, 6)))


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
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
