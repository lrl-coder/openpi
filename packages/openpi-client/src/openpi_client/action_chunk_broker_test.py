import numpy as np
import pytest

from openpi_client import action_chunk_broker
from openpi_client import base_policy


class _Frapolicy(base_policy.BasePolicy):
    def __init__(self):
        self.plan_calls = 0
        self.reflex_calls = 0
        self.last_reflex_obs = None

    def infer(self, obs):
        if obs.get("fra_mode", False):
            self.reflex_calls += 1
            self.last_reflex_obs = obs
            nominal = np.asarray(obs["fra_nominal_action"])
            corrected = nominal.copy()
            corrected[:2] += np.asarray(obs["force_history_global"])[-1, 0]
            return {
                "actions": corrected[None, :],
                "fra": {"gate": np.full(2, 0.5)},
            }

        chunk_index = self.plan_calls
        self.plan_calls += 1
        arm = np.arange(6, dtype=np.float64).reshape(3, 2) + 10 * chunk_index
        gripper = np.full((3, 1), 0.25 + chunk_index)
        return {"actions": np.concatenate([arm, gripper], axis=-1)}


def _observation(force: float, *, reset: bool = False):
    return {
        "observation/state": np.array([0.0, 0.0, 0.25, force]),
        "force": np.array([force]),
        "reset": reset,
    }


def test_fra_continuously_corrects_arm_and_keeps_gripper_nominal():
    policy = _Frapolicy()
    broker = action_chunk_broker.ForceReflexActionChunkBroker(
        policy,
        action_horizon=3,
        arm_joint_dim=2,
        force_dim=1,
        force_history_len=4,
    )

    first = broker.infer(_observation(0.5))
    second = broker.infer(_observation(-0.25))

    np.testing.assert_allclose(first["actions"], [0.5, 1.5, 0.25])
    np.testing.assert_allclose(second["actions"], [1.75, 2.75, 0.25])
    np.testing.assert_allclose(first["fra"]["residual"], [0.5, 0.5])
    assert first["fra"]["chunk_index"] == 0
    assert second["fra"]["chunk_index"] == 1
    assert policy.plan_calls == 1
    assert policy.reflex_calls == 2


def test_fra_updates_causal_force_history_and_replans_only_after_chunk_end():
    policy = _Frapolicy()
    broker = action_chunk_broker.ForceReflexActionChunkBroker(
        policy,
        action_horizon=3,
        arm_joint_dim=2,
        force_dim=1,
        force_history_len=3,
    )

    for force in (1.0, 2.0, 3.0, 4.0):
        broker.infer(_observation(force))

    assert policy.plan_calls == 2
    assert policy.reflex_calls == 4
    np.testing.assert_allclose(policy.last_reflex_obs["force_history_global"][:, 0], [2.0, 3.0, 4.0])


def test_fra_enforces_arm_joint_limits_after_correction():
    broker = action_chunk_broker.ForceReflexActionChunkBroker(
        _Frapolicy(),
        action_horizon=3,
        arm_joint_dim=2,
        force_dim=1,
        joint_lower_limits=[-0.2, -0.2],
        joint_upper_limits=[0.2, 0.2],
    )

    result = broker.infer(_observation(10.0))

    np.testing.assert_allclose(result["actions"], [0.2, 0.2, 0.25])
    np.testing.assert_allclose(result["fra"]["residual"], [0.2, -0.8])


def test_fra_reset_clears_chunk_and_force_history():
    policy = _Frapolicy()
    broker = action_chunk_broker.ForceReflexActionChunkBroker(
        policy,
        action_horizon=3,
        arm_joint_dim=2,
        force_dim=1,
        force_history_len=3,
    )
    broker.infer(_observation(1.0))
    reset_result = broker.infer(_observation(5.0, reset=True))

    assert policy.plan_calls == 2
    assert reset_result["fra"]["chunk_index"] == 0
    np.testing.assert_allclose(policy.last_reflex_obs["force_history_global"], np.full((3, 1), 5.0))


def test_fra_requires_valid_joint_limits():
    with pytest.raises(ValueError, match="provided together"):
        action_chunk_broker.ForceReflexActionChunkBroker(
            _Frapolicy(),
            action_horizon=3,
            arm_joint_dim=2,
            force_dim=1,
            joint_lower_limits=[-1.0, -1.0],
        )
