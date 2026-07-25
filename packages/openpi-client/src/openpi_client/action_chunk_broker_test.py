import numpy as np
import pytest

from openpi_client import action_chunk_broker
from openpi_client import base_policy


class _ChunkPolicy(base_policy.BasePolicy):
    def __init__(self, force_predictions):
        self.force_predictions = [np.asarray(x, dtype=np.float64) for x in force_predictions]
        self.calls = 0

    def infer(self, obs):
        chunk_index = self.calls
        self.calls += 1
        return {
            "actions": np.full((3, 2), chunk_index, dtype=np.float64),
            "force_prediction": self.force_predictions[chunk_index],
        }


def test_normalized_force_residual():
    score = action_chunk_broker.normalized_force_residual(
        np.array([2.0, 6.0]),
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
    )
    np.testing.assert_allclose(score, 2.5)


def test_calibrate_corace_threshold_uses_chunk_max_and_conformal_rank():
    predicted = np.zeros((4, 2, 1))
    observed = np.array([[[1.0], [0.0]], [[2.0], [0.0]], [[3.0], [0.0]], [[4.0], [0.0]]])

    threshold = action_chunk_broker.calibrate_corace_threshold(
        observed,
        predicted,
        np.ones(1),
        alpha=0.4,
    )

    assert threshold == 9.0


def test_calibrate_corace_threshold_requires_enough_chunks():
    observed = np.zeros((4, 2, 1))

    with pytest.raises(ValueError, match="insufficient"):
        action_chunk_broker.calibrate_corace_threshold(
            observed,
            np.zeros_like(observed),
            np.ones(1),
            alpha=0.05,
        )


def test_corace_keeps_chunk_when_aligned_residual_is_small():
    predictions = [np.array([[1.0], [2.0], [3.0]])]
    policy = _ChunkPolicy(predictions)
    broker = action_chunk_broker.CoRACEActionChunkBroker(
        policy,
        action_horizon=3,
        residual_threshold=1.0,
        force_scale=[1.0],
    )

    first = broker.infer({"force": np.array([0.0])})
    second = broker.infer({"force": np.array([1.5])})

    assert policy.calls == 1
    assert first["corace"]["residual_score"] is None
    assert second["corace"]["prediction_index"] == 0
    assert second["corace"]["residual_score"] == pytest.approx(0.25)
    assert not second["corace"]["triggered"]


def test_corace_discards_suffix_and_replans_on_residual_violation():
    predictions = [
        np.array([[1.0], [2.0], [3.0]]),
        np.array([[10.0], [11.0], [12.0]]),
    ]
    policy = _ChunkPolicy(predictions)
    broker = action_chunk_broker.CoRACEActionChunkBroker(
        policy,
        action_horizon=3,
        residual_threshold=1.0,
        force_scale=[1.0],
    )

    first = broker.infer({"force": np.array([0.0])})
    replanned = broker.infer({"force": np.array([4.0])})

    np.testing.assert_array_equal(first["actions"], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(replanned["actions"], np.array([1.0, 1.0]))
    assert policy.calls == 2
    assert replanned["corace"]["triggered"]
    assert replanned["corace"]["prediction_index"] == 0
    assert replanned["corace"]["replan_count"] == 1


def test_corace_accounts_for_measurement_delay():
    policy = _ChunkPolicy([np.array([[1.0], [2.0], [3.0]])])
    broker = action_chunk_broker.CoRACEActionChunkBroker(
        policy,
        action_horizon=3,
        residual_threshold=1.0,
        force_scale=[1.0],
        measurement_delay_steps=1,
    )

    broker.infer({"force": np.array([0.0])})
    delayed = broker.infer({"force": np.array([100.0])})
    aligned = broker.infer({"force": np.array([1.0])})

    assert delayed["corace"]["residual_score"] is None
    assert aligned["corace"]["prediction_index"] == 0
    assert aligned["corace"]["residual_score"] == 0.0


def test_corace_requires_force_prediction():
    class _MissingPredictionPolicy(base_policy.BasePolicy):
        def infer(self, obs):
            return {"actions": np.zeros((3, 2))}

    broker = action_chunk_broker.CoRACEActionChunkBroker(
        _MissingPredictionPolicy(),
        action_horizon=3,
        residual_threshold=1.0,
        force_scale=[1.0],
    )

    with pytest.raises(KeyError, match="return_force_prediction"):
        broker.infer({"force": np.array([0.0])})
