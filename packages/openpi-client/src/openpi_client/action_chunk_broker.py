import math
from typing import Dict, Optional, Sequence, Union

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


def normalized_force_residual(
    observed_force: np.ndarray,
    predicted_force: np.ndarray,
    force_scale: np.ndarray,
) -> np.ndarray:
    """Compute a dimensionless, per-step force residual.

    The last axis is the force axis. ``force_scale`` may contain one scale per
    force axis or one scale per horizon step and force axis.
    """
    observed_force = np.asarray(observed_force, dtype=np.float64)
    predicted_force = np.asarray(predicted_force, dtype=np.float64)
    force_scale = np.asarray(force_scale, dtype=np.float64)
    if observed_force.ndim == 0 or force_scale.ndim == 0:
        raise ValueError("Force arrays must have a force axis.")
    if observed_force.shape != predicted_force.shape:
        raise ValueError(
            f"Observed and predicted force must have the same shape; got "
            f"{observed_force.shape} and {predicted_force.shape}."
        )
    if force_scale.shape[-1] != observed_force.shape[-1]:
        raise ValueError(f"force_scale has {force_scale.shape[-1]} axes but force has {observed_force.shape[-1]}.")
    if not np.all(np.isfinite(observed_force)) or not np.all(np.isfinite(predicted_force)):
        raise ValueError("Observed and predicted force must be finite.")
    if not np.all(np.isfinite(force_scale)) or np.any(force_scale <= 0):
        raise ValueError("force_scale must contain finite positive values.")
    return np.mean(np.square((observed_force - predicted_force) / force_scale), axis=-1)


def estimate_force_scale(
    observed_force: np.ndarray,
    predicted_force: np.ndarray,
    *,
    minimum_scale: float = 1e-3,
) -> np.ndarray:
    """Estimate one RMS residual scale per force axis from a training split."""
    observed_force = np.asarray(observed_force, dtype=np.float64)
    predicted_force = np.asarray(predicted_force, dtype=np.float64)
    if observed_force.shape != predicted_force.shape or observed_force.ndim < 2:
        raise ValueError("Force arrays must have the same shape [..., force_dim].")
    if minimum_scale <= 0:
        raise ValueError("minimum_scale must be positive.")
    residual = observed_force - predicted_force
    if not np.all(np.isfinite(residual)):
        raise ValueError("Force arrays must be finite.")
    reduce_axes = tuple(range(residual.ndim - 1))
    return np.maximum(np.sqrt(np.mean(np.square(residual), axis=reduce_axes)), minimum_scale)


def calibrate_corace_threshold(
    observed_force: np.ndarray,
    predicted_force: np.ndarray,
    force_scale: np.ndarray,
    *,
    alpha: float = 0.05,
) -> float:
    """Calibrate a chunk-level split-conformal trigger threshold.

    Inputs have shape ``[num_chunks, horizon, force_dim]``. Each calibration
    score is the maximum normalized residual within one nominal chunk, so the
    returned threshold controls the probability of any false interruption in a
    chunk under the usual exchangeability assumption.
    """
    observed_force = np.asarray(observed_force)
    predicted_force = np.asarray(predicted_force)
    if observed_force.shape != predicted_force.shape or observed_force.ndim != 3:
        raise ValueError("Calibration force arrays must have shape [num_chunks, horizon, force_dim].")
    if observed_force.shape[0] == 0:
        raise ValueError("At least one calibration chunk is required.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")

    step_scores = normalized_force_residual(observed_force, predicted_force, np.asarray(force_scale))
    chunk_scores = np.max(step_scores, axis=1)
    rank = math.ceil((chunk_scores.shape[0] + 1) * (1.0 - alpha))
    if rank > chunk_scores.shape[0]:
        raise ValueError(
            f"{chunk_scores.shape[0]} calibration chunks are insufficient for alpha={alpha}; "
            f"at least {math.ceil(1.0 / alpha) - 1} are required."
        )
    return float(np.sort(chunk_scores)[rank - 1])


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0


class CoRACEActionChunkBroker(_base_policy.BasePolicy):
    """Contact-Residual Adaptive Chunk Execution.

    After each issued action, the broker compares the newly measured force with
    the force predicted for that exact action index. A calibrated residual
    violation discards the unexecuted suffix and requests a new chunk from the
    latest observation.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        residual_threshold: float,
        force_scale: Union[np.ndarray, Sequence[float]],
        *,
        force_key: str = "force",
        force_prediction_key: str = "force_prediction",
        measurement_delay_steps: int = 0,
    ):
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        if not np.isfinite(residual_threshold) or residual_threshold < 0:
            raise ValueError("residual_threshold must be finite and non-negative.")
        if measurement_delay_steps < 0:
            raise ValueError("measurement_delay_steps must be non-negative.")

        force_scale_array = np.asarray(force_scale, dtype=np.float64)
        if force_scale_array.ndim not in (1, 2):
            raise ValueError("force_scale must have shape [force_dim] or [horizon, force_dim].")
        if not np.all(np.isfinite(force_scale_array)) or np.any(force_scale_array <= 0):
            raise ValueError("force_scale must contain finite positive values.")
        if force_scale_array.ndim == 2 and force_scale_array.shape[0] < action_horizon:
            raise ValueError("A horizon-dependent force_scale must cover action_horizon.")

        self._policy = policy
        self._action_horizon = action_horizon
        self._residual_threshold = float(residual_threshold)
        self._force_scale = force_scale_array
        self._force_key = force_key
        self._force_prediction_key = force_prediction_key
        self._measurement_delay_steps = int(measurement_delay_steps)

        self._cur_step = 0
        self._last_results: Optional[Dict] = None
        self._replan_count = 0

    def _request_chunk(self, obs: Dict) -> None:
        results = self._policy.infer(obs)
        if "actions" not in results:
            raise KeyError("CoRACE requires policy output key 'actions'.")
        if self._force_prediction_key not in results:
            raise KeyError(
                f"CoRACE requires policy output key {self._force_prediction_key!r}; "
                "start the policy server with return_force_prediction=True."
            )

        actions = np.asarray(results["actions"])
        force_prediction = np.asarray(results[self._force_prediction_key])
        if actions.ndim < 2 or actions.shape[0] < self._action_horizon:
            raise ValueError("The returned action chunk does not cover action_horizon.")
        if force_prediction.ndim != 2 or force_prediction.shape[0] < self._action_horizon:
            raise ValueError("force_prediction must have shape [horizon, force_dim] and cover action_horizon.")
        if force_prediction.shape[-1] != self._force_scale.shape[-1]:
            raise ValueError("force_prediction and force_scale must have the same force dimension.")

        self._last_results = results
        self._cur_step = 0

    def _aligned_residual(self, obs: Dict) -> tuple:
        prediction_index = self._cur_step - 1 - self._measurement_delay_steps
        if prediction_index < 0:
            return None, None
        if self._force_key not in obs:
            raise KeyError(f"CoRACE requires observation key {self._force_key!r}.")
        assert self._last_results is not None
        observed_force = np.asarray(obs[self._force_key])
        predicted_force = np.asarray(self._last_results[self._force_prediction_key])[prediction_index]
        if observed_force.ndim != 1:
            raise ValueError(f"Observation {self._force_key!r} must have shape [force_dim].")
        force_scale = self._force_scale[prediction_index] if self._force_scale.ndim == 2 else self._force_scale
        score = normalized_force_residual(observed_force, predicted_force, force_scale)
        return prediction_index, float(score)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        reset = bool(np.asarray(obs.get("reset", False)))
        if reset:
            self._last_results = None
            self._cur_step = 0
            self._replan_count = 0

        triggered = False
        prediction_index = None
        residual_score = None
        if self._last_results is not None:
            prediction_index, residual_score = self._aligned_residual(obs)
            if residual_score is not None and residual_score > self._residual_threshold:
                self._last_results = None
                self._cur_step = 0
                self._replan_count += 1
                triggered = True

        if self._last_results is None:
            self._request_chunk(obs)

        assert self._last_results is not None
        results = {
            key: (np.asarray(value)[self._cur_step] if key == "actions" else value)
            for key, value in self._last_results.items()
            if key != self._force_prediction_key
        }
        results["corace"] = {
            "triggered": triggered,
            "residual_score": residual_score,
            "residual_threshold": self._residual_threshold,
            "prediction_index": prediction_index,
            "replan_count": self._replan_count,
        }

        self._cur_step += 1
        if self._cur_step >= self._action_horizon:
            self._last_results = None
            self._cur_step = 0
        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
        self._replan_count = 0
