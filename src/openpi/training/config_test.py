import numpy as np

from openpi.models import model as _model
from openpi.training import config as _config


def test_flexiv_fra_transform_builds_stale_chunk_feedback_sequences():
    raw_state = np.arange(7 * 14, dtype=np.float32).reshape(7, 14)
    data = {
        "observation/state": raw_state,
        "observation/image": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "actions": np.zeros((3, 8), dtype=np.float32),
        "prompt": "insert",
    }
    transform = _config.FlexivPumpInputs(
        model_type=_model.ModelType.PI0,
        force_history_global_len=4,
        force_history_local_len=2,
        fra_training=True,
    )

    result = transform(data)

    np.testing.assert_array_equal(result["state"], raw_state[3, :8])
    np.testing.assert_array_equal(result["force"], raw_state[3, 8:14])
    np.testing.assert_array_equal(result["fra_joint_states"], raw_state[3:6, :7])
    assert result["fra_force_history"].shape == (3, 4, 6)
    np.testing.assert_array_equal(result["fra_force_history"][0], raw_state[0:4, 8:14])
    np.testing.assert_array_equal(result["fra_force_history"][2], raw_state[2:6, 8:14])


def test_flexiv_output_keeps_gripper_and_drops_model_padding():
    outputs = _config.FlexivPumpOutputs(action_dim=8)({"actions": np.ones((2, 32), dtype=np.float32)})

    assert outputs["actions"].shape == (2, 8)


def test_flexiv_fra_stage_two_config_freezes_base_and_loads_stage_one():
    config = _config.get_config("pi0_flexiv_pump_1bottle_inputForce_lora_fra_stage2")

    assert config.model.fra_enabled
    assert config.model.fra_training_only
    assert config.model.fra_arm_dim == 7
    assert config.num_train_steps == 10_000
    assert config.lr_schedule.peak_lr == 1e-4
    assert config.weight_loader.params_path.endswith("/params")
