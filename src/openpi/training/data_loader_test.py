import dataclasses
from typing import ClassVar

import jax
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class _FakeLeRobotMetadata:
    repo_id = "fake/repo"
    total_episodes = 50
    info: ClassVar[dict] = {"splits": {"train": "0:40", "test": "40:50"}}


def test_episodes_from_lerobot_split():
    assert _data_loader._episodes_from_lerobot_split(_FakeLeRobotMetadata(), "train") == list(range(40))  # noqa: SLF001
    assert _data_loader._episodes_from_lerobot_split(_FakeLeRobotMetadata(), "test") == list(range(40, 50))  # noqa: SLF001


def test_patch_lerobot_episode_data_index_for_nonzero_episode_ids():
    class FakeDataset:
        def __init__(self):
            self.episodes = list(range(40, 50))
            self.episode_data_index = {
                "from": torch.arange(0, 100, 10),
                "to": torch.arange(10, 110, 10),
            }

    dataset = FakeDataset()
    _data_loader._patch_lerobot_episode_data_index(dataset)  # noqa: SLF001

    assert dataset.episode_data_index["from"][40] == 0
    assert dataset.episode_data_index["to"][49] == 100


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
