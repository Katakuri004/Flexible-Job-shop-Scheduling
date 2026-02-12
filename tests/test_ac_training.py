from pathlib import Path

import torch

from src.ac_training import TrainingConfig, run_training
from src.seed_utils import SeedConfig


def test_ac_training_short_run() -> None:
    """
    Smoke test: run a very short training loop and check metric shapes.
    """
    cfg = TrainingConfig(
        instance_path=Path("data/brandimarte_mk_toy.txt"),
        seed_config=SeedConfig(base_seed=18000),
        num_episodes=5,
        max_steps_per_episode=32,
        log_interval=5,
        device="cpu",
    )

    metrics = run_training(cfg)

    # Basic structural checks
    assert len(metrics["episode_rewards"]) == cfg.num_episodes
    assert len(metrics["episode_makespans"]) == cfg.num_episodes
    assert len(metrics["episode_lengths"]) == cfg.num_episodes
    assert len(metrics["episode_values"]) == cfg.num_episodes
    assert len(metrics["episode_advantages"]) == cfg.num_episodes
    assert len(metrics["losses"]) == cfg.num_episodes
    assert len(metrics["value_losses"]) == cfg.num_episodes

    # Numerical sanity: rewards, makespans, and losses should be finite
    for key in ["episode_rewards", "episode_makespans", "losses", "value_losses"]:
        tensor = torch.tensor(metrics[key], dtype=torch.float32)
        assert torch.isfinite(tensor).all(), f"Non-finite values found in {key}"

