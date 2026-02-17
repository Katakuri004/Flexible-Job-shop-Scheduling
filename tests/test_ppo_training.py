from pathlib import Path

import torch

from src.ppo_training import PPOConfig, run_ppo_training
from src.seed_utils import SeedConfig


def test_ppo_training_smoke() -> None:
    """
    Smoke test for PPO training loop on the toy instance.
    """
    cfg = PPOConfig(
        instance_path=Path("data/brandimarte_mk_toy.txt"),
        seed_config=SeedConfig(base_seed=19000),
        num_epochs=5,
        episodes_per_epoch=2,
        max_steps_per_episode=32,
        ppo_update_iters=2,
        device="cpu",
    )

    metrics, _ = run_ppo_training(cfg)

    assert len(metrics["epoch_losses"]) == cfg.num_epochs
    assert len(metrics["epoch_value_losses"]) == cfg.num_epochs
    assert len(metrics["epoch_entropies"]) == cfg.num_epochs
    assert len(metrics["epoch_mean_makespans"]) == cfg.num_epochs

    # numerical sanity
    for key in ["epoch_losses", "epoch_value_losses", "epoch_entropies", "epoch_mean_makespans"]:
        t = torch.tensor(metrics[key], dtype=torch.float32)
        assert torch.isfinite(t).all(), f"Non-finite values in {key}"

