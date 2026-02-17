from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .fjsp_env import FJSPEnv, FJSPEnvConfig
from .graph_builder import build_graph_from_env_state
from .marl_policy import FJSPActorCritic
from .seed_utils import SeedConfig, set_global_seeds


@dataclass
class TrainingConfig:
    """
    Configuration for simple actor-critic training on the FJSP environment.
    """

    instance_path: Path
    seed_config: SeedConfig

    # Optimization hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0

    # Training schedule
    num_episodes: int = 200
    max_steps_per_episode: int = 64
    log_interval: int = 10

    # Device
    device: str = "cpu"

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_prefix: str = "ac_fjsp"


@dataclass
class EpisodeMetrics:
    total_reward: float
    final_makespan: float
    length: int
    mean_value: float
    mean_advantage: float


def _ensure_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _discount_returns(rewards: List[float], gamma: float) -> List[float]:
    returns: List[float] = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.append(g)
    returns.reverse()
    return returns


def run_episode(
    env: FJSPEnv,
    policy: FJSPActorCritic,
    config: TrainingConfig,
    device: torch.device,
) -> Tuple[EpisodeMetrics, Dict[str, Any]]:
    """
    Collect one episode of experience using the current policy.

    Returns episode-level metrics and a dict containing tensors
    needed for the policy update.
    """

    obs = env.reset()
    done = False

    log_probs: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    rewards: List[float] = []

    step = 0
    last_makespan: float = 0.0

    while not done and step < config.max_steps_per_episode:
        assert env._step_jobs is not None
        assert env._step_machines is not None

        graph = build_graph_from_env_state(env._step_jobs, env._step_machines)
        feasible_actions = obs["feasible_actions"]

        # Invariant: there should always be at least one feasible action
        assert len(feasible_actions) > 0, "No feasible actions available during episode."

        # Policy action (keep graph for backpropagation)
        # forward uses numpy arrays; policy will move tensors to device internally via encode_state_with_gnn
        output = policy.forward(graph, feasible_actions)
        logits = output["action_logits"]
        value = output["value"]

        # Numerical sanity checks
        assert torch.isfinite(logits).all(), "Non-finite logits encountered."
        assert torch.isfinite(value).all(), "Non-finite value encountered."

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        action_idx = int(action.item())
        assert 0 <= action_idx < len(feasible_actions), "Policy chose invalid action index."

        obs, reward, done, info = env.step(action_idx)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)

        if done:
            # When done, env.last_makespan should be set
            assert env.last_makespan is not None
            last_makespan = float(env.last_makespan)

        step += 1

    # Ensure episode termination
    assert done or step >= config.max_steps_per_episode, "Episode ended prematurely without done flag."

    # Compute returns and advantages
    returns = _discount_returns(rewards, config.gamma)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    values_t = torch.stack(values).to(device).view(-1)
    log_probs_t = torch.stack(log_probs).to(device).view(-1)

    # Simple advantage: return - value
    advantages = returns_t - values_t

    # Metrics
    total_reward = float(sum(rewards))
    mean_value = float(values_t.mean().item())
    mean_advantage = float(advantages.mean().item())

    metrics = EpisodeMetrics(
        total_reward=total_reward,
        final_makespan=last_makespan,
        length=step,
        mean_value=mean_value,
        mean_advantage=mean_advantage,
    )

    trajectory = {
        "log_probs": log_probs_t,
        "values": values_t,
        "returns": returns_t,
        "advantages": advantages,
    }
    return metrics, trajectory


def update_policy(
    policy: FJSPActorCritic,
    optimizer: torch.optim.Optimizer,
    trajectory: Dict[str, torch.Tensor],
    config: TrainingConfig,
) -> Dict[str, float]:
    """
    Perform a single actor-critic update using collected trajectory.
    """
    log_probs = trajectory["log_probs"]
    values = trajectory["values"]
    returns = trajectory["returns"]
    advantages = trajectory["advantages"]

    # Normalize advantages for stability
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Actor loss (policy gradient)
    actor_loss = -(log_probs * advantages.detach()).mean()

    # Critic loss (value function regression)
    value_loss = nn.functional.mse_loss(values, returns)

    # Entropy bonus (encourage exploration)
    with torch.no_grad():
        entropy_estimate = -log_probs.mean()

    loss = actor_loss + config.value_coef * value_loss - config.entropy_coef * entropy_estimate

    # Sanity-check loss: skip update if non-finite
    if not torch.isfinite(loss):
        return {
            "loss": float("nan"),
            "actor_loss": float(actor_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_estimate.item()),
            "grad_norm": float("nan"),
        }

    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping and sanity checks
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
    if not torch.isfinite(grad_norm):
        return {
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_estimate.item()),
            "grad_norm": float("nan"),
        }

    optimizer.step()

    return {
        "loss": float(loss.item()),
        "actor_loss": float(actor_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy_estimate.item()),
        "grad_norm": float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
    }


def save_checkpoint(
    policy: FJSPActorCritic,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    episode: int,
) -> Path:
    """
    Save a training checkpoint (policy + optimizer state).
    """
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.checkpoint_dir / f"{config.checkpoint_prefix}_ep{episode}.pt"
    torch.save(
        {
            "episode": episode,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        ckpt_path,
    )
    return ckpt_path


def load_checkpoint(
    path: Path,
    policy: FJSPActorCritic,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint from disk.

    Returns a dict with 'episode', 'config', and optionally 'optimizer_state_dict'.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return {
        "episode": ckpt["episode"],
        "config": ckpt.get("config"),
        "optimizer_state_dict": ckpt.get("optimizer_state_dict"),
    }


def run_training(config: TrainingConfig) -> Dict[str, List[float]]:
    """
    Run a full training loop with the simple actor-critic algorithm.

    Returns logged metrics for downstream analysis/plotting.
    """
    # Global seed discipline
    set_global_seeds(config.seed_config)
    print(
        f"[Seeds] python={config.seed_config.resolved_python_seed()}, "
        f"numpy={config.seed_config.resolved_numpy_seed()}, "
        f"torch={config.seed_config.resolved_torch_seed()}"
    )

    device = _ensure_device(config.device)

    env_config = FJSPEnvConfig(
        instance_path=config.instance_path,
        seed_config=config.seed_config,
    )
    env = FJSPEnv(env_config)

    policy = FJSPActorCritic()
    policy.to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    # Logging buffers
    episode_rewards: List[float] = []
    episode_makespans: List[float] = []
    episode_lengths: List[int] = []
    episode_values: List[float] = []
    episode_advantages: List[float] = []
    losses: List[float] = []
    value_losses: List[float] = []

    for ep in range(1, config.num_episodes + 1):
        metrics, trajectory = run_episode(env, policy, config, device)
        stats = update_policy(policy, optimizer, trajectory, config)

        # Log (use 0.0 placeholder when update was skipped due to non-finite loss)
        episode_rewards.append(metrics.total_reward)
        episode_makespans.append(metrics.final_makespan)
        episode_lengths.append(metrics.length)
        episode_values.append(metrics.mean_value)
        episode_advantages.append(metrics.mean_advantage)
        losses.append(stats["loss"] if np.isfinite(stats["loss"]) else 0.0)
        value_losses.append(stats["value_loss"] if np.isfinite(stats["value_loss"]) else 0.0)

        # Basic numerical sanity checks
        if not np.isfinite(metrics.total_reward) or not np.isfinite(metrics.final_makespan):
            print(f"[Episode {ep}] WARNING: Non-finite reward or makespan.")
        if not np.isfinite(stats["loss"]):
            print(f"[Episode {ep}] WARNING: Non-finite loss (update was skipped).")

        if ep % config.log_interval == 0 or ep == 1:
            print(
                f"[Episode {ep}/{config.num_episodes}] "
                f"Reward={metrics.total_reward:.2f}, "
                f"Makespan={metrics.final_makespan:.2f}, "
                f"Len={metrics.length}, "
                f"Loss={stats['loss']:.4f}, "
                f"ValueLoss={stats['value_loss']:.4f}"
            )
            save_checkpoint(policy, optimizer, config, ep)

    return {
        "episode_rewards": episode_rewards,
        "episode_makespans": episode_makespans,
        "episode_lengths": episode_lengths,
        "episode_values": episode_values,
        "episode_advantages": episode_advantages,
        "losses": losses,
        "value_losses": value_losses,
    }


__all__ = [
    "TrainingConfig",
    "EpisodeMetrics",
    "run_episode",
    "update_policy",
    "save_checkpoint",
    "load_checkpoint",
    "run_training",
]

