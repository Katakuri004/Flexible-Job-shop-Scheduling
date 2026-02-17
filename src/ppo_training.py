from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .fjsp_env import FJSPEnv, FJSPEnvConfig
from .graph_builder import build_graph_from_env_state
from .marl_policy import FJSPActorCritic
from .seed_utils import SeedConfig, set_global_seeds


@dataclass
class PPOConfig:
    """
    Configuration for PPO training with GAE on the FJSP environment.
    """

    instance_path: Path
    seed_config: SeedConfig

    # Optimization hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0

    # Training schedule
    num_epochs: int = 50
    episodes_per_epoch: int = 4
    max_steps_per_episode: int = 64
    ppo_update_iters: int = 4  # number of passes over collected batch

    # Device
    device: str = "cpu"


def _ensure_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE advantages and returns for a trajectory.

    rewards, values, dones are 1D tensors of same length.
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=torch.float32, device=rewards.device)
    last_adv = 0.0
    last_value = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t].item())
        delta = rewards[t] + gamma * last_value * mask - values[t]
        last_adv = delta + gamma * lam * last_adv * mask
        advantages[t] = last_adv
        last_value = values[t].item()

    returns = advantages + values
    return advantages, returns


def collect_ppo_batch(
    env: FJSPEnv,
    policy: FJSPActorCritic,
    config: PPOConfig,
) -> Dict[str, torch.Tensor]:
    """
    Collect a batch of trajectories for PPO.
    """
    device = _ensure_device(config.device)

    # For PPO we keep per-step graphs and feasible action lists so that
    # we can recompute logits/values under the current policy during updates.
    graphs: List[Any] = []
    feasibles: List[List[Tuple[int, int, int]]] = []
    actions: List[int] = []
    log_probs: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    rewards: List[float] = []
    dones: List[float] = []

    episode_makespans: List[float] = []

    episodes_collected = 0
    while episodes_collected < config.episodes_per_epoch:
        obs = env.reset()
        done = False
        step = 0

        while not done and step < config.max_steps_per_episode:
            assert env._step_jobs is not None
            assert env._step_machines is not None

            graph = build_graph_from_env_state(env._step_jobs, env._step_machines)
            feasible = obs["feasible_actions"]
            assert feasible, "PPO: no feasible actions."

            out = policy.forward(graph, feasible)
            logits = out["action_logits"]
            value = out["value"]

            assert torch.isfinite(logits).all()
            assert torch.isfinite(value).all()

            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            a_idx = int(action.item())
            assert 0 <= a_idx < len(feasible)

            next_obs, reward, done, info = env.step(a_idx)

            graphs.append(graph)
            feasibles.append(list(feasible))
            actions.append(a_idx)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(float(done))

            obs = next_obs
            step += 1

            if done:
                assert env.last_makespan is not None
                episode_makespans.append(float(env.last_makespan))

        episodes_collected += 1

    # Convert to tensors
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    log_probs_t = torch.stack(log_probs).to(device).view(-1)
    values_t = torch.stack(values).to(device).view(-1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

    advantages_t, returns_t = _compute_gae(
        rewards=rewards_t,
        values=values_t,
        dones=dones_t,
        gamma=config.gamma,
        lam=config.gae_lambda,
    )

    # Normalize advantages
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    batch = {
        "graphs": graphs,
        "feasibles": feasibles,
        "actions": actions_t,
        "old_log_probs": log_probs_t,
        "old_values": values_t,
        "advantages": advantages_t,
        "returns": returns_t,
    }
    stats = {
        "episode_makespans": episode_makespans,
    }
    return {"batch": batch, "stats": stats}


def ppo_update(
    policy: FJSPActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    config: PPOConfig,
) -> Dict[str, float]:
    """
    Run multiple PPO update iterations on a collected batch.
    """
    device = _ensure_device(config.device)
    graphs = batch["graphs"]
    feasibles = batch["feasibles"]
    actions = batch["actions"]
    old_log_probs = batch["old_log_probs"].detach()
    old_values = batch["old_values"].detach()
    advantages = batch["advantages"].detach()
    returns = batch["returns"].detach()

    num_steps = actions.shape[0]

    total_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0

    for _ in range(config.ppo_update_iters):
        policy_losses: List[torch.Tensor] = []
        value_losses: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []

        for t in range(num_steps):
            graph = graphs[t]
            feasible = feasibles[t]
            action_idx = actions[t].to(device)

            out = policy.forward(graph, feasible)
            logits = out["action_logits"]
            value_pred = out["value"]

            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action_idx)

            ratio = (log_prob - old_log_probs[t]).exp()
            surr1 = ratio * advantages[t]
            surr2 = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * advantages[t]
            policy_loss = -torch.min(surr1, surr2)

            v_loss = nn.functional.mse_loss(value_pred, returns[t])
            entropy_estimate = -log_prob

            policy_losses.append(policy_loss)
            value_losses.append(v_loss)
            entropies.append(entropy_estimate)

        policy_loss_mean = torch.stack(policy_losses).mean()
        value_loss_mean = torch.stack(value_losses).mean()
        entropy_mean = torch.stack(entropies).mean()

        loss = policy_loss_mean + config.value_coef * value_loss_mean - config.entropy_coef * entropy_mean

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
        assert torch.isfinite(grad_norm)
        optimizer.step()

        total_loss += float(loss.item())
        total_value_loss += float(value_loss_mean.item())
        total_entropy += float(entropy_mean.item())

    avg_loss = total_loss / float(config.ppo_update_iters)
    avg_value_loss = total_value_loss / float(config.ppo_update_iters)
    avg_entropy = total_entropy / float(config.ppo_update_iters)

    return {
        "loss": avg_loss,
        "value_loss": avg_value_loss,
        "entropy": avg_entropy,
        "num_steps": float(num_steps),
    }


def run_ppo_training(config: PPOConfig) -> Tuple[Dict[str, List[float]], FJSPActorCritic]:
    """
    High-level PPO training loop. Returns (metrics_dict, trained_policy) for evaluation.
    """
    set_global_seeds(config.seed_config)
    device = _ensure_device(config.device)

    env_cfg = FJSPEnvConfig(
        instance_path=config.instance_path,
        seed_config=config.seed_config,
    )
    env = FJSPEnv(env_cfg)

    policy = FJSPActorCritic()
    policy.to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    epoch_losses: List[float] = []
    epoch_value_losses: List[float] = []
    epoch_entropies: List[float] = []
    epoch_mean_makespans: List[float] = []

    for epoch in range(1, config.num_epochs + 1):
        collected = collect_ppo_batch(env, policy, config)
        batch = collected["batch"]
        stats = collected["stats"]

        update_stats = ppo_update(policy, optimizer, batch, config)

        # Aggregate episode stats
        makespans = np.array(stats["episode_makespans"], dtype=float)
        mean_makespan = float(makespans.mean()) if makespans.size > 0 else float("nan")

        epoch_losses.append(update_stats["loss"])
        epoch_value_losses.append(update_stats["value_loss"])
        epoch_entropies.append(update_stats["entropy"])
        epoch_mean_makespans.append(mean_makespan)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[PPO Epoch {epoch}/{config.num_epochs}] "
                f"MeanMakespan={mean_makespan:.2f}, "
                f"Loss={update_stats['loss']:.4f}, "
                f"ValueLoss={update_stats['value_loss']:.4f}"
            )

    metrics = {
        "epoch_losses": epoch_losses,
        "epoch_value_losses": epoch_value_losses,
        "epoch_entropies": epoch_entropies,
        "epoch_mean_makespans": epoch_mean_makespans,
    }
    return metrics, policy


__all__ = ["PPOConfig", "collect_ppo_batch", "run_ppo_training"]

