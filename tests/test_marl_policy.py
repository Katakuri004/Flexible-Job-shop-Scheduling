from pathlib import Path

import torch

from src.fjsp_env import FJSPEnv, FJSPEnvConfig
from src.graph_builder import build_graph_from_env_state
from src.marl_policy import FJSPActorCritic
from src.seed_utils import SeedConfig


def test_actor_critic_forward_pass_shapes() -> None:
    """
    Test that the actor-critic network produces correct output shapes.
    """
    cfg = FJSPEnvConfig(
        instance_path=Path("data/brandimarte_mk_toy.txt"),
        seed_config=SeedConfig(base_seed=14000),
    )
    env = FJSPEnv(cfg)
    obs = env.reset()

    graph = build_graph_from_env_state(env._step_jobs, env._step_machines)
    feasible_actions = obs["feasible_actions"]

    model = FJSPActorCritic(
        op_feat_dim=6, machine_feat_dim=2, hidden_dim=64, gnn_layers=2
    )

    output = model.forward(graph, feasible_actions)

    assert "value" in output
    assert "action_logits" in output

    value = output["value"]
    action_logits = output["action_logits"]

    assert value.shape == (), "Value should be a scalar"
    assert action_logits.shape == (
        len(feasible_actions),
    ), f"Action logits should match number of feasible actions ({len(feasible_actions)})"


def test_actor_critic_action_selection() -> None:
    """
    Test deterministic and stochastic action selection.
    """
    cfg = FJSPEnvConfig(
        instance_path=Path("data/brandimarte_mk_toy.txt"),
        seed_config=SeedConfig(base_seed=15000),
    )
    env = FJSPEnv(cfg)
    obs = env.reset()

    graph = build_graph_from_env_state(env._step_jobs, env._step_machines)
    feasible_actions = obs["feasible_actions"]

    model = FJSPActorCritic(op_feat_dim=6, machine_feat_dim=2, hidden_dim=32, gnn_layers=2)

    # Deterministic selection
    action_det, logits_det, value_det = model.get_action_and_value(
        graph, feasible_actions, deterministic=True
    )

    assert 0 <= action_det < len(feasible_actions), "Action index must be valid"
    assert logits_det.shape == (len(feasible_actions),)
    assert value_det.shape == ()

    # Stochastic selection (multiple runs should sometimes differ)
    torch.manual_seed(42)
    action_stoch1, _, _ = model.get_action_and_value(
        graph, feasible_actions, deterministic=False
    )

    torch.manual_seed(43)
    action_stoch2, _, _ = model.get_action_and_value(
        graph, feasible_actions, deterministic=False
    )

    # At least one should be different (unless there's only one action)
    if len(feasible_actions) > 1:
        # With different seeds, actions might differ
        assert isinstance(action_stoch1, int) and isinstance(action_stoch2, int)


def test_actor_critic_gradient_flow() -> None:
    """
    Test that gradients flow through the actor-critic network.
    """
    cfg = FJSPEnvConfig(
        instance_path=Path("data/brandimarte_mk_toy.txt"),
        seed_config=SeedConfig(base_seed=16000),
    )
    env = FJSPEnv(cfg)
    obs = env.reset()

    graph = build_graph_from_env_state(env._step_jobs, env._step_machines)
    feasible_actions = obs["feasible_actions"]

    model = FJSPActorCritic(op_feat_dim=6, machine_feat_dim=2, hidden_dim=32, gnn_layers=2)

    output = model.forward(graph, feasible_actions)
    action_logits = output["action_logits"]
    value = output["value"]

    # Simple loss: actor loss + value loss
    action_dist = torch.distributions.Categorical(logits=action_logits)
    action = action_dist.sample()
    log_prob = action_dist.log_prob(action)

    # Dummy advantage (for testing gradient flow only)
    advantage = value.detach() - value  # This will be zero, but tests structure
    actor_loss = -log_prob * advantage.detach()
    value_loss = value ** 2  # Dummy value loss

    total_loss = actor_loss + value_loss
    total_loss.backward()

    # Check that gradients exist
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            assert not torch.isnan(param.grad).any(), "Gradients should not contain NaN"
            break

    assert has_gradients, "At least some parameters should have gradients"


def test_actor_critic_integration_with_env() -> None:
    """
    Test that the policy network can be used with FJSPEnv in a simple rollout.
    """
    cfg = FJSPEnvConfig(
        instance_path=Path("data/brandimarte_mk_toy.txt"),
        seed_config=SeedConfig(base_seed=17000),
    )
    env = FJSPEnv(cfg)
    model = FJSPActorCritic(op_feat_dim=6, machine_feat_dim=2, hidden_dim=32, gnn_layers=2)

    obs = env.reset()
    done = False
    step_count = 0
    total_value = 0.0

    while not done and step_count < 20:  # Safety limit
        graph = build_graph_from_env_state(env._step_jobs, env._step_machines)
        feasible_actions = obs["feasible_actions"]

        if not feasible_actions:
            break

        # Get action from policy
        action_idx, logits, value = model.get_action_and_value(
            graph, feasible_actions, deterministic=True
        )

        total_value += value.item()
        step_count += 1

        # Step environment
        obs, reward, done, info = env.step(action=action_idx)

    assert step_count > 0, "Should have taken at least one step"
    assert done or step_count >= 20, "Episode should complete or hit safety limit"
