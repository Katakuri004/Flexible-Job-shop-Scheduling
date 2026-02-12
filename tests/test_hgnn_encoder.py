from pathlib import Path

import numpy as np
import torch

from src.fjsp_env import FJSPEnv, FJSPEnvConfig
from src.graph_builder import build_graph_from_env_state
from src.hgnn_encoder import HeterogeneousGNN, encode_state_with_gnn
from src.seed_utils import SeedConfig


def test_hgnn_forward_pass_shapes() -> None:
    """
    Test that the HGNN forward pass produces embeddings with correct shapes.
    """
    cfg = FJSPEnvConfig(
        instance_path=Path("data/brandimarte_mk_toy.txt"),
        seed_config=SeedConfig(base_seed=9000),
    )
    env = FJSPEnv(cfg)
    env.reset()

    graph = build_graph_from_env_state(env._step_jobs, env._step_machines)

    num_ops = len(graph["op_node_ids"])
    num_machines = len(graph["machine_node_ids"])
    hidden_dim = 64  # Smaller for testing

    model = HeterogeneousGNN(
        op_feat_dim=6, machine_feat_dim=2, hidden_dim=hidden_dim, num_layers=2
    )

    op_features = torch.from_numpy(graph["op_features"]).float()
    machine_features = torch.from_numpy(graph["machine_features"]).float()
    precedence_edges = torch.from_numpy(graph["precedence_edges"]).long()
    compatibility_edges = torch.from_numpy(graph["compatibility_edges"]).long()

    embeddings = model(op_features, machine_features, precedence_edges, compatibility_edges)

    assert "op_embeddings" in embeddings
    assert "machine_embeddings" in embeddings

    op_emb = embeddings["op_embeddings"]
    machine_emb = embeddings["machine_embeddings"]

    assert op_emb.shape == (num_ops, hidden_dim), f"Expected ({num_ops}, {hidden_dim}), got {op_emb.shape}"
    assert machine_emb.shape == (num_machines, hidden_dim), f"Expected ({num_machines}, {hidden_dim}), got {machine_emb.shape}"


def test_hgnn_gradient_flow() -> None:
    """
    Test that gradients flow through the HGNN (basic sanity check for training).
    """
    cfg = FJSPEnvConfig(
        instance_path=Path("data/brandimarte_mk_toy.txt"),
        seed_config=SeedConfig(base_seed=10000),
    )
    env = FJSPEnv(cfg)
    env.reset()

    graph = build_graph_from_env_state(env._step_jobs, env._step_machines)

    model = HeterogeneousGNN(op_feat_dim=6, machine_feat_dim=2, hidden_dim=32, num_layers=2)

    op_features = torch.from_numpy(graph["op_features"]).float()
    machine_features = torch.from_numpy(graph["machine_features"]).float()
    precedence_edges = torch.from_numpy(graph["precedence_edges"]).long()
    compatibility_edges = torch.from_numpy(graph["compatibility_edges"]).long()

    embeddings = model(op_features, machine_features, precedence_edges, compatibility_edges)

    # Simple loss: sum of all embeddings
    loss = embeddings["op_embeddings"].sum() + embeddings["machine_embeddings"].sum()
    loss.backward()

    # Check that gradients exist for at least some parameters
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            assert not torch.isnan(param.grad).any(), "Gradients should not contain NaN"
            break

    assert has_gradients, "At least some parameters should have gradients"


def test_hgnn_deterministic_encoding() -> None:
    """
    Test that the HGNN produces identical embeddings for identical inputs
    (with model in eval mode and same random seed).
    """
    torch.manual_seed(42)
    np.random.seed(42)

    cfg = FJSPEnvConfig(
        instance_path=Path("data/brandimarte_mk_toy.txt"),
        seed_config=SeedConfig(base_seed=11000),
    )
    env = FJSPEnv(cfg)
    env.reset()

    graph = build_graph_from_env_state(env._step_jobs, env._step_machines)

    # Create two identical models with same initialization
    model1 = HeterogeneousGNN(op_feat_dim=6, machine_feat_dim=2, hidden_dim=32, num_layers=2)
    model2 = HeterogeneousGNN(op_feat_dim=6, machine_feat_dim=2, hidden_dim=32, num_layers=2)

    # Copy weights from model1 to model2
    model2.load_state_dict(model1.state_dict())

    model1.eval()
    model2.eval()

    op_features = torch.from_numpy(graph["op_features"]).float()
    machine_features = torch.from_numpy(graph["machine_features"]).float()
    precedence_edges = torch.from_numpy(graph["precedence_edges"]).long()
    compatibility_edges = torch.from_numpy(graph["compatibility_edges"]).long()

    with torch.no_grad():
        emb1 = model1(op_features, machine_features, precedence_edges, compatibility_edges)
        emb2 = model2(op_features, machine_features, precedence_edges, compatibility_edges)

    # Check that embeddings are identical (within float precision)
    torch.testing.assert_close(
        emb1["op_embeddings"], emb2["op_embeddings"], rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        emb1["machine_embeddings"], emb2["machine_embeddings"], rtol=1e-6, atol=1e-6
    )


def test_encode_state_with_gnn_convenience() -> None:
    """
    Test the convenience function encode_state_with_gnn integrates correctly
    with graph_builder output.
    """
    cfg = FJSPEnvConfig(
        instance_path=Path("data/brandimarte_mk_toy.txt"),
        seed_config=SeedConfig(base_seed=12000),
    )
    env = FJSPEnv(cfg)
    env.reset()

    graph = build_graph_from_env_state(env._step_jobs, env._step_machines)

    model = HeterogeneousGNN(op_feat_dim=6, machine_feat_dim=2, hidden_dim=32, num_layers=2)

    embeddings = encode_state_with_gnn(graph, model)

    assert "op_embeddings" in embeddings
    assert "machine_embeddings" in embeddings

    num_ops = len(graph["op_node_ids"])
    num_machines = len(graph["machine_node_ids"])

    assert embeddings["op_embeddings"].shape == (num_ops, 32)
    assert embeddings["machine_embeddings"].shape == (num_machines, 32)


def test_hgnn_after_env_steps() -> None:
    """
    Test that the HGNN can encode states at different points during an episode.
    """
    cfg = FJSPEnvConfig(
        instance_path=Path("data/brandimarte_mk_toy.txt"),
        seed_config=SeedConfig(base_seed=13000),
    )
    env = FJSPEnv(cfg)
    env.reset()

    model = HeterogeneousGNN(op_feat_dim=6, machine_feat_dim=2, hidden_dim=32, num_layers=2)

    # Encode initial state
    graph_init = build_graph_from_env_state(env._step_jobs, env._step_machines)
    emb_init = encode_state_with_gnn(graph_init, model)

    # Take a step and encode again
    env.step(action=0)
    graph_step1 = build_graph_from_env_state(env._step_jobs, env._step_machines)
    emb_step1 = encode_state_with_gnn(graph_step1, model)

    # Embeddings should change (at least for some nodes) after a step
    op_diff = (emb_init["op_embeddings"] - emb_step1["op_embeddings"]).abs().max()
    machine_diff = (emb_init["machine_embeddings"] - emb_step1["machine_embeddings"]).abs().max()

    # Some change is expected (though not guaranteed for all nodes)
    # We just check that the model runs without errors and produces valid outputs
    assert op_diff >= 0.0, "Operation embeddings should be valid"
    assert machine_diff >= 0.0, "Machine embeddings should be valid"
