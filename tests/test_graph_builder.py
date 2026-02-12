from pathlib import Path

import numpy as np

from src.fjsp_env import FJSPEnv, FJSPEnvConfig
from src.graph_builder import build_graph_from_env_state
from src.seed_utils import SeedConfig


def test_graph_builder_basic_structure() -> None:
    """
    Test that graph construction produces correct node/edge counts and
    basic structural invariants on the toy Brandimarte instance.
    """
    cfg = FJSPEnvConfig(
        instance_path=Path("data/brandimarte_mk_toy.txt"),
        seed_config=SeedConfig(base_seed=5000),
    )
    env = FJSPEnv(cfg)
    obs = env.reset()

    assert env._step_jobs is not None
    assert env._step_machines is not None

    graph = build_graph_from_env_state(env._step_jobs, env._step_machines)

    # Check node counts
    assert len(graph["op_node_ids"]) == 6, "Toy instance has 3 jobs × 2 ops = 6 operations"
    assert len(graph["machine_node_ids"]) == 2, "Toy instance has 2 machines"

    # Check feature shapes
    assert graph["op_features"].shape == (6, 6), "6 ops × 6 features"
    assert graph["machine_features"].shape == (2, 2), "2 machines × 2 features"

    # Check precedence edges: should be (num_ops - 1) per job
    # Job 0: 1 edge, Job 1: 1 edge, Job 2: 1 edge = 3 total
    prec_edges = graph["precedence_edges"]
    assert prec_edges.shape[0] == 2, "Precedence edges must be [2, num_edges]"
    assert prec_edges.shape[1] == 3, "3 precedence edges (one per job)"

    # Check compatibility edges: bidirectional, so count depends on op-machine pairs
    compat_edges = graph["compatibility_edges"]
    assert compat_edges.shape[0] == 2, "Compatibility edges must be [2, num_edges]"
    assert compat_edges.shape[1] > 0, "Must have at least some compatibility edges"

    # Verify operation node IDs are unique and cover all (job_id, op_index)
    op_ids_set = set(graph["op_node_ids"])
    assert len(op_ids_set) == 6, "All operation nodes must be unique"
    expected_op_ids = {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)}
    assert op_ids_set == expected_op_ids, "Operation IDs must match expected set"

    # Verify machine node IDs
    assert set(graph["machine_node_ids"]) == {0, 1}, "Machine IDs must be {0, 1}"


def test_graph_builder_feature_values_initial_state() -> None:
    """
    Test that feature values are correct at initial state (reset).
    """
    cfg = FJSPEnvConfig(
        instance_path=Path("data/brandimarte_mk_toy.txt"),
        seed_config=SeedConfig(base_seed=6000),
    )
    env = FJSPEnv(cfg)
    env.reset()

    graph = build_graph_from_env_state(env._step_jobs, env._step_machines)

    op_features = graph["op_features"]
    op_node_ids = graph["op_node_ids"]

    # At reset, all ops should be unscheduled, and the first op of each job should be "current"
    for idx, (job_id, op_idx) in enumerate(op_node_ids):
        feat = op_features[idx]
        is_scheduled, is_current, is_future, remaining_ops, op_index_norm, num_compatible = feat

        assert is_scheduled == 0.0, f"Op ({job_id}, {op_idx}) should be unscheduled at reset"
        assert is_current == (1.0 if op_idx == 0 else 0.0), f"Only first op of each job should be current"
        assert is_future == (1.0 if op_idx > 0 else 0.0), f"Future ops should have is_future=1.0"
        assert remaining_ops > 0.0, "Remaining ops must be positive"
        assert 0.0 <= op_index_norm <= 1.0, "Normalized op index must be in [0, 1]"
        assert num_compatible > 0.0, "Each op must have at least one compatible machine"

    # Machine features: available_at should be 0.0 at reset
    machine_features = graph["machine_features"]
    for idx, m_id in enumerate(graph["machine_node_ids"]):
        available_at, compatible_ops_count = machine_features[idx]
        assert available_at == 0.0, f"Machine {m_id} should be available at time 0.0"
        assert compatible_ops_count > 0.0, "Each machine should have compatible ops"


def test_graph_builder_deterministic_across_runs() -> None:
    """
    Test that graph construction is deterministic: same env state produces
    identical graph structures and features.
    """
    cfg = FJSPEnvConfig(
        instance_path=Path("data/brandimarte_mk_toy.txt"),
        seed_config=SeedConfig(base_seed=7000),
    )

    env1 = FJSPEnv(cfg)
    env1.reset()
    graph1 = build_graph_from_env_state(env1._step_jobs, env1._step_machines)

    env2 = FJSPEnv(cfg)
    env2.reset()
    graph2 = build_graph_from_env_state(env2._step_jobs, env2._step_machines)

    # Compare node IDs
    assert graph1["op_node_ids"] == graph2["op_node_ids"]
    assert graph1["machine_node_ids"] == graph2["machine_node_ids"]

    # Compare features (with small tolerance for float comparisons)
    np.testing.assert_array_almost_equal(graph1["op_features"], graph2["op_features"], decimal=6)
    np.testing.assert_array_almost_equal(
        graph1["machine_features"], graph2["machine_features"], decimal=6
    )

    # Compare edge indices
    np.testing.assert_array_equal(graph1["precedence_edges"], graph2["precedence_edges"])
    np.testing.assert_array_equal(graph1["compatibility_edges"], graph2["compatibility_edges"])


def test_graph_builder_after_some_steps() -> None:
    """
    Test graph construction after taking a few steps in the environment.
    """
    cfg = FJSPEnvConfig(
        instance_path=Path("data/brandimarte_mk_toy.txt"),
        seed_config=SeedConfig(base_seed=8000),
    )
    env = FJSPEnv(cfg)
    env.reset()

    # Take 2 steps
    env.step(action=0)
    env.step(action=0)

    graph = build_graph_from_env_state(env._step_jobs, env._step_machines)

    # Some operations should now be scheduled
    op_features = graph["op_features"]
    scheduled_count = sum(1 for feat in op_features if feat[0] > 0.5)
    assert scheduled_count >= 2, "At least 2 operations should be scheduled after 2 steps"

    # Machine availability should have increased
    machine_features = graph["machine_features"]
    max_available = max(feat[0] for feat in machine_features)
    assert max_available > 0.0, "At least one machine should have processed an operation"
