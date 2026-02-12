from pathlib import Path

from src.brandimarte_parser import load_brandimarte_instance
from src.seed_utils import SeedConfig, set_global_seeds
from src.simpy_fjsp_env import SimPyFJSPEnvironment


def test_brandimarte_parser_basic_invariants() -> None:
    """Parser should correctly load the toy Brandimarte-style instance."""

    set_global_seeds(SeedConfig(base_seed=101))

    path = Path("data/brandimarte_mk_toy.txt")
    instance = load_brandimarte_instance(path)

    assert instance.num_jobs == 3
    assert instance.num_machines == 2
    assert len(instance.jobs) == 3

    for job in instance.jobs:
        # Each job in the toy instance has 2 operations.
        assert len(job.operations) == 2
        for op in job.operations:
            assert op.compatible_machines, "Each op must have at least one machine."
            for m_id in op.compatible_machines:
                assert 0 <= m_id < instance.num_machines


def test_simpy_fjsp_env_runs_and_respects_invariants() -> None:
    """
    Integration test for the general SimPy FJSP environment:

    - Load the toy Brandimarte-style instance.
    - Run with the built-in greedy machine selection rule.
    - Rely on environment invariants to catch overlaps/precedence violations.
    """

    set_global_seeds(SeedConfig(base_seed=202))

    path = Path("data/brandimarte_mk_toy.txt")
    instance = load_brandimarte_instance(path)
    env = SimPyFJSPEnvironment(instance)

    env.run_greedy_earliest_machine()

    # All jobs must be complete and makespan finite.
    assert all(job.is_complete() for job in env.jobs)
    assert env.makespan > 0.0

    # Each machine should have at least one operation scheduled.
    for machine in env.machines.values():
        assert len(machine.schedule) > 0

