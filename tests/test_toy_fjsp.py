import math

from src.seed_utils import SeedConfig, set_global_seeds
from src.toy_fjsp import ToyFJSPEnvironment
from src.simpy_toy_fjsp import SimPyToyFJSPEnvironment


def test_fifo_schedule_invariants_toy_instance() -> None:
    """
    Integration-style test for the toy deterministic FJSP:

    - Constructs the canonical toy instance.
    - Runs the built-in FIFO scheduler.
    - Relies on the environment's internal invariants to raise if anything is wrong.
    """

    set_global_seeds(SeedConfig(base_seed=123))

    env = ToyFJSPEnvironment.create_toy_instance()
    env.run_with_fifo()

    # Basic sanity checks beyond internal assertions.
    # 1) All jobs should be complete.
    assert all(job.is_complete() for job in env.jobs)

    # 2) Makespan should be finite and positive.
    assert isinstance(env.current_time, int)
    assert env.current_time > 0

    # 3) There should be at least one operation on each machine.
    for machine in env.machines.values():
        assert len(machine.schedule) > 0


def test_reset_restores_unscheduled_state() -> None:
    """
    Unit/integration test for reset behavior:

    - After running a schedule, calling reset should erase all temporal
      fields while preserving the static problem definition.
    """

    set_global_seeds(SeedConfig(base_seed=456))

    env = ToyFJSPEnvironment.create_toy_instance()
    env.run_with_fifo()

    # Capture a non-zero makespan to ensure we actually ran the schedule.
    assert env.current_time > 0

    env.reset()

    # After reset, current_time should be zero and no operation should be scheduled.
    assert env.current_time == 0
    for job in env.jobs:
        for op in job.operations:
            assert op.start_time is None
            assert op.completion_time is None

    # Machine schedules should be empty but machines should still exist.
    assert len(env.machines) > 0
    for machine in env.machines.values():
        assert machine.busy_until == 0
        assert machine.schedule == []


def test_simpy_toy_matches_toy_fifo_schedule() -> None:
    """
    Integration test: the SimPy-based toy environment must reproduce
    the same schedule as the deterministic ToyFJSPEnvironment with FIFO.

    We compare, for each machine:
    - The ordered list of (job_id, op_index).
    - The makespan (max completion time).

    Minor floating-point differences in start/completion times are tolerated
    via rounding.
    """

    set_global_seeds(SeedConfig(base_seed=789))

    # Baseline schedule from the non-SimPy environment.
    toy_env = ToyFJSPEnvironment.create_toy_instance()
    toy_env.run_with_fifo()

    toy_schedule: dict[int, list[tuple[int, int, int, int]]] = {}
    for m_id, machine in toy_env.machines.items():
        sorted_ops = sorted(machine.schedule, key=lambda op: op.start_time or 0)
        toy_schedule[m_id] = [
            (
                op.job_id,
                op.op_index,
                int(op.start_time or 0),
                int(op.completion_time or 0),
            )
            for op in sorted_ops
        ]

    toy_makespan = toy_env.current_time

    # SimPy schedule.
    simpy_env = SimPyToyFJSPEnvironment()
    simpy_env.run()
    simpy_schedule = simpy_env.export_schedule()

    simpy_makespan = max(
        (op[3] for ops in simpy_schedule.values() for op in ops), default=0.0
    )

    # Compare per-machine job/op ordering.
    assert set(toy_schedule.keys()) == set(simpy_schedule.keys())

    for m_id in toy_schedule.keys():
        toy_ops = [(j, o) for (j, o, _, _) in toy_schedule[m_id]]
        simpy_ops = [(j, o) for (j, o, _, _) in simpy_schedule[m_id]]
        assert (
            toy_ops == simpy_ops
        ), f"Mismatch in operation order on machine {m_id}: {toy_ops} vs {simpy_ops}"

    # Compare makespans within a small tolerance.
    assert abs(float(toy_makespan) - float(simpy_makespan)) < 1e-6

