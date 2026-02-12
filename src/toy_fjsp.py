from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Operation:
    """
    Single operation in a job for the toy, deterministic FJSP.

    Assumptions (toy setting):
    - Each operation has a fixed processing time on exactly one machine.
    - No alternative machines or failures at this stage.
    """

    job_id: int
    op_index: int
    machine_id: int
    processing_time: int

    start_time: Optional[int] = None
    completion_time: Optional[int] = None

    def is_scheduled(self) -> bool:
        return self.start_time is not None and self.completion_time is not None


@dataclass
class Job:
    """Sequence of operations that must execute in order."""

    job_id: int
    operations: List[Operation] = field(default_factory=list)

    def current_operation(self) -> Optional[Operation]:
        for op in self.operations:
            if not op.is_scheduled():
                return op
        return None

    def is_complete(self) -> bool:
        return self.current_operation() is None


@dataclass
class Machine:
    """
    Simple deterministic machine with a single-server queue.

    For the toy model we assume:
    - No breakdowns.
    - One operation at a time.
    """

    machine_id: int
    busy_until: int = 0
    schedule: List[Operation] = field(default_factory=list)

    def can_start_at(self, ready_time: int) -> int:
        """Earliest feasible start time respecting current schedule."""
        return max(self.busy_until, ready_time)

    def assign(self, operation: Operation, ready_time: int) -> None:
        start_time = self.can_start_at(ready_time)
        completion_time = start_time + operation.processing_time

        operation.start_time = start_time
        operation.completion_time = completion_time

        self.busy_until = completion_time
        self.schedule.append(operation)


class ToyFJSPEnvironment:
    """
    Deterministic, failure-free FJSP toy environment with a built-in FIFO dispatcher.

    This class is deliberately simple and self-contained to serve as:
    - A reference implementation for later SimPy and RL environments.
    - A target for invariants and unit/integration tests.
    """

    def __init__(self, jobs: List[Job], machines: Dict[int, Machine]):
        assert jobs, "Environment must be initialized with at least one job."
        assert machines, "Environment must be initialized with at least one machine."

        self.jobs = jobs
        self.machines = machines
        self.current_time: int = 0

        self._check_initial_invariants()

    @classmethod
    def create_toy_instance(cls) -> "ToyFJSPEnvironment":
        """
        Create a small, fully deterministic instance:

        - 2 machines: M0, M1
        - 3 jobs, each with 2 operations
        - Each operation is bound to exactly one machine

        The numbers are chosen so that manual verification is easy.
        """
        # Define operations per job: (machine_id, processing_time)
        job_specs: List[List[Tuple[int, int]]] = [
            # Job 0
            [(0, 3), (1, 2)],
            # Job 1
            [(1, 4), (0, 1)],
            # Job 2
            [(0, 2), (1, 3)],
        ]

        jobs: List[Job] = []
        for j_id, ops in enumerate(job_specs):
            operations: List[Operation] = []
            for op_index, (m_id, p_time) in enumerate(ops):
                operations.append(
                    Operation(
                        job_id=j_id,
                        op_index=op_index,
                        machine_id=m_id,
                        processing_time=p_time,
                    )
                )
            jobs.append(Job(job_id=j_id, operations=operations))

        machines: Dict[int, Machine] = {
            0: Machine(machine_id=0),
            1: Machine(machine_id=1),
        }

        return cls(jobs=jobs, machines=machines)

    def reset(self) -> None:
        """Reset all temporal information while keeping the static structure."""
        self.current_time = 0
        for job in self.jobs:
            for op in job.operations:
                op.start_time = None
                op.completion_time = None
        for machine in self.machines.values():
            machine.busy_until = 0
            machine.schedule.clear()

        self._check_initial_invariants()

    def run_with_fifo(self) -> None:
        """
        Run a deterministic schedule with a simple FIFO policy:

        - At each decision point, consider jobs in job_id order.
        - For each job, if its next operation is ready (all predecessors done),
          assign it to its required machine as soon as that machine is free.

        This is not meant to be optimal—just a predictable baseline.
        """
        self.reset()

        # Continue until all jobs are finished.
        while not all(job.is_complete() for job in self.jobs):
            # Determine the next earliest scheduling event across all jobs.
            progress_made = False

            for job in sorted(self.jobs, key=lambda j: j.job_id):
                op = job.current_operation()
                if op is None:
                    continue

                # Precedence is enforced by Job.current_operation order.
                machine = self.machines[op.machine_id]

                # Job is ready at the completion of its previous op (or time 0).
                if op.op_index == 0:
                    ready_time = 0
                else:
                    prev_op = job.operations[op.op_index - 1]
                    assert (
                        prev_op.completion_time is not None
                    ), "Previous operation must be complete before scheduling the next."
                    ready_time = prev_op.completion_time

                machine.assign(op, ready_time=ready_time)
                self.current_time = max(self.current_time, op.completion_time or 0)
                progress_made = True

            # Safety net to avoid infinite loops in case of logic bugs.
            assert progress_made, "No progress made in scheduling loop; check logic."

        self._check_schedule_invariants()

    # -------------------------------------------------------------------------
    # Invariants and consistency checks
    # -------------------------------------------------------------------------
    def _check_initial_invariants(self) -> None:
        """Basic sanity checks on the static problem definition."""
        # Jobs must have at least one operation.
        assert all(job.operations for job in self.jobs), "Each job must have operations."

        # Operation indices within a job must be contiguous and start at 0.
        for job in self.jobs:
            expected_indices = list(range(len(job.operations)))
            actual_indices = [op.op_index for op in job.operations]
            assert (
                actual_indices == expected_indices
            ), f"Job {job.job_id} operation indices must be contiguous starting at 0."

        # All referenced machines must exist.
        machine_ids = set(self.machines.keys())
        for job in self.jobs:
            for op in job.operations:
                assert (
                    op.machine_id in machine_ids
                ), f"Operation references unknown machine {op.machine_id}."

    def _check_schedule_invariants(self) -> None:
        """Verify that the produced schedule is temporally and logically consistent."""
        # 1) Every operation must be scheduled exactly once.
        for job in self.jobs:
            for op in job.operations:
                assert op.is_scheduled(), "All operations must be scheduled."
                assert (
                    op.start_time is not None
                    and op.completion_time is not None
                    and op.start_time < op.completion_time
                ), "Operation must have a positive processing interval."

        # 2) Precedence: operations within each job must be non-overlapping and ordered.
        for job in self.jobs:
            completion_times = [
                op.completion_time for op in job.operations
            ]
            start_times = [op.start_time for op in job.operations]
            assert all(
                t is not None for t in completion_times
            ), "All job operations must have completion times."
            assert all(
                s is not None for s in start_times
            ), "All job operations must have start times."

            for i in range(1, len(job.operations)):
                prev_op = job.operations[i - 1]
                curr_op = job.operations[i]
                assert (
                    prev_op.completion_time <= curr_op.start_time
                ), f"Job {job.job_id} precedence violated between ops {i-1} and {i}."

        # 3) Machine capacity: no overlapping operations on the same machine.
        for machine in self.machines.values():
            sorted_ops = sorted(
                machine.schedule, key=lambda op: op.start_time or 0
            )
            for i in range(1, len(sorted_ops)):
                prev_op = sorted_ops[i - 1]
                curr_op = sorted_ops[i]
                assert (
                    prev_op.completion_time <= curr_op.start_time
                ), f"Machine {machine.machine_id} has overlapping operations."

        # 4) Global makespan consistency: environment current_time must match last completion.
        all_completion_times = [
            op.completion_time
            for job in self.jobs
            for op in job.operations
        ]
        assert all_completion_times, "There must be at least one scheduled operation."
        max_completion = max(t for t in all_completion_times if t is not None)
        assert (
            self.current_time == max_completion
        ), "Environment current_time must equal global makespan."


__all__ = [
    "Operation",
    "Job",
    "Machine",
    "ToyFJSPEnvironment",
]

