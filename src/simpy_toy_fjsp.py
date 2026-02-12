from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import simpy


@dataclass
class SimPyOperation:
    job_id: int
    op_index: int
    machine_id: int
    processing_time: int

    start_time: Optional[float] = None
    completion_time: Optional[float] = None


@dataclass
class SimPyJob:
    job_id: int
    operations: List[SimPyOperation] = field(default_factory=list)

    def is_complete(self) -> bool:
        return all(op.completion_time is not None for op in self.operations)


@dataclass
class SimPyMachine:
    machine_id: int
    resource: simpy.Resource
    schedule: List[SimPyOperation] = field(default_factory=list)


class SimPyToyFJSPEnvironment:
    """
    SimPy-based reimplementation of the toy deterministic FJSP environment.

    This environment is designed so that, under the same FIFO job ordering
    and processing times, it produces the *same* schedule as
    `ToyFJSPEnvironment` in `src.toy_fjsp`.
    """

    def __init__(self) -> None:
        # We mirror the static structure of `ToyFJSPEnvironment.create_toy_instance`.
        self.env = simpy.Environment()

        # Define operations per job: (machine_id, processing_time)
        job_specs: List[List[Tuple[int, int]]] = [
            [(0, 3), (1, 2)],  # Job 0
            [(1, 4), (0, 1)],  # Job 1
            [(0, 2), (1, 3)],  # Job 2
        ]

        self.jobs: List[SimPyJob] = []
        for j_id, ops in enumerate(job_specs):
            operations: List[SimPyOperation] = []
            for op_index, (m_id, p_time) in enumerate(ops):
                operations.append(
                    SimPyOperation(
                        job_id=j_id,
                        op_index=op_index,
                        machine_id=m_id,
                        processing_time=p_time,
                    )
                )
            self.jobs.append(SimPyJob(job_id=j_id, operations=operations))

        self.machines: Dict[int, SimPyMachine] = {
            0: SimPyMachine(machine_id=0, resource=simpy.Resource(self.env, capacity=1)),
            1: SimPyMachine(machine_id=1, resource=simpy.Resource(self.env, capacity=1)),
        }

        self._check_static_invariants()

    def reset(self) -> None:
        """Recreate the environment with the same static problem data."""
        # SimPy environments cannot be trivially rewound; recreate.
        self.__init__()

    # ------------------------------------------------------------------
    # Simulation logic
    # ------------------------------------------------------------------
    def _job_process(self, job: SimPyJob):
        """
        SimPy process for a single job.

        We enforce precedence by iterating operations in order and
        requesting the required machine sequentially.
        """
        for op in job.operations:
            machine = self.machines[op.machine_id]
            with machine.resource.request() as req:
                yield req
                op.start_time = self.env.now
                yield self.env.timeout(op.processing_time)
                op.completion_time = self.env.now
                machine.schedule.append(op)

    def run(self) -> None:
        """Run the SimPy environment until all jobs complete."""
        for job in self.jobs:
            self.env.process(self._job_process(job))

        self.env.run()
        self._check_schedule_invariants()

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------
    def _check_static_invariants(self) -> None:
        assert self.jobs, "There must be at least one job."
        assert self.machines, "There must be at least one machine."

        for job in self.jobs:
            assert job.operations, f"Job {job.job_id} must have operations."
            expected_indices = list(range(len(job.operations)))
            actual_indices = [op.op_index for op in job.operations]
            assert (
                expected_indices == actual_indices
            ), f"Job {job.job_id} operation indices must be contiguous from 0."

            for op in job.operations:
                assert (
                    op.machine_id in self.machines
                ), f"Operation references unknown machine {op.machine_id}."

    def _check_schedule_invariants(self) -> None:
        # 1) Every operation must be scheduled exactly once.
        for job in self.jobs:
            for op in job.operations:
                assert (
                    op.start_time is not None and op.completion_time is not None
                ), "All operations must be scheduled."
                assert (
                    op.start_time < op.completion_time
                ), "Operation must have positive processing time."

        # 2) Precedence within each job.
        for job in self.jobs:
            for i in range(1, len(job.operations)):
                prev_op = job.operations[i - 1]
                curr_op = job.operations[i]
                assert (
                    prev_op.completion_time <= curr_op.start_time
                ), f"Job {job.job_id} precedence violated between ops {i-1} and {i}."

        # 3) Machine capacity: no overlaps.
        for machine in self.machines.values():
            sorted_ops = sorted(
                machine.schedule, key=lambda op: op.start_time or 0.0
            )
            for i in range(1, len(sorted_ops)):
                prev_op = sorted_ops[i - 1]
                curr_op = sorted_ops[i]
                assert (
                    prev_op.completion_time <= curr_op.start_time
                ), f"Machine {machine.machine_id} has overlapping operations."

    # ------------------------------------------------------------------
    # Helper for tests
    # ------------------------------------------------------------------
    def export_schedule(self) -> Dict[int, List[Tuple[int, int, float, float]]]:
        """
        Export schedule as a per-machine list of:
        (job_id, op_index, start_time, completion_time), sorted by start time.
        """
        result: Dict[int, List[Tuple[int, int, float, float]]] = {}
        for m_id, machine in self.machines.items():
            sorted_ops = sorted(
                machine.schedule, key=lambda op: op.start_time or 0.0
            )
            result[m_id] = [
                (
                    op.job_id,
                    op.op_index,
                    float(op.start_time or 0.0),
                    float(op.completion_time or 0.0),
                )
                for op in sorted_ops
            ]
        return result


__all__ = ["SimPyToyFJSPEnvironment", "SimPyJob", "SimPyMachine", "SimPyOperation"]

