from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import simpy

from .brandimarte_parser import ParsedInstance, ParsedJob, ParsedOperation


@dataclass
class FJSPOperation:
    job_id: int
    op_index: int
    compatible_machines: List[int]
    processing_times: Dict[int, int]

    assigned_machine: Optional[int] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None


@dataclass
class FJSPJob:
    job_id: int
    operations: List[FJSPOperation] = field(default_factory=list)

    def is_complete(self) -> bool:
        return all(op.completion_time is not None for op in self.operations)


@dataclass
class FJSPMachine:
    machine_id: int
    resource: simpy.Resource
    schedule: List[FJSPOperation] = field(default_factory=list)


class SimPyFJSPEnvironment:
    """
    General SimPy-based FJSP environment built on ParsedInstance.

    For now this environment provides:
    - A fixed, greedy dispatching rule (earliest available compatible machine).
    - Strong invariants and an exportable schedule.

    Later, this can be extended with explicit decision points and an RL API.
    """

    def __init__(self, instance: ParsedInstance):
        self.instance = instance
        self.env = simpy.Environment()

        # Create jobs and operations from ParsedInstance.
        self.jobs: List[FJSPJob] = []
        for parsed_job in instance.jobs:
            ops: List[FJSPOperation] = []
            for parsed_op in parsed_job.operations:
                ops.append(
                    FJSPOperation(
                        job_id=parsed_op.job_id,
                        op_index=parsed_op.op_index,
                        compatible_machines=list(parsed_op.compatible_machines),
                        processing_times=dict(parsed_op.processing_times),
                    )
                )
            self.jobs.append(FJSPJob(job_id=parsed_job.job_id, operations=ops))

        # Machines are zero-based indices; capacity 1 for each.
        self.machines: Dict[int, FJSPMachine] = {
            m_id: FJSPMachine(
                machine_id=m_id, resource=simpy.Resource(self.env, capacity=1)
            )
            for m_id in range(instance.num_machines)
        }

        self._check_static_invariants()

    def reset(self) -> None:
        """Recreate the underlying SimPy environment and clear dynamic state."""
        self.__init__(self.instance)

    # ------------------------------------------------------------------
    # Simulation logic
    # ------------------------------------------------------------------
    def _job_process(
        self,
        job: FJSPJob,
        machine_selector: Callable[[FJSPOperation], int],
    ):
        """
        SimPy process for a single job.

        machine_selector: given an operation, returns the chosen machine_id
        from op.compatible_machines.
        """
        for op in job.operations:
            m_id = machine_selector(op)
            assert (
                m_id in op.compatible_machines
            ), "Selected machine must be compatible with operation."
            machine = self.machines[m_id]

            with machine.resource.request() as req:
                yield req
                op.assigned_machine = m_id
                op.start_time = self.env.now
                p_time = op.processing_times[m_id]
                assert p_time > 0, "Processing time must be positive."
                yield self.env.timeout(p_time)
                op.completion_time = self.env.now
                machine.schedule.append(op)

    def run_greedy_earliest_machine(self) -> None:
        """
        Run the environment with a simple greedy rule:

        - For each operation, choose the compatible machine with the
          smallest index. Since each machine has capacity 1 and jobs
          are sequences in time, this is enough to generate a valid
          schedule without deadlocks.
        """

        def selector(op: FJSPOperation) -> int:
            return min(op.compatible_machines)

        for job in self.jobs:
            self.env.process(self._job_process(job, selector))

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
            ), f"Job {job.job_id} operations must be contiguous from 0."

            for op in job.operations:
                assert (
                    op.compatible_machines
                ), "Each operation must be compatible with at least one machine."
                for m_id in op.compatible_machines:
                    assert (
                        m_id in self.machines
                    ), f"Operation references unknown machine {m_id}."
                assert set(op.compatible_machines) == set(
                    op.processing_times.keys()
                ), "compatible_machines and processing_times keys must match."

    def _check_schedule_invariants(self) -> None:
        # 1) Every operation must be scheduled exactly once.
        for job in self.jobs:
            for op in job.operations:
                assert (
                    op.start_time is not None
                    and op.completion_time is not None
                    and op.assigned_machine is not None
                ), "All operations must be scheduled with assigned machine."
                assert (
                    op.start_time < op.completion_time
                ), "Operation must have positive duration."

        # 2) Precedence within each job.
        for job in self.jobs:
            for i in range(1, len(job.operations)):
                prev_op = job.operations[i - 1]
                curr_op = job.operations[i]
                assert (
                    prev_op.completion_time <= curr_op.start_time
                ), f"Job {job.job_id} precedence violated between ops {i-1} and {i}."

        # 3) Machine capacity: no overlapping operations.
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
    # Helpers
    # ------------------------------------------------------------------
    def export_schedule(self) -> Dict[int, List[Tuple[int, int, float, float]]]:
        """
        Export schedule per machine as:
          machine_id -> [(job_id, op_index, start_time, completion_time), ...]
        """
        out: Dict[int, List[Tuple[int, int, float, float]]] = {}
        for m_id, machine in self.machines.items():
            sorted_ops = sorted(
                machine.schedule, key=lambda op: op.start_time or 0.0
            )
            out[m_id] = [
                (
                    op.job_id,
                    op.op_index,
                    float(op.start_time or 0.0),
                    float(op.completion_time or 0.0),
                )
                for op in sorted_ops
            ]
        return out

    @property
    def makespan(self) -> float:
        """Return the final completion time of all operations."""
        all_times: List[float] = []
        for job in self.jobs:
            for op in job.operations:
                if op.completion_time is not None:
                    all_times.append(float(op.completion_time))
        return max(all_times) if all_times else 0.0


__all__ = [
    "SimPyFJSPEnvironment",
    "FJSPJob",
    "FJSPMachine",
    "FJSPOperation",
]

