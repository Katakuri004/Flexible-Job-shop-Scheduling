from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class ParsedOperation:
    job_id: int
    op_index: int
    compatible_machines: List[int]
    processing_times: Dict[int, int]


@dataclass
class ParsedJob:
    job_id: int
    operations: List[ParsedOperation]


@dataclass
class ParsedInstance:
    num_jobs: int
    num_machines: int
    jobs: List[ParsedJob]


def load_brandimarte_instance(path: str | Path) -> ParsedInstance:
    """
    Parse a Brandimarte-style FJSP instance file.

    Format (simplified):
      Line 1: num_jobs num_machines avg_machines_per_op
      For each job j in [0, num_jobs):
        Line: num_operations
        For each operation o in job:
          Line: num_capable machine_id1 time1 machine_id2 time2 ...

    This loader is kept minimal but includes basic invariants so that
    malformed inputs fail fast.
    """

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Brandimarte instance not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not lines:
        raise ValueError(f"Brandimarte instance file is empty: {p}")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError("Header must contain at least num_jobs and num_machines.")

    num_jobs = int(header[0])
    num_machines = int(header[1])

    jobs: List[ParsedJob] = []
    line_idx = 1

    for j in range(num_jobs):
        if line_idx >= len(lines):
            raise ValueError(f"Unexpected end of file while reading job {j}.")

        num_ops = int(lines[line_idx])
        line_idx += 1

        operations: List[ParsedOperation] = []
        for o in range(num_ops):
            if line_idx >= len(lines):
                raise ValueError(
                    f"Unexpected end of file while reading op {o} of job {j}."
                )

            tokens = [int(tok) for tok in lines[line_idx].split()]
            line_idx += 1

            if not tokens:
                raise ValueError(
                    f"Empty operation line for op {o} of job {j} in file {p}."
                )

            num_capable = tokens[0]
            if len(tokens) != 1 + 2 * num_capable:
                raise ValueError(
                    f"Inconsistent num_capable for op {o} of job {j}: "
                    f"expected {1 + 2 * num_capable} ints, got {len(tokens)}."
                )

            compatible_machines: List[int] = []
            processing_times: Dict[int, int] = {}

            for i in range(num_capable):
                m_id = tokens[1 + 2 * i]
                p_time = tokens[2 + 2 * i]
                if not (1 <= m_id <= num_machines):
                    raise ValueError(
                        f"Machine id {m_id} out of range in job {j}, op {o}."
                    )
                # Store machine ids as zero-based internally.
                m_idx = m_id - 1
                compatible_machines.append(m_idx)
                processing_times[m_idx] = p_time

            operations.append(
                ParsedOperation(
                    job_id=j,
                    op_index=o,
                    compatible_machines=compatible_machines,
                    processing_times=processing_times,
                )
            )

        jobs.append(ParsedJob(job_id=j, operations=operations))

    _check_instance_invariants(num_jobs, num_machines, jobs)
    return ParsedInstance(num_jobs=num_jobs, num_machines=num_machines, jobs=jobs)


def _check_instance_invariants(
    num_jobs: int, num_machines: int, jobs: List[ParsedJob]
) -> None:
    assert len(jobs) == num_jobs, "Number of jobs in file does not match header."
    assert num_machines > 0, "There must be at least one machine."

    for job in jobs:
        assert job.operations, f"Job {job.job_id} must have at least one operation."
        expected_indices = list(range(len(job.operations)))
        actual_indices = [op.op_index for op in job.operations]
        assert (
            expected_indices == actual_indices
        ), f"Job {job.job_id} operations must have contiguous indices starting at 0."

        for op in job.operations:
            assert op.compatible_machines, "Each operation must have at least one machine."
            for m_id in op.compatible_machines:
                assert 0 <= m_id < num_machines, "Machine index out of bounds."
            assert set(op.compatible_machines) == set(
                op.processing_times.keys()
            ), "compatible_machines and processing_times keys must match."


__all__ = ["ParsedOperation", "ParsedJob", "ParsedInstance", "load_brandimarte_instance"]

