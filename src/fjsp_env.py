from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .brandimarte_parser import load_brandimarte_instance
from .seed_utils import SeedConfig, set_global_seeds
from .simpy_fjsp_env import SimPyFJSPEnvironment


@dataclass
class FJSPEnvConfig:
    """
    Configuration for the high-level FJSPEnv wrapper.

    At this stage the environment exposes an episode-level API where a
    single `step()` call runs the whole greedy schedule. This is mainly
    to:
    - Standardize reset/step/seed signatures.
    - Validate deterministic replay and logging.
    A finer-grained decision API can be layered on later.
    """

    instance_path: Path
    seed_config: SeedConfig


class FJSPEnv:
    """
    Gym-like wrapper around SimPyFJSPEnvironment.

    API:
      - seed(config): sets global RNG seeds for deterministic replay.
      - reset(): reloads the instance and returns an initial observation.
      - step(action): runs the built-in greedy schedule (action is ignored
        for now), returns (obs, reward, done, info).

    The design prioritizes:
      - Determinism under fixed seeds.
      - Simple, inspectable observations for testing and logging.
    """

    def __init__(self, config: FJSPEnvConfig):
        self.config = config
        self._seed_config: SeedConfig = config.seed_config
        self._instance_path: Path = config.instance_path

        self._simpy_env: Optional[SimPyFJSPEnvironment] = None
        self._last_schedule: Optional[Dict[int, Any]] = None
        self._last_makespan: Optional[float] = None

        # Step-wise state
        self._step_jobs: Optional[List[Dict[str, Any]]] = None
        self._step_machines: Optional[List[Dict[str, Any]]] = None
        self._step_actions: Optional[List[Tuple[int, int, int]]] = None  # (job_id, op_index, machine_id)
        self._step_done: bool = False

    # ------------------------------------------------------------------
    # Seeding and reset
    # ------------------------------------------------------------------
    def seed(self, seed_config: Optional[SeedConfig] = None) -> None:
        if seed_config is not None:
            self._seed_config = seed_config
        set_global_seeds(self._seed_config)

    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to its initial state.

        Returns a simple observation containing static problem info that
        is useful for tests and logging.
        """
        self.seed()  # Ensure RNGs are in a known state.

        instance = load_brandimarte_instance(self._instance_path)
        # Keep SimPy environment available for cross-checks or baselines if needed.
        self._simpy_env = SimPyFJSPEnvironment(instance)
        self._last_schedule = None
        self._last_makespan = None

        # Initialize step-wise state.
        self._step_jobs = []
        for job in instance.jobs:
            self._step_jobs.append(
                {
                    "job_id": job.job_id,
                    "num_ops": len(job.operations),
                    "next_op_index": 0,
                    "op_machines": [list(op.compatible_machines) for op in job.operations],
                    "op_times": [dict(op.processing_times) for op in job.operations],
                    "op_start": [None] * len(job.operations),
                    "op_end": [None] * len(job.operations),
                }
            )

        self._step_machines = [
            {"machine_id": m_id, "available_at": 0.0}
            for m_id in range(instance.num_machines)
        ]
        self._step_done = False

        obs = self._build_observation()
        return obs

    # ------------------------------------------------------------------
    # Step-wise helpers
    # ------------------------------------------------------------------
    def _compute_feasible_actions(self) -> List[Tuple[int, int, int]]:
        """
        Compute all feasible (job_id, op_index, machine_id) assignments.

        For now, an operation is considered ready if it is the next
        unscheduled operation of its job; we do not delay actions based
        on machine availability, since start times will be computed as
        max(job_ready_time, machine_available_at).
        """
        assert self._step_jobs is not None

        actions: List[Tuple[int, int, int]] = []
        for job in self._step_jobs:
            j_id = job["job_id"]
            next_idx = job["next_op_index"]
            if next_idx >= job["num_ops"]:
                continue
            machines = job["op_machines"][next_idx]
            for m_id in machines:
                actions.append((j_id, next_idx, m_id))
        return actions

    def _apply_action(self, action_index: int) -> None:
        """
        Apply the selected action index to update step-wise schedule.
        """
        assert self._step_jobs is not None
        assert self._step_machines is not None
        assert self._step_actions is not None

        assert 0 <= action_index < len(self._step_actions), "Invalid action index."
        job_id, op_idx, machine_id = self._step_actions[action_index]

        job = next(j for j in self._step_jobs if j["job_id"] == job_id)
        assert job["next_op_index"] == op_idx, "Action must select next operation of job."

        # Compute job readiness time.
        if op_idx == 0:
            job_ready = 0.0
        else:
            prev_end = job["op_end"][op_idx - 1]
            assert prev_end is not None
            job_ready = float(prev_end)

        machine = self._step_machines[machine_id]
        m_ready = float(machine["available_at"])

        start_time = max(job_ready, m_ready)
        proc_time = float(job["op_times"][op_idx][machine_id])
        assert proc_time > 0.0
        end_time = start_time + proc_time

        job["op_start"][op_idx] = start_time
        job["op_end"][op_idx] = end_time
        job["next_op_index"] += 1

        machine["available_at"] = end_time

        # If all operations are complete, mark done.
        self._step_done = all(j["next_op_index"] >= j["num_ops"] for j in self._step_jobs)

    def _build_observation(self) -> Dict[str, Any]:
        """
        Build a simple observation structure capturing:
        - job-level progress
        - machine availability
        - feasible actions and an action mask
        """
        assert self._step_jobs is not None
        assert self._step_machines is not None

        self._step_actions = self._compute_feasible_actions()

        jobs_obs = [
            {
                "job_id": j["job_id"],
                "next_op_index": j["next_op_index"],
                "num_ops": j["num_ops"],
            }
            for j in self._step_jobs
        ]
        machines_obs = [
            {"machine_id": m["machine_id"], "available_at": float(m["available_at"])}
            for m in self._step_machines
        ]

        action_mask = [True] * len(self._step_actions)

        return {
            "jobs": jobs_obs,
            "machines": machines_obs,
            "feasible_actions": self._step_actions,
            "action_mask": action_mask,
        }

    # ------------------------------------------------------------------
    # Step API
    # ------------------------------------------------------------------
    def step(self, action: Optional[int] = None) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Step-wise scheduling:

        - `action` is an index into the current feasible_actions list.
        - The environment applies this assignment, updates internal state,
          and advances logical time accordingly.
        - When all operations are scheduled, `done` becomes True and
          the final makespan and schedule are reported.
        """
        if self._step_jobs is None or self._step_machines is None:
            # Allow calling step() without an explicit reset for convenience.
            self.reset()

        assert self._step_actions is not None

        # Default to the first available action if none is provided (for testing).
        if action is None:
            action_index = 0
        else:
            action_index = int(action)

        self._apply_action(action_index)

        obs = self._build_observation()

        if self._step_done:
            # Build schedule representation and makespan.
            assert self._step_jobs is not None
            makespan = 0.0
            schedule: Dict[int, List[Tuple[int, int, float, float]]] = {}

            for m in self._step_machines or []:
                m_id = m["machine_id"]
                schedule[m_id] = []

            for job in self._step_jobs:
                j_id = job["job_id"]
                for op_idx in range(job["num_ops"]):
                    start = job["op_start"][op_idx]
                    end = job["op_end"][op_idx]
                    assert start is not None and end is not None

                    # For the step-wise env we do not track which machine actually processed
                    # the operation beyond what is implied by processing times; to keep this
                    # simple and deterministic, we recompute the chosen machine from
                    # op_machines/op_times by matching duration on the earliest machine.
                    # NOTE: this is limited to the toy/benchmark scale and can be
                    # refined later if needed.
                    machines = job["op_machines"][op_idx]
                    proc_times = job["op_times"][op_idx]
                    # Use the smallest machine id among those that match (end-start).
                    duration = end - start
                    chosen_m = min(
                        m_id for m_id in machines if float(proc_times[m_id]) == duration
                    )

                    schedule[chosen_m].append((j_id, op_idx, float(start), float(end)))
                    makespan = max(makespan, float(end))

            # Sort each machine's operations by start time.
            for m_id, ops in schedule.items():
                schedule[m_id] = sorted(ops, key=lambda x: x[2])

            self._last_schedule = schedule
            self._last_makespan = makespan

            reward = -makespan
            done = True
            info = {"schedule": schedule}
        else:
            reward = 0.0
            done = False
            info = {}

        return obs, float(reward), done, info

    # ------------------------------------------------------------------
    # Introspection helpers for testing/debugging
    # ------------------------------------------------------------------
    @property
    def last_schedule(self) -> Optional[Dict[int, Any]]:
        return self._last_schedule

    @property
    def last_makespan(self) -> Optional[float]:
        return self._last_makespan


__all__ = ["FJSPEnv", "FJSPEnvConfig"]

