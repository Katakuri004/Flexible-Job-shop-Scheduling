## Feature Registry – MARL FJSP Project

This document tracks the mapping from **requirements** → **implementation** → **tests** to support auditability and traceability.

Each feature is labeled with a stable ID (`R-*`) and links to the relevant code and tests.

---

### R-ENV-TOY-01 – Deterministic Toy FJSP Environment

- **Requirement**: Provide a minimal, deterministic, failure-free FJSP environment suitable for manual reasoning, invariants, and early tests.
- **Implementation**:
  - `ToyFJSPEnvironment`, `Job`, `Operation`, `Machine` in `src/toy_fjsp.py`
- **Key Invariants**:
  - Jobs have contiguous operation indices starting at 0.
  - All referenced machines exist.
  - After scheduling:
    - Every operation is scheduled exactly once with a positive processing interval.
    - Job precedence is respected (no operation starts before its predecessor completes).
    - No machine processes more than one operation at a time (no temporal overlaps).
    - Environment `current_time` equals the global makespan.
- **Tests**:
  - `tests/test_toy_fjsp.py::test_fifo_schedule_invariants_toy_instance`
  - `tests/test_toy_fjsp.py::test_reset_restores_unscheduled_state`

---

### R-SEED-01 – Seed Discipline for Deterministic Replay (Pre-RL)

- **Requirement**: Establish a centralized way to set seeds for stochastic components (at this stage: Python and NumPy) to prepare for deterministic replay and reproducible tests.
- **Implementation**:
  - `SeedConfig`, `set_global_seeds` in `src/seed_utils.py`
- **Tests**:
  - Indirectly exercised via:
    - `tests/test_toy_fjsp.py` (both tests call `set_global_seeds` before environment construction and execution)
  - Dedicated seed-behavior tests can be added later when randomness is introduced into the environment.

---

### R-ENV-SIMPY-TOY-02 – SimPy Toy FJSP Environment Equivalent to Deterministic Toy

- **Requirement**: Provide a SimPy-based digital twin of the toy deterministic FJSP instance that reproduces the same schedule and makespan, serving as a bridge toward more complex SimPy environments.
- **Implementation**:
  - `SimPyToyFJSPEnvironment`, `SimPyJob`, `SimPyMachine`, `SimPyOperation` in `src/simpy_toy_fjsp.py`
- **Key Invariants**:
  - Jobs and operations mirror the structure of `ToyFJSPEnvironment.create_toy_instance` (3 jobs × 2 machines × 2 operations).
  - Within each job, operations execute in strict sequence (no precedence violations).
  - Each machine has capacity 1 with no overlapping operations.
  - `export_schedule()` returns per-machine ordered lists of `(job_id, op_index, start_time, completion_time)` consistent with SimPy execution.
- **Tests**:
  - `tests/test_toy_fjsp.py::test_simpy_toy_matches_toy_fifo_schedule` (equivalence of job/operation ordering and makespan between deterministic and SimPy toy environments).

---

### R-PARSE-BR-03 – Brandimarte-Style Instance Parser

- **Requirement**: Parse Brandimarte-style FJSP benchmark files into a structured in-memory representation with strong invariants and fail-fast behavior on malformed data.
- **Implementation**:
  - `ParsedOperation`, `ParsedJob`, `ParsedInstance`, and `load_brandimarte_instance` in `src/brandimarte_parser.py`
  - Example instance file `data/brandimarte_mk_toy.txt` (3 jobs, 2 machines, 2 operations per job).
- **Key Invariants**:
  - Number of jobs in file matches header `num_jobs`.
  - Each job has contiguous operation indices starting from 0.
  - Each operation has at least one compatible machine and positive processing times.
  - Machine IDs in the file are within `[1, num_machines]` and converted to zero-based indices internally.
  - `compatible_machines` set equals the keys of `processing_times`.
- **Tests**:
  - `tests/test_brandimarte_simpy_env.py::test_brandimarte_parser_basic_invariants`

---

### R-ENV-SIMPY-GEN-04 – General SimPy FJSP Environment

- **Requirement**: Provide a general SimPy-based FJSP environment that can execute schedules derived from parsed Brandimarte instances using a simple built-in dispatching rule, with strong invariants guaranteeing schedule correctness.
- **Implementation**:
  - `SimPyFJSPEnvironment`, `FJSPOperation`, `FJSPJob`, `FJSPMachine` in `src/simpy_fjsp_env.py`
- **Key Invariants**:
  - Each job’s operations are executed in order with no precedence violations.
  - Each machine has capacity 1 with no overlapping operations in its `schedule`.
  - Every operation is assigned to a compatible machine, with positive processing time.
  - `makespan` reflects the latest completion time across all operations.
- **Tests**:
  - `tests/test_brandimarte_simpy_env.py::test_simpy_fjsp_env_runs_and_respects_invariants`

---

### R-ENV-GYM-05 – Gym-Style FJSPEnv Wrapper with Deterministic Replay

- **Requirement**: Provide a high-level environment wrapper with a standardized `reset/step/seed` API over the SimPy FJSP environment, enabling deterministic, seed-controlled episodes and serving as the integration point for MARL algorithms.
- **Implementation**:
  - `FJSPEnvConfig`, `FJSPEnv` in `src/fjsp_env.py`
- **Behavior**:
  - `seed(SeedConfig)`: applies global seeds via `set_global_seeds` to ensure deterministic replay.
  - `reset()`: reloads a Brandimarte-style instance, constructs a `SimPyFJSPEnvironment`, and returns static problem information (`num_jobs`, `num_machines`, `total_operations`).
  - `step(action)`: currently ignores `action` and runs the built-in greedy schedule, returning:
    - `obs`: includes `makespan`, `num_jobs`, `num_machines`.
    - `reward`: `-makespan`.
    - `done`: always `True` (episode-level step).
    - `info`: includes the full schedule exported from the underlying SimPy env.
- **Tests**:
  - `tests/test_fjsp_env_api.py::test_fjsp_env_reset_and_step_api`
  - `tests/test_fjsp_env_api.py::test_fjsp_env_deterministic_replay_with_seed`




