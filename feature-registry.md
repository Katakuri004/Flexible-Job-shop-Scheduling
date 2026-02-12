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

---

### R-GRAPH-STATE-06 – Heterogeneous Graph Construction from FJSPEnv State

- **Requirement**: Convert the step-wise internal state of `FJSPEnv` into a heterogeneous graph representation suitable for GNN-based MARL policies, with operation nodes, machine nodes, precedence edges, and compatibility edges.
- **Implementation**:
  - `build_graph_from_env_state` in `src/graph_builder.py`
- **Graph Schema**:
  - **Node types**:
    - Operation nodes: one per `(job_id, op_index)` with 6 features: `[is_scheduled, is_current, is_future, remaining_ops_in_job, op_index_normalized, num_compatible_machines]`
    - Machine nodes: one per `machine_id` with 2 features: `[available_at, compatible_ops_count]`
  - **Edge types**:
    - Precedence edges: directed `(job_id, op_k) → (job_id, op_{k+1})` for consecutive operations within each job
    - Compatibility edges: bidirectional `(operation) ↔ (machine)` if the machine can process that operation
- **Key Invariants**:
  - Number of operation nodes equals total operations across all jobs
  - Precedence edges exist exactly `(num_ops - 1)` per job
  - Compatibility edges match `op_machines` from env state (bidirectional, so each op-machine pair creates 2 edges)
  - Feature vectors have consistent shapes: `op_features` is `[num_ops, 6]`, `machine_features` is `[num_machines, 2]`
- **Tests**:
  - `tests/test_graph_builder.py::test_graph_builder_basic_structure`
  - `tests/test_graph_builder.py::test_graph_builder_feature_values_initial_state`
  - `tests/test_graph_builder.py::test_graph_builder_deterministic_across_runs`
  - `tests/test_graph_builder.py::test_graph_builder_after_some_steps`

---

### R-HGNN-ENC-07 – Heterogeneous GNN Encoder for FJSP State Graphs

- **Requirement**: Provide a lightweight heterogeneous GNN encoder that processes the graph representation from `build_graph_from_env_state` and produces node embeddings suitable for MARL policy networks.
- **Implementation**:
  - `HeterogeneousGNN`, `encode_state_with_gnn` in `src/hgnn_encoder.py`
- **Architecture**:
  - **Initial embeddings**: separate linear layers for operation features (6-dim) and machine features (2-dim) → hidden_dim (default 128)
  - **Message passing layers** (configurable `num_layers`, default 2):
    - Precedence message passing: operations aggregate messages from predecessor operations via mean pooling
    - Compatibility message passing: bidirectional op↔machine message passing (operations receive from machines, machines receive from operations)
  - **Normalization**: LayerNorm and dropout after each message passing layer, with residual connections
  - **Output**: node embeddings for operations `[num_ops, hidden_dim]` and machines `[num_machines, hidden_dim]`
- **Key Design Decisions**:
  - Pure PyTorch implementation (no PyTorch Geometric dependency) for simplicity and easier debugging
  - Unified node space for compatibility edges: machine indices offset by `num_ops` to distinguish from operation indices
  - Mean pooling for message aggregation (simple but effective for small instances)
- **Tests**:
  - `tests/test_hgnn_encoder.py::test_hgnn_forward_pass_shapes` (validates output tensor shapes)
  - `tests/test_hgnn_encoder.py::test_hgnn_gradient_flow` (confirms gradients flow and contain no NaN)
  - `tests/test_hgnn_encoder.py::test_hgnn_deterministic_encoding` (identical models produce identical embeddings)
  - `tests/test_hgnn_encoder.py::test_encode_state_with_gnn_convenience` (integration with graph_builder)
  - `tests/test_hgnn_encoder.py::test_hgnn_after_env_steps` (encoding works at different episode stages)

---

### R-MARL-POLICY-08 – Actor-Critic Policy Network for FJSP Scheduling

- **Requirement**: Provide a MARL policy network (actor-critic) that uses the HGNN encoder to produce action logits over feasible actions and state value estimates, enabling RL-based scheduling decisions.
- **Implementation**:
  - `FJSPActorCritic` in `src/marl_policy.py`
- **Architecture**:
  - **Shared HGNN encoder**: Uses `HeterogeneousGNN` to produce operation and machine node embeddings
  - **Actor head**: Scores feasible actions by concatenating operation and machine embeddings for each `(job_id, op_index, machine_id)` action, then passes through MLP to produce action logits
  - **Critic head**: Estimates state value by pooling operation and machine embeddings (mean pooling) and passing through MLP
  - **Action selection**: Supports both deterministic (argmax) and stochastic (Categorical sampling) modes
- **Key Design Decisions**:
  - Single-agent policy (not yet dual-agent MAPPO) for initial validation
  - Dynamic action space: action logits are computed only for currently feasible actions from `FJSPEnv`
  - Action scoring combines embeddings from the specific operation and machine involved in each action
  - State value is computed from global graph representation (pooled embeddings)
- **Integration**:
  - Works with `build_graph_from_env_state` output and `FJSPEnv`'s `feasible_actions` list
  - `get_action_and_value` convenience method provides action selection, logits, and value in one call
- **Tests**:
  - `tests/test_marl_policy.py::test_actor_critic_forward_pass_shapes` (validates output tensor shapes match feasible actions)
  - `tests/test_marl_policy.py::test_actor_critic_action_selection` (tests deterministic and stochastic action selection)
  - `tests/test_marl_policy.py::test_actor_critic_gradient_flow` (confirms gradients flow through actor and critic heads)
  - `tests/test_marl_policy.py::test_actor_critic_integration_with_env` (end-to-end integration with FJSPEnv in a simple rollout)




