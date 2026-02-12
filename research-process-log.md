## Research Process Log

### Entry 1 – Initial Review and Planning
- **Timestamp**: 2026-02-10
- **Goal**: Understand existing proposal and technical plan; identify weaknesses and define a feasible, implementation-focused roadmap.
- **Inputs Reviewed**:
  - `plan-v1.md` (1173 lines; detailed technical roadmap with GNN + MARL).
  - `MARL for Robust Flexible Job Shops - Google Docs.pdf` (original 4-page research proposal).
- **Key Actions**:
  - Read and synthesized both documents to map the evolution from the original CNP-based MAS idea to the current GNN + MAPPO/QMIX concept.
  - Identified major conceptual and practical risk areas (scope, novelty focus, algorithmic complexity, and evaluation design).
  - Prepared to return a consolidated critique plus a streamlined, high-feasibility implementation plan for discussion.
- **Preliminary Metrics**:
  - Documents analyzed: 2
  - Core research dimensions identified: 4 (robustness definition, MARL formulation, environment design, experimental comparison).
  - Indicative project length from current plan: 14–16 weeks.
- **Observations**:
  - The core motivation (decentralized, robust scheduling under failures) is strong and well-aligned with Industry 4.0.
  - The current technical plan is state-of-the-art but quite heavy for a single-student project unless scope is narrowed and milestones are carefully prioritized.
  - Novelty is present (robustness focus with failure-aware MARL), but it needs to be framed more crisply against recent GNN+MARL FJSP work.
- **Next Step**:
  - Formulate: (a) a weaknesses/risks list, and (b) a concrete, staged implementation plan that is clearly achievable with limited resources and time, to review and refine with the student.

### Entry 2 – Integration of `marl-research.pdf` into `plan-v1.md`
- **Timestamp**: 2026-02-10
- **Goal**: Incorporate insights from `marl-research.pdf` (critique + SOTA HGNN-MAPPO framework + robustness metrics + RTX 4070 constraints) into an improved, execution-ready implementation roadmap in `plan-v1.md`.
- **Inputs Reviewed**:
  - `marl-research.pdf` (41-page research report with theoretical foundations, robustness metrics such as CVaR and SRM, and hardware-aware training strategies).
  - Existing `plan-v1.md` roadmap (Phases 1–4 with environment, GNN, MARL, and experimentation).
- **Key Actions**:
  - Parsed the introductory, architectural, robustness, and implementation sections of `marl-research.pdf`, focusing on: heterogeneous graph state design, dual-agent MAPPO, curriculum over failure rates, and GPU optimization for an RTX 4070.
  - Replaced the original Section `## 7. IMPLEMENTATION ROADMAP` in `plan-v1.md` with a new **“Implementation Roadmap (v2 – MARL + Robustness Focused)”** that:
    - Explicitly structures the work into four phases (Digital Twin & Baselines; HGNN+MAPPO Core; Robustness Evaluation; Analysis & Writing).
    - Adds concrete tasks for Brandimarte integration, failure/degradation modeling, and strong non-learning baselines (GA + PDRs).
    - Integrates HGNN+MAPPO dual-agent policy design, curriculum learning over disruption levels, and RTX 4070–specific optimizations (AMP, gradient clipping/checkpointing).
    - Includes a dedicated robustness phase implementing variance, CVaR, SRM-like metrics, and nervousness, plus focused ablations (encoder, coordination, reward).
  - Ensured that the new roadmap remains consistent with the original research objectives (self-healing, decentralized robustness) while aligning terminology and methods with the 2024–2025 SOTA literature summarized in the PDF.
- **Metrics and Results**:
  - Sections updated: 1 (Section 7 of `plan-v1.md` fully rewritten).
  - Phases defined: 4 main phases with 10+ sub-blocks and ~40 concrete checklist items.
  - Robustness metrics explicitly incorporated: variance/std, CVaR\(_{95\%}\), SRM-style resilience, nervousness.
  - Hardware considerations: explicit constraints and optimizations for NVIDIA RTX 4070 (8GB VRAM) now captured in the plan.
- **Observations**:
  - The new roadmap is more tightly coupled to a clear central claim: HGNN-MAPPO can deliver higher robustness (CVaR, SRM) than GA/PDR baselines under machine failures, on realistic hardware.
  - The plan is still ambitious, but work is now sequenced so that a minimal end-to-end system (environment + baselines + a first MARL policy) can be completed early, reducing risk.
  - Robustness and stability metrics are now treated as first-class design targets rather than afterthoughts.
- **Next Step**:
  - Align this roadmap with your real-world constraints (available weeks, GPU access, desired depth of ablations), then potentially prune or prioritize phases/subtasks before coding.

### Entry 3 – Step 1 Implementation: Deterministic Toy FJSP + Testing & Traceability Scaffolding
- **Timestamp**: 2026-02-10
- **Goal**: Implement the first, failure-free toy FJSP environment with strong invariants, basic test coverage, and initial traceability/seed discipline to serve as a reference for later SimPy and MARL work.
- **Files Added**:
  - `src/toy_fjsp.py`: Defines `Operation`, `Job`, `Machine`, and `ToyFJSPEnvironment` with built-in invariants and a deterministic FIFO scheduler (`run_with_fifo`).
  - `src/seed_utils.py`: Provides `SeedConfig` and `set_global_seeds` for centralized seed management (Python + NumPy) ahead of later RL work.
  - `tests/test_toy_fjsp.py`: Initial unit/integration tests verifying toy environment behavior (`test_fifo_schedule_invariants_toy_instance`, `test_reset_restores_unscheduled_state`).
  - `feature-registry.md`: Introduces the feature registry with entries `R-ENV-TOY-01` (toy environment) and `R-SEED-01` (seed discipline).
- **Key Actions**:
  - Designed a small, fully deterministic FJSP instance (3 jobs × 2 machines, 2 operations per job) with fixed processing times to allow manual reasoning.
  - Implemented a simple FIFO scheduling policy in `ToyFJSPEnvironment.run_with_fifo`, enforcing job precedence via operation indices and machine capacity via `Machine.assign`.
  - Embedded invariants directly in the environment:
    - Static invariants on initialization/reset (jobs non-empty, contiguous operation indices, machines exist).
    - Schedule invariants after `run_with_fifo` (every operation scheduled once, no precedence violations, no machine overlaps, environment `current_time` equals global makespan).
  - Added tests exercising both scheduling and reset behavior, treating the environment’s assertions as fail-fast guards for logical errors.
  - Created the feature registry and populated it with requirement → implementation → tests mappings for the toy environment and seed utilities.
- **Metrics and Observations**:
  - New code modules: 2 (`toy_fjsp`, `seed_utils`), new test module: 1, new documentation file: 1.
  - The toy environment is deterministic and side-effect free apart from its internal state, making it suitable as a baseline for later SimPy integration and regression tests.
  - The current tests offer basic unit/integration coverage; property-based tests and performance checks will be layered on later steps once more complexity is introduced.
- **Next Step**:
  - Proceed to Step 2: re-create this toy instance using SimPy with a similar API, and verify equivalence of makespan and schedules, extending the test suite and feature registry accordingly.

### Entry 4 – Step 2 Implementation: SimPy Toy FJSP + Equivalence Tests
- **Timestamp**: 2026-02-12
- **Goal**: Create a SimPy-based toy FJSP environment that mirrors the deterministic toy instance, enforce invariants, and verify schedule equivalence via automated tests, further strengthening reproducibility and traceability.
- **Files Added/Updated**:
  - Added `src/__init__.py` to make `src` a proper package for clean `pytest` imports.
  - Added `src/simpy_toy_fjsp.py` implementing `SimPyToyFJSPEnvironment`, `SimPyJob`, `SimPyMachine`, and `SimPyOperation` using `simpy.Environment` and `simpy.Resource`.
  - Updated `tests/test_toy_fjsp.py` to import the SimPy toy environment and added `test_simpy_toy_matches_toy_fifo_schedule`.
  - Updated `feature-registry.md` with `R-ENV-SIMPY-TOY-02` describing the SimPy toy environment and its associated tests.
  - Added `pytest.ini` to ensure `src` is on `PYTHONPATH` during test runs.
- **Key Actions**:
  - Installed the `simpy` package and implemented a process-based model where each job is a SimPy process that:
    - Requests its required machine (capacity-1 `Resource`).
    - Records `start_time` and `completion_time` for each operation.
  - Embedded static and schedule invariants into `SimPyToyFJSPEnvironment` analogous to those in `ToyFJSPEnvironment` (job structure, precedence, machine capacity, positive processing intervals).
  - Implemented `export_schedule()` to expose per-machine ordered lists of `(job_id, op_index, start_time, completion_time)` for comparison.
  - Wrote an equivalence test that:
    - Runs the deterministic toy environment with FIFO and extracts its per-machine schedule and makespan.
    - Runs the SimPy toy environment and exports its schedule.
    - Asserts equality of job/operation ordering per machine and equality of makespan up to a small floating-point tolerance.
  - Fixed test collection/import issues by:
    - Adding `src/__init__.py`.
    - Adding `pytest.ini` with `pythonpath = .`, enabling `from src.*` imports.
- **Metrics and Results**:
  - Test suite: 3 tests passing in ~0.3s (`pytest`), including the new equivalence test.
  - Verified that SimPy and deterministic toy environments produce identical machine-wise job/op sequences and makespans for the canonical toy instance.
  - Confirmed that the basic test pyramid now includes:
    - Unit/integration tests for deterministic toy env and reset behavior.
    - Integration/equivalence test spanning both deterministic and SimPy environments.
- **Next Step**:
  - Build on this SimPy foundation to introduce a more general `FJSPEnv` wrapper and begin moving toward configurable instances (Brandimarte parsing) while continuing to extend invariants, tests, and the feature registry.

### Entry 5 – Step 3 Implementation: Brandimarte Parser + General SimPy FJSP Environment
- **Timestamp**: 2026-02-12
- **Goal**: Introduce a general SimPy-based FJSP environment driven by Brandimarte-style benchmark instances, with strong invariants, a simple built-in dispatching rule, and tests that validate parsing and schedule correctness.
- **Files Added/Updated**:
  - Added `data/brandimarte_mk_toy.txt`: a small, synthetic Brandimarte-style instance (3 jobs, 2 machines, 2 operations/job) used for parser and environment tests.
  - Added `src/brandimarte_parser.py`: implements `ParsedOperation`, `ParsedJob`, `ParsedInstance`, and `load_brandimarte_instance` with fail-fast invariants.
  - Added `src/simpy_fjsp_env.py`: implements `SimPyFJSPEnvironment` plus `FJSPOperation`, `FJSPJob`, and `FJSPMachine` for general FJSP instances with a greedy machine-selection rule.
  - Added `tests/test_brandimarte_simpy_env.py`: parser invariants test and integration test for the general SimPy FJSP environment.
  - Updated `feature-registry.md` with `R-PARSE-BR-03` (Brandimarte parser) and `R-ENV-SIMPY-GEN-04` (general SimPy environment).
- **Key Actions**:
  - Designed a minimal Brandimarte-style instance file consistent with the documented format and with structure analogous to the toy examples (3 jobs × 2 machines × 2 operations).
  - Implemented `load_brandimarte_instance` to:
    - Parse header and per-job/per-operation lines.
    - Convert 1-based machine IDs to zero-based internal indices.
    - Enforce invariants on job counts, contiguous op indices, machine ranges, and matching `compatible_machines`/`processing_times` keys.
  - Built `SimPyFJSPEnvironment` that:
    - Instantiates jobs and operations from a `ParsedInstance`.
    - Uses a simple greedy rule (smallest-index compatible machine) in `run_greedy_earliest_machine`.
    - Embeds invariants for precedence, capacity, and operation scheduling (assigned machine, positive duration, no overlaps).
    - Exposes `export_schedule()` and `makespan` for analysis and future regression tests.
  - Added tests that:
    - Validate basic parser invariants on the toy Brandimarte instance.
    - Run the general SimPy FJSP env with the greedy rule and assert that all jobs complete, makespan is positive, and each machine processes at least one operation (with env invariants acting as fail-fast guards).
  - Ran the full test suite (`pytest`): all 5 tests passed (toy deterministic env, SimPy toy equivalence, Brandimarte parser, and general SimPy env).
- **Metrics and Observations**:
  - Total tests: 5, all passing in <0.2s, covering deterministic toy env, SimPy toy, Brandimarte parsing, and general SimPy env execution.
  - The pipeline from a Brandimarte-style text file → parsed instance → SimPy environment → schedule is now in place with explicit invariants and reproducible tests.
  - The general SimPy env currently uses a built-in greedy dispatching rule; this can later be replaced or wrapped by RL decision logic without changing the underlying parsing and invariants.
- **Next Step**:
  - Introduce a Gym-like `FJSPEnv` wrapper around the SimPy environment (with `reset/step/seed`), define observation/action abstractions for small instances, and extend tests and the feature registry to cover deterministic replay and API contracts in preparation for MARL integration.

### Entry 6 – Step 4 Implementation: Gym-Style FJSPEnv Wrapper + Deterministic Replay
- **Timestamp**: 2026-02-12
- **Goal**: Wrap the general SimPy FJSP environment in a Gym-like `reset/step/seed` API, enforce deterministic replay via seeds, and validate the contract with focused tests to prepare for MARL integration.
- **Files Added/Updated**:
  - Added `src/fjsp_env.py`: defines `FJSPEnvConfig` and `FJSPEnv`, a high-level wrapper over `SimPyFJSPEnvironment`.
  - Added `tests/test_fjsp_env_api.py`: tests for API behavior and deterministic replay.
  - Updated `feature-registry.md` with `R-ENV-GYM-05` documenting the wrapper, behaviors, and associated tests.
- **Key Actions**:
  - Designed `FJSPEnvConfig` to bind an instance path (`data/brandimarte_mk_toy.txt`) and a `SeedConfig`, centralizing configuration for reproducible episodes.
  - Implemented `FJSPEnv` with:
    - `seed()`: uses `set_global_seeds` to reset Python and NumPy RNGs.
    - `reset()`: reloads the Brandimarte-style instance via `load_brandimarte_instance`, constructs a `SimPyFJSPEnvironment`, clears cached schedule/makespan, and returns a static observation (`num_jobs`, `num_machines`, `total_operations`).
    - `step(action)`: currently ignores `action`, runs `run_greedy_earliest_machine()`, and returns:
      - `obs`: includes `makespan`, `num_jobs`, `num_machines`.
      - `reward`: `-makespan`.
      - `done`: `True` (episode-level step).
      - `info`: `{"schedule": exported_schedule}`; also caches `last_schedule` and `last_makespan` for introspection.
  - Wrote API tests to:
    - Validate basic shapes and semantics (`test_fjsp_env_reset_and_step_api`).
    - Confirm deterministic replay (`test_fjsp_env_deterministic_replay_with_seed`) by running two independent env instances with the same config and asserting equal makespans and identical schedules.
  - Ran the full test suite (`pytest`): 7 tests now pass, covering toy envs, SimPy toy equivalence, Brandimarte parser, general SimPy env, and the new Gym-style wrapper.
- **Metrics and Observations**:
  - Test count increased from 5 to 7, all passing in ~0.15s.
  - The environment stack now supports:
    - Text instance file → parsed instance → SimPy environment → Gym-style wrapper with deterministic seed discipline.
  - The current `step()` remains episode-level (single-step episodes), which is acceptable for early MARL plumbing and can be refined into multi-step decision points later without breaking the existing tests and traceability.
- **Next Step**:
  - Begin defining a richer observation/action design for small instances (e.g., selecting `(job, machine)` pairs at decision points) and incrementally refactor the environment toward a genuine step-wise MARL interface, while preserving invariants, deterministic tests, and feature registry mappings.

### Entry 7 – Step 5 Implementation: Step-Wise FJSPEnv for MARL-ready Scheduling
- **Timestamp**: 2026-02-12
- **Goal**: Refine `FJSPEnv` from a single-step, episode-level wrapper into a genuinely step-wise scheduling environment with a discrete action space, while maintaining invariants, deterministic replay, and API stability.
- **Files Updated**:
  - `src/fjsp_env.py`: extended `FJSPEnv` with internal step-wise state (`_step_jobs`, `_step_machines`, `_step_actions`) and redefined `reset()` and `step()` accordingly.
  - `tests/test_fjsp_env_api.py`: updated to exercise the new multi-step API and deterministic replay under a fixed policy.
- **Key Actions**:
  - Designed a step-wise scheduling model that:
    - Maintains per-job state (`next_op_index`, per-op compatible machines and processing times, start/end times).
    - Tracks per-machine availability (`available_at`), used to compute start/end times for each scheduled operation.
    - At each step, computes a list of feasible actions as `(job_id, op_index, machine_id)` for the next unscheduled operation of each job and all its compatible machines.
  - Implemented:
    - `_compute_feasible_actions()`: builds the current feasible action set.
    - `_apply_action(action_index)`: applies the chosen assignment, updating job and machine timelines and checking that actions always select the next operation in a job.
    - `_build_observation()`: returns an observation with:
      - `jobs`: minimal progress info per job (`job_id`, `next_op_index`, `num_ops`).
      - `machines`: current `available_at` times for each machine.
      - `feasible_actions`: the list of `(job_id, op_index, machine_id)` triplets.
      - `action_mask`: simple boolean mask over feasible actions (all `True` in this version).
  - Modified `reset()` to:
    - Build the step-wise state from the parsed Brandimarte instance (while still constructing `SimPyFJSPEnvironment` for potential cross-checking).
    - Return the initial observation instead of static counts.
  - Reworked `step(action)` to:
    - Interpret `action` as an index into `feasible_actions` (defaulting to `0` for tests).
    - Apply that action, rebuild the observation, and:
      - If all operations are scheduled, construct a per-machine schedule and makespan, cache them (`last_schedule`, `last_makespan`), and return a terminal transition with reward `-makespan`.
      - Otherwise, return a non-terminal transition with reward `0.0`.
- **Tests and Results**:
  - Updated `tests/test_fjsp_env_api.py` to:
    - Use `_run_fixed_policy_episode` (always `action=0`) to drive episodes to completion, asserting:
      - Observation structure (`jobs`, `machines`, `feasible_actions`, `action_mask`) on reset.
      - Terminal state provides a schedule and that `last_makespan == -reward == makespan`.
    - Verify deterministic replay: two independent envs with the same config and policy produce identical rewards and schedules.
  - Full `pytest` run: 7 tests passing in ~0.3s, confirming:
    - Toy deterministic env + SimPy toy equivalence.
    - Brandimarte parser and general SimPy env invariants.
    - New step-wise `FJSPEnv` API behavior and determinism.
- **Observations**:
  - The environment is now structurally ready for MARL: actions choose among a finite, dynamically changing feasible set, and observations expose job/machine progress and that set.
  - The step-wise env currently reconstructs the machine assignment when building a final schedule by matching durations; this is sufficient for the toy/benchmark scale but may be refined later for more complex instances.
- **Next Step**:
  - Add a cross-check test that aligns the step-wise fixed policy schedule with the existing SimPy greedy environment on the toy instance, and then begin defining a more informative observation encoding suitable for GNN-based policies (while keeping the current simple one for baseline MARL experiments).

### Entry 8 – Step 6 Implementation: Heterogeneous Graph Construction from FJSPEnv State
- **Timestamp**: 2026-02-12
- **Goal**: Implement graph construction utilities that convert `FJSPEnv` step-wise internal state into a heterogeneous graph representation (operation nodes, machine nodes, precedence/compatibility edges) suitable for GNN-based MARL policies, with strong invariants and deterministic behavior.
- **Files Added/Updated**:
  - Added `src/graph_builder.py`: implements `build_graph_from_env_state` that constructs a heterogeneous graph from `FJSPEnv` internal state (`_step_jobs`, `_step_machines`).
  - Added `tests/test_graph_builder.py`: 4 tests covering graph structure, feature values, determinism, and behavior after environment steps.
  - Updated `feature-registry.md` with `R-GRAPH-STATE-06` documenting the graph schema, invariants, and associated tests.
- **Key Actions**:
  - Defined a heterogeneous graph schema:
    - **Operation nodes**: one per `(job_id, op_index)` with 6 features:
      - `is_scheduled` (0.0/1.0), `is_current` (1.0 if this is the next op of its job), `is_future` (1.0 if op_index > next_op_index)
      - `remaining_ops_in_job`, `op_index_normalized` (op_idx / num_ops), `num_compatible_machines`
    - **Machine nodes**: one per `machine_id` with 2 features:
      - `available_at` (time when machine becomes free), `compatible_ops_count` (count of remaining unscheduled ops compatible with this machine)
    - **Precedence edges**: directed `(job_id, op_k) → (job_id, op_{k+1})` for consecutive operations
    - **Compatibility edges**: bidirectional `(operation) ↔ (machine)` if machine can process that operation
  - Implemented `build_graph_from_env_state` to:
    - Enumerate all operation nodes with stable IDs `(job_id, op_index)` and compute features from env state
    - Enumerate machine nodes and compute features (availability, compatible ops count)
    - Build precedence edge index array `[2, num_prec_edges]` connecting consecutive ops within each job
    - Build compatibility edge index array `[2, num_compat_edges]` connecting ops to compatible machines (bidirectional)
    - Return a dictionary with `op_node_ids`, `machine_node_ids`, `op_features`, `machine_features`, `precedence_edges`, `compatibility_edges`
  - Embedded invariants:
    - Operation node count equals total operations
    - Precedence edge count equals `sum(num_ops - 1)` per job
    - Feature arrays have expected shapes (`[num_ops, 6]` and `[num_machines, 2]`)
    - All operation IDs are unique and cover all `(job_id, op_index)` pairs
- **Tests and Results**:
  - `test_graph_builder_basic_structure`: validates node/edge counts, feature shapes, and structural invariants on the toy Brandimarte instance
  - `test_graph_builder_feature_values_initial_state`: checks that feature values are correct at reset (unscheduled ops, first op of each job marked as "current", machines available at 0.0)
  - `test_graph_builder_deterministic_across_runs`: verifies that identical env states produce identical graph structures and feature arrays (within float tolerance)
  - `test_graph_builder_after_some_steps`: confirms that after taking steps, scheduled ops and machine availability are correctly reflected in features
  - Full `pytest` run: 11 tests passing (up from 7), including 4 new graph builder tests, all passing in ~0.22s
- **Observations**:
  - The graph representation is deterministic and can be constructed at any point during an episode, making it suitable for GNN-based policy networks
  - Feature engineering is minimal but captures essential scheduling state (scheduled/current/future ops, machine availability, compatibility)
  - The bidirectional compatibility edges allow GNNs to propagate information both from operations to machines and vice versa
- **Next Step**:
  - Implement a minimal heterogeneous GNN encoder (using PyTorch Geometric or a lightweight PyTorch-only version) that processes these graphs and produces node embeddings, with unit tests for forward pass, gradient flow, and deterministic encoding.

### Entry 9 – Step 7 Implementation: Heterogeneous GNN Encoder
- **Timestamp**: 2026-02-12
- **Goal**: Implement a lightweight heterogeneous GNN encoder that processes the graph representation from `build_graph_from_env_state` and produces node embeddings suitable for MARL policy networks, with strong tests for forward pass, gradient flow, and deterministic behavior.
- **Files Added/Updated**:
  - Added `src/hgnn_encoder.py`: implements `HeterogeneousGNN` (pure PyTorch, no PyG dependency) and convenience function `encode_state_with_gnn`.
  - Added `tests/test_hgnn_encoder.py`: 5 tests covering forward pass shapes, gradient flow, deterministic encoding, convenience function integration, and encoding at different episode stages.
  - Updated `src/graph_builder.py`: modified compatibility edge construction to offset machine indices by `num_ops`, creating a unified node space (operations: 0..num_ops-1, machines: num_ops..num_ops+num_machines-1) for cleaner message passing.
  - Updated `feature-registry.md` with `R-HGNN-ENC-07` documenting the encoder architecture, design decisions, and associated tests.
- **Key Actions**:
  - Designed a lightweight heterogeneous GNN architecture:
    - Separate initial embedding layers for operations (6 features → hidden_dim) and machines (2 features → hidden_dim)
    - Multi-layer message passing with relation-specific layers:
      - Precedence message passing: operations aggregate messages from predecessor operations via mean pooling
      - Compatibility message passing: bidirectional op↔machine aggregation (operations receive from machines, machines receive from operations)
    - Layer normalization, dropout, and residual connections after each message passing layer
    - Outputs node embeddings for both node types: `op_embeddings [num_ops, hidden_dim]` and `machine_embeddings [num_machines, hidden_dim]`
  - Fixed graph builder compatibility edge indexing:
    - Modified `build_graph_from_env_state` to offset machine indices by `num_ops` in compatibility edges, creating a unified node space
    - Updated HGNN message passing to correctly handle this unified space (operations: 0..num_ops-1, machines: num_ops..num_ops+num_machines-1)
  - Implemented `encode_state_with_gnn` convenience function that:
    - Takes graph data from `build_graph_from_env_state` and a `HeterogeneousGNN` model
    - Converts numpy arrays to torch tensors
    - Runs forward pass in eval mode and returns embeddings
  - Added comprehensive tests:
    - Forward pass shape validation
    - Gradient flow check (loss backward, verify gradients exist and contain no NaN)
    - Deterministic encoding (identical models produce identical embeddings)
    - Integration with graph builder
    - Encoding at different episode stages (initial state vs after steps)
- **Tests and Results**:
  - Full `pytest` run: 16 tests passing (up from 11), including 5 new HGNN encoder tests, all passing in ~2.1s
  - Verified that:
    - HGNN produces correct output shapes for operation and machine embeddings
    - Gradients flow through the network (essential for training)
    - Encoding is deterministic under fixed model weights and inputs
    - The encoder integrates cleanly with the graph builder pipeline
- **Observations**:
  - The pure PyTorch implementation is sufficient for small-to-medium instances and avoids PyG dependency complexity at this stage
  - The unified node space (offset machine indices) simplifies message passing logic and makes the code more maintainable
  - The encoder is ready to be integrated into MARL policy networks (actor-critic heads can consume the node embeddings)
- **Next Step**:
  - Begin implementing a simple MARL policy network (e.g., a basic actor-critic) that uses the HGNN encoder to produce action logits from graph embeddings, starting with a single-agent or simplified dual-agent setup to validate the end-to-end pipeline before full MAPPO implementation.

---

## Entry 10: MARL Actor-Critic Policy Network Implementation

**Date**: 2026-02-10  
**Objective**: Implement a basic actor-critic policy network that uses the HGNN encoder to produce action logits and state value estimates for FJSP scheduling.

**Actions Taken**:
1. **Created `src/marl_policy.py`**:
   - Implemented `FJSPActorCritic` class with:
     - Shared `HeterogeneousGNN` encoder for node embeddings
     - Actor head: MLP that scores feasible actions by combining operation and machine embeddings
     - Critic head: MLP that estimates state value from pooled node embeddings
     - `forward()` method: computes action logits (if feasible_actions provided) and state value
     - `get_action_and_value()` convenience method: selects action (deterministic or stochastic) and returns logits and value
   - Design decisions:
     - Single-agent policy (not yet dual-agent MAPPO) for initial validation
     - Dynamic action space: action logits computed only for currently feasible actions
     - Action scoring: concatenates embeddings from the specific operation and machine involved in each action
     - State value: computed from global graph representation (mean-pooled embeddings)

2. **Created `tests/test_marl_policy.py`**:
   - `test_actor_critic_forward_pass_shapes`: Validates output tensor shapes (value is scalar, action_logits match number of feasible actions)
   - `test_actor_critic_action_selection`: Tests deterministic (argmax) and stochastic (Categorical sampling) action selection
   - `test_actor_critic_gradient_flow`: Confirms gradients flow through actor and critic heads without NaN
   - `test_actor_critic_integration_with_env`: End-to-end integration test with `FJSPEnv` in a simple rollout

3. **Updated `feature-registry.md`**:
   - Added `R-MARL-POLICY-08` entry documenting the actor-critic policy network requirement, implementation, architecture, and tests

**Results**:
- All 4 new policy tests pass
- Full test suite: 20 tests passing (16 existing + 4 new)
- Policy network successfully integrates with `FJSPEnv`, `build_graph_from_env_state`, and `HeterogeneousGNN`
- Gradient flow confirmed through both actor and critic heads

**Observations**:
- The actor-critic architecture cleanly separates action scoring (actor) from state value estimation (critic)
- Dynamic action space handling (scoring only feasible actions) works well with the current `FJSPEnv` API
- The policy network is ready for training loop integration (next step would be implementing a training algorithm like PPO or MAPPO)
- The single-agent policy serves as a foundation that can be extended to dual-agent MAPPO later

**Next Step**:
- Implement a basic training loop (e.g., PPO or simplified MAPPO) that:
  - Collects rollouts using the actor-critic policy
  - Computes advantages (e.g., GAE)
  - Updates policy and value networks via policy gradient loss
  - Includes logging and checkpointing for experiment tracking
- Alternatively, first implement a simple baseline (e.g., random policy or rule-based heuristic) to establish performance benchmarks before RL training




