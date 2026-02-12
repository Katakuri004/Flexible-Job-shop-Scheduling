from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def build_graph_from_env_state(
    step_jobs: List[Dict[str, Any]], step_machines: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build a heterogeneous graph representation from FJSPEnv step-wise state.

    Graph Schema:
    - Node types:
      * Operation nodes: one per (job_id, op_index)
      * Machine nodes: one per machine_id
    - Edge types:
      * Precedence edges: (job_id, op_k) -> (job_id, op_{k+1}) [directed]
      * Compatibility edges: (operation) <-> (machine) [undirected, bidirectional]

    Returns a dictionary with:
    - 'op_node_ids': List[(job_id, op_index)] - stable IDs for operation nodes
    - 'machine_node_ids': List[int] - machine IDs
    - 'op_features': np.ndarray [num_ops, feat_dim] - feature vectors for operations
    - 'machine_features': np.ndarray [num_machines, feat_dim] - feature vectors for machines
    - 'precedence_edges': np.ndarray [2, num_prec_edges] - edge index for precedence
    - 'compatibility_edges': np.ndarray [2, num_compat_edges] - edge index for compatibility

    Invariants enforced:
    - Number of op nodes equals total operations across all jobs
    - Precedence edges exist exactly between consecutive ops in each job
    - Compatibility edges match op_machines from env state
    """
    assert step_jobs, "step_jobs must be non-empty"
    assert step_machines, "step_machines must be non-empty"

    # Build operation node IDs and features
    op_node_ids: List[Tuple[int, int]] = []
    op_features_list: List[List[float]] = []

    for job in step_jobs:
        job_id = job["job_id"]
        num_ops = job["num_ops"]
        next_op_idx = job["next_op_index"]
        op_machines = job["op_machines"]
        op_times = job["op_times"]
        op_start = job["op_start"]
        op_end = job["op_end"]

        for op_idx in range(num_ops):
            op_node_ids.append((job_id, op_idx))

            # Feature: [is_scheduled, is_current, is_future, remaining_ops_in_job, op_index_normalized]
            is_scheduled = 1.0 if op_start[op_idx] is not None else 0.0
            is_current = 1.0 if op_idx == next_op_idx else 0.0
            is_future = 1.0 if op_idx > next_op_idx else 0.0
            remaining_ops = float(num_ops - op_idx)
            op_index_norm = float(op_idx) / max(num_ops, 1.0)

            # Additional: number of compatible machines (flexibility)
            num_compatible = float(len(op_machines[op_idx]))

            op_features_list.append(
                [is_scheduled, is_current, is_future, remaining_ops, op_index_norm, num_compatible]
            )

        total_ops = len(op_node_ids)
        op_features = np.array(op_features_list, dtype=np.float32)
        assert op_features.shape == (total_ops, 6), f"Expected shape ({total_ops}, 6), got {op_features.shape}"

    # Build machine node IDs and features
    machine_node_ids: List[int] = [m["machine_id"] for m in step_machines]
    num_machines = len(machine_node_ids)
    machine_features_list: List[List[float]] = []

    for machine in step_machines:
        m_id = machine["machine_id"]
        available_at = float(machine["available_at"])

        # Count remaining operations compatible with this machine
        compatible_ops_count = 0
        for job in step_jobs:
            op_machines = job["op_machines"]
            next_op_idx = job["next_op_index"]
            for op_idx in range(next_op_idx, job["num_ops"]):
                if m_id in op_machines[op_idx]:
                    compatible_ops_count += 1

        machine_features_list.append([available_at, float(compatible_ops_count)])

    machine_features = np.array(machine_features_list, dtype=np.float32)
    assert machine_features.shape == (num_machines, 2), f"Expected shape ({num_machines}, 2), got {machine_features.shape}"

    # Build precedence edges: (job_id, op_k) -> (job_id, op_{k+1})
    precedence_edges: List[Tuple[int, int]] = []
    op_id_to_idx: Dict[Tuple[int, int], int] = {op_id: idx for idx, op_id in enumerate(op_node_ids)}

    for job in step_jobs:
        job_id = job["job_id"]
        num_ops = job["num_ops"]
        for op_idx in range(num_ops - 1):
            src_op_id = (job_id, op_idx)
            tgt_op_id = (job_id, op_idx + 1)
            assert src_op_id in op_id_to_idx and tgt_op_id in op_id_to_idx
            precedence_edges.append((op_id_to_idx[src_op_id], op_id_to_idx[tgt_op_id]))

    precedence_edge_index = np.array(precedence_edges, dtype=np.int64).T
    assert precedence_edge_index.shape[0] == 2, "Precedence edges must be [2, num_edges]"

    # Build compatibility edges: (operation) <-> (machine) [bidirectional]
    compatibility_edges: List[Tuple[int, int]] = []
    machine_id_to_idx: Dict[int, int] = {m_id: idx for idx, m_id in enumerate(machine_node_ids)}

    for job in step_jobs:
        job_id = job["job_id"]
        op_machines = job["op_machines"]
        num_ops = job["num_ops"]
        for op_idx in range(num_ops):
            op_node_idx = op_id_to_idx[(job_id, op_idx)]
            for m_id in op_machines[op_idx]:
                m_node_idx = machine_id_to_idx[m_id]
                # Bidirectional: op -> machine and machine -> op
                compatibility_edges.append((op_node_idx, m_node_idx))
                compatibility_edges.append((m_node_idx, op_node_idx))

    compatibility_edge_index = np.array(compatibility_edges, dtype=np.int64).T
    assert compatibility_edge_index.shape[0] == 2, "Compatibility edges must be [2, num_edges]"

    # Invariant checks
    assert len(op_node_ids) == total_ops, "Operation node count mismatch"
    assert len(machine_node_ids) == num_machines, "Machine node count mismatch"
    assert len(precedence_edges) == sum(
        job["num_ops"] - 1 for job in step_jobs
    ), "Precedence edge count should be sum(num_ops - 1) per job"

    return {
        "op_node_ids": op_node_ids,
        "machine_node_ids": machine_node_ids,
        "op_features": op_features,
        "machine_features": machine_features,
        "precedence_edges": precedence_edge_index,
        "compatibility_edges": compatibility_edge_index,
    }


__all__ = ["build_graph_from_env_state"]
