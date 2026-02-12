from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


class HeterogeneousGNN(nn.Module):
    """
    Lightweight heterogeneous GNN encoder for FJSP state graphs.

    Architecture:
    - Separate embedding layers for operation and machine nodes
    - Relation-specific message passing layers for precedence and compatibility edges
    - Multi-layer processing with residual connections and layer normalization
    - Outputs node embeddings for operations and machines

    This is a pure PyTorch implementation (no PyTorch Geometric dependency)
    suitable for small-to-medium instances. Can be upgraded to PyG later.
    """

    def __init__(
        self,
        op_feat_dim: int = 6,
        machine_feat_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.op_feat_dim = op_feat_dim
        self.machine_feat_dim = machine_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Initial embedding layers
        self.op_embedding = nn.Linear(op_feat_dim, hidden_dim)
        self.machine_embedding = nn.Linear(machine_feat_dim, hidden_dim)

        # Message passing layers for each relation type
        self.precedence_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.compatibility_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        # Layer normalization and dropout
        self.op_layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.machine_layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        op_features: torch.Tensor,
        machine_features: torch.Tensor,
        precedence_edges: torch.Tensor,
        compatibility_edges: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the heterogeneous GNN.

        Parameters
        ----------
        op_features : torch.Tensor [num_ops, op_feat_dim]
            Feature vectors for operation nodes
        machine_features : torch.Tensor [num_machines, machine_feat_dim]
            Feature vectors for machine nodes
        precedence_edges : torch.Tensor [2, num_prec_edges]
            Edge index for precedence relations (op -> op)
        compatibility_edges : torch.Tensor [2, num_compat_edges]
            Edge index for compatibility relations (op <-> machine)

        Returns
        -------
        Dict with:
        - 'op_embeddings': torch.Tensor [num_ops, hidden_dim]
        - 'machine_embeddings': torch.Tensor [num_machines, hidden_dim]
        """
        # Initial embeddings
        op_emb = self.op_embedding(op_features)
        machine_emb = self.machine_embedding(machine_features)

        # Multi-layer message passing
        for layer_idx in range(self.num_layers):
            # Precedence message passing (operation -> operation)
            op_emb_prec = self._message_passing_precedence(
                op_emb, precedence_edges, self.precedence_layers[layer_idx]
            )

            # Compatibility message passing (operation <-> machine)
            op_emb_compat, machine_emb_compat = self._message_passing_compatibility(
                op_emb,
                machine_emb,
                compatibility_edges,
                self.compatibility_layers[layer_idx],
            )

            # Combine messages for operations (precedence + compatibility)
            op_emb = op_emb + op_emb_prec + op_emb_compat
            op_emb = self.op_layer_norms[layer_idx](op_emb)
            op_emb = self.dropout(op_emb)

            # Update machine embeddings
            machine_emb = machine_emb + machine_emb_compat
            machine_emb = self.machine_layer_norms[layer_idx](machine_emb)
            machine_emb = self.dropout(machine_emb)

        return {
            "op_embeddings": op_emb,
            "machine_embeddings": machine_emb,
        }

    def _message_passing_precedence(
        self,
        op_emb: torch.Tensor,
        edge_index: torch.Tensor,
        linear: nn.Linear,
    ) -> torch.Tensor:
        """
        Message passing along precedence edges (op -> op).

        Simple aggregation: for each target op, aggregate messages from
        predecessor ops via mean pooling.
        """
        if edge_index.shape[1] == 0:
            return torch.zeros_like(op_emb)

        src_idx = edge_index[0]  # Source operation indices
        tgt_idx = edge_index[1]  # Target operation indices

        # Get source embeddings
        src_emb = op_emb[src_idx]
        src_emb_transformed = linear(src_emb)

        # Aggregate messages per target (mean pooling)
        num_ops = op_emb.shape[0]
        aggregated = torch.zeros_like(op_emb)
        counts = torch.zeros(num_ops, dtype=torch.long, device=op_emb.device)

        for i, tgt in enumerate(tgt_idx):
            aggregated[tgt] += src_emb_transformed[i]
            counts[tgt] += 1

        # Avoid division by zero
        counts = counts.clamp(min=1).float().unsqueeze(-1)
        aggregated = aggregated / counts

        return aggregated

    def _message_passing_compatibility(
        self,
        op_emb: torch.Tensor,
        machine_emb: torch.Tensor,
        edge_index: torch.Tensor,
        linear: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Message passing along compatibility edges (op <-> machine).

        The compatibility edges use a unified node space:
        - Operation nodes: 0..num_ops-1
        - Machine nodes: num_ops..num_ops+num_machines-1 (offset by num_ops)

        Edges come in bidirectional pairs:
        - (op_idx, m_idx + num_ops): operation -> machine
        - (m_idx + num_ops, op_idx): machine -> operation
        """
        if edge_index.shape[1] == 0:
            return torch.zeros_like(op_emb), torch.zeros_like(machine_emb)

        src_idx = edge_index[0]
        tgt_idx = edge_index[1]

        num_ops = op_emb.shape[0]
        num_machines = machine_emb.shape[0]

        # Separate edges into op->machine and machine->op
        op_to_machine_msgs = []
        machine_to_op_msgs = []

        for i in range(edge_index.shape[1]):
            src = src_idx[i].item()
            tgt = tgt_idx[i].item()

            if src < num_ops and tgt >= num_ops:
                # op -> machine: src is op index, tgt is machine index (offset)
                m_idx = tgt - num_ops
                op_to_machine_msgs.append((src, m_idx))
            elif src >= num_ops and tgt < num_ops:
                # machine -> op: src is machine index (offset), tgt is op index
                m_idx = src - num_ops
                machine_to_op_msgs.append((m_idx, tgt))

        # Aggregate messages for machines (from operations)
        machine_msg_agg = torch.zeros_like(machine_emb)
        machine_counts = torch.zeros(num_machines, dtype=torch.long, device=machine_emb.device)

        for op_idx, m_idx in op_to_machine_msgs:
            if op_idx < num_ops and m_idx < num_machines:
                msg = linear(op_emb[op_idx])
                machine_msg_agg[m_idx] += msg
                machine_counts[m_idx] += 1

        machine_counts = machine_counts.clamp(min=1).float().unsqueeze(-1)
        machine_msg_agg = machine_msg_agg / machine_counts

        # Aggregate messages for operations (from machines)
        op_msg_agg = torch.zeros_like(op_emb)
        op_counts = torch.zeros(num_ops, dtype=torch.long, device=op_emb.device)

        for m_idx, op_idx in machine_to_op_msgs:
            if m_idx < num_machines and op_idx < num_ops:
                msg = linear(machine_emb[m_idx])
                op_msg_agg[op_idx] += msg
                op_counts[op_idx] += 1

        op_counts = op_counts.clamp(min=1).float().unsqueeze(-1)
        op_msg_agg = op_msg_agg / op_counts

        return op_msg_agg, machine_msg_agg


def encode_state_with_gnn(
    graph_data: Dict[str, np.ndarray],
    model: HeterogeneousGNN,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to encode a graph state using the GNN model.

    Parameters
    ----------
    graph_data : Dict from build_graph_from_env_state
    model : HeterogeneousGNN
    device : Optional torch device (defaults to CPU)

    Returns
    -------
    Dict with 'op_embeddings' and 'machine_embeddings' tensors
    """
    if device is None:
        device = torch.device("cpu")

    # Convert numpy arrays to torch tensors
    op_features = torch.from_numpy(graph_data["op_features"]).float().to(device)
    machine_features = torch.from_numpy(graph_data["machine_features"]).float().to(device)
    precedence_edges = torch.from_numpy(graph_data["precedence_edges"]).long().to(device)
    compatibility_edges = torch.from_numpy(graph_data["compatibility_edges"]).long().to(device)

    model.eval()
    with torch.no_grad():
        embeddings = model(
            op_features, machine_features, precedence_edges, compatibility_edges
        )

    return embeddings


__all__ = ["HeterogeneousGNN", "encode_state_with_gnn"]
