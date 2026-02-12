from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .graph_builder import build_graph_from_env_state
from .hgnn_encoder import HeterogeneousGNN, encode_state_with_gnn


class FJSPActorCritic(nn.Module):
    """
    Actor-Critic policy network for FJSP scheduling.

    Architecture:
    - Shared HGNN encoder produces node embeddings
    - Actor head: scores feasible actions by combining operation and machine embeddings
    - Critic head: estimates state value from pooled node embeddings

    This is a simplified single-agent policy (not yet dual-agent MAPPO).
    Actions are indices into the current feasible_actions list from FJSPEnv.
    """

    def __init__(
        self,
        op_feat_dim: int = 6,
        machine_feat_dim: int = 2,
        hidden_dim: int = 128,
        gnn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Shared GNN encoder
        self.gnn = HeterogeneousGNN(
            op_feat_dim=op_feat_dim,
            machine_feat_dim=machine_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            dropout=dropout,
        )

        # Actor head: scores actions by combining op and machine embeddings
        # For an action (job_id, op_index, machine_id), we combine:
        # - op_embedding[op_node_idx] and machine_embedding[machine_idx]
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Single scalar score per action
        )

        # Critic head: estimates state value from global graph representation
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # op_pool + machine_pool
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # State value
        )

    def forward(
        self,
        graph_data: Dict[str, np.ndarray],
        feasible_actions: Optional[List[Tuple[int, int, int]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through actor-critic network.

        Parameters
        ----------
        graph_data : Dict from build_graph_from_env_state
        feasible_actions : Optional list of (job_id, op_index, machine_id) tuples
            If None, action logits will not be computed (critic-only forward)

        Returns
        -------
        Dict with:
        - 'action_logits': torch.Tensor [num_feasible_actions] (if feasible_actions provided)
        - 'value': torch.Tensor [1] (state value estimate)
        """
        # Encode graph with GNN
        embeddings = encode_state_with_gnn(graph_data, self.gnn)
        op_emb = embeddings["op_embeddings"]  # [num_ops, hidden_dim]
        machine_emb = embeddings["machine_embeddings"]  # [num_machines, hidden_dim]

        # Build mapping from (job_id, op_index) to op_node_idx
        op_node_ids = graph_data["op_node_ids"]
        op_id_to_idx = {op_id: idx for idx, op_id in enumerate(op_node_ids)}

        # Critic: global state value from pooled embeddings
        op_pool = op_emb.mean(dim=0)  # [hidden_dim]
        machine_pool = machine_emb.mean(dim=0)  # [hidden_dim]
        global_state = torch.cat([op_pool, machine_pool], dim=-1)  # [hidden_dim * 2]
        value = self.critic_head(global_state).squeeze(-1)  # [1] -> scalar

        result = {"value": value}

        # Actor: score each feasible action
        if feasible_actions is not None:
            action_scores = []
            for job_id, op_idx, machine_id in feasible_actions:
                op_node_id = (job_id, op_idx)
                if op_node_id not in op_id_to_idx:
                    # This shouldn't happen if feasible_actions is correct, but handle gracefully
                    action_scores.append(torch.tensor(-1e9, device=op_emb.device))
                    continue

                op_node_idx = op_id_to_idx[op_node_id]
                if machine_id >= machine_emb.shape[0]:
                    action_scores.append(torch.tensor(-1e9, device=op_emb.device))
                    continue

                # Combine op and machine embeddings for this action
                op_emb_action = op_emb[op_node_idx]  # [hidden_dim]
                machine_emb_action = machine_emb[machine_id]  # [hidden_dim]
                action_embedding = torch.cat([op_emb_action, machine_emb_action], dim=-1)  # [hidden_dim * 2]

                # Score via actor head
                score = self.actor_head(action_embedding).squeeze(-1)  # scalar
                action_scores.append(score)

            action_logits = torch.stack(action_scores)  # [num_feasible_actions]
            result["action_logits"] = action_logits

        return result

    def get_action_and_value(
        self,
        graph_data: Dict[str, np.ndarray],
        feasible_actions: List[Tuple[int, int, int]],
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Convenience method: get action, logits, and value in one call.

        Parameters
        ----------
        graph_data : Dict from build_graph_from_env_state
        feasible_actions : List of (job_id, op_index, machine_id) tuples
        deterministic : If True, select argmax; else sample from softmax

        Returns
        -------
        (action_index, action_logits, value)
        """
        output = self.forward(graph_data, feasible_actions)
        action_logits = output["action_logits"]
        value = output["value"]

        if deterministic:
            action_index = action_logits.argmax().item()
        else:
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action_index = action_dist.sample().item()

        return action_index, action_logits, value


__all__ = ["FJSPActorCritic"]
