import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PDGCNConfig
from ..mlp import build_mlp


class EdgeBlock(nn.Module):
    def __init__(self, config: PDGCNConfig):
        super().__init__()
        self.config = config
        self.message_mlp = build_mlp(
            3 * config.hidden_size,
            config.hidden_size,
            config.hidden_size,
            layer_norm=config.layer_norm,
            dropout=config.dropout,
        )
        self.aniso_mlp = build_mlp(
            2,
            config.hidden_size,
            config.hidden_size,
            layer_norm=False,
            dropout=config.dropout,
        )
        self.last_alpha = None
        self.last_aniso_gate = None

    def forward(self, graph, raw_edge_attr):
        sender, receiver = graph.edge_index
        sender_attr = graph.x[sender]
        receiver_attr = graph.x[receiver]
        # Deviation from the plan: the planned formula is m_ij = MLP_msg(h_j, e_ij^(0))
        # (using only the receiver-side neighbor feature h_j), while this implementation
        # passes both sender and receiver features.
        # This is a PIGNN-style "two-sided message" with stronger expressiveness.
        # It is physically harmless, but not fully consistent with the design formula.
        # It is recommended to document the actual implementation explicitly to avoid
        # inconsistencies between training code and paper documentation.
        message_input = torch.cat([sender_attr, receiver_attr, graph.edge_attr], dim=-1)
        message = self.message_mlp(message_input)

        alpha = self._upwind_gate(raw_edge_attr)
        aniso_gate = self._aniso_gate(raw_edge_attr, message)
        graph.edge_attr = alpha * aniso_gate * message

        self.last_alpha = alpha.detach()
        self.last_aniso_gate = aniso_gate.detach()
        return graph

    def _upwind_gate(self, raw_edge_attr):
        cos_theta = raw_edge_attr[:, 4:5]
        return F.relu(1.0 + float(self.config.gamma_upwind) * cos_theta)

    def _aniso_gate(self, raw_edge_attr, message):
        if not self.config.use_aniso_gate:
            return torch.ones_like(message)
        gate_input = raw_edge_attr[:, 5:7].to(device=message.device, dtype=message.dtype)
        # Sigmoid outputs values in the range [0, 1]. Is this constraint important,
        # and is sigmoid necessary here?

        # If you prioritize maximum physical interpretability, you can remove this MLP
        # and directly broadcast raw_edge_attr[:, 6:7] (i.e., $\cos^2\phi$) to 64 dimensions
        # and multiply it with the message. If you prioritize lower loss, keeping the
        # current MLP + sigmoid design is usually a better choice.
        return torch.sigmoid(self.aniso_mlp(gate_input))
