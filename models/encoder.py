import torch
import torch.nn as nn
from torch_geometric.data import Data

from .config import PDGCNConfig
from .mlp import build_mlp


class Encoder(nn.Module):
    def __init__(self, config: PDGCNConfig):
        super().__init__()
        self.config = config
        self.node_encoder = build_mlp(
            config.encoder_node_input_size,
            config.hidden_size,
            config.hidden_size,
            layer_norm=config.layer_norm,
            dropout=config.dropout,
        )
        self.edge_encoder = build_mlp(
            config.edge_input_size,
            config.hidden_size,
            config.hidden_size,
            layer_norm=config.layer_norm,
            dropout=config.dropout,
        )

    def forward(self, graph: Data) -> Data:
        graph = _copy_data(graph)
        node_attr = graph.x
        if self.config.include_global:
            node_attr = _append_global_condition(node_attr, graph.global_attr)

        graph.x = self.node_encoder(node_attr)
        graph.edge_attr = self.edge_encoder(graph.edge_attr)
        return graph


def _append_global_condition(node_attr: torch.Tensor, global_attr: torch.Tensor) -> torch.Tensor:
    if global_attr is None:
        raise ValueError("graph.global_attr is required when include_global=True.")

    global_flat = global_attr.reshape(1, -1).to(device=node_attr.device, dtype=node_attr.dtype)
    global_per_node = global_flat.expand(node_attr.shape[0], -1)
    return torch.cat([node_attr, global_per_node], dim=-1)


def _copy_data(graph: Data) -> Data:
    return Data(**{key: graph[key] for key in graph.keys})
