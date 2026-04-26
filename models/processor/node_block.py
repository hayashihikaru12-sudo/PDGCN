import torch
import torch.nn as nn
from torch_scatter import scatter_add

from ..config import PDGCNConfig
from ..mlp import build_mlp


class NodeBlock(nn.Module):
    def __init__(self, config: PDGCNConfig):
        super().__init__()
        self.update_mlp = build_mlp(
            2 * config.hidden_size,
            config.hidden_size,
            config.hidden_size,
            layer_norm=config.layer_norm,
            dropout=config.dropout,
        )

    def forward(self, graph):
        _, receiver = graph.edge_index
        num_nodes = graph.num_nodes
        aggregated = scatter_add(graph.edge_attr, receiver, dim=0, dim_size=num_nodes)
        update_input = torch.cat([graph.x, aggregated], dim=-1)
        graph.x = self.update_mlp(update_input)
        return graph
