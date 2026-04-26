from .edge_block import EdgeBlock
from .node_block import NodeBlock

import torch.nn as nn


class GnBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.edge_block = EdgeBlock(config)
        self.node_block = NodeBlock(config)

    def forward(self, graph, raw_edge_attr):
        node_residual = graph.x
        edge_residual = graph.edge_attr

        graph = self.edge_block(graph, raw_edge_attr)
        graph.edge_attr = graph.edge_attr + edge_residual

        graph = self.node_block(graph)
        graph.x = graph.x + node_residual
        return graph


class Processor(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Stack GnBlock `message_passing_num` times to perform multi-step message passing.
        self.blocks = nn.ModuleList([GnBlock(config) for _ in range(config.message_passing_num)])

    def forward(self, graph, raw_edge_attr):
        for block in self.blocks:
            graph = block(graph, raw_edge_attr)
        return graph


__all__ = ["EdgeBlock", "GnBlock", "NodeBlock", "Processor"]
