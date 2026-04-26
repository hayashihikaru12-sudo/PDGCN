import torch.nn as nn

from .config import PDGCNConfig
from .mlp import build_mlp


class Decoder(nn.Module):
    def __init__(self, config: PDGCNConfig):
        super().__init__()
        self.decoder = build_mlp(
            config.hidden_size,
            config.hidden_size,
            config.output_size,
            layer_norm=False,
            dropout=config.dropout,
        )

    def forward(self, graph):
        return self.decoder(graph.x)
