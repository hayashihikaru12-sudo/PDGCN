import torch.nn as nn

from .config import PDGCNConfig
from .decoder import Decoder
from .encoder import Encoder
from .processor import Processor


class PDGCN(nn.Module):
    def __init__(self, config: PDGCNConfig = None):
        super().__init__()
        self.config = config or PDGCNConfig()
        self.encoder = Encoder(self.config)
        self.processor = Processor(self.config)
        self.decoder = Decoder(self.config)

    def forward(self, graph):
        raw_edge_attr = graph.edge_attr
        graph = self.encoder(graph)
        graph = self.processor(graph, raw_edge_attr)
        return self.decoder(graph)
