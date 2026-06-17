import torch.nn as nn

from .config import PDGCNConfig
from .decoder import Decoder
from .encoder import Encoder
from .processor import Processor


class PDGCN(nn.Module):
    def __init__(self, config: PDGCNConfig = None):
        """初始化 PD-GCN 主模型。

        参数:
            config: 可选 ``PDGCNConfig``；若为 ``None``，使用默认配置。

        返回:
            None。实例包含 ``encoder``、``processor`` 和 ``decoder`` 三个子模块。
        """

        super().__init__()
        self.config = config or PDGCNConfig()
        self.encoder = Encoder(self.config)
        self.processor = Processor(self.config)
        self.decoder = Decoder(self.config)

    def forward(self, graph):
        """执行一次 PD-GCN 前向传播并预测温度残差。

        参数:
            graph: ``torch_geometric.data.Data`` 图对象，需包含 ``x`` 形状
                ``[N, node_input_size]``、``edge_index`` 形状 ``[2, E]``、
                ``edge_attr`` 形状 ``[E, 7]``，以及可选的 ``global_attr``。

        返回:
            温度残差张量 ``delta_T*``，形状 ``[N, output_size]``，默认 ``[N, 1]``。
        """

        raw_edge_attr = graph.edge_attr
        graph = self.encoder(graph)
        graph = self.processor(graph, raw_edge_attr)
        return self.decoder(graph)
