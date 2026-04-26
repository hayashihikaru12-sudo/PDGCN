import torch.nn as nn

from .config import PDGCNConfig
from .mlp import build_mlp


class Decoder(nn.Module):
    def __init__(self, config: PDGCNConfig):
        """初始化温度残差解码器。

        参数:
            config: ``PDGCNConfig``，提供隐空间维度、输出维度和 dropout 配置。

        返回:
            None。实例会创建从 ``hidden_size`` 到 ``output_size`` 的 MLP。
        """

        super().__init__()
        self.decoder = build_mlp(
            config.hidden_size,
            config.hidden_size,
            config.output_size,
            layer_norm=False,
            dropout=config.dropout,
        )

    def forward(self, graph):
        """将节点隐状态解码为温度残差。

        参数:
            graph: PyG ``Data`` 图对象，``x`` 为节点隐状态，形状 ``[N, hidden_size]``。

        返回:
            温度残差张量，形状 ``[N, output_size]``，默认表示 ``delta_T*``。
        """

        return self.decoder(graph.x)
