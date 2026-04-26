import torch
import torch.nn as nn
from torch_scatter import scatter_add

from ..config import PDGCNConfig
from ..mlp import build_mlp


class NodeBlock(nn.Module):
    def __init__(self, config: PDGCNConfig):
        """初始化节点状态更新模块。

        参数:
            config: ``PDGCNConfig``，提供隐空间维度、归一化和 dropout 配置。

        返回:
            None。实例会创建接收 ``[node, aggregated_message]`` 的更新 MLP。
        """

        super().__init__()
        self.update_mlp = build_mlp(
            2 * config.hidden_size,
            config.hidden_size,
            config.hidden_size,
            layer_norm=config.layer_norm,
            dropout=config.dropout,
        )

    def forward(self, graph):
        """聚合入边消息并更新节点隐状态。

        参数:
            graph: PyG ``Data`` 图对象，``x`` 形状 ``[N, hidden_size]``，
                ``edge_index`` 形状 ``[2, E]``，``edge_attr`` 形状
                ``[E, hidden_size]``。

        返回:
            更新后的 ``graph``，其中 ``x`` 形状仍为 ``[N, hidden_size]``。
        """

        _, receiver = graph.edge_index
        num_nodes = graph.num_nodes
        aggregated = scatter_add(graph.edge_attr, receiver, dim=0, dim_size=num_nodes)
        update_input = torch.cat([graph.x, aggregated], dim=-1)
        graph.x = self.update_mlp(update_input)
        return graph
