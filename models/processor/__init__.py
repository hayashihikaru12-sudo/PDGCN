from .edge_block import EdgeBlock
from .node_block import NodeBlock

import torch.nn as nn


class GnBlock(nn.Module):
    def __init__(self, config):
        """初始化一层图网络消息传递块。

        参数:
            config: ``PDGCNConfig``，用于创建边更新块和节点更新块。

        返回:
            None。实例包含 ``EdgeBlock`` 和 ``NodeBlock``。
        """

        super().__init__()
        self.edge_block = EdgeBlock(config)
        self.node_block = NodeBlock(config)

    def forward(self, graph, raw_edge_attr):
        """执行一层边更新和节点更新，并加入残差连接。

        参数:
            graph: PyG ``Data`` 图对象，节点和边特征均已编码到隐空间。
            raw_edge_attr: 原始边特征张量，形状 ``[E, 7]``，供物理门控使用。

        返回:
            更新后的 ``graph``，``x`` 和 ``edge_attr`` 仍处于隐空间维度。
        """

        node_residual = graph.x
        edge_residual = graph.edge_attr

        graph = self.edge_block(graph, raw_edge_attr)
        graph.edge_attr = graph.edge_attr + edge_residual

        graph = self.node_block(graph)
        graph.x = graph.x + node_residual
        return graph


class Processor(nn.Module):
    def __init__(self, config):
        """初始化多层图消息传递处理器。

        参数:
            config: ``PDGCNConfig``，其中 ``message_passing_num`` 指定堆叠层数。

        返回:
            None。实例包含 ``message_passing_num`` 个 ``GnBlock``。
        """

        super().__init__()
        # 堆叠 message_passing_num 个 GnBlock，以执行多步消息传递。
        self.blocks = nn.ModuleList([GnBlock(config) for _ in range(config.message_passing_num)])

    def forward(self, graph, raw_edge_attr):
        """顺序执行多层 PD-GCN 消息传递。

        参数:
            graph: PyG ``Data`` 图对象，``x`` 和 ``edge_attr`` 已为隐空间张量。
            raw_edge_attr: 原始边特征张量，形状 ``[E, 7]``。

        返回:
            经过多层消息传递后的 ``graph``，节点隐状态用于后续解码。
        """

        for block in self.blocks:
            graph = block(graph, raw_edge_attr)
        return graph


__all__ = ["EdgeBlock", "GnBlock", "NodeBlock", "Processor"]
