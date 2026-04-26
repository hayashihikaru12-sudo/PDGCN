from typing import Dict

import torch


TEMPERATURE_SLICE = slice(6, 7)
HEAT_SOURCE_SLICE = slice(7, 8)


def clone_graph_with_temperature(graph, temperature_star):
    """复制图对象并替换节点温度特征。

    参数:
        graph: PyG ``Data`` 图对象，``x`` 至少包含第 7 列温度特征。
        temperature_star: 新的无量纲温度张量，形状 ``[N, 1]``。

    返回:
        新的图对象，``x[:, 6:7]`` 已替换为 ``temperature_star``，
        其他字段继承自输入图。
    """

    cloned = graph.clone()
    cloned.x = graph.x.clone()
    cloned.x[:, TEMPERATURE_SLICE] = temperature_star.to(device=cloned.x.device, dtype=cloned.x.dtype)
    return cloned


def graph_temperature(graph):
    """读取图中的节点无量纲温度特征。

    参数:
        graph: PyG ``Data`` 图对象，``x`` 形状 ``[N, >=7]``。

    返回:
        温度特征张量，形状 ``[N, 1]``，对应 ``x[:, 6:7]``。
    """

    return graph.x[:, TEMPERATURE_SLICE]


def graph_heat_source(graph):
    """读取图中的节点无量纲热源特征。

    参数:
        graph: PyG ``Data`` 图对象，``x`` 形状 ``[N, >=8]``。

    返回:
        热源特征张量，形状 ``[N, 1]``，对应 ``x[:, 7:8]``。
    """

    return graph.x[:, HEAT_SOURCE_SLICE]


def graph_boundary_nodes(graph) -> Dict[str, torch.Tensor]:
    """从图对象中提取边界节点索引字典。

    参数:
        graph: PyG ``Data`` 图对象，需包含 ``upwind_nodes``、``side_nodes``、
            ``downwind_nodes`` 属性。

    返回:
        字典 ``{"upwind": ..., "side": ..., "downwind": ...}``，
        每个值为一维 ``torch.LongTensor`` 节点索引。
    """

    return {
        "upwind": graph.upwind_nodes,
        "side": graph.side_nodes,
        "downwind": graph.downwind_nodes,
    }


def graph_to_device(graph, device):
    """将图对象移动到指定设备。

    参数:
        graph: PyG ``Data`` 图对象。
        device: 目标设备；若为 ``None`` 则不移动。

    返回:
        位于目标设备的图对象；当 ``device=None`` 时返回原对象。
    """

    if device is None:
        return graph
    return graph.to(device)
