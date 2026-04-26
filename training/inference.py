from typing import Sequence, Union

import torch

from data.dimensionless import temperature_from_dimensionless

from .graph_utils import clone_graph_with_temperature, graph_boundary_nodes, graph_to_device
from .warmup import pseudo_time_relax_initial_temperature
from pde import apply_dirichlet_boundary


@torch.no_grad()
def rollout(
    model,
    graph_init_or_seq: Union[object, Sequence],
    steps: int,
    scale_params,
    *,
    return_dimensionless: bool = False,
    warmup_steps: int = 0,
):
    """使用训练好的模型进行自回归滚动推理。

    参数:
        model: PD-GCN 模型，输入单步图并输出 ``delta_T*``，推理期间自动切换到 eval。
        graph_init_or_seq: 单个图对象或图对象序列；若为单图，则所有步复用同一图结构。
        steps: 推理步数，必须为正整数。
        scale_params: ``ScaleParams`` 实例，用于将无量纲温度还原为真实温度。
        return_dimensionless: 是否同时返回无量纲温度序列。
        warmup_steps: 可选 PD-GCN 伪时间松弛步数；默认 ``0`` 表示直接使用图中温度。

    返回:
        若 ``return_dimensionless=False``，返回真实温度张量，形状 ``[steps, N, 1]``；
        否则返回字典，包含 ``temperature`` 和 ``temperature_star`` 两个同形状张量。
    """

    if int(steps) <= 0:
        raise ValueError(f"steps must be positive, got {steps}.")

    model_device = next(model.parameters()).device
    graphs = _as_graph_sequence(graph_init_or_seq, int(steps), model_device)
    current_temperature = pseudo_time_relax_initial_temperature(model, graphs[0], int(warmup_steps))
    predictions_star = []

    was_training = model.training
    model.eval()
    try:
        for step in range(int(steps)):
            graph = graphs[step]
            graph_step = clone_graph_with_temperature(graph, current_temperature)
            delta_temperature = model(graph_step)
            next_temperature = current_temperature + delta_temperature
            next_temperature = apply_dirichlet_boundary(
                next_temperature,
                graph_boundary_nodes(graph_step),
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            predictions_star.append(next_temperature)
            current_temperature = next_temperature
    finally:
        if was_training:
            model.train()

    temperature_star = torch.stack(predictions_star, dim=0)
    temperature = temperature_from_dimensionless(temperature_star, scale_params)
    if return_dimensionless:
        return {"temperature": temperature, "temperature_star": temperature_star}
    return temperature


def _as_graph_sequence(graph_init_or_seq, steps: int, device):
    """将单图或图序列规范化为指定长度的图序列。

    参数:
        graph_init_or_seq: 单个 PyG 图对象，或长度不少于 ``steps`` 的列表/元组。
        steps: 需要的推理步数。
        device: 图对象应移动到的目标设备。

    返回:
        长度为 ``steps`` 的图对象列表，每个图已移动到 ``device``。
    """

    if isinstance(graph_init_or_seq, (list, tuple)):
        if len(graph_init_or_seq) < steps:
            raise ValueError(f"graph sequence length {len(graph_init_or_seq)} is shorter than steps={steps}.")
        return [graph_to_device(graph, device) for graph in graph_init_or_seq[:steps]]
    graph = graph_to_device(graph_init_or_seq, device)
    return [graph] * steps
