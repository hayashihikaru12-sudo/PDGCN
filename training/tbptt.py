from typing import List, Sequence

import torch

from pde import apply_dirichlet_boundary, total_loss

from .graph_utils import clone_graph_with_temperature, graph_boundary_nodes, graph_heat_source, graph_temperature


def iter_tbptt_windows(graph_seq: Sequence, window_size: int):
    """按固定长度切分 TBPTT 时间窗口。

    参数:
        graph_seq: 图序列，长度为 ``T``，每个元素通常为 PyG ``Data`` 对象。
        window_size: 每个截断窗口的最大长度 ``k``，必须为正整数。

    返回:
        生成器；每次产出一个列表窗口，长度不超过 ``window_size``，
        并保持原始时间顺序。
    """

    if int(window_size) <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    for start in range(0, len(graph_seq), int(window_size)):
        window = list(graph_seq[start : start + int(window_size)])
        if window:
            yield window


def rollout_window(model, window: Sequence, initial_temperature_star):
    """在一个 TBPTT 窗口内自回归滚动预测温度。

    参数:
        model: PD-GCN 模型，输入单步图对象并输出 ``delta_T*``，形状 ``[N, 1]``。
        window: 图对象序列，长度为 ``k``。
        initial_temperature_star: 窗口初始无量纲温度，形状 ``[N, 1]``。

    返回:
        ``(prediction_seq, final_temperature)`` 二元组：
        ``prediction_seq`` 形状 ``[k, N, 1]``，为每步预测温度；
        ``final_temperature`` 形状 ``[N, 1]``，为窗口末温度。
    """

    if not window:
        raise ValueError("window must contain at least one graph.")

    predictions = []
    current_temperature = initial_temperature_star
    for graph in window:
        graph_step = clone_graph_with_temperature(graph, current_temperature)
        delta_temperature = model(graph_step)
        next_temperature = current_temperature + delta_temperature
        next_temperature = apply_dirichlet_boundary(
            next_temperature,
            graph_boundary_nodes(graph_step),
            value=getattr(model.config, "dirichlet_temperature_star", 0.0),
        )
        predictions.append(next_temperature)
        current_temperature = next_temperature

    return torch.stack(predictions, dim=0), current_temperature


def train_tbptt_window(model, window: Sequence, initial_temperature_star):
    """计算单个 TBPTT 窗口的物理损失。

    参数:
        model: PD-GCN 模型，需带有 ``config`` 属性以读取物理损失参数。
        window: 图对象序列，长度为 ``k``。
        initial_temperature_star: 窗口初始无量纲温度，形状 ``[N, 1]``。

    返回:
        ``(loss, final_temperature)`` 二元组：
        ``loss`` 为标量张量，可直接反向传播；
        ``final_temperature`` 形状 ``[N, 1]``，用于下一个窗口的初值。
    """

    prediction_seq, final_temperature = rollout_window(model, window, initial_temperature_star)
    losses = []
    current_temperature = initial_temperature_star

    for step, graph in enumerate(window):
        prediction = prediction_seq[step]
        loss = total_loss(
            T_next=prediction,
            T_current=current_temperature,
            v_scan_star=graph.global_attr,
            Q_star=graph_heat_source(graph),
            dt_star=model.config.dt_star,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            boundary_nodes=graph_boundary_nodes(graph),
            inverse_pe=model.config.inverse_pe,
            pi_q=model.config.pi_q,
            k_ratio=model.config.k_ratio,
            lambda_outflow=model.config.lambda_outflow,
            dirichlet_temperature_star=model.config.dirichlet_temperature_star,
            thermal_loss_beta=model.config.thermal_loss_beta,
            thermal_loss_base_temperature_star=model.config.thermal_loss_base_temperature_star,
            residual_time_scheme=model.config.residual_time_scheme,
        )
        losses.append(loss)
        current_temperature = prediction

    return torch.stack(losses).mean(), final_temperature


def initial_temperature_from_graph_seq(graph_seq: Sequence):
    """从图序列首帧读取初始无量纲温度。

    参数:
        graph_seq: 非空图对象序列，首个图的 ``x[:, 6:7]`` 为初温。

    返回:
        初始温度张量，形状 ``[N, 1]``。
    """

    if not graph_seq:
        raise ValueError("graph_seq must contain at least one graph.")
    return graph_temperature(graph_seq[0])
