from typing import Dict, Iterable, Optional

import torch

from .residual import (
    _as_time_node,
    _broadcast_to_match,
    _restore_layout,
    compute_pde_residual,
)


def apply_dirichlet_boundary(T, boundary_nodes: Optional[Dict[str, torch.Tensor]], *, value: float = 0.0):
    """将迎风和侧边界节点钳制为 Dirichlet 温度。

    参数:
        T: 无量纲温度张量，允许形状 ``[N]``、``[N, 1]``、``[K, N]``
            或 ``[K, N, 1]``。
        boundary_nodes: 边界节点字典；函数使用其中 ``upwind`` 和 ``side``。
        value: 无量纲 Dirichlet 温度值，默认 ``0.0`` 表示环境温度。

    返回:
        与 ``T`` 形状一致的新张量；指定边界节点已被替换为 ``value``。
    """

    nodes = _concat_boundary_nodes(boundary_nodes, ("upwind", "side"), device=torch.as_tensor(T).device)
    if nodes.numel() == 0:
        return T.clone()

    output = T.clone()
    fill_value = torch.as_tensor(value, device=output.device, dtype=output.dtype)
    if output.ndim == 1:
        output[nodes] = fill_value
    elif output.ndim == 2 and output.shape[1] == 1:
        output[nodes, :] = fill_value
    elif output.ndim == 2:
        output[:, nodes] = fill_value
    elif output.ndim == 3 and output.shape[2] == 1:
        output[:, nodes, :] = fill_value
    else:
        raise ValueError(f"T must have shape [N], [N, 1], [K, N], or [K, N, 1], got {tuple(output.shape)}.")
    return output


def compute_outflow_loss(T, edge_index, edge_attr, outflow_nodes, *, eps: float = 1e-12):
    """计算尾迹出流边界的无量纲 Neumann 软约束损失。

    参数:
        T: 无量纲温度张量，允许形状 ``[N]``、``[N, 1]``、``[K, N]``
            或 ``[K, N, 1]``。
        edge_index: 图边索引，形状 ``[2, E]``。
        edge_attr: 原始边特征，形状 ``[E, >=7]``，使用 ``d`` 和 ``cos_theta``。
        outflow_nodes: 尾迹边界节点索引，一维张量或可转为张量的序列。
        eps: 分母下界，用于避免除零。

    返回:
        标量张量，表示出流边界法向温度梯度平方均值。
    """

    T_2d, _ = _as_time_node(T, name="T")
    device = T_2d.device
    dtype = T_2d.dtype
    outflow_nodes = torch.as_tensor(outflow_nodes, device=device, dtype=torch.long).reshape(-1)
    if outflow_nodes.numel() == 0:
        return T_2d.new_zeros(())

    edge_index = edge_index.to(device=device)
    edge_attr = edge_attr.to(device=device, dtype=dtype)
    sender = edge_index[0]
    receiver = edge_index[1]
    distance = edge_attr[:, 3].clamp_min(eps)
    weight = torch.relu(edge_attr[:, 4])

    edge_gradient = weight.reshape(1, -1) * (T_2d[:, receiver] - T_2d[:, sender]) / distance.reshape(1, -1)
    weighted_gradient_sum = torch.zeros_like(T_2d)
    weight_sum = torch.zeros_like(T_2d)
    weighted_gradient_sum.index_add_(1, receiver, edge_gradient)
    weight_sum.index_add_(1, receiver, weight.reshape(1, -1).expand(T_2d.shape[0], -1))

    normal_gradient = weighted_gradient_sum[:, outflow_nodes] / weight_sum[:, outflow_nodes].clamp_min(eps)
    return normal_gradient.square().mean()


def compute_zero_source_anchor_loss(delta_T_star, boundary_nodes=None):
    """计算冷态零源条件下 PD-GCN 输出增量的内部节点均方误差。

    参数:
        delta_T_star: PD-GCN 在 ``Q*=0, T*=T_ref*`` 图上的输出张量，
            形状 ``[N]``、``[N, 1]``、``[K, N]`` 或 ``[K, N, 1]``。
        boundary_nodes: 边界节点字典；只对内部节点统计锚定损失。

    返回:
        标量张量，等于 ``mean((delta_T_star)^2)``，仅在内部节点上聚合。
    """

    delta_2d, _ = _as_time_node(delta_T_star, name="delta_T_star")
    interior_nodes = _interior_nodes(delta_2d.shape[1], boundary_nodes, device=delta_2d.device)
    if interior_nodes.numel() == 0:
        return delta_2d.square().mean()
    return delta_2d[:, interior_nodes].square().mean()


def compute_graph_gradient_loss(T, edge_index, edge_attr, boundary_nodes=None, *, eps: float = 1e-12):
    """计算内部边上的一阶图梯度平滑损失。"""

    T_2d, _ = _as_time_node(T, name="T")
    device = T_2d.device
    dtype = T_2d.dtype
    edge_index = edge_index.to(device=device)
    edge_attr = edge_attr.to(device=device, dtype=dtype)
    if edge_index.numel() == 0:
        return T_2d.new_zeros(())

    sender = edge_index[0]
    receiver = edge_index[1]
    interior_nodes = _interior_nodes(T_2d.shape[1], boundary_nodes, device=device)
    if interior_nodes.numel() == 0:
        return T_2d.new_zeros(())

    interior_mask = torch.zeros(T_2d.shape[1], device=device, dtype=torch.bool)
    interior_mask[interior_nodes] = True
    edge_mask = interior_mask[sender] & interior_mask[receiver]
    if not bool(edge_mask.any().item()):
        return T_2d.new_zeros(())

    distance = edge_attr[edge_mask, 3].clamp_min(eps)
    gradient = (T_2d[:, receiver[edge_mask]] - T_2d[:, sender[edge_mask]]) / distance.reshape(1, -1)
    return gradient.square().mean()


def total_loss(
    *,
    T_next,
    T_current,
    v_scan_star,
    Q_star=None,
    dt_star,
    edge_index,
    edge_attr,
    boundary_nodes: Optional[Dict[str, torch.Tensor]] = None,
    inverse_pe: float = 1.0,
    pi_q: float = 1.0,
    k_ratio: float = 0.05,
    lambda_outflow: float = 1.0,
    gradient_regularization: float = 0.0,
    dirichlet_temperature_star: float = 0.0,
    thermal_loss_beta: float = 0.0,
    thermal_loss_base_temperature_star=0.0,
    residual_time_scheme: str = "explicit",
    zero_source_anchor_delta=None,
    zero_source_anchor_weight: float = 0.0,
    return_components: bool = False,
    eps: float = 1e-12,
):
    """计算 PD-GCN 纯物理总损失。

    参数:
        T_next: 下一步无量纲温度预测，形状 ``[N]``、``[N, 1]``、
            ``[K, N]`` 或 ``[K, N, 1]``。
        T_current: 当前无量纲温度，形状同 ``T_next`` 或可广播到 ``T_next``。
        v_scan_star: 无量纲扫描速度，标量或长度为 ``K`` 的张量。
        Q_star: 兼容旧调用的保留参数；无源残差中不再使用。
        dt_star: 无量纲时间步长。
        edge_index: 图边索引，形状 ``[2, E]``。
        edge_attr: 原始边特征，形状 ``[E, >=7]``。
        boundary_nodes: 边界节点字典，使用 ``upwind``、``side`` 和 ``downwind``。
        inverse_pe: 佩克莱特数倒数。
        pi_q: 兼容旧调用的保留参数；无源残差中不再使用。
        k_ratio: 横向/纵向导热系数比。
        lambda_outflow: 出流边界损失权重。
        gradient_regularization: 图梯度平滑损失权重，用于抑制预测温度的高频振荡。
        dirichlet_temperature_star: 硬 Dirichlet 边界的无量纲温度值。
        thermal_loss_beta: 兼容旧调用的保留参数；无源残差中不再使用。
        thermal_loss_base_temperature_star: 兼容旧调用的保留参数；无源残差中不再使用。
        residual_time_scheme: PDE 空间项和热耗散项的时间离散方式；
            ``explicit`` 使用当前温度，``backward`` 使用预测温度。
        zero_source_anchor_delta: 冷态零源图上 PD-GCN 输出的无量纲温度增量
            ``delta_T0*``，用于零源锚定损失。形状与 ``T_next`` 兼容；若为 ``None``
            或权重为 ``0``，则锚定损失为 ``0`` 且不入总损失。
        zero_source_anchor_weight: 零源锚定损失权重 ``lambda_zero``，默认 ``0``。
            当大于 ``0`` 且提供 ``zero_source_anchor_delta`` 时，
            ``lambda_zero * mean((delta_T0*)^2)`` 累加进总损失。
        return_components: 是否返回损失分量和中间张量。
        eps: 数值下界，用于防止除零。

    返回:
        若 ``return_components=False``，返回标量总损失张量；
        否则返回字典，包含 ``loss_total``、``loss_pde``、``loss_outflow``、
        ``loss_smooth``、``residual`` 和 ``T_next_bc``。
    """

    T_next_bc = apply_dirichlet_boundary(
        T_next,
        boundary_nodes,
        value=dirichlet_temperature_star,
    )
    residual = compute_pde_residual(
        T_next=T_next_bc,
        T_current=T_current,
        v_scan_star=v_scan_star,
        dt_star=dt_star,
        edge_index=edge_index,
        edge_attr=edge_attr,
        Q_star=Q_star,
        inverse_pe=inverse_pe,
        pi_q=pi_q,
        k_ratio=k_ratio,
        thermal_loss_beta=thermal_loss_beta,
        thermal_loss_base_temperature_star=thermal_loss_base_temperature_star,
        residual_time_scheme=residual_time_scheme,
        eps=eps,
    )

    residual_2d, _ = _as_time_node(residual, name="residual")
    T_next_bc_2d, t_next_layout = _as_time_node(T_next_bc, name="T_next_bc")
    thermal_loss_term_2d = torch.zeros_like(T_next_bc_2d.to(device=residual_2d.device, dtype=residual_2d.dtype))

    interior_nodes = _interior_nodes(residual_2d.shape[1], boundary_nodes, device=residual_2d.device)
    if interior_nodes.numel() == 0:
        loss_pde = residual_2d.square().mean()
        loss_beta = thermal_loss_term_2d.square().mean()
    else:
        loss_pde = residual_2d[:, interior_nodes].square().mean()
        loss_beta = thermal_loss_term_2d[:, interior_nodes].square().mean()

    outflow_nodes = _concat_boundary_nodes(boundary_nodes, ("downwind",), device=residual_2d.device)
    loss_outflow = compute_outflow_loss(T_next_bc, edge_index, edge_attr, outflow_nodes, eps=eps)
    loss_smooth = compute_graph_gradient_loss(T_next_bc, edge_index, edge_attr, boundary_nodes, eps=eps)
    if zero_source_anchor_delta is not None:
        loss_zero_source_anchor = compute_zero_source_anchor_loss(
            zero_source_anchor_delta,
            boundary_nodes=boundary_nodes,
        ).to(device=residual_2d.device, dtype=residual_2d.dtype)
    else:
        loss_zero_source_anchor = residual_2d.new_zeros(())
    loss_total = (
        loss_pde
        + float(lambda_outflow) * loss_outflow
        + float(gradient_regularization) * loss_smooth
        + float(zero_source_anchor_weight) * loss_zero_source_anchor
    )

    if not return_components:
        return loss_total

    return {
        "loss_total": loss_total,
        "loss_pde": loss_pde,
        "loss_transport": loss_pde,
        "loss_outflow": loss_outflow,
        "loss_beta": loss_beta,
        "loss_smooth": loss_smooth,
        "loss_zero_source_anchor": loss_zero_source_anchor,
        "residual": residual,
        "T_next_bc": T_next_bc,
        "thermal_loss_term": _restore_layout(thermal_loss_term_2d, t_next_layout),
    }


def _concat_boundary_nodes(boundary_nodes: Optional[Dict[str, torch.Tensor]], names: Iterable[str], *, device):
    """从边界字典中合并指定类别节点。

    参数:
        boundary_nodes: 边界节点字典，值为一维索引张量；可为 ``None``。
        names: 要合并的边界名称序列。
        device: 返回索引张量所在设备。

    返回:
        一维 ``torch.LongTensor``，包含指定边界类别的唯一索引；
        若没有可用边界则返回空张量。
    """

    if not boundary_nodes:
        return torch.empty(0, device=device, dtype=torch.long)
    selected = [
        torch.as_tensor(boundary_nodes[name], device=device, dtype=torch.long).reshape(-1)
        for name in names
        if name in boundary_nodes and boundary_nodes[name] is not None
    ]
    if not selected:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.unique(torch.cat(selected, dim=0))


def _interior_nodes(num_nodes: int, boundary_nodes: Optional[Dict[str, torch.Tensor]], *, device):
    """计算不属于任意边界的内部节点索引。

    参数:
        num_nodes: 图节点数量 ``N``。
        boundary_nodes: 边界节点字典；使用 ``upwind``、``side`` 和 ``downwind``。
        device: 返回索引张量所在设备。

    返回:
        一维 ``torch.LongTensor``，包含内部节点索引；若未提供边界则返回所有节点。
    """

    all_nodes = torch.arange(num_nodes, device=device, dtype=torch.long)
    boundary = _concat_boundary_nodes(boundary_nodes, ("upwind", "side", "downwind"), device=device)
    if boundary.numel() == 0:
        return all_nodes
    mask = torch.ones(num_nodes, device=device, dtype=torch.bool)
    mask[boundary] = False
    return all_nodes[mask]
