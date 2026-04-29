from typing import Dict, Iterable, Optional

import torch

from .residual import _as_time_node, compute_pde_residual


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


def total_loss(
    *,
    T_next,
    T_current,
    v_scan_star,
    Q_star,
    dt_star,
    edge_index,
    edge_attr,
    boundary_nodes: Optional[Dict[str, torch.Tensor]] = None,
    inverse_pe: float = 1.0,
    pi_q: float = 1.0,
    k_ratio: float = 0.05,
    lambda_outflow: float = 1.0,
    dirichlet_temperature_star: float = 0.0,
    thermal_loss_beta: float = 0.0,
    thermal_loss_base_temperature_star=0.0,
    residual_time_scheme: str = "explicit",
    return_components: bool = False,
    eps: float = 1e-12,
):
    """计算 PD-GCN 纯物理总损失。

    参数:
        T_next: 下一步无量纲温度预测，形状 ``[N]``、``[N, 1]``、
            ``[K, N]`` 或 ``[K, N, 1]``。
        T_current: 当前无量纲温度，形状同 ``T_next`` 或可广播到 ``T_next``。
        v_scan_star: 无量纲扫描速度，标量或长度为 ``K`` 的张量。
        Q_star: 无量纲热源张量，形状同温度或可广播。
        dt_star: 无量纲时间步长。
        edge_index: 图边索引，形状 ``[2, E]``。
        edge_attr: 原始边特征，形状 ``[E, >=7]``。
        boundary_nodes: 边界节点字典，使用 ``upwind``、``side`` 和 ``downwind``。
        inverse_pe: 佩克莱特数倒数。
        pi_q: 无量纲热源强度系数。
        k_ratio: 横向/纵向导热系数比。
        lambda_outflow: 出流边界损失权重。
        dirichlet_temperature_star: 硬 Dirichlet 边界的无量纲温度值。
        thermal_loss_beta: 无量纲层间等效热耗散系数 ``beta``。
        thermal_loss_base_temperature_star: 无量纲基底温度 ``T_base*``，
            单层训练默认 ``0.0`` 表示冷源。
        residual_time_scheme: PDE 空间项和热耗散项的时间离散方式；
            ``explicit`` 使用当前温度，``backward`` 使用预测温度。
        return_components: 是否返回损失分量和中间张量。
        eps: 数值下界，用于防止除零。

    返回:
        若 ``return_components=False``，返回标量总损失张量；
        否则返回字典，包含 ``loss_total``、``loss_pde``、``loss_outflow``、
        ``residual`` 和 ``T_next_bc``。
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
        Q_star=Q_star,
        dt_star=dt_star,
        edge_index=edge_index,
        edge_attr=edge_attr,
        inverse_pe=inverse_pe,
        pi_q=pi_q,
        k_ratio=k_ratio,
        thermal_loss_beta=thermal_loss_beta,
        thermal_loss_base_temperature_star=thermal_loss_base_temperature_star,
        residual_time_scheme=residual_time_scheme,
        eps=eps,
    )

    residual_2d, _ = _as_time_node(residual, name="residual")
    interior_nodes = _interior_nodes(residual_2d.shape[1], boundary_nodes, device=residual_2d.device)
    if interior_nodes.numel() == 0:
        loss_pde = residual_2d.square().mean()
    else:
        loss_pde = residual_2d[:, interior_nodes].square().mean()

    outflow_nodes = _concat_boundary_nodes(boundary_nodes, ("downwind",), device=residual_2d.device)
    loss_outflow = compute_outflow_loss(T_next_bc, edge_index, edge_attr, outflow_nodes, eps=eps)
    loss_total = loss_pde + float(lambda_outflow) * loss_outflow

    if not return_components:
        return loss_total

    return {
        "loss_total": loss_total,
        "loss_pde": loss_pde,
        "loss_outflow": loss_outflow,
        "residual": residual,
        "T_next_bc": T_next_bc,
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
