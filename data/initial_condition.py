from typing import Dict, Optional

import torch


def generate_initial_temperature(
    edge_index: torch.Tensor,
    q_star: torch.Tensor,
    boundary_nodes: Dict[str, torch.Tensor],
    num_nodes: int,
    M: int = 20,
    dtau: float = 0.1,
    diffusion: float = 0.2,
    source_weight: float = 1.0,
    upwind_bias: float = 0.1,
    x_star: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """通过 legacy 图扩散松弛生成无量纲准稳态初始温度。

    训练入口默认使用当前 PD-GCN 权重执行伪时间 warmup；本函数仅作为
    数据层无模型依赖的构图 fallback。

    参数:
        edge_index: 图边索引，形状 ``[2, E]``，第一行为 source，第二行为 receiver。
        q_star: 无量纲热源张量，形状 ``[N]`` 或 ``[N, 1]``。
        boundary_nodes: 边界节点字典，至少可包含 ``upwind`` 和 ``side``。
        num_nodes: 图节点数量 ``N``。
        M: 松弛迭代次数；为 ``0`` 时直接返回冷态零温度。
        dtau: 伪时间步长，用于更新松弛温度。
        diffusion: 图扩散项权重。
        source_weight: 热源项权重。
        upwind_bias: 沿局部 x 方向的迎风权重偏置强度。
        x_star: 可选无量纲坐标张量，形状 ``[N, 3]``；提供时用于构造迎风偏置。

    返回:
        无量纲初始温度张量，形状 ``[N, 1]``；迎风和侧边界节点被钳制为 ``0``。
    """

    if M < 0:
        raise ValueError(f"M must be non-negative, got {M}.")
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")

    device = q_star.device
    dtype = q_star.dtype
    q = q_star.reshape(num_nodes, 1)
    temperature = torch.zeros((num_nodes, 1), device=device, dtype=dtype)

    source, receiver = edge_index[0], edge_index[1]
    degree = torch.zeros((num_nodes, 1), device=device, dtype=dtype)
    degree.index_add_(0, receiver, torch.ones((receiver.numel(), 1), device=device, dtype=dtype))
    degree = degree.clamp_min(1.0)

    edge_weight = torch.ones((receiver.numel(), 1), device=device, dtype=dtype)
    if x_star is not None and upwind_bias != 0:
        x = x_star.reshape(num_nodes, -1)[:, 0:1]
        dx = x[receiver] - x[source]
        edge_weight = torch.relu(1.0 + float(upwind_bias) * torch.sign(dx))

    clamped_nodes = _concat_unique_boundary(boundary_nodes, ("upwind", "side"), device=device)

    for _ in range(M):
        neighbor_delta = (temperature[source] - temperature[receiver]) * edge_weight
        diffusion_sum = torch.zeros_like(temperature)
        diffusion_sum.index_add_(0, receiver, neighbor_delta)
        laplace_like = diffusion_sum / degree

        temperature = temperature + float(dtau) * (float(diffusion) * laplace_like + float(source_weight) * q)
        temperature = temperature.clamp_min(0.0)
        if clamped_nodes.numel() > 0:
            temperature[clamped_nodes] = 0.0

    return temperature


def _concat_unique_boundary(boundary_nodes, names, *, device):
    """合并指定边界类别的节点索引并去重。

    参数:
        boundary_nodes: 边界节点字典，值为一维索引张量。
        names: 要合并的边界名称序列，例如 ``("upwind", "side")``。
        device: 输出索引张量所在设备。

    返回:
        一维 ``torch.LongTensor``，包含指定边界类别的唯一节点索引；
        若没有匹配边界，则返回空张量。
    """

    selected = [boundary_nodes[name].to(device=device, dtype=torch.long) for name in names if name in boundary_nodes]
    if not selected:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.unique(torch.cat(selected, dim=0))
