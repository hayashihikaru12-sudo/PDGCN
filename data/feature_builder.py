from typing import Optional

import torch
from torch_geometric.data import Data

from .dimensionless import (
    ScaleParams,
    coordinates_to_dimensionless,
    heat_source_to_dimensionless,
    temperature_to_dimensionless,
    velocity_to_dimensionless,
)
from .initial_condition import generate_initial_temperature
from .loader import GraphRawData
from .velocity import tangent_velocity_direction


def build_node_type(num_nodes: int, boundary_nodes, *, device=None) -> torch.Tensor:
    """构建节点类型标记。

    参数:
        num_nodes: 图中的节点数量 ``N``。
        boundary_nodes: 边界节点字典，包含 ``upwind``、``downwind``、``side``
            三类一维索引张量。
        device: 可选目标设备。

    返回:
        ``torch.LongTensor``，形状 ``[N]``；内部节点为 ``0``，任意边界节点为 ``1``。
    """

    node_type = torch.zeros(num_nodes, dtype=torch.long, device=device)
    boundary_tensors = [boundary_nodes[name].to(device=device, dtype=torch.long) for name in ("upwind", "downwind", "side")]
    boundary_index = torch.unique(torch.cat(boundary_tensors, dim=0))
    node_type[boundary_index] = 1
    return node_type


def build_node_features(nodes_star, fibers, temperature_star, q_star) -> torch.Tensor:
    """拼接 PD-GCN 节点输入特征。

    参数:
        nodes_star: 无量纲节点坐标，形状 ``[N, 3]``。
        fibers: 节点纤维方向，形状 ``[N, 3]``，函数内部会归一化为单位向量。
        temperature_star: 无量纲节点温度，形状 ``[N]`` 或 ``[N, 1]``。
        q_star: 无量纲节点热源强度，形状 ``[N]`` 或 ``[N, 1]``。

    返回:
        节点特征张量，形状 ``[N, 8]``，列为
        ``[x*, y*, z*, fx, fy, fz, T*, Q*]``。
    """

    fibers_unit = _normalize_vectors(fibers)
    return torch.cat(
        [
            nodes_star,
            fibers_unit,
            temperature_star.reshape(nodes_star.shape[0], 1),
            q_star.reshape(nodes_star.shape[0], 1),
        ],
        dim=-1,
    )


def build_edge_features(nodes_star, edge_index, fibers, normals, velocity_direction, eps: float = 1e-12) -> torch.Tensor:
    """根据节点坐标和纤维方向构建 PD-GCN 边特征。

    参数:
        nodes_star: 无量纲节点坐标，形状 ``[N, 3]``。
        edge_index: 图边索引，形状 ``[2, E]``，第一行为 source，第二行为 receiver。
        fibers: 节点纤维方向，形状 ``[N, 3]``。
        normals: 节点曲面法向，形状 ``[N, 3]``。
        velocity_direction: 文件级速度方向，形状 ``[3]``，会投影到接收节点切平面。
        eps: 距离和向量范数下界，用于避免除零。

    返回:
        边特征张量，形状 ``[E, 7]``，列为
        ``[dx, dy, dz, d, cos_theta, cos_phi, cos_phi_sq]``。
    """

    source, receiver = edge_index[0], edge_index[1]
    delta = nodes_star[receiver] - nodes_star[source]
    distance = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(eps)
    direction = delta / distance

    tangent_velocity = tangent_velocity_direction(velocity_direction, normals, eps=eps)
    cos_theta = torch.sum(tangent_velocity[receiver] * direction, dim=-1, keepdim=True).clamp(-1.0, 1.0)

    fibers_unit = _normalize_vectors(fibers, eps=eps)
    fiber_mid = _normalize_vectors(fibers_unit[source] + fibers_unit[receiver], eps=eps)
    cos_phi = torch.sum(fiber_mid * direction, dim=-1, keepdim=True).clamp(-1.0, 1.0)
    cos_phi_sq = cos_phi.square()

    return torch.cat([delta, distance, cos_theta, cos_phi, cos_phi_sq], dim=-1)


def build_global_condition(scan_velocity, scale_params: ScaleParams, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """构建全局工艺条件特征。

    参数:
        scan_velocity: 真实扫描速度，类型可为标量或张量。
        scale_params: ``ScaleParams`` 实例，使用其中 ``v0`` 转换为无量纲速度。
        device: 可选目标设备。
        dtype: 目标数据类型，默认为 ``torch.float32``。

    返回:
        形状 ``[1]`` 的张量，表示无量纲扫描速度 ``v_scan*``。
    """

    return velocity_to_dimensionless(scan_velocity, scale_params, device=device, dtype=dtype).reshape(1)


def build_graph(
    raw_data: GraphRawData,
    scale_params: ScaleParams,
    scan_velocity,
    initial_temperature: Optional[torch.Tensor] = None,
    relaxation_steps: int = 20,
) -> Data:
    """将单帧原始数据转换为 PyG ``Data`` 图对象。

    参数:
        raw_data: ``GraphRawData``，包含原始坐标、纤维方向、热源、边和边界节点。
        scale_params: ``ScaleParams`` 实例，用于坐标、温度、热源和速度无量纲化。
        scan_velocity: 真实扫描速度，会转换为 ``graph.global_attr``。
        initial_temperature: 可选真实温度张量，形状 ``[N]`` 或 ``[N, 1]``；
            若为 ``None``，则使用数据层 legacy 图扩散方法填充图中初温。
            训练入口会默认用当前 PD-GCN 权重重新生成 warmup 初温。
        relaxation_steps: 未提供初温时 legacy 图扩散方法的迭代步数。

    返回:
        ``torch_geometric.data.Data`` 图对象，包含：
        ``x`` 形状 ``[N, 8]``、``edge_index`` 形状 ``[2, E]``、
        ``edge_attr`` 形状 ``[E, 7]``、``global_attr`` 形状 ``[1]``，
        以及边界节点索引属性。
    """

    device = raw_data.xyz.device
    dtype = raw_data.xyz.dtype

    nodes_star = coordinates_to_dimensionless(raw_data.xyz, scale_params)
    q_star = heat_source_to_dimensionless(raw_data.q, scale_params)

    if initial_temperature is None:
        temperature_star = generate_initial_temperature(
            raw_data.edge_index,
            q_star,
            raw_data.boundary_nodes,
            raw_data.xyz.shape[0],
            M=relaxation_steps,
            x_star=nodes_star,
        )
    else:
        temperature = initial_temperature.to(device=device, dtype=dtype)
        temperature_star = temperature_to_dimensionless(temperature.reshape(raw_data.xyz.shape[0], 1), scale_params)

    node_features = build_node_features(nodes_star, raw_data.fiber, temperature_star, q_star)
    edge_features = build_edge_features(
        nodes_star,
        raw_data.edge_index,
        raw_data.fiber,
        raw_data.normal,
        raw_data.velocity_direction,
        eps=scale_params.eps,
    )
    node_type = build_node_type(raw_data.xyz.shape[0], raw_data.boundary_nodes, device=device)
    global_attr = build_global_condition(scan_velocity, scale_params, device=device, dtype=dtype)

    graph = Data(
        x=node_features,
        edge_index=raw_data.edge_index,
        edge_attr=edge_features,
        node_type=node_type,
        global_attr=global_attr,
        pos=nodes_star,
    )
    graph.num_nodes = raw_data.xyz.shape[0]
    graph.frame_idx = raw_data.frame_idx
    graph.num_frames = raw_data.num_frames
    graph.normal = raw_data.normal
    graph.velocity_direction = raw_data.velocity_direction
    graph.upwind_nodes = raw_data.boundary_nodes["upwind"]
    graph.downwind_nodes = raw_data.boundary_nodes["downwind"]
    graph.side_nodes = raw_data.boundary_nodes["side"]

    return graph


def _normalize_vectors(vectors, eps: float = 1e-12):
    """将向量沿最后一维归一化为单位向量。

    参数:
        vectors: 向量张量，形状通常为 ``[N, 3]`` 或 ``[E, 3]``。
        eps: 范数下界，用于避免零向量导致除零。

    返回:
        与 ``vectors`` 形状一致的单位向量张量。
    """

    norm = torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)
    return vectors / norm
