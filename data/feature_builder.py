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


def build_node_type(num_nodes: int, boundary_nodes, *, device=None) -> torch.Tensor:
    node_type = torch.zeros(num_nodes, dtype=torch.long, device=device)
    boundary_tensors = [boundary_nodes[name].to(device=device, dtype=torch.long) for name in ("upwind", "downwind", "side")]
    boundary_index = torch.unique(torch.cat(boundary_tensors, dim=0))
    node_type[boundary_index] = 1
    return node_type


def build_node_features(nodes_star, fibers, temperature_star, q_star) -> torch.Tensor:
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


def build_edge_features(nodes_star, edge_index, fibers, eps: float = 1e-12) -> torch.Tensor:
    source, receiver = edge_index[0], edge_index[1]
    delta = nodes_star[receiver] - nodes_star[source]
    distance = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(eps)
    direction = delta / distance

    cos_theta = direction[:, 0:1]

    fibers_unit = _normalize_vectors(fibers, eps=eps)
    fiber_mid = _normalize_vectors(fibers_unit[source] + fibers_unit[receiver], eps=eps)
    cos_phi = torch.sum(fiber_mid * direction, dim=-1, keepdim=True).clamp(-1.0, 1.0)
    cos_phi_sq = cos_phi.square()

    return torch.cat([delta, distance, cos_theta, cos_phi, cos_phi_sq], dim=-1)


def build_global_condition(scan_velocity, scale_params: ScaleParams, *, device=None, dtype=torch.float32) -> torch.Tensor:
    return velocity_to_dimensionless(scan_velocity, scale_params, device=device, dtype=dtype).reshape(1)


def build_graph(
    raw_data: GraphRawData,
    scale_params: ScaleParams,
    scan_velocity,
    initial_temperature: Optional[torch.Tensor] = None,
    relaxation_steps: int = 20,
) -> Data:
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
    edge_features = build_edge_features(nodes_star, raw_data.edge_index, raw_data.fiber, eps=scale_params.eps)
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
    graph.upwind_nodes = raw_data.boundary_nodes["upwind"]
    graph.downwind_nodes = raw_data.boundary_nodes["downwind"]
    graph.side_nodes = raw_data.boundary_nodes["side"]

    return graph


def _normalize_vectors(vectors, eps: float = 1e-12):
    norm = torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)
    return vectors / norm
