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
    """Generate a dimensionless quasi-steady initial temperature by graph relaxation."""

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
    selected = [boundary_nodes[name].to(device=device, dtype=torch.long) for name in names if name in boundary_nodes]
    if not selected:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.unique(torch.cat(selected, dim=0))
