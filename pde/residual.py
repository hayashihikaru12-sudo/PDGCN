from typing import Tuple

import torch


def compute_pde_residual(
    T_next,
    T_current,
    v_scan_star,
    Q_star,
    dt_star,
    edge_index,
    edge_attr,
    *,
    inverse_pe: float = 1.0,
    pi_q: float = 1.0,
    k_ratio: float = 0.05,
    eps: float = 1e-12,
):
    """Compute the dimensionless graph PDE residual at each node.

    Edge feature layout is [dx, dy, dz, d, cos_theta, cos_phi, cos_phi_sq].
    Inputs may be single-step tensors ([N], [N, 1]) or TBPTT windows
    ([K, N], [K, N, 1]). The returned residual follows the T_next shape.
    """

    T_next_2d, layout = _as_time_node(T_next, name="T_next")
    T_current_2d, _ = _as_time_node(T_current, name="T_current")
    Q_star_2d, _ = _as_time_node(Q_star, name="Q_star")
    T_current_2d = _broadcast_to_match(T_current_2d, T_next_2d, name="T_current")
    Q_star_2d = _broadcast_to_match(Q_star_2d, T_next_2d, name="Q_star")
    _validate_graph(edge_index, edge_attr, T_next_2d.shape[1])

    device = T_next_2d.device
    dtype = T_next_2d.dtype
    T_current_2d = T_current_2d.to(device=device, dtype=dtype)
    Q_star_2d = Q_star_2d.to(device=device, dtype=dtype)
    edge_index = edge_index.to(device=device)
    edge_attr = edge_attr.to(device=device, dtype=dtype)

    sender = edge_index[0]
    receiver = edge_index[1]
    distance = edge_attr[:, 3].clamp_min(eps)
    cos_theta = edge_attr[:, 4]
    cos_phi_sq = edge_attr[:, 6]
    upwind_weight = torch.relu(cos_theta)
    k_edge = cos_phi_sq + float(k_ratio) * (1.0 - cos_phi_sq)

    T_i = T_next_2d[:, receiver]
    T_j = T_next_2d[:, sender]

    v_scan = _as_time_scalar(v_scan_star, T_next_2d.shape[0], device=device, dtype=dtype)
    convection_edge = v_scan * upwind_weight.reshape(1, -1) * (T_i - T_j) / distance.reshape(1, -1)
    diffusion_edge = k_edge.reshape(1, -1) * (T_j - T_i) / distance.square().reshape(1, -1)

    convection = torch.zeros_like(T_next_2d)
    diffusion = torch.zeros_like(T_next_2d)
    convection.index_add_(1, receiver, convection_edge)
    diffusion.index_add_(1, receiver, diffusion_edge)

    transient = (T_next_2d - T_current_2d) / _as_scalar_tensor(dt_star, device=device, dtype=dtype).clamp_min(eps)
    residual = transient + convection - float(inverse_pe) * diffusion - float(pi_q) * Q_star_2d
    return _restore_layout(residual, layout)


def _as_time_node(value, *, name: str) -> Tuple[torch.Tensor, str]:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 1:
        return tensor.reshape(1, -1), "node"
    if tensor.ndim == 2:
        if tensor.shape[1] == 1:
            return tensor.reshape(1, tensor.shape[0]), "node_col"
        return tensor, "time_node"
    if tensor.ndim == 3 and tensor.shape[2] == 1:
        return tensor[:, :, 0], "time_node_col"
    raise ValueError(f"{name} must have shape [N], [N, 1], [K, N], or [K, N, 1], got {tuple(tensor.shape)}.")


def _restore_layout(value: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "node":
        return value[0]
    if layout == "node_col":
        return value[0].reshape(-1, 1)
    if layout == "time_node":
        return value
    if layout == "time_node_col":
        return value.unsqueeze(-1)
    raise ValueError(f"Unsupported layout: {layout}.")


def _broadcast_to_match(value, target, *, name: str):
    if value.shape == target.shape:
        return value
    if value.shape[0] == 1 and value.shape[1] == target.shape[1]:
        return value.expand(target.shape[0], -1)
    raise ValueError(f"{name} shape {tuple(value.shape)} must match T_next shape {tuple(target.shape)}.")


def _validate_graph(edge_index, edge_attr, num_nodes: int):
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")
    if edge_attr.ndim != 2 or edge_attr.shape[1] < 7:
        raise ValueError(f"edge_attr must have shape [E, >=7], got {tuple(edge_attr.shape)}.")
    if edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_attr must have one row per edge_index column.")
    if edge_index.numel() > 0:
        min_index = int(edge_index.min().item())
        max_index = int(edge_index.max().item())
        if min_index < 0 or max_index >= num_nodes:
            raise ValueError(f"edge_index values must be within [0, {num_nodes - 1}], got [{min_index}, {max_index}].")


def _as_time_scalar(value, num_steps: int, *, device, dtype):
    tensor = _as_scalar_tensor(value, device=device, dtype=dtype)
    if tensor.numel() == 1:
        return tensor.reshape(1, 1)
    if tensor.numel() == num_steps:
        return tensor.reshape(num_steps, 1)
    raise ValueError(f"v_scan_star must be scalar or have {num_steps} values, got {tensor.numel()}.")


def _as_scalar_tensor(value, *, device, dtype):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)
