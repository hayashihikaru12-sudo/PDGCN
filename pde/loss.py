from typing import Dict, Iterable, Optional

import torch

from .residual import _as_time_node, compute_pde_residual


def apply_dirichlet_boundary(T, boundary_nodes: Optional[Dict[str, torch.Tensor]], *, value: float = 0.0):
    """Clamp upwind and side boundary nodes to the dimensionless Dirichlet value."""

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
    """Compute the dimensionless soft Neumann outflow loss on downwind nodes."""

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
    return_components: bool = False,
    eps: float = 1e-12,
):
    """Compute L_total = L_PDE + lambda_outflow * L_outflow."""

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
    all_nodes = torch.arange(num_nodes, device=device, dtype=torch.long)
    boundary = _concat_boundary_nodes(boundary_nodes, ("upwind", "side", "downwind"), device=device)
    if boundary.numel() == 0:
        return all_nodes
    mask = torch.ones(num_nodes, device=device, dtype=torch.bool)
    mask[boundary] = False
    return all_nodes[mask]
