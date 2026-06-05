from typing import Dict, Optional

import torch

from .residual import _as_time_node, _restore_layout


def project_non_heating_delta(
    delta_T_star,
    boundary_nodes: Optional[Dict[str, torch.Tensor]] = None,
    *,
    eps: float = 1e-12,
):
    """Project source-free in-plane increments without changing their signs.

    The projection is applied only on non-boundary internal nodes. For each
    time/layer slice, positive increments are scaled down only when their
    total exceeds the negative-increment magnitude:

    ``delta_proj = alpha * relu(delta_raw) - relu(-delta_raw)``.

    With uniformly sampled nodes, the unweighted node sum is used. Boundary
    nodes are left unchanged.
    """

    delta_2d, layout = _as_time_node(delta_T_star, name="delta_T_star")
    projected = delta_2d.clone()
    internal_mask = _internal_node_mask(projected.shape[1], boundary_nodes, device=projected.device)
    if not bool(internal_mask.any().item()):
        return _restore_layout(projected, layout)

    internal_delta = projected[:, internal_mask]
    positive = torch.relu(internal_delta)
    negative = torch.relu(-internal_delta)
    positive_sum = positive.sum(dim=1, keepdim=True)
    negative_sum = negative.sum(dim=1, keepdim=True)
    scale = torch.where(
        positive_sum > negative_sum,
        negative_sum / positive_sum.clamp_min(float(eps)),
        torch.ones_like(positive_sum),
    )
    projected[:, internal_mask] = scale * positive - negative
    return _restore_layout(projected, layout)


def _internal_node_mask(num_nodes: int, boundary_nodes: Optional[Dict[str, torch.Tensor]], *, device):
    mask = torch.ones(int(num_nodes), device=device, dtype=torch.bool)
    if not boundary_nodes:
        return mask

    selected = [
        torch.as_tensor(boundary_nodes[name], device=device, dtype=torch.long).reshape(-1)
        for name in ("upwind", "side", "downwind")
        if name in boundary_nodes and boundary_nodes[name] is not None
    ]
    if not selected:
        return mask

    boundary = torch.unique(torch.cat(selected, dim=0))
    if boundary.numel() > 0:
        mask[boundary] = False
    return mask
