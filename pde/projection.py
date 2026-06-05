from typing import Dict, Optional

import torch

from .residual import _as_time_node, _restore_layout


def project_non_heating_delta(delta_T_star, boundary_nodes: Optional[Dict[str, torch.Tensor]] = None):
    """Project source-free in-plane increments to remove positive mean drift.

    The projection is applied only on non-boundary internal nodes. For each
    time/layer slice, a positive internal-node mean is subtracted from all
    internal increments:

    ``delta_proj = delta_raw - relu(mean_internal(delta_raw))``.
    """

    delta_2d, layout = _as_time_node(delta_T_star, name="delta_T_star")
    projected = delta_2d.clone()
    internal_mask = _internal_node_mask(projected.shape[1], boundary_nodes, device=projected.device)
    if not bool(internal_mask.any().item()):
        return _restore_layout(projected, layout)

    internal_delta = projected[:, internal_mask]
    correction = torch.relu(internal_delta.mean(dim=1, keepdim=True))
    projected[:, internal_mask] = internal_delta - correction
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
