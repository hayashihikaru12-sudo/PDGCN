import numpy as np
import torch


LOCAL_VELOCITY_FRAME = "nip_local_velocity_side_normal"


def resolve_velocity_direction_local(h5_file):
    """Resolve the HDF5 file-level velocity direction in local coordinates."""

    if "velocity_direction_local" in h5_file.attrs:
        direction = np.asarray(h5_file.attrs["velocity_direction_local"], dtype=np.float32)
    elif str(h5_file.attrs.get("coordinate_frame", "")) == LOCAL_VELOCITY_FRAME:
        direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        raise KeyError(
            "HDF5 file must provide root attr 'velocity_direction_local' when coordinate_frame "
            f"is not {LOCAL_VELOCITY_FRAME!r}."
        )
    if direction.shape != (3,):
        raise ValueError(f"velocity_direction_local must have shape [3], got {direction.shape}.")
    if not np.all(np.isfinite(direction)):
        raise ValueError("velocity_direction_local must contain finite values.")
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        raise ValueError("velocity_direction_local must be non-zero.")
    return direction / norm


def tangent_velocity_direction(velocity_direction, normals, *, eps: float = 1e-12):
    """Project a velocity direction onto each node tangent plane and normalize it."""

    normal_unit = _normalize_vectors(normals, eps=eps)
    velocity = torch.as_tensor(velocity_direction, device=normal_unit.device, dtype=normal_unit.dtype)
    if velocity.ndim == 1:
        velocity = velocity.reshape(1, 3).expand_as(normal_unit)
    elif velocity.shape != normal_unit.shape:
        raise ValueError(
            "velocity_direction must have shape [3] or match normals shape, "
            f"got {tuple(velocity.shape)} for normals {tuple(normal_unit.shape)}."
        )
    velocity_unit = _normalize_vectors(velocity, eps=eps)
    tangent = velocity_unit - (velocity_unit * normal_unit).sum(dim=-1, keepdim=True) * normal_unit
    tangent_norm = torch.linalg.norm(tangent, dim=-1, keepdim=True)
    if bool((tangent_norm <= float(eps)).any().item()):
        raise ValueError("velocity direction is nearly parallel to at least one surface normal.")
    return tangent / tangent_norm


def _normalize_vectors(vectors, *, eps: float):
    norm = torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)
    return vectors / norm
