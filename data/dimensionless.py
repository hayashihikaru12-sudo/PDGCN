from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class ScaleParams:
    """Characteristic scales used by the PD-GCN data pipeline.

    K0, rho, Cp are optional: they do not participate in tensor-level
    nondimensionalization but are required by ``derive_pde_constants`` to
    compute the scalar coefficients ``inverse_pe`` and ``pi_q``.
    """

    L0: float
    v0: float
    T_amb: float
    delta_T0: float
    Q0: float
    K0: Optional[float] = None
    rho: Optional[float] = None
    Cp: Optional[float] = None
    eps: float = 1e-12

    def __post_init__(self):
        positive_fields = ("L0", "v0", "delta_T0", "Q0", "eps")
        for field_name in positive_fields:
            value = float(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}.")
        for field_name in ("K0", "rho", "Cp"):
            value = getattr(self, field_name)
            if value is not None and float(value) <= 0:
                raise ValueError(f"{field_name} must be positive when provided, got {value}.")


def derive_pde_constants(scale_params: "ScaleParams") -> Tuple[float, float]:
    """Compute (inverse_pe, pi_q) from characteristic scales.

    ``1/Pe = K0 / (rho * Cp * v0 * L0)`` and
    ``pi_q = Q0 * L0 / (rho * Cp * v0 * delta_T0)``.
    Requires ``K0``, ``rho``, and ``Cp`` to be set on ``scale_params``.
    """

    missing = [name for name in ("K0", "rho", "Cp") if getattr(scale_params, name) is None]
    if missing:
        raise ValueError(
            f"derive_pde_constants requires {missing} to be set on ScaleParams."
        )

    rho = float(scale_params.rho)
    Cp = float(scale_params.Cp)
    v0 = float(scale_params.v0)
    L0 = float(scale_params.L0)
    K0 = float(scale_params.K0)
    delta_T0 = float(scale_params.delta_T0)
    Q0 = float(scale_params.Q0)

    denom = rho * Cp * v0
    if denom <= 0:
        raise ValueError("rho * Cp * v0 must be positive to derive PDE constants.")

    inverse_pe = K0 / (denom * L0)
    pi_q = (Q0 * L0) / (denom * delta_T0)
    return inverse_pe, pi_q


def coordinates_to_dimensionless(coordinates, scale_params: ScaleParams):
    return coordinates / scale_params.L0


def coordinates_from_dimensionless(coordinates_star, scale_params: ScaleParams):
    return coordinates_star * scale_params.L0


def temperature_to_dimensionless(temperature, scale_params: ScaleParams):
    return (temperature - scale_params.T_amb) / scale_params.delta_T0


def temperature_from_dimensionless(temperature_star, scale_params: ScaleParams):
    return temperature_star * scale_params.delta_T0 + scale_params.T_amb


def heat_source_to_dimensionless(q, scale_params: ScaleParams):
    return q / scale_params.Q0


def heat_source_from_dimensionless(q_star, scale_params: ScaleParams):
    return q_star * scale_params.Q0


def velocity_to_dimensionless(scan_velocity, scale_params: ScaleParams, *, device=None, dtype=torch.float32):
    if torch.is_tensor(scan_velocity):
        velocity = scan_velocity
        if device is not None:
            velocity = velocity.to(device=device)
        if dtype is not None:
            velocity = velocity.to(dtype=dtype)
    else:
        velocity = torch.tensor(scan_velocity, device=device, dtype=dtype)
    return velocity / scale_params.v0


def to_dimensionless(node_features, edge_features, global_condition, scale_params: ScaleParams):
    """Convert raw PD-GCN feature tensors to the dimensionless feature layout.

    Expected node layout: [x, y, z, fx, fy, fz, T, Q].
    Expected edge layout: [dx, dy, dz, d, cos_theta, cos_phi, cos_phi_sq].
    Expected global layout: [scan_velocity].
    """

    node_star = node_features.clone()
    edge_star = edge_features.clone()

    node_star[:, 0:3] = coordinates_to_dimensionless(node_star[:, 0:3], scale_params)
    node_star[:, 6:7] = temperature_to_dimensionless(node_star[:, 6:7], scale_params)
    node_star[:, 7:8] = heat_source_to_dimensionless(node_star[:, 7:8], scale_params)

    edge_star[:, 0:4] = edge_star[:, 0:4] / scale_params.L0
    global_star = velocity_to_dimensionless(
        global_condition,
        scale_params,
        device=global_condition.device if torch.is_tensor(global_condition) else None,
        dtype=global_condition.dtype if torch.is_tensor(global_condition) else torch.float32,
    ).reshape(-1)

    return node_star, edge_star, global_star
