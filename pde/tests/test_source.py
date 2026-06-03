import torch

from data.dimensionless import ScaleParams
from pde.source import compute_surface_source_delta_star, compute_surface_source_delta_star_from_physical


def test_surface_source_delta_star_uses_split_source_coefficient():
    q_surface_star = torch.tensor([[0.0], [0.5], [1.0]])

    delta = compute_surface_source_delta_star(
        q_surface_star,
        dt_star=0.2,
        source_coefficient=3.0,
        absorptivity=0.5,
    )

    assert torch.allclose(delta, torch.tensor([[0.0], [0.15], [0.30]]), atol=1e-6)


def test_surface_source_delta_from_physical_matches_heat_capacity_formula():
    scale = ScaleParams(
        L0=2.0,
        v0=4.0,
        T_amb=300.0,
        delta_T0=10.0,
        Q0=100.0,
        rho=2.0,
        Cp=5.0,
        heat_source_effective_thickness=0.5,
        heat_source_absorptivity=0.25,
    )
    q_surface = torch.tensor([[100.0], [200.0]])

    delta = compute_surface_source_delta_star_from_physical(q_surface, scale, dt=0.1)

    assert torch.allclose(delta, torch.tensor([[0.05], [0.10]]), atol=1e-6)
