import torch

from pde import project_non_heating_delta


def test_project_non_heating_delta_scales_positive_internal_sum():
    delta = torch.tensor([[10.0], [2.0], [-1.0], [10.0]])
    boundary_nodes = {
        "upwind": torch.tensor([0]),
        "side": torch.tensor([3]),
        "downwind": torch.empty(0, dtype=torch.long),
    }

    projected = project_non_heating_delta(delta, boundary_nodes)

    expected = torch.tensor([[10.0], [1.0], [-1.0], [10.0]])
    assert torch.allclose(projected, expected, atol=1e-6)
    assert torch.allclose(projected[1:3].sum(), torch.tensor(0.0), atol=1e-6)
    assert torch.equal(torch.sign(projected[1:3]), torch.sign(delta[1:3]))


def test_project_non_heating_delta_zeroes_positive_sum_without_negative_budget():
    delta = torch.tensor([[10.0], [2.0], [4.0], [10.0]])
    boundary_nodes = {
        "upwind": torch.tensor([0]),
        "side": torch.tensor([3]),
        "downwind": torch.empty(0, dtype=torch.long),
    }

    projected = project_non_heating_delta(delta, boundary_nodes)

    expected = torch.tensor([[10.0], [0.0], [0.0], [10.0]])
    assert torch.allclose(projected, expected, atol=1e-6)
    assert torch.allclose(projected[1:3].sum(), torch.tensor(0.0), atol=1e-6)


def test_project_non_heating_delta_keeps_non_positive_internal_mean():
    delta = torch.tensor([[10.0], [-3.0], [1.0], [10.0]])
    boundary_nodes = {
        "upwind": torch.tensor([0]),
        "side": torch.tensor([3]),
        "downwind": torch.empty(0, dtype=torch.long),
    }

    projected = project_non_heating_delta(delta, boundary_nodes)

    assert torch.allclose(projected, delta, atol=1e-6)


def test_project_non_heating_delta_applies_per_layer():
    delta = torch.tensor([[[1.0], [3.0]], [[-2.0], [0.0]]])

    projected = project_non_heating_delta(delta)

    expected = torch.tensor([[[0.0], [0.0]], [[-2.0], [0.0]]])
    assert torch.allclose(projected, expected, atol=1e-6)
    assert torch.all(projected.sum(dim=1) <= 1e-6)


def test_project_non_heating_delta_returns_clone_without_internal_nodes():
    delta = torch.tensor([[1.0], [2.0]])
    boundary_nodes = {
        "upwind": torch.tensor([0]),
        "side": torch.empty(0, dtype=torch.long),
        "downwind": torch.tensor([1]),
    }

    projected = project_non_heating_delta(delta, boundary_nodes)

    assert torch.allclose(projected, delta, atol=1e-6)
    assert projected.data_ptr() != delta.data_ptr()
