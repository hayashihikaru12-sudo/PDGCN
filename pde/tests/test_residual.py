import torch

from pde.residual import compute_pde_residual


def _edge_attr(distance, cos_theta, cos_phi_sq):
    rows = []
    for d, theta, phi_sq in zip(distance, cos_theta, cos_phi_sq):
        rows.append([0.0, 0.0, 0.0, d, theta, 0.0, phi_sq])
    return torch.tensor(rows, dtype=torch.float32)


def test_compute_pde_residual_matches_hand_calculation():
    edge_index = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)
    edge_attr = _edge_attr(
        distance=[2.0, 1.0],
        cos_theta=[1.0, -1.0],
        cos_phi_sq=[1.0, 0.0],
    )
    T_next = torch.tensor([[1.0], [3.0], [5.0]])
    T_current = torch.tensor([[0.0], [1.0], [2.0]])
    Q_star = torch.full((3, 1), 0.1)

    residual = compute_pde_residual(
        T_next=T_next,
        T_current=T_current,
        v_scan_star=2.0,
        Q_star=Q_star,
        dt_star=2.0,
        edge_index=edge_index,
        edge_attr=edge_attr,
        inverse_pe=0.5,
        pi_q=2.0,
        k_ratio=0.1,
    )

    expected = torch.tensor([[0.3], [2.95], [1.3]])
    assert residual.shape == T_next.shape
    assert torch.allclose(residual, expected, atol=1e-6)


def test_compute_pde_residual_supports_tbptt_window_shape():
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_attr = _edge_attr(distance=[1.0], cos_theta=[1.0], cos_phi_sq=[1.0])
    T_next = torch.tensor([[[1.0], [2.0]], [[2.0], [4.0]]])
    T_current = torch.zeros_like(T_next)
    Q_star = torch.zeros((2, 1))

    residual = compute_pde_residual(
        T_next=T_next,
        T_current=T_current,
        v_scan_star=torch.tensor([1.0, 2.0]),
        Q_star=Q_star,
        dt_star=1.0,
        edge_index=edge_index,
        edge_attr=edge_attr,
        inverse_pe=1.0,
        pi_q=1.0,
        k_ratio=0.05,
    )

    assert residual.shape == T_next.shape
    assert torch.isfinite(residual).all()
