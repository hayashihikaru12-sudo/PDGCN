import torch
import unittest

from pde.loss import apply_dirichlet_boundary, compute_graph_gradient_loss, compute_outflow_loss, total_loss


def _edge_attr(distance, cos_theta, cos_phi_sq=None):
    """构造用于损失测试的边特征张量。

    参数:
        distance: 边距离列表，长度为 ``E``。
        cos_theta: 边与扫描方向夹角余弦列表，长度为 ``E``。
        cos_phi_sq: 可选余弦平方列表；若为 ``None``，默认全部为 ``1.0``。

    返回:
        ``torch.FloatTensor``，形状 ``[E, 7]``。
    """

    if cos_phi_sq is None:
        cos_phi_sq = [1.0] * len(distance)
    rows = []
    for d, theta, phi_sq in zip(distance, cos_theta, cos_phi_sq):
        rows.append([0.0, 0.0, 0.0, d, theta, 0.0, phi_sq])
    return torch.tensor(rows, dtype=torch.float32)


def test_apply_dirichlet_boundary_clamps_upwind_and_side_nodes():
    """验证 Dirichlet 硬边界只钳制迎风和侧边界节点。

    参数:
        None。

    返回:
        None。断言钳制结果和原输入未被原地修改。
    """

    T = torch.tensor([[5.0], [6.0], [7.0], [8.0]])
    boundary_nodes = {
        "upwind": torch.tensor([0]),
        "side": torch.tensor([3]),
        "downwind": torch.tensor([2]),
    }

    clamped = apply_dirichlet_boundary(T, boundary_nodes, value=0.0)

    assert torch.allclose(clamped, torch.tensor([[0.0], [6.0], [7.0], [0.0]]))
    assert torch.allclose(T, torch.tensor([[5.0], [6.0], [7.0], [8.0]]))


def test_compute_outflow_loss_matches_weighted_gradient():
    """验证出流损失与手工加权梯度计算一致。

    参数:
        None。

    返回:
        None。断言标量损失值。
    """

    edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    edge_attr = _edge_attr(distance=[1.0, 2.0], cos_theta=[1.0, 3.0])
    T = torch.tensor([[1.0], [3.0], [7.0]])

    loss = compute_outflow_loss(T, edge_index, edge_attr, torch.tensor([2]))

    expected_gradient = ((7.0 - 1.0) / 1.0 * 1.0 + (7.0 - 3.0) / 2.0 * 3.0) / (1.0 + 3.0)
    assert torch.allclose(loss, torch.tensor(expected_gradient**2), atol=1e-6)


def test_compute_graph_gradient_loss_uses_only_internal_edges():
    """验证图梯度平滑损失只统计两端均为内部节点的边。"""

    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = _edge_attr(distance=[1.0, 2.0, 1.0], cos_theta=[0.0, 0.0, 0.0])
    T = torch.tensor([[0.0], [2.0], [8.0], [10.0]])
    boundary_nodes = {
        "upwind": torch.tensor([0]),
        "side": torch.empty(0, dtype=torch.long),
        "downwind": torch.tensor([3]),
    }

    loss = compute_graph_gradient_loss(T, edge_index, edge_attr, boundary_nodes)

    assert torch.allclose(loss, torch.tensor(9.0), atol=1e-6)


def test_total_loss_returns_scalar_and_components_are_consistent():
    """验证总损失为标量且分量加权关系正确。

    参数:
        None。

    返回:
        None。断言总损失、PDE 损失、出流损失和边界钳制结果。
    """

    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = _edge_attr(distance=[1.0, 1.0], cos_theta=[1.0, 1.0])
    T_next = torch.tensor([[9.0], [2.0], [4.0], [8.0]])
    T_current = torch.tensor([[0.0], [1.0], [3.0], [0.0]])
    boundary_nodes = {
        "upwind": torch.tensor([0]),
        "side": torch.tensor([3]),
        "downwind": torch.tensor([2]),
    }

    components = total_loss(
        T_next=T_next,
        T_current=T_current,
        v_scan_star=1.0,
        dt_star=1.0,
        edge_index=edge_index,
        edge_attr=edge_attr,
        boundary_nodes=boundary_nodes,
        inverse_pe=0.0,
        k_ratio=0.05,
        lambda_outflow=0.25,
        return_components=True,
    )

    assert components["loss_total"].ndim == 0
    assert torch.allclose(components["T_next_bc"], torch.tensor([[0.0], [2.0], [4.0], [0.0]]))
    assert torch.allclose(
        components["loss_total"],
        components["loss_pde"] + 0.25 * components["loss_outflow"],
    )
    assert "loss_smooth" in components


def test_total_loss_adds_weighted_graph_gradient_regularization():
    """验证总损失会按权重加入图梯度平滑正则。"""

    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_attr = _edge_attr(distance=[2.0], cos_theta=[0.0])
    T_next = torch.tensor([[1.0], [5.0]])

    components = total_loss(
        T_next=T_next,
        T_current=T_next,
        v_scan_star=0.0,
        dt_star=1.0,
        edge_index=edge_index,
        edge_attr=edge_attr,
        inverse_pe=0.0,
        lambda_outflow=0.0,
        gradient_regularization=0.25,
        return_components=True,
    )

    assert torch.allclose(components["loss_smooth"], torch.tensor(4.0), atol=1e-6)
    assert torch.allclose(components["loss_total"], torch.tensor(1.0), atol=1e-6)


def test_total_loss_smooth_component_is_zero_without_internal_edges():
    """验证空边或边界边不会产生平滑正则损失。"""

    T_next = torch.tensor([[1.0], [5.0]])
    empty_components = total_loss(
        T_next=T_next,
        T_current=T_next,
        v_scan_star=0.0,
        dt_star=1.0,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 7), dtype=torch.float32),
        inverse_pe=0.0,
        gradient_regularization=1.0,
        return_components=True,
    )
    assert torch.allclose(empty_components["loss_smooth"], torch.tensor(0.0))

    boundary_components = total_loss(
        T_next=T_next,
        T_current=T_next,
        v_scan_star=0.0,
        dt_star=1.0,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=_edge_attr(distance=[1.0], cos_theta=[0.0]),
        boundary_nodes={
            "upwind": torch.tensor([0]),
            "side": torch.empty(0, dtype=torch.long),
            "downwind": torch.tensor([1]),
        },
        inverse_pe=0.0,
        gradient_regularization=1.0,
        return_components=True,
    )
    assert torch.allclose(boundary_components["loss_smooth"], torch.tensor(0.0))


def test_total_loss_pde_component_is_source_free_transport_term():
    """验证 PDE 损失分量不再包含热源项或热耗散项。"""

    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 7), dtype=torch.float32)
    T_next = torch.tensor([[2.0], [4.0]])
    T_current = torch.tensor([[1.0], [1.0]])

    components = total_loss(
        T_next=T_next,
        T_current=T_current,
        v_scan_star=0.0,
        dt_star=1.0,
        edge_index=edge_index,
        edge_attr=edge_attr,
        inverse_pe=0.0,
        k_ratio=0.05,
        lambda_outflow=0.0,
        thermal_loss_beta=0.5,
        thermal_loss_base_temperature_star=0.5,
        return_components=True,
    )

    expected_residual = torch.tensor([[1.0], [3.0]])
    expected_loss = expected_residual.square().mean()
    assert torch.allclose(components["residual"], expected_residual, atol=1e-6)
    assert torch.allclose(components["loss_pde"], expected_loss, atol=1e-6)
    assert torch.allclose(components["loss_total"], expected_loss, atol=1e-6)


def test_total_loss_can_use_backward_residual_time_scheme():
    """验证总损失可切换为后向残差时间离散方式。"""

    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 7), dtype=torch.float32)
    T_next = torch.tensor([[2.0], [4.0]])
    T_current = torch.tensor([[1.0], [1.0]])

    components = total_loss(
        T_next=T_next,
        T_current=T_current,
        v_scan_star=0.0,
        dt_star=1.0,
        edge_index=edge_index,
        edge_attr=edge_attr,
        inverse_pe=0.0,
        k_ratio=0.05,
        lambda_outflow=0.0,
        thermal_loss_beta=0.5,
        thermal_loss_base_temperature_star=0.5,
        residual_time_scheme="backward",
        return_components=True,
    )

    expected_residual = torch.tensor([[1.0], [3.0]])
    assert torch.allclose(components["residual"], expected_residual, atol=1e-6)


class TotalLossComponentTests(unittest.TestCase):
    def test_return_components_includes_beta_without_adding_it_to_total(self):
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 7), dtype=torch.float32)
        T_next = torch.tensor([[2.0], [4.0]])
        T_current = torch.tensor([[1.0], [1.0]])

        components = total_loss(
            T_next=T_next,
            T_current=T_current,
            v_scan_star=0.0,
            dt_star=1.0,
            edge_index=edge_index,
            edge_attr=edge_attr,
            lambda_outflow=0.5,
            thermal_loss_beta=0.25,
            thermal_loss_base_temperature_star=1.0,
            residual_time_scheme="backward",
            return_components=True,
        )

        self.assertIn("loss_beta", components)
        self.assertIn("thermal_loss_term", components)
        self.assertTrue(torch.allclose(components["loss_beta"], torch.tensor(0.0)))
        self.assertTrue(
            torch.allclose(
                components["loss_total"],
                components["loss_pde"] + 0.5 * components["loss_outflow"],
            )
        )

    def test_beta_monitor_loss_is_zero_when_beta_is_zero(self):
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 7), dtype=torch.float32)
        components = total_loss(
            T_next=torch.tensor([[2.0], [4.0]]),
            T_current=torch.tensor([[1.0], [1.0]]),
            v_scan_star=0.0,
            dt_star=1.0,
            edge_index=edge_index,
            edge_attr=edge_attr,
            thermal_loss_beta=0.0,
            return_components=True,
        )

        self.assertTrue(torch.allclose(components["loss_beta"], torch.tensor(0.0)))
