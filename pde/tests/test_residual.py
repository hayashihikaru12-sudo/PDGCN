import torch

from pde.residual import compute_pde_residual


def _edge_attr(distance, cos_theta, cos_phi_sq):
    """构造用于残差测试的边特征张量。

    参数:
        distance: 边距离列表，长度为 ``E``。
        cos_theta: 边与扫描方向夹角余弦列表，长度为 ``E``。
        cos_phi_sq: 边与纤维方向夹角余弦平方列表，长度为 ``E``。

    返回:
        ``torch.FloatTensor``，形状 ``[E, 7]``，仅测试所需列被赋值。
    """

    rows = []
    for d, theta, phi_sq in zip(distance, cos_theta, cos_phi_sq):
        rows.append([0.0, 0.0, 0.0, d, theta, 0.0, phi_sq])
    return torch.tensor(rows, dtype=torch.float32)


def test_compute_pde_residual_matches_hand_calculation():
    """验证 PDE 残差计算与手工推导结果一致。

    参数:
        None。

    返回:
        None。断言失败时由测试框架报告错误。
    """

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

    expected = torch.tensor([[0.3], [1.875], [1.3]])
    assert residual.shape == T_next.shape
    assert torch.allclose(residual, expected, atol=1e-6)


def test_compute_pde_residual_supports_tbptt_window_shape():
    """验证 PDE 残差函数支持 TBPTT 时间窗口输入形状。

    参数:
        None。

    返回:
        None。断言输出形状和有限性。
    """

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


def test_compute_pde_residual_includes_single_layer_thermal_loss():
    """验证 PDE 残差包含单层等效热耗散项 beta * (T* - T_base*)。"""

    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 7), dtype=torch.float32)
    T_next = torch.tensor([[2.0], [4.0]])
    T_current = torch.tensor([[1.0], [1.0]])
    Q_star = torch.zeros_like(T_next)

    residual = compute_pde_residual(
        T_next=T_next,
        T_current=T_current,
        v_scan_star=0.0,
        Q_star=Q_star,
        dt_star=1.0,
        edge_index=edge_index,
        edge_attr=edge_attr,
        inverse_pe=0.0,
        pi_q=0.0,
        k_ratio=0.05,
        thermal_loss_beta=0.5,
        thermal_loss_base_temperature_star=0.5,
    )

    expected = torch.tensor([[1.25], [3.25]])
    assert torch.allclose(residual, expected, atol=1e-6)


def test_compute_pde_residual_can_use_backward_time_scheme():
    """验证后向残差会用预测温度计算空间项和热耗散项。"""

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
        residual_time_scheme="backward",
    )

    expected = torch.tensor([[0.3], [2.95], [1.3]])
    assert torch.allclose(residual, expected, atol=1e-6)
