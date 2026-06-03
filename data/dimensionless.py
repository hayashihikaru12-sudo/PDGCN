from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class ScaleParams:
    """PD-GCN 数据流水线使用的特征尺度参数。

    ``K0``、``rho``、``Cp`` 和 ``heat_source_effective_thickness`` 是可选参数：
    它们不参与节点张量级无量纲化，但 ``derive_pde_constants`` 需要使用它们
    计算曲面内输运系数和显式表面热源系数。
    """

    L0: float
    v0: float
    T_amb: float
    delta_T0: float
    Q0: float
    K0: Optional[float] = None
    rho: Optional[float] = None
    Cp: Optional[float] = None
    heat_source_effective_thickness: Optional[float] = None
    heat_source_absorptivity: float = 1.0
    eps: float = 1e-12

    def __post_init__(self):
        """校验无量纲化特征尺度参数。

        参数:
            self: ``ScaleParams`` 实例，包含长度、速度、温度、热源等标尺；
                ``K0``、``rho``、``Cp`` 可选，但若提供则必须为正数。

        返回:
            None。校验失败时抛出 ``ValueError``。
        """

        positive_fields = ("L0", "v0", "delta_T0", "Q0", "eps")
        for field_name in positive_fields:
            value = float(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}.")
        for field_name in ("K0", "rho", "Cp", "heat_source_effective_thickness"):
            value = getattr(self, field_name)
            if value is not None and float(value) <= 0:
                raise ValueError(f"{field_name} must be positive when provided, got {value}.")
        if float(self.heat_source_absorptivity) < 0:
            raise ValueError(
                "heat_source_absorptivity must be non-negative, "
                f"got {self.heat_source_absorptivity}."
            )


def derive_pde_constants(scale_params: "ScaleParams") -> Tuple[float, float]:
    """根据特征尺度计算 ``(inverse_pe, source_coefficient)``。

    ``1/Pe = K0 / (rho * Cp * v0 * L0)``。
    当 ``Q0`` 表示表面热流尺度 ``W/m^2`` 时，显式热源系数为
    ``Q0 * L0 / (rho * Cp * v0 * h_eff * delta_T0)``，满足
    ``delta_T_Q* = eta * source_coefficient * dt_star * q_surface*``。
    调用前必须在 ``scale_params`` 中设置 ``K0``、``rho`` 和 ``Cp``。

    参数:
        scale_params: ``ScaleParams`` 实例，必须设置 ``K0``、``rho`` 和 ``Cp``；
            其余字段提供无量纲化所需的特征标尺。

    返回:
        ``(inverse_pe, source_coefficient)`` 二元组，均为 Python ``float``；
        分别表示佩克莱特数倒数和显式表面热源温升系数。
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
    if scale_params.heat_source_effective_thickness is None:
        source_coefficient = (Q0 * L0) / (denom * delta_T0)
    else:
        source_coefficient = (Q0 * L0) / (
            denom * float(scale_params.heat_source_effective_thickness) * delta_T0
        )
    return inverse_pe, source_coefficient


def coordinates_to_dimensionless(coordinates, scale_params: ScaleParams):
    """将真实坐标转换为无量纲坐标。

    参数:
        coordinates: 坐标张量或数组，形状通常为 ``[N, 3]``，单位与 ``L0`` 一致。
        scale_params: ``ScaleParams`` 实例，使用其中的 ``L0`` 作为长度标尺。

    返回:
        与 ``coordinates`` 形状一致的无量纲坐标，数值为 ``coordinates / L0``。
    """

    return coordinates / scale_params.L0


def coordinates_from_dimensionless(coordinates_star, scale_params: ScaleParams):
    """将无量纲坐标还原为真实坐标。

    参数:
        coordinates_star: 无量纲坐标张量或数组，形状通常为 ``[N, 3]``。
        scale_params: ``ScaleParams`` 实例，使用其中的 ``L0`` 作为长度标尺。

    返回:
        与 ``coordinates_star`` 形状一致的真实坐标，数值为 ``coordinates_star * L0``。
    """

    return coordinates_star * scale_params.L0


def temperature_to_dimensionless(temperature, scale_params: ScaleParams):
    """将真实温度转换为无量纲温度。

    参数:
        temperature: 温度张量或数组，形状通常为 ``[N, 1]``，单位为真实温度单位。
        scale_params: ``ScaleParams`` 实例，使用 ``T_amb`` 和 ``delta_T0``。

    返回:
        与 ``temperature`` 形状一致的无量纲温度 ``(T - T_amb) / delta_T0``。
    """

    return (temperature - scale_params.T_amb) / scale_params.delta_T0


def temperature_from_dimensionless(temperature_star, scale_params: ScaleParams):
    """将无量纲温度还原为真实温度。

    参数:
        temperature_star: 无量纲温度张量或数组，形状通常为 ``[N, 1]`` 或
            ``[K, N, 1]``。
        scale_params: ``ScaleParams`` 实例，使用 ``T_amb`` 和 ``delta_T0``。

    返回:
        与 ``temperature_star`` 形状一致的真实温度 ``T* * delta_T0 + T_amb``。
    """

    return temperature_star * scale_params.delta_T0 + scale_params.T_amb


def heat_source_to_dimensionless(q, scale_params: ScaleParams):
    """将真实表面热流转换为无量纲热源标记。

    参数:
        q: 表面热流张量或数组，形状通常为 ``[N, 1]``，单位 ``W/m^2``。
        scale_params: ``ScaleParams`` 实例，使用其中的 ``Q0`` 作为表面热流标尺。

    返回:
        与 ``q`` 形状一致的无量纲表面热流 ``q / Q0``。
    """

    return q / scale_params.Q0


def heat_source_from_dimensionless(q_star, scale_params: ScaleParams):
    """将无量纲表面热流还原为真实表面热流。

    参数:
        q_star: 无量纲表面热流张量或数组，形状通常为 ``[N, 1]``。
        scale_params: ``ScaleParams`` 实例，使用其中的 ``Q0`` 作为表面热流标尺。

    返回:
        与 ``q_star`` 形状一致的真实表面热流 ``q_star * Q0``。
    """

    return q_star * scale_params.Q0


def velocity_to_dimensionless(scan_velocity, scale_params: ScaleParams, *, device=None, dtype=torch.float32):
    """将扫描速度转换为无量纲速度张量。

    参数:
        scan_velocity: 标量、序列或 ``torch.Tensor``，表示真实扫描速度。
        scale_params: ``ScaleParams`` 实例，使用其中的 ``v0`` 作为速度标尺。
        device: 可选目标设备，例如 ``torch.device("cpu")`` 或 CUDA 设备。
        dtype: 可选目标数据类型，默认为 ``torch.float32``。

    返回:
        ``torch.Tensor``，形状继承自输入，数值为 ``scan_velocity / v0``。
    """

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
    """将原始 PD-GCN 特征张量转换为无量纲特征布局。

    重要提示:
        本函数只接收带真实单位的原始特征张量。不要对 ``build_graph``
        返回的 ``graph.x``、``graph.edge_attr`` 或 ``graph.global_attr``
        再调用本函数。``build_graph`` 已经完成坐标、边位移/距离、
        温度和扫描速度的无量纲化。若对 ``build_graph`` 构造出的图
        再调用本函数，``edge_features[:, 0:4]`` 会被第二次除以 ``L0``，
        从而破坏 PDE 尺度。

    期望节点布局: [x, y, z, fx, fy, fz, T]。
    期望边布局: [dx, dy, dz, d, cos_theta, cos_phi, cos_phi_sq]。
    期望全局条件布局: [scan_velocity]。

    参数:
        node_features: 原始节点特征张量，形状 ``[N, 7]``，列含义为
            ``[x, y, z, fx, fy, fz, T]``。
        edge_features: 原始边特征张量，形状 ``[E, 7]``，列含义为
            ``[dx, dy, dz, d, cos_theta, cos_phi, cos_phi_sq]``。
        global_condition: 全局工艺条件，形状通常为 ``[1]``，表示扫描速度。
        scale_params: ``ScaleParams`` 实例，提供坐标、温度和速度标尺。

    返回:
        ``(node_star, edge_star, global_star)`` 三元组：
        ``node_star`` 形状 ``[N, 7]``，坐标/温度已无量纲化；
        ``edge_star`` 形状 ``[E, 7]``，边位移和距离已无量纲化；
        ``global_star`` 形状 ``[G]``，全局速度已无量纲化。
    """

    node_star = node_features.clone()
    edge_star = edge_features.clone()

    node_star[:, 0:3] = coordinates_to_dimensionless(node_star[:, 0:3], scale_params)
    node_star[:, 6:7] = temperature_to_dimensionless(node_star[:, 6:7], scale_params)

    # 警告: 此处的 edge_features[:, 0:4] 必须仍然带真实长度单位。
    # 不要传入 build_graph/build_edge_features(nodes_star, ...) 生成的 edge_attr，
    # 因为其中的边位移和边距离已经是无量纲量。
    edge_star[:, 0:4] = edge_star[:, 0:4] / scale_params.L0
    global_star = velocity_to_dimensionless(
        global_condition,
        scale_params,
        device=global_condition.device if torch.is_tensor(global_condition) else None,
        dtype=global_condition.dtype if torch.is_tensor(global_condition) else torch.float32,
    ).reshape(-1)

    return node_star, edge_star, global_star
