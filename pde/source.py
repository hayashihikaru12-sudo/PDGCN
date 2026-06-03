import torch


def compute_surface_source_delta_star(
    q_surface_star,
    *,
    dt_star: float,
    source_coefficient: float,
    absorptivity: float = 1.0,
):
    """将无量纲表面热流显式转换为无量纲温升。

    该函数实现
    ``delta_T_Q* = eta * source_coefficient * dt_star * q_surface*``，
    其中 ``source_coefficient = Q0 * L0 / (rho * Cp * v0 * h_eff * delta_T0)``。
    """

    if float(dt_star) <= 0:
        raise ValueError(f"dt_star must be positive, got {dt_star}.")
    if float(source_coefficient) < 0:
        raise ValueError(f"source_coefficient must be non-negative, got {source_coefficient}.")
    if float(absorptivity) < 0:
        raise ValueError(f"absorptivity must be non-negative, got {absorptivity}.")
    return (
        torch.as_tensor(q_surface_star)
        * float(absorptivity)
        * float(source_coefficient)
        * float(dt_star)
    )


def compute_surface_source_delta_star_from_physical(
    q_surface,
    scale_params,
    *,
    dt_star=None,
    dt=None,
    absorptivity=None,
):
    """将真实表面热流 ``q''`` 显式转换为无量纲温升。

    ``q_surface`` 单位为 ``W/m^2``。调用者可传真实时间步 ``dt``，也可传
    ``dt_star``，函数会使用 ``dt = dt_star * L0 / v0``。
    """

    if dt is None:
        if dt_star is None:
            raise ValueError("Either dt or dt_star must be provided.")
        if float(dt_star) <= 0:
            raise ValueError(f"dt_star must be positive, got {dt_star}.")
        dt = float(dt_star) * float(scale_params.L0) / float(scale_params.v0)
    if float(dt) <= 0:
        raise ValueError(f"dt must be positive, got {dt}.")

    missing = [
        name
        for name in ("rho", "Cp", "heat_source_effective_thickness")
        if getattr(scale_params, name) is None
    ]
    if missing:
        raise ValueError(
            "Explicit surface source requires "
            f"{missing} to be set on ScaleParams."
        )

    eta = (
        float(scale_params.heat_source_absorptivity)
        if absorptivity is None
        else float(absorptivity)
    )
    if eta < 0:
        raise ValueError(f"absorptivity must be non-negative, got {eta}.")

    denominator = (
        float(scale_params.rho)
        * float(scale_params.Cp)
        * float(scale_params.heat_source_effective_thickness)
        * float(scale_params.delta_T0)
    )
    if denominator <= 0:
        raise ValueError("rho * Cp * heat_source_effective_thickness * delta_T0 must be positive.")
    return torch.as_tensor(q_surface) * eta * float(dt) / denominator
