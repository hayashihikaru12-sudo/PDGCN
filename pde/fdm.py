import torch


def compute_layer_fdm_coefficient(
    *,
    dt_star: float,
    inverse_pe: float,
    k_ratio: float,
    layer_spacing_star: float,
):
    """Compute the 1D through-thickness FDM coefficient C_n."""

    if float(dt_star) <= 0:
        raise ValueError(f"dt_star must be positive, got {dt_star}.")
    if float(inverse_pe) < 0:
        raise ValueError(f"inverse_pe must be non-negative, got {inverse_pe}.")
    if float(k_ratio) < 0:
        raise ValueError(f"k_ratio must be non-negative, got {k_ratio}.")
    if float(layer_spacing_star) <= 0:
        raise ValueError(f"layer_spacing_star must be positive, got {layer_spacing_star}.")
    return float(k_ratio) * float(dt_star) * float(inverse_pe) / (float(layer_spacing_star) ** 2)


def compute_layer_fdm_delta(
    temperature_star,
    *,
    dt_star: float,
    inverse_pe: float,
    k_ratio: float,
    layer_spacing_star: float,
):
    """Compute explicit through-thickness 1D FDM increments for diagnostics/tests."""

    tensor = torch.as_tensor(temperature_star)
    if tensor.ndim == 2:
        temperature_2d = tensor
        restore_col = False
    elif tensor.ndim == 3 and tensor.shape[2] == 1:
        temperature_2d = tensor[:, :, 0]
        restore_col = True
    else:
        raise ValueError(
            "temperature_star must have shape [L, N] or [L, N, 1], "
            f"got {tuple(tensor.shape)}."
        )

    if temperature_2d.shape[0] < 2:
        raise ValueError(f"temperature_star must contain at least 2 layers, got {temperature_2d.shape[0]}.")

    coefficient = compute_layer_fdm_coefficient(
        dt_star=dt_star,
        inverse_pe=inverse_pe,
        k_ratio=k_ratio,
        layer_spacing_star=layer_spacing_star,
    )
    delta = torch.zeros_like(temperature_2d)
    delta[0] = coefficient * (temperature_2d[1] - temperature_2d[0])
    if temperature_2d.shape[0] > 2:
        delta[1:-1] = coefficient * (temperature_2d[:-2] - 2.0 * temperature_2d[1:-1] + temperature_2d[2:])

    if restore_col:
        return delta.unsqueeze(-1)
    return delta


def compute_layer_implicit_fdm_step(
    temperature_star,
    *,
    dt_star: float,
    inverse_pe: float,
    k_ratio: float,
    layer_spacing_star: float,
    bottom_temperature_star: float = 0.0,
):
    """Apply one unconditionally stable implicit through-thickness FDM step.

    The bottom layer is treated as a fixed Dirichlet boundary. Active layers
    solve the Backward Euler system ``(I - C_n D) u_next = u_current`` in
    temperature relative to the bottom boundary.
    """

    tensor = torch.as_tensor(temperature_star)
    if tensor.ndim == 2:
        temperature_2d = tensor
        restore_col = False
    elif tensor.ndim == 3 and tensor.shape[2] == 1:
        temperature_2d = tensor[:, :, 0]
        restore_col = True
    else:
        raise ValueError(
            "temperature_star must have shape [L, N] or [L, N, 1], "
            f"got {tuple(tensor.shape)}."
        )

    layer_count = int(temperature_2d.shape[0])
    if layer_count < 2:
        raise ValueError(f"temperature_star must contain at least 2 layers, got {layer_count}.")

    coefficient = compute_layer_fdm_coefficient(
        dt_star=dt_star,
        inverse_pe=inverse_pe,
        k_ratio=k_ratio,
        layer_spacing_star=layer_spacing_star,
    )
    bottom = torch.as_tensor(
        bottom_temperature_star,
        device=temperature_2d.device,
        dtype=temperature_2d.dtype,
    )
    active = temperature_2d[:-1]
    relative_active = active - bottom
    matrix = _implicit_fdm_matrix(
        active.shape[0],
        coefficient=coefficient,
        device=temperature_2d.device,
        dtype=temperature_2d.dtype,
    )
    next_relative = torch.linalg.solve(matrix, relative_active)

    output = torch.empty_like(temperature_2d)
    output[:-1] = next_relative + bottom
    output[-1] = bottom

    if restore_col:
        return output.unsqueeze(-1)
    return output


def _implicit_fdm_matrix(active_layers: int, *, coefficient: float, device, dtype):
    if int(active_layers) < 1:
        raise ValueError(f"active_layers must be positive, got {active_layers}.")
    matrix = torch.eye(int(active_layers), device=device, dtype=dtype)
    coefficient_value = torch.as_tensor(float(coefficient), device=device, dtype=dtype)

    matrix[0, 0] = 1.0 + coefficient_value
    if int(active_layers) > 1:
        matrix[0, 1] = -coefficient_value

    for layer_index in range(1, int(active_layers)):
        matrix[layer_index, layer_index] = 1.0 + 2.0 * coefficient_value
        matrix[layer_index, layer_index - 1] = -coefficient_value
        if layer_index + 1 < int(active_layers):
            matrix[layer_index, layer_index + 1] = -coefficient_value
    return matrix


__all__ = ["compute_layer_fdm_coefficient", "compute_layer_fdm_delta", "compute_layer_implicit_fdm_step"]
