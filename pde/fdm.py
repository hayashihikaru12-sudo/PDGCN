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


def compute_fin_cooling_gamma(*, inverse_pe: float, r_char_star: float):
    """Compute the dimensionless transient-fin cooling rate gamma*."""

    if float(inverse_pe) < 0:
        raise ValueError(f"inverse_pe must be non-negative, got {inverse_pe}.")
    if float(r_char_star) <= 0:
        raise ValueError(f"r_char_star must be positive, got {r_char_star}.")
    return float(inverse_pe) / (float(r_char_star) ** 2)


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
    fin_cooling_gamma_star=None,
    fin_cooling_skip_top_layers: int = 0,
    layer_interface_scales=None,
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
    fin_cooling_diagonal = _fin_cooling_diagonal(
        fin_cooling_gamma_star,
        dt_star=dt_star,
    )
    if fin_cooling_gamma_star is not None and int(fin_cooling_skip_top_layers) != 0:
        raise ValueError("fin_cooling_skip_top_layers must be 0 when fin_cooling_gamma_star is set.")
    matrix = _implicit_fdm_matrix(
        active.shape[0],
        coefficient=coefficient,
        device=temperature_2d.device,
        dtype=temperature_2d.dtype,
        fin_cooling_diagonal=fin_cooling_diagonal,
        fin_cooling_skip_top_layers=fin_cooling_skip_top_layers,
        layer_interface_scales=layer_interface_scales,
    )
    next_relative = torch.linalg.solve(matrix, relative_active)

    output = torch.empty_like(temperature_2d)
    output[:-1] = next_relative + bottom
    output[-1] = bottom

    if restore_col:
        return output.unsqueeze(-1)
    return output


def _fin_cooling_diagonal(fin_cooling_gamma_star, *, dt_star: float):
    if fin_cooling_gamma_star is None:
        return 0.0
    gamma = torch.as_tensor(fin_cooling_gamma_star)
    if gamma.ndim > 1:
        raise ValueError(
            "fin_cooling_gamma_star must be a non-negative scalar or 1D sequence, "
            f"got shape {tuple(gamma.shape)}."
        )
    if bool((gamma < 0).any().item()):
        raise ValueError(
            "fin_cooling_gamma_star must be non-negative when set, "
            f"got {fin_cooling_gamma_star}."
        )
    diagonal = gamma * float(dt_star)
    if diagonal.ndim == 0:
        return float(diagonal.item())
    return diagonal


def _implicit_fdm_matrix(
    active_layers: int,
    *,
    coefficient: float,
    device,
    dtype,
    fin_cooling_diagonal: float = 0.0,
    fin_cooling_skip_top_layers: int = 0,
    layer_interface_scales=None,
):
    if int(active_layers) < 1:
        raise ValueError(f"active_layers must be positive, got {active_layers}.")
    skip_layers = _as_non_negative_integer(
        fin_cooling_skip_top_layers,
        "fin_cooling_skip_top_layers",
    )
    matrix = torch.eye(int(active_layers), device=device, dtype=dtype)
    coefficient_value = torch.as_tensor(float(coefficient), device=device, dtype=dtype)
    interface_values = coefficient_value * _expand_layer_interface_scales(
        layer_interface_scales,
        active_layers=int(active_layers),
        device=device,
        dtype=dtype,
    )
    fin_values = _expand_fin_cooling_diagonal(
        fin_cooling_diagonal,
        active_layers=int(active_layers),
        device=device,
        dtype=dtype,
    )
    if skip_layers > 0:
        fin_values = fin_values.clone()
        fin_values[: min(skip_layers, int(active_layers))] = 0.0

    matrix[0, 0] = 1.0 + interface_values[0] + fin_values[0]
    if int(active_layers) > 1:
        matrix[0, 1] = -interface_values[0]

    for layer_index in range(1, int(active_layers)):
        interface_up = interface_values[layer_index - 1]
        interface_down = interface_values[layer_index]
        matrix[layer_index, layer_index] = 1.0 + interface_up + interface_down + fin_values[layer_index]
        matrix[layer_index, layer_index - 1] = -interface_up
        if layer_index + 1 < int(active_layers):
            matrix[layer_index, layer_index + 1] = -interface_down
    return matrix


def _expand_layer_interface_scales(layer_interface_scales, *, active_layers: int, device, dtype):
    if layer_interface_scales is None:
        return torch.ones(int(active_layers), device=device, dtype=dtype)
    values = torch.as_tensor(layer_interface_scales, device=device, dtype=dtype)
    if values.ndim == 0:
        if float(values.item()) < 0:
            raise ValueError(f"layer_interface_scales must be non-negative, got {layer_interface_scales}.")
        return values.expand(int(active_layers))
    if values.ndim != 1:
        raise ValueError(
            "layer_interface_scales must be a non-negative scalar or 1D sequence, "
            f"got shape {tuple(values.shape)}."
        )
    if int(values.numel()) != int(active_layers):
        raise ValueError(
            "layer_interface_scales sequence length must match the through-thickness interface count "
            f"{active_layers}, got {int(values.numel())}."
        )
    if bool((values < 0).any().item()):
        raise ValueError(f"layer_interface_scales must be non-negative, got {layer_interface_scales}.")
    return values


def _expand_fin_cooling_diagonal(fin_cooling_diagonal, *, active_layers: int, device, dtype):
    values = torch.as_tensor(fin_cooling_diagonal, device=device, dtype=dtype)
    if values.ndim == 0:
        return values.expand(int(active_layers))
    if values.ndim != 1:
        raise ValueError(
            "fin_cooling_gamma_star must be a non-negative scalar or 1D sequence, "
            f"got shape {tuple(values.shape)}."
        )
    if int(values.numel()) != int(active_layers):
        raise ValueError(
            "fin_cooling_gamma_star sequence length must match the active layer count "
            f"{active_layers}, got {int(values.numel())}."
        )
    return values


def _as_non_negative_integer(value, name: str):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer, got {value}.")
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {value}.") from exc
    if int_value < 0 or int_value != value:
        raise ValueError(f"{name} must be a non-negative integer, got {value}.")
    return int_value


__all__ = [
    "compute_fin_cooling_gamma",
    "compute_layer_fdm_coefficient",
    "compute_layer_fdm_delta",
    "compute_layer_implicit_fdm_step",
]
