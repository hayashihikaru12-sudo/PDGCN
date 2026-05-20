import torch


def compute_layer_fdm_coefficient(
    *,
    dt_star: float,
    inverse_pe: float,
    k_ratio: float,
    layer_spacing_star: float,
):
    """Compute the explicit 1D through-thickness FDM coefficient C_n."""

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
    """Compute through-thickness 1D FDM increments for [layer, node, 1] temperatures."""

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
