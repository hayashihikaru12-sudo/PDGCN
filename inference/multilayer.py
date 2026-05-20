from typing import Sequence, Union

import torch
from torch_geometric.data import Data

from data.dimensionless import temperature_from_dimensionless
from pde import apply_dirichlet_boundary
from training.graph_utils import graph_boundary_nodes, graph_to_device
from training.warmup import pseudo_time_relax_initial_temperature

from .fdm import compute_layer_fdm_coefficient, compute_layer_fdm_delta


@torch.no_grad()
def rollout_multilayer_fdm(
    model,
    graph_init_or_seq: Union[object, Sequence],
    steps: int,
    scale_params,
    *,
    num_layers: int,
    layer_spacing: float,
    return_dimensionless: bool = False,
    return_all: bool = True,
    writer=None,
    warmup_steps: int = 0,
    initial_temperature_star=None,
    bottom_temperature_star: float = 0.0,
    top_heat_source_only: bool = True,
    allow_unstable_fdm: bool = False,
):
    """Run multilayer PD-GCN inference coupled with explicit 1D FDM in thickness."""

    steps = int(steps)
    num_layers = int(num_layers)
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}.")
    if num_layers < 2:
        raise ValueError(f"num_layers must be at least 2, got {num_layers}.")
    if float(layer_spacing) <= 0:
        raise ValueError(f"layer_spacing must be positive, got {layer_spacing}.")
    if int(warmup_steps) < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")

    model_device = next(model.parameters()).device
    graph0 = _graph_for_step(graph_init_or_seq, 0, steps, model_device)
    current_temperature = _initial_multilayer_temperature(
        model,
        graph0,
        num_layers,
        initial_temperature_star=initial_temperature_star,
        warmup_steps=int(warmup_steps),
        bottom_temperature_star=bottom_temperature_star,
    )

    layer_spacing_star = float(layer_spacing) / float(scale_params.L0)
    fdm_coefficient = compute_layer_fdm_coefficient(
        dt_star=getattr(model.config, "dt_star", 1.0),
        inverse_pe=getattr(model.config, "inverse_pe", 1.0),
        k_ratio=getattr(model.config, "k_ratio", 0.0),
        layer_spacing_star=layer_spacing_star,
    )
    if not bool(allow_unstable_fdm) and fdm_coefficient > 0.5:
        raise ValueError(
            "Explicit FDM coefficient C_n must be <= 0.5 for stable rollout; "
            f"got {fdm_coefficient}. Increase layer_spacing, reduce dt_star, or set allow_unstable_fdm=True."
        )

    outputs = []
    was_training = model.training
    model.eval()
    try:
        for step in range(steps):
            graph = graph0 if step == 0 else _graph_for_step(graph_init_or_seq, step, steps, model_device)
            graph_step = _build_multilayer_graph(
                graph,
                current_temperature,
                top_heat_source_only=top_heat_source_only,
            )
            delta_net = model(graph_step).reshape(num_layers, graph.num_nodes, -1)
            if delta_net.shape[-1] != 1:
                raise ValueError(f"model output_size must be 1 for thermal rollout, got {delta_net.shape[-1]}.")

            delta_fdm = compute_layer_fdm_delta(
                current_temperature,
                dt_star=getattr(model.config, "dt_star", 1.0),
                inverse_pe=getattr(model.config, "inverse_pe", 1.0),
                k_ratio=getattr(model.config, "k_ratio", 0.0),
                layer_spacing_star=layer_spacing_star,
            )
            compensation = (
                float(getattr(model.config, "thermal_loss_beta", 0.0))
                * current_temperature
                * float(getattr(model.config, "dt_star", 1.0))
            )
            next_temperature = current_temperature + delta_net + compensation + delta_fdm
            next_temperature = apply_dirichlet_boundary(
                next_temperature,
                graph_boundary_nodes(graph),
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            next_temperature[-1, :, :] = torch.as_tensor(
                bottom_temperature_star,
                device=next_temperature.device,
                dtype=next_temperature.dtype,
            )

            output = (
                next_temperature
                if return_dimensionless
                else temperature_from_dimensionless(next_temperature, scale_params)
            )
            if writer is not None:
                writer(step, output.detach().cpu())
            if return_all:
                outputs.append(output.detach().cpu())
            current_temperature = next_temperature
    finally:
        if was_training:
            model.train()

    if return_all:
        return torch.stack(outputs, dim=0)
    return None


def _graph_for_step(graph_init_or_seq, step: int, steps: int, device):
    if callable(graph_init_or_seq) and not isinstance(graph_init_or_seq, Data):
        return graph_to_device(graph_init_or_seq(int(step)), device)
    if isinstance(graph_init_or_seq, (list, tuple)):
        if len(graph_init_or_seq) < steps:
            raise ValueError(f"graph sequence length {len(graph_init_or_seq)} is shorter than steps={steps}.")
        return graph_to_device(graph_init_or_seq[int(step)], device)
    return graph_to_device(graph_init_or_seq, device)


def _initial_multilayer_temperature(
    model,
    graph,
    num_layers: int,
    *,
    initial_temperature_star,
    warmup_steps: int,
    bottom_temperature_star: float,
):
    device = graph.x.device
    dtype = graph.x.dtype
    if initial_temperature_star is not None:
        initial = torch.as_tensor(initial_temperature_star, device=device, dtype=dtype)
        if initial.shape != (num_layers, graph.num_nodes, 1):
            raise ValueError(
                "initial_temperature_star must have shape "
                f"({num_layers}, {graph.num_nodes}, 1), got {tuple(initial.shape)}."
            )
        output = initial.clone()
    else:
        if warmup_steps > 0:
            top_temperature = pseudo_time_relax_initial_temperature(model, graph, int(warmup_steps))
        else:
            top_temperature = graph.x[:, 6:7]
        output = torch.full(
            (num_layers, graph.num_nodes, 1),
            float(bottom_temperature_star),
            device=device,
            dtype=dtype,
        )
        output[0] = top_temperature.to(device=device, dtype=dtype)
    output[-1, :, :] = torch.as_tensor(bottom_temperature_star, device=device, dtype=dtype)
    return output


def _build_multilayer_graph(graph, temperature_star, *, top_heat_source_only: bool):
    num_layers, num_nodes, _ = temperature_star.shape
    edge_index = graph.edge_index
    edge_count = edge_index.shape[1]
    device = graph.x.device

    x = graph.x.repeat(num_layers, 1)
    x[:, 6:7] = temperature_star.reshape(num_layers * num_nodes, 1).to(device=device, dtype=x.dtype)
    if top_heat_source_only and num_layers > 1:
        x[num_nodes:, 7:8] = 0.0

    offsets = (torch.arange(num_layers, device=device, dtype=edge_index.dtype) * num_nodes).reshape(num_layers, 1, 1)
    edge_index_batched = (edge_index.reshape(1, 2, edge_count) + offsets).permute(1, 0, 2).reshape(
        2,
        num_layers * edge_count,
    )
    edge_attr = graph.edge_attr.repeat(num_layers, 1)

    data = Data(
        x=x,
        edge_index=edge_index_batched,
        edge_attr=edge_attr,
        global_attr=graph.global_attr,
    )
    data.num_nodes = num_layers * num_nodes
    if hasattr(graph, "node_type"):
        data.node_type = graph.node_type.repeat(num_layers)
    if getattr(graph, "pos", None) is not None:
        data.pos = graph.pos.repeat(num_layers, 1)
    return data
