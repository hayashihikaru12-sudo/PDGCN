from typing import List, Sequence

import torch

from pde import apply_dirichlet_boundary, total_loss

from .graph_utils import clone_graph_with_temperature, graph_boundary_nodes, graph_heat_source, graph_temperature


def iter_tbptt_windows(graph_seq: Sequence, window_size: int):
    if int(window_size) <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    for start in range(0, len(graph_seq), int(window_size)):
        window = list(graph_seq[start : start + int(window_size)])
        if window:
            yield window


def rollout_window(model, window: Sequence, initial_temperature_star):
    if not window:
        raise ValueError("window must contain at least one graph.")

    predictions = []
    current_temperature = initial_temperature_star
    for graph in window:
        graph_step = clone_graph_with_temperature(graph, current_temperature)
        delta_temperature = model(graph_step)
        next_temperature = current_temperature + delta_temperature
        next_temperature = apply_dirichlet_boundary(
            next_temperature,
            graph_boundary_nodes(graph_step),
            value=getattr(model.config, "dirichlet_temperature_star", 0.0),
        )
        predictions.append(next_temperature)
        current_temperature = next_temperature

    return torch.stack(predictions, dim=0), current_temperature


def train_tbptt_window(model, window: Sequence, initial_temperature_star):
    prediction_seq, final_temperature = rollout_window(model, window, initial_temperature_star)
    losses = []
    current_temperature = initial_temperature_star

    for step, graph in enumerate(window):
        prediction = prediction_seq[step]
        loss = total_loss(
            T_next=prediction,
            T_current=current_temperature,
            v_scan_star=graph.global_attr,
            Q_star=graph_heat_source(graph),
            dt_star=model.config.dt_star,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            boundary_nodes=graph_boundary_nodes(graph),
            inverse_pe=model.config.inverse_pe,
            pi_q=model.config.pi_q,
            k_ratio=model.config.k_ratio,
            lambda_outflow=model.config.lambda_outflow,
            dirichlet_temperature_star=model.config.dirichlet_temperature_star,
        )
        losses.append(loss)
        current_temperature = prediction

    return torch.stack(losses).mean(), final_temperature


def initial_temperature_from_graph_seq(graph_seq: Sequence):
    if not graph_seq:
        raise ValueError("graph_seq must contain at least one graph.")
    return graph_temperature(graph_seq[0])
