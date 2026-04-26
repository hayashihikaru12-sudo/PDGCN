from typing import Dict

import torch


TEMPERATURE_SLICE = slice(6, 7)
HEAT_SOURCE_SLICE = slice(7, 8)


def clone_graph_with_temperature(graph, temperature_star):
    cloned = graph.clone()
    cloned.x = graph.x.clone()
    cloned.x[:, TEMPERATURE_SLICE] = temperature_star.to(device=cloned.x.device, dtype=cloned.x.dtype)
    return cloned


def graph_temperature(graph):
    return graph.x[:, TEMPERATURE_SLICE]


def graph_heat_source(graph):
    return graph.x[:, HEAT_SOURCE_SLICE]


def graph_boundary_nodes(graph) -> Dict[str, torch.Tensor]:
    return {
        "upwind": graph.upwind_nodes,
        "side": graph.side_nodes,
        "downwind": graph.downwind_nodes,
    }


def graph_to_device(graph, device):
    if device is None:
        return graph
    return graph.to(device)
