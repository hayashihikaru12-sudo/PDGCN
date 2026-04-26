from typing import Optional, Sequence

import torch

from .config import TrainConfig
from .graph_utils import graph_to_device
from .tbptt import initial_temperature_from_graph_seq, iter_tbptt_windows, train_tbptt_window


def train(model, graph_seq: Sequence, config: TrainConfig, optimizer: Optional[torch.optim.Optimizer] = None):
    if not graph_seq:
        raise ValueError("graph_seq must contain at least one graph.")

    device = torch.device(config.device) if config.device is not None else next(model.parameters()).device
    model.to(device)
    graph_seq = [graph_to_device(graph, device) for graph in graph_seq]

    if optimizer is None:
        optimizer = _build_optimizer(model, config)

    history = []
    for epoch in range(int(config.epochs)):
        model.train()
        current_temperature = initial_temperature_from_graph_seq(graph_seq).detach()
        epoch_losses = []

        for window in iter_tbptt_windows(graph_seq, config.tbptt_window):
            optimizer.zero_grad()
            loss, final_temperature = train_tbptt_window(model, window, current_temperature)
            loss.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))
            current_temperature = final_temperature.detach()

        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        history.append({"epoch": epoch, "loss": mean_loss, "window_losses": epoch_losses})

    return history


def _build_optimizer(model, config: TrainConfig):
    if config.optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=float(config.lr))
    raise ValueError(f"Unsupported optimizer: {config.optimizer}.")
