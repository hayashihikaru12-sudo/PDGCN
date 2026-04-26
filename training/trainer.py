from typing import Optional, Sequence

import torch

from .config import TrainConfig
from .graph_utils import graph_to_device
from .tbptt import initial_temperature_from_graph_seq, iter_tbptt_windows, train_tbptt_window
from .warmup import pseudo_time_relax_initial_temperature


def train(model, graph_seq: Sequence, config: TrainConfig, optimizer: Optional[torch.optim.Optimizer] = None):
    """使用 TBPTT 和物理损失训练 PD-GCN 模型。

    参数:
        model: 待训练模型，输入单步图并输出 ``delta_T*``；需包含 ``config`` 属性。
        graph_seq: 非空图序列，每个图包含节点特征、边特征、全局速度和边界索引。
        config: ``TrainConfig``，提供学习率、轮数、窗口长度、warmup、设备和梯度裁剪配置。
        optimizer: 可选 PyTorch 优化器；若为 ``None``，根据 ``config`` 创建 Adam。

    返回:
        训练历史列表；每个元素是字典，包含 ``epoch``、该轮平均 ``loss`` 和
        ``window_losses`` 列表。
    """

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
        if int(config.warmup_steps) > 0:
            current_temperature = pseudo_time_relax_initial_temperature(
                model,
                graph_seq[0],
                int(config.warmup_steps),
            )
        else:
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
        epoch_record = {"epoch": epoch, "loss": mean_loss, "window_losses": epoch_losses}
        history.append(epoch_record)
        if config.loss_threshold is not None and mean_loss < float(config.loss_threshold):
            epoch_record["stopped_early"] = True
            epoch_record["stop_reason"] = "loss_threshold"
            break

    return history


def _build_optimizer(model, config: TrainConfig):
    """根据训练配置创建优化器。

    参数:
        model: 待优化的 PyTorch 模型。
        config: ``TrainConfig``，使用其中的 ``optimizer`` 和 ``lr`` 字段。

    返回:
        PyTorch 优化器实例；当前仅支持 ``torch.optim.Adam``。
    """

    if config.optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=float(config.lr))
    raise ValueError(f"Unsupported optimizer: {config.optimizer}.")
