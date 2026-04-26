from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(model, optimizer, path, *, epoch: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None):
    """保存模型、优化器和实验元信息 checkpoint。

    参数:
        model: PyTorch 模型，使用其 ``state_dict`` 保存权重。
        optimizer: 可选 PyTorch 优化器；若不为 ``None``，保存其 ``state_dict``。
        path: checkpoint 输出路径，类型可为字符串或 ``Path``。
        epoch: 可选训练轮数或检查点对应 epoch。
        metadata: 可选字典，用于保存尺度参数、配置或实验备注等附加信息。

    返回:
        None。函数会在 ``path`` 写入 ``torch.save`` 格式文件。
    """

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metadata": metadata or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path, *, map_location="cpu"):
    """从 checkpoint 恢复模型和优化器状态。

    参数:
        model: 待加载权重的 PyTorch 模型。
        optimizer: 可选 PyTorch 优化器；若 checkpoint 中包含优化器状态则恢复。
        path: checkpoint 文件路径。
        map_location: ``torch.load`` 的设备映射参数，默认加载到 CPU。

    返回:
        checkpoint 字典，包含 ``model``、``optimizer``、``epoch`` 和 ``metadata`` 等键。
    """

    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
