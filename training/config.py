from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-4
    optimizer: str = "Adam"
    epochs: int = 1
    tbptt_window: int = 20
    warmup_steps: int = 30
    grad_clip_norm: Optional[float] = None
    loss_threshold: Optional[float] = None
    device: Optional[str] = None
    resume_from_checkpoint: bool = False
    resume_checkpoint_path: Optional[str] = None
    resume_optimizer_state: bool = True

    def __post_init__(self):
        """校验训练超参数。

        参数:
            self: ``TrainConfig`` 实例，包含学习率、优化器类型、训练轮数、
                TBPTT 窗口长度、伪时间 warmup 步数、梯度裁剪阈值、
                提前停止 loss 阈值和设备配置。

        返回:
            None。若配置非法则抛出 ``ValueError``。
        """

        if float(self.lr) <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}.")
        if self.optimizer != "Adam":
            raise ValueError(f"Only Adam optimizer is supported, got {self.optimizer}.")
        if int(self.epochs) <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}.")
        if int(self.tbptt_window) <= 0:
            raise ValueError(f"tbptt_window must be positive, got {self.tbptt_window}.")
        if int(self.warmup_steps) < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}.")
        if self.grad_clip_norm is not None and float(self.grad_clip_norm) <= 0:
            raise ValueError(f"grad_clip_norm must be positive when set, got {self.grad_clip_norm}.")
        if self.loss_threshold is not None and float(self.loss_threshold) <= 0:
            raise ValueError(f"loss_threshold must be positive when set, got {self.loss_threshold}.")
        if not isinstance(self.resume_from_checkpoint, bool):
            raise ValueError(f"resume_from_checkpoint must be a boolean, got {self.resume_from_checkpoint!r}.")
        if self.resume_checkpoint_path is not None and not str(self.resume_checkpoint_path).strip():
            raise ValueError("resume_checkpoint_path must be non-empty when set.")
        if not isinstance(self.resume_optimizer_state, bool):
            raise ValueError(f"resume_optimizer_state must be a boolean, got {self.resume_optimizer_state!r}.")
