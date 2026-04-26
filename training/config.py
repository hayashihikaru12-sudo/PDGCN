from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-4
    optimizer: str = "Adam"
    epochs: int = 1
    tbptt_window: int = 20
    grad_clip_norm: Optional[float] = None
    device: Optional[str] = None

    def __post_init__(self):
        if float(self.lr) <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}.")
        if self.optimizer != "Adam":
            raise ValueError(f"Only Adam optimizer is supported, got {self.optimizer}.")
        if int(self.epochs) <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}.")
        if int(self.tbptt_window) <= 0:
            raise ValueError(f"tbptt_window must be positive, got {self.tbptt_window}.")
        if self.grad_clip_norm is not None and float(self.grad_clip_norm) <= 0:
            raise ValueError(f"grad_clip_norm must be positive when set, got {self.grad_clip_norm}.")
