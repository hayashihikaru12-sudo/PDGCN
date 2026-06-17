import math


def build_lr_scheduler(optimizer, config):
    scheduler = str(config.lr_scheduler).strip().lower()
    if scheduler == "none":
        return NullLRScheduler(optimizer)
    if scheduler == "warmup_cosine":
        return WarmupCosineLRScheduler(
            optimizer,
            base_lr=float(config.lr),
            min_lr=float(config.min_lr),
            total_epochs=int(config.epochs),
            warmup_epochs=int(config.lr_warmup_epochs),
        )
    if scheduler == "plateau":
        return PlateauLRScheduler(
            optimizer,
            min_lr=float(config.min_lr),
            patience=int(config.lr_patience),
            factor=float(config.lr_factor),
        )
    raise ValueError(f"Unsupported lr_scheduler: {config.lr_scheduler}.")


def optimizer_lr(optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]["lr"])


def _set_optimizer_lr(optimizer, lr: float):
    lr = float(lr)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


class NullLRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def begin_epoch(self, epoch_index: int) -> float:
        return optimizer_lr(self.optimizer)

    def end_epoch(self, metric: float) -> float:
        return optimizer_lr(self.optimizer)

    def state_dict(self):
        return {}

    def load_state_dict(self, state: dict):
        pass


class WarmupCosineLRScheduler:
    def __init__(self, optimizer, *, base_lr: float, min_lr: float, total_epochs: int, warmup_epochs: int):
        if total_epochs <= 0:
            raise ValueError(f"total_epochs must be positive, got {total_epochs}.")
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = min(max(int(warmup_epochs), 0), self.total_epochs)

    def begin_epoch(self, epoch_index: int) -> float:
        return _set_optimizer_lr(self.optimizer, self.lr_for_epoch(epoch_index))

    def end_epoch(self, metric: float) -> float:
        return optimizer_lr(self.optimizer)

    def lr_for_epoch(self, epoch_index: int) -> float:
        epoch_index = min(max(int(epoch_index), 0), self.total_epochs - 1)
        if self.warmup_epochs > 0 and epoch_index < self.warmup_epochs:
            progress = float(epoch_index + 1) / float(self.warmup_epochs)
            return self.min_lr + (self.base_lr - self.min_lr) * progress

        if self.warmup_epochs >= self.total_epochs:
            return self.base_lr

        if self.warmup_epochs == 0:
            denominator = max(self.total_epochs - 1, 1)
            progress = float(epoch_index) / float(denominator)
        else:
            remaining_epochs = max(self.total_epochs - self.warmup_epochs, 1)
            progress = float(epoch_index - self.warmup_epochs + 1) / float(remaining_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def state_dict(self):
        return {}

    def load_state_dict(self, state: dict):
        pass


class PlateauLRScheduler:
    def __init__(self, optimizer, *, min_lr: float, patience: int, factor: float):
        if int(patience) <= 0:
            raise ValueError(f"patience must be positive, got {patience}.")
        if not 0.0 < float(factor) < 1.0:
            raise ValueError(f"factor must be in (0, 1), got {factor}.")
        self.optimizer = optimizer
        self.min_lr = float(min_lr)
        self.patience = int(patience)
        self.factor = float(factor)
        self.best = None
        self.bad_epochs = 0

    def begin_epoch(self, epoch_index: int) -> float:
        return optimizer_lr(self.optimizer)

    def end_epoch(self, metric: float) -> float:
        metric = float(metric)
        if self.best is None or metric < self.best:
            self.best = metric
            self.bad_epochs = 0
            return optimizer_lr(self.optimizer)

        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            current_lr = optimizer_lr(self.optimizer)
            next_lr = max(current_lr * self.factor, self.min_lr)
            if next_lr < current_lr:
                _set_optimizer_lr(self.optimizer, next_lr)
            self.bad_epochs = 0
        return optimizer_lr(self.optimizer)

    def state_dict(self):
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict):
        self.best = state.get("best")
        self.bad_epochs = state.get("bad_epochs", 0)
