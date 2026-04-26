import json
from pathlib import Path
from typing import Callable, Optional


class LossMonitor:
    """训练过程中打印 epoch loss，并刷新轻量 history 记录。"""

    def __init__(
        self,
        *,
        total_epochs: Optional[int] = None,
        history_path=None,
        print_fn: Callable[[str], None] = print,
    ):
        self.total_epochs = int(total_epochs) if total_epochs is not None else None
        self.history_path = Path(history_path) if history_path is not None else None
        self.print_fn = print_fn
        self.records = []

    def __call__(self, epoch_record):
        epoch = int(epoch_record["epoch"])
        loss = float(epoch_record["loss"])
        self.records.append({"epoch": epoch, "loss": loss})

        if self.total_epochs is None:
            self.print_fn(f"Epoch {epoch + 1} - loss={loss:.8g}")
        else:
            self.print_fn(f"Epoch {epoch + 1}/{self.total_epochs} - loss={loss:.8g}")

        self._write_history()

    def _write_history(self):
        if self.history_path is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(
            json.dumps({"history": self.records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
