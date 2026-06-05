import json
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np


EPOCH_METRIC_FIELDS = (
    "epoch",
    "loss_total",
    "loss_pde",
    "loss_outflow",
    "loss_beta",
    "loss_smooth",
    "loss_zero_source_anchor",
    "temperature_mean",
    "temperature_max",
    "temperature_min",
    "temperature_var",
)
SLICE_METRIC_FIELDS = (
    "epoch",
    "slice_index",
    "loss_total",
    "loss_pde",
    "loss_outflow",
    "loss_beta",
    "loss_smooth",
    "loss_zero_source_anchor",
    "temperature_mean",
    "temperature_max",
    "temperature_min",
    "temperature_var",
)
INTEGER_FIELDS = {"epoch", "slice_index"}


class LossMonitor:
    """Print epoch loss and keep the lightweight JSON history."""

    def __init__(
        self,
        *,
        total_epochs: Optional[int] = None,
        history_path=None,
        print_fn: Callable[[str], None] = print,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.total_epochs = int(total_epochs) if total_epochs is not None else None
        self.history_path = Path(history_path) if history_path is not None else None
        self.print_fn = print_fn
        self.clock = clock
        self.start_time = float(self.clock())
        self.last_epoch_time = self.start_time
        self.records = []

    def __call__(self, epoch_record):
        epoch = int(epoch_record["epoch"])
        loss = float(epoch_record["loss"])
        self.records.append({"epoch": epoch, "loss": loss})

        self.print_fn(
            _format_epoch_message(
                epoch=epoch,
                loss=loss,
                total_epochs=self.total_epochs,
                timing=self._tick_timing(epoch),
            )
        )

        self._write_history()

    def _write_history(self):
        if self.history_path is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(
            json.dumps({"history": self.records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _tick_timing(self, epoch: int):
        return _tick_timing(
            clock=self.clock,
            start_time=self.start_time,
            last_epoch_time=self.last_epoch_time,
            total_epochs=self.total_epochs,
            epoch=epoch,
            update_last_epoch_time=self._set_last_epoch_time,
        )

    def _set_last_epoch_time(self, value: float):
        self.last_epoch_time = float(value)


class TrainingProcessMonitor:
    """Write training monitor records and snapshots to a single HDF5 file.

    Training deliberately does not create visualization files. Use
    ``python -m training.visualize_monitor --monitor-data ...`` to export VTK files.
    """

    def __init__(
        self,
        *,
        total_epochs: Optional[int] = None,
        history_path=None,
        figures_dir=None,
        metrics_path=None,
        interval_epochs: int = 10,
        temperature_frame_index: Optional[int] = None,
        h5_files=None,
        scale_params=None,
        model_config=None,
        train_config=None,
        print_fn: Callable[[str], None] = print,
        clock: Callable[[], float] = time.perf_counter,
    ):
        if int(interval_epochs) <= 0:
            raise ValueError(f"interval_epochs must be positive, got {interval_epochs}.")
        self.total_epochs = int(total_epochs) if total_epochs is not None else None
        self.history_path = Path(history_path) if history_path is not None else None
        base_dir = self.history_path.parent if self.history_path is not None else Path.cwd()
        self.figures_dir = Path(figures_dir) if figures_dir is not None else base_dir / "figures"
        self.metrics_path = Path(metrics_path) if metrics_path is not None else base_dir / "metrics" / "monitor_data.h5"
        self.interval_epochs = int(interval_epochs)
        self.temperature_frame_index = (
            int(temperature_frame_index) if temperature_frame_index is not None else None
        )
        self.h5_files = [str(path) for path in (h5_files or [])]
        self.scale_params = _json_ready_value(_dataclass_or_value(scale_params))
        self.model_config = _json_ready_value(_dataclass_or_value(model_config))
        self.train_config = _json_ready_value(_dataclass_or_value(train_config))
        self.print_fn = print_fn
        self.clock = clock
        self.start_time = float(self.clock())
        self.last_epoch_time = self.start_time
        self.records = []
        self.slice_records = []
        self._initialize_hdf5()

    def __call__(self, epoch_record, monitor_payload=None):
        record = _json_ready_record(epoch_record)
        self.records.append(record)

        epoch = int(record["epoch"])
        loss = float(record.get("loss_total", record.get("loss", 0.0)))
        self.print_fn(
            _format_epoch_message(
                epoch=epoch,
                loss=loss,
                total_epochs=self.total_epochs,
                timing=self._tick_timing(epoch),
            )
        )

        snapshot = (monitor_payload or {}).get("snapshot")
        with h5py.File(self.metrics_path, "a") as h5_file:
            _append_metric_record(h5_file["epoch_metrics"], EPOCH_METRIC_FIELDS, record)
            if snapshot and self._should_store_snapshot(epoch):
                _write_snapshot(
                    h5_file.require_group("epoch_snapshots"),
                    f"epoch_{epoch + 1:04d}",
                    snapshot,
                    epoch=epoch,
                    slice_index=None,
                    kind="epoch",
                )
        self._write_history()

    def record_slice(self, slice_record, monitor_payload=None):
        record = _json_ready_record(slice_record)
        self.slice_records.append(record)

        snapshot = (monitor_payload or {}).get("snapshot")
        with h5py.File(self.metrics_path, "a") as h5_file:
            _append_metric_record(h5_file["slice_metrics"], SLICE_METRIC_FIELDS, record)
            if snapshot:
                epoch = int(record.get("epoch", 0))
                slice_index = int(record.get("slice_index", 0))
                _write_snapshot(
                    h5_file.require_group("slice_snapshots"),
                    f"epoch_{epoch + 1:04d}_slice_{slice_index + 1:04d}",
                    snapshot,
                    epoch=epoch,
                    slice_index=slice_index,
                    kind="slice",
                )
        self._write_history()

    def _should_store_snapshot(self, epoch: int) -> bool:
        return int(epoch) == 0 or (int(epoch) + 1) % self.interval_epochs == 0

    def _initialize_hdf5(self):
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.metrics_path, "w") as h5_file:
            h5_file.attrs["schema_version"] = "1.0"
            h5_file.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
            h5_file.attrs["figures_dir"] = str(self.figures_dir)
            h5_file.attrs["interval_epochs"] = self.interval_epochs
            h5_file.attrs["temperature_frame_index"] = (
                -1 if self.temperature_frame_index is None else self.temperature_frame_index
            )
            h5_file.attrs["h5_files"] = json.dumps(self.h5_files, ensure_ascii=False)
            h5_file.attrs["scale_params_json"] = json.dumps(self.scale_params, ensure_ascii=False)
            h5_file.attrs["model_config_json"] = json.dumps(self.model_config, ensure_ascii=False)
            h5_file.attrs["train_config_json"] = json.dumps(self.train_config, ensure_ascii=False)
            _create_metric_group(h5_file, "epoch_metrics", EPOCH_METRIC_FIELDS)
            _create_metric_group(h5_file, "slice_metrics", SLICE_METRIC_FIELDS)
            h5_file.require_group("epoch_snapshots")
            h5_file.require_group("slice_snapshots")

    def _write_history(self):
        if self.history_path is None:
            return
        payload = {"history": self.records, "slice_records": self.slice_records}
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _tick_timing(self, epoch: int):
        return _tick_timing(
            clock=self.clock,
            start_time=self.start_time,
            last_epoch_time=self.last_epoch_time,
            total_epochs=self.total_epochs,
            epoch=epoch,
            update_last_epoch_time=self._set_last_epoch_time,
        )

    def _set_last_epoch_time(self, value: float):
        self.last_epoch_time = float(value)


def _format_epoch_message(*, epoch: int, loss: float, total_epochs: Optional[int], timing):
    if total_epochs is None:
        prefix = f"Epoch {int(epoch) + 1}"
    else:
        prefix = f"Epoch {int(epoch) + 1}/{int(total_epochs)}"
    parts = [
        prefix,
        f"loss={float(loss):.8g}",
        f"epoch_time={_format_duration(timing['epoch_seconds'])}",
        f"elapsed={_format_duration(timing['elapsed_seconds'])}",
    ]
    if timing.get("eta_seconds") is not None:
        parts.append(f"eta={_format_duration(timing['eta_seconds'])}")
    return " - ".join(parts)


def _tick_timing(*, clock, start_time: float, last_epoch_time: float, total_epochs, epoch: int, update_last_epoch_time):
    now = float(clock())
    epoch_seconds = max(0.0, now - float(last_epoch_time))
    elapsed_seconds = max(0.0, now - float(start_time))
    update_last_epoch_time(now)
    eta_seconds = None
    if total_epochs is not None:
        completed_epochs = max(int(epoch) + 1, 1)
        remaining_epochs = max(int(total_epochs) - completed_epochs, 0)
        eta_seconds = (elapsed_seconds / completed_epochs) * remaining_epochs
    return {
        "epoch_seconds": epoch_seconds,
        "elapsed_seconds": elapsed_seconds,
        "eta_seconds": eta_seconds,
    }


def _format_duration(seconds: float):
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _create_metric_group(h5_file, group_name: str, fields):
    group = h5_file.require_group(group_name)
    for field in fields:
        dtype = np.int64 if field in INTEGER_FIELDS else np.float64
        group.create_dataset(
            field,
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype=dtype,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
    return group


def _append_metric_record(group, fields, record):
    for field in fields:
        dataset = group[field]
        dataset.resize((dataset.shape[0] + 1,))
        if field in INTEGER_FIELDS:
            dataset[-1] = int(record.get(field, -1))
        else:
            value = record.get(field)
            dataset[-1] = float(value) if value is not None else np.nan


def _write_snapshot(parent_group, group_name: str, snapshot, *, epoch: int, slice_index, kind: str):
    if group_name in parent_group:
        del parent_group[group_name]
    group = parent_group.create_group(group_name)
    frame_index = int(snapshot.get("frame_index", 0))
    group.attrs["frame_index"] = frame_index
    group.attrs["epoch"] = int(epoch)
    group.attrs["slice_index"] = -1 if slice_index is None else int(slice_index)
    group.attrs["kind"] = str(kind)

    coords = np.asarray(snapshot["coords"], dtype=np.float32)
    residual = np.asarray(snapshot["residual"], dtype=np.float32).reshape(-1)
    temperature = np.asarray(snapshot["temperature"], dtype=np.float32).reshape(-1)
    edge_index = snapshot.get("edge_index")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"snapshot coords must have shape [N, 3], got {coords.shape}.")
    if residual.shape[0] != coords.shape[0] or temperature.shape[0] != coords.shape[0]:
        raise ValueError("snapshot residual/temperature length must match coords.")

    _create_array_dataset(group, "coords", coords)
    _create_array_dataset(group, "residual", residual)
    _create_array_dataset(group, "temperature", temperature)
    if edge_index is not None:
        edge_index = np.asarray(edge_index, dtype=np.int64)
        if edge_index.ndim != 2 or 2 not in edge_index.shape:
            raise ValueError(f"snapshot edge_index must have shape [2, E] or [E, 2], got {edge_index.shape}.")
        if edge_index.shape[0] != 2:
            edge_index = edge_index.T
        _create_array_dataset(group, "edge_index", edge_index)


def _create_array_dataset(group, name: str, values: np.ndarray):
    chunk_rows = min(max(int(values.shape[0]), 1), 65536)
    chunks = (chunk_rows,) if values.ndim == 1 else (chunk_rows, values.shape[1])
    group.create_dataset(
        name,
        data=values,
        chunks=chunks,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )


def _json_ready_record(record):
    return {
        key: _json_ready_value(value)
        for key, value in dict(record).items()
        if not str(key).startswith("_")
    }


def _json_ready_value(value):
    try:
        import torch

        if torch.is_tensor(value):
            if value.numel() == 1:
                return float(value.detach().cpu())
            return value.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready_value(item) for item in value]
    return str(value)


def _dataclass_or_value(value):
    if value is not None and is_dataclass(value):
        return asdict(value)
    return value
