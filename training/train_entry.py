import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from data import HDF5FrameReader, build_static_cache
from data.hdf5_units import length_mm_to_m, velocity_mm_per_s_to_m_per_s
from data.static_cache import META_FILE, STATIC_FILE
from models import PDGCN

from training.checkpoint import save_checkpoint
from training.monitor import LossMonitor, TrainingProcessMonitor
from training.run_config import (
    load_run_config,
    pdgcn_config_from_scale,
    run_config_to_dict,
)
from training.static_topology import (
    GpuFeatureBuilder,
    StaticGraphState,
    evaluate_static_topology_sequence,
    train_static_topology_sequences,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pdgcn_train.example.json"


def run_training_from_config(config_path):
    """从统一 JSON 配置执行固定拓扑端到端训练。"""

    config_path = Path(config_path)
    run_config = load_run_config(config_path)
    if len(run_config.datasets) > 1:
        raise NotImplementedError(
            "The classified config schema can list multiple datasets, but "
            "run_training_from_config currently supports one dataset per run."
        )
    base_dir = config_path.resolve().parent

    h5_dir = _resolve_path(base_dir, run_config.data.h5_dir)
    h5_paths = discover_hdf5_files(h5_dir)
    first_h5_path = h5_paths[0]
    cache_dir = _resolve_path(base_dir, run_config.data.cache_dir)
    checkpoint_path = _resolve_path(base_dir, run_config.data.checkpoint_path)
    history_path = (
        _resolve_path(base_dir, run_config.data.history_path)
        if run_config.data.history_path is not None
        else checkpoint_path.with_suffix(".history.json")
    )

    scale_params = run_config.scale.to_scale_params()
    timing = derive_timing_from_hdf5(
        first_h5_path,
        scale_params,
        scan_velocity=run_config.data.scan_velocity,
    )
    model_config = pdgcn_config_from_scale(
        scale_params,
        dt=timing["dt"],
        model_overrides=run_config.model,
    )
    train_config = run_config.training

    _ensure_static_cache(
        first_h5_path,
        cache_dir,
        scale_params,
        scan_velocity=run_config.data.scan_velocity,
    )

    device = train_config.device
    static_state = StaticGraphState.from_cache(cache_dir, device=device)
    model = PDGCN(model_config).to(static_state.device)
    feature_builder = GpuFeatureBuilder(static_state, scale_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_config.lr))

    readers = [
        HDF5FrameReader(
            h5_path,
            expected_num_nodes=static_state.num_nodes,
            scale_params=scale_params,
            scan_velocity=run_config.data.scan_velocity,
        )
        for h5_path in h5_paths
    ]
    monitor_frame_index = (
        run_config.monitoring.temperature_frame_index
        if run_config.monitoring.temperature_frame_index is not None
        else readers[0].num_frames // 2
    )
    monitor_frame_index = min(int(monitor_frame_index), readers[0].num_frames - 1)
    monitor = _build_monitor(
        run_config,
        history_path,
        h5_paths=h5_paths,
        scale_params=scale_params,
        model_config=model_config,
        train_config=train_config,
        temperature_frame_index=monitor_frame_index,
    )

    def slice_callback(slice_context):
        if not run_config.monitoring.enabled or not hasattr(monitor, "record_slice"):
            return
        slice_record, slice_payload = evaluate_static_topology_sequence(
            model,
            readers[0],
            static_state,
            feature_builder,
            train_config,
            epoch=int(slice_context["epoch"]),
            slice_index=int(slice_context["slice_index"]),
            monitor_frame_index=monitor_frame_index,
        )
        monitor.record_slice(slice_record, slice_payload)

    try:
        history = train_static_topology_sequences(
            model,
            readers,
            static_state,
            feature_builder,
            train_config,
            optimizer=optimizer,
            monitor_callback=monitor if run_config.monitoring.enabled else None,
            epoch_callback=None if run_config.monitoring.enabled else monitor,
            slice_callback=slice_callback if run_config.monitoring.enabled else None,
            monitor_frame_index=monitor_frame_index,
        )
    finally:
        for reader in readers:
            reader.close()

    metadata = {
        "run_config": run_config_to_dict(run_config),
        "scale_params": asdict(scale_params),
        "hdf5_timing": timing,
        "h5_files": [str(path) for path in h5_paths],
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "history": history,
    }
    slice_records = getattr(monitor, "slice_records", [])
    save_checkpoint(
        model,
        optimizer,
        checkpoint_path,
        epoch=int(history[-1]["epoch"]) if history else -1,
        metadata=metadata,
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(
            {"history": history, "slice_records": slice_records, "metadata": metadata},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "monitor_data_path": str(getattr(monitor, "metrics_path", "")),
        "cache_dir": str(cache_dir),
        "h5_files": [str(path) for path in h5_paths],
        "model_config": model_config,
        "scale_params": scale_params,
    }


def _build_monitor(
    run_config,
    history_path,
    *,
    h5_paths=None,
    scale_params=None,
    model_config=None,
    train_config=None,
    temperature_frame_index=None,
):
    if not run_config.monitoring.enabled:
        return LossMonitor(
            total_epochs=int(run_config.training.epochs),
            history_path=history_path,
        )

    figures_dir = (
        _resolve_path(history_path.parent, run_config.monitoring.figures_dir)
        if run_config.monitoring.figures_dir is not None
        else history_path.parent / "figures"
    )
    metrics_path = (
        _resolve_path(history_path.parent, run_config.monitoring.metrics_path)
        if run_config.monitoring.metrics_path is not None
        else history_path.parent / "metrics" / "monitor_data.h5"
    )
    return TrainingProcessMonitor(
        total_epochs=int(run_config.training.epochs),
        history_path=history_path,
        figures_dir=figures_dir,
        metrics_path=metrics_path,
        interval_epochs=int(run_config.monitoring.interval_epochs),
        temperature_frame_index=temperature_frame_index,
        h5_files=h5_paths,
        scale_params=scale_params,
        model_config=model_config,
        train_config=train_config,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train PD-GCN from a unified JSON config.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a PD-GCN training JSON config.",
    )
    args = parser.parse_args(argv)
    result = run_training_from_config(args.config)
    final_loss = result["history"][-1]["loss"] if result["history"] else None
    print(f"checkpoint: {result['checkpoint_path']}")
    print(f"history: {result['history_path']}")
    if result.get("monitor_data_path"):
        print(f"monitor_data: {result['monitor_data_path']}")
    print(f"final_loss: {final_loss}")
    return 0


def discover_hdf5_files(h5_dir):
    h5_dir = Path(h5_dir)
    if not h5_dir.exists():
        raise FileNotFoundError(f"HDF5 directory not found: {h5_dir}")
    if not h5_dir.is_dir():
        raise NotADirectoryError(f"h5_dir must be a directory: {h5_dir}")
    h5_paths = sorted(
        [path for path in h5_dir.iterdir() if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}],
        key=lambda path: _natural_sort_key(path.name),
    )
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files found in directory: {h5_dir}")
    return h5_paths


def _ensure_static_cache(h5_path, cache_dir, scale_params, *, scan_velocity):
    cache_dir = Path(cache_dir)
    required = [cache_dir / name for name in (STATIC_FILE, META_FILE)]
    if all(path.exists() for path in required):
        return cache_dir
    return build_static_cache(
        h5_path,
        cache_dir,
        scale_params,
        scan_velocity=scan_velocity,
    )


def derive_timing_from_hdf5(h5_path, scale_params, *, scan_velocity=None, tolerance: float = 1e-6):
    """从 HDF5 切片文件派生文件级真实时间步和无量纲时间步。"""

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5_file:
        if "velocity_speed" not in h5_file.attrs:
            raise ValueError(f"HDF5 file {h5_path} is missing root attr 'velocity_speed'.")
        native_velocity = float(h5_file.attrs["velocity_speed"])
        velocity = float(velocity_mm_per_s_to_m_per_s(native_velocity))
        if velocity <= 0:
            raise ValueError(f"HDF5 file {h5_path} has non-positive velocity_speed={native_velocity}.")
        if scan_velocity is not None and not np.isclose(
            float(scan_velocity),
            velocity,
            rtol=tolerance,
            atol=tolerance,
        ):
            raise ValueError(
                "Configured scan_velocity must match HDF5 velocity_speed when dt is "
                f"derived from file timing; got scan_velocity={scan_velocity}, "
                f"velocity_speed={velocity} m/s."
            )

        if "dynamic/xyz" not in h5_file:
            raise ValueError(f"HDF5 file {h5_path} is missing dataset 'dynamic/xyz'.")
        num_frames = int(h5_file["dynamic/xyz"].shape[0])
        if num_frames < 2:
            raise ValueError(f"HDF5 file {h5_path} must contain at least two frames.")

        native_step_distance = _read_file_step_distance(
            h5_file,
            h5_path,
            expected_intervals=num_frames - 1,
            tolerance=tolerance,
        )
        step_distance = float(length_mm_to_m(native_step_distance))
        if "path/slice_path_length" in h5_file:
            native_slice_path_length = float(np.asarray(h5_file["path/slice_path_length"][()]))
            slice_path_length = float(length_mm_to_m(native_slice_path_length))
            expected_length = step_distance * float(num_frames - 1)
            if not np.isclose(slice_path_length, expected_length, rtol=tolerance, atol=tolerance):
                raise ValueError(
                    "HDF5 slice_path_length must equal heat_center_step_distance * "
                    f"(num_frames - 1); got {slice_path_length} vs {expected_length}."
                )
        else:
            native_slice_path_length = None
            slice_path_length = None

    dt = step_distance / velocity
    dt_star = dt / (float(scale_params.L0) / float(scale_params.v0))
    return {
        "source": "hdf5:path/heat_center_step_distance",
        "dt": float(dt),
        "dt_star": float(dt_star),
        "step_distance": float(step_distance),
        "velocity_speed": float(velocity),
        "num_frames": int(num_frames),
        "slice_path_length": slice_path_length,
        "native_step_distance_mm": float(native_step_distance),
        "native_velocity_speed_mm_per_s": float(native_velocity),
        "native_slice_path_length_mm": native_slice_path_length,
    }


def _read_file_step_distance(h5_file, h5_path, *, expected_intervals: int, tolerance: float):
    if "path/heat_center_step_distance" not in h5_file:
        raise ValueError(f"HDF5 file {h5_path} is missing dataset 'path/heat_center_step_distance'.")

    raw = np.asarray(h5_file["path/heat_center_step_distance"][()], dtype=np.float64)
    if raw.ndim == 0:
        step_distance = float(raw)
    else:
        if raw.size == 0:
            raise ValueError(f"HDF5 file {h5_path} has empty heat_center_step_distance.")
        if raw.size != int(expected_intervals):
            raise ValueError(
                "HDF5 heat_center_step_distance array must have length num_frames - 1; "
                f"got {raw.size}, expected {expected_intervals}."
            )
        if not np.all(np.isfinite(raw)):
            raise ValueError(f"HDF5 file {h5_path} has non-finite heat_center_step_distance values.")
        min_step = float(np.min(raw))
        max_step = float(np.max(raw))
        if not np.isclose(min_step, max_step, rtol=tolerance, atol=tolerance):
            raise ValueError(
                "HDF5 heat_center_step_distance must be constant within a slice; "
                f"got min={min_step}, max={max_step}."
            )
        step_distance = float(np.mean(raw))

    if not np.isfinite(step_distance) or step_distance <= 0:
        raise ValueError(
            f"HDF5 file {h5_path} has non-positive heat_center_step_distance={step_distance}."
        )
    return step_distance


def _resolve_path(base_dir: Path, value) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _natural_sort_key(value: str):
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


if __name__ == "__main__":
    raise SystemExit(main())
