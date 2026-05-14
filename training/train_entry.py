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

from data import FrameMemmapReader, build_static_cache
from data.static_cache import DYNAMIC_NODE_FILE, GLOBAL_FILE, META_FILE, STATIC_FILE
from models import PDGCN

from training.checkpoint import save_checkpoint
from training.monitor import LossMonitor
from training.run_config import (
    load_run_config,
    pdgcn_config_from_scale,
    run_config_to_dict,
)
from training.static_topology import GpuFeatureBuilder, StaticGraphState, train_static_topology


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

    h5_path = _resolve_path(base_dir, run_config.data.h5_path)
    cache_dir = _resolve_path(base_dir, run_config.data.cache_dir)
    checkpoint_path = _resolve_path(base_dir, run_config.data.checkpoint_path)
    history_path = (
        _resolve_path(base_dir, run_config.data.history_path)
        if run_config.data.history_path is not None
        else checkpoint_path.with_suffix(".history.json")
    )

    scale_params = run_config.scale.to_scale_params()
    timing = derive_timing_from_hdf5(
        h5_path,
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
        h5_path,
        cache_dir,
        scale_params,
        overwrite=run_config.data.overwrite_cache,
        scan_velocity=run_config.data.scan_velocity,
    )

    device = train_config.device
    static_state = StaticGraphState.from_cache(cache_dir, device=device)
    model = PDGCN(model_config).to(static_state.device)
    feature_builder = GpuFeatureBuilder(static_state, scale_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_config.lr))

    reader = FrameMemmapReader(cache_dir)
    monitor = LossMonitor(
        total_epochs=int(train_config.epochs),
        history_path=history_path,
    )
    try:
        history = train_static_topology(
            model,
            reader,
            static_state,
            feature_builder,
            train_config,
            optimizer=optimizer,
            epoch_callback=monitor,
        )
    finally:
        reader.close()

    metadata = {
        "run_config": run_config_to_dict(run_config),
        "scale_params": asdict(scale_params),
        "hdf5_timing": timing,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "history": history,
    }
    save_checkpoint(
        model,
        optimizer,
        checkpoint_path,
        epoch=int(history[-1]["epoch"]) if history else -1,
        metadata=metadata,
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps({"history": history, "metadata": metadata}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "cache_dir": str(cache_dir),
        "model_config": model_config,
        "scale_params": scale_params,
    }


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
    print(f"final_loss: {final_loss}")
    return 0


def _ensure_static_cache(h5_path, cache_dir, scale_params, *, overwrite: bool, scan_velocity):
    cache_dir = Path(cache_dir)
    required = [cache_dir / name for name in (STATIC_FILE, DYNAMIC_NODE_FILE, GLOBAL_FILE, META_FILE)]
    if not overwrite and all(path.exists() for path in required):
        return cache_dir
    return build_static_cache(
        h5_path,
        cache_dir,
        scale_params,
        scan_velocity=scan_velocity,
        overwrite=overwrite,
    )


def derive_timing_from_hdf5(h5_path, scale_params, *, scan_velocity=None, tolerance: float = 1e-6):
    """从 HDF5 切片文件派生文件级真实时间步和无量纲时间步。"""

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5_file:
        if "velocity_speed" not in h5_file.attrs:
            raise ValueError(f"HDF5 file {h5_path} is missing root attr 'velocity_speed'.")
        velocity = float(h5_file.attrs["velocity_speed"])
        if velocity <= 0:
            raise ValueError(f"HDF5 file {h5_path} has non-positive velocity_speed={velocity}.")
        if scan_velocity is not None and not np.isclose(
            float(scan_velocity),
            velocity,
            rtol=tolerance,
            atol=tolerance,
        ):
            raise ValueError(
                "Configured scan_velocity must match HDF5 velocity_speed when dt is "
                f"derived from file timing; got scan_velocity={scan_velocity}, "
                f"velocity_speed={velocity}."
            )

        if "dynamic/xyz" not in h5_file:
            raise ValueError(f"HDF5 file {h5_path} is missing dataset 'dynamic/xyz'.")
        num_frames = int(h5_file["dynamic/xyz"].shape[0])
        if num_frames < 2:
            raise ValueError(f"HDF5 file {h5_path} must contain at least two frames.")

        step_distance = _read_file_step_distance(
            h5_file,
            h5_path,
            expected_intervals=num_frames - 1,
            tolerance=tolerance,
        )
        if "path/slice_path_length" in h5_file:
            slice_path_length = float(np.asarray(h5_file["path/slice_path_length"][()]))
            expected_length = step_distance * float(num_frames - 1)
            if not np.isclose(slice_path_length, expected_length, rtol=tolerance, atol=tolerance):
                raise ValueError(
                    "HDF5 slice_path_length must equal heat_center_step_distance * "
                    f"(num_frames - 1); got {slice_path_length} vs {expected_length}."
                )
        else:
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


if __name__ == "__main__":
    raise SystemExit(main())
