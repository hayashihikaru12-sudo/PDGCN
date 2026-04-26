import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

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
    model_config = pdgcn_config_from_scale(
        scale_params,
        dt=run_config.scale.dt,
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


def _resolve_path(base_dir: Path, value) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
