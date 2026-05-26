import json
from dataclasses import MISSING, asdict, fields
from pathlib import Path

import h5py
import numpy as np
import torch

from data import HDF5Loader, build_graph
from data.dimensionless import temperature_from_dimensionless
from models import PDGCN, PDGCNConfig
from training.run_config import load_run_config, pdgcn_config_from_scale
from training.train_entry import derive_timing_from_hdf5, discover_hdf5_files
from visualization import write_polydata_vtk

from .config import InferenceRunConfig
from .fdm import compute_layer_fdm_coefficient
from .multilayer import rollout_multilayer_fdm


def run_multilayer_inference_from_config(config_path, *, checkpoint=None, h5_path=None, output_path=None):
    """Run multilayer PD-GCN + 1D FDM inference from an inference JSON config."""

    config_path = Path(config_path)
    run_config, inference_config, training_base_dir, inference_base_dir, training_config_path = (
        load_inference_run_context(config_path)
    )

    if int(inference_config.dataset_index) >= len(run_config.datasets):
        raise IndexError(
            f"inference.dataset_index={inference_config.dataset_index} exceeds "
            f"datasets length {len(run_config.datasets)}."
        )

    dataset = run_config.datasets[int(inference_config.dataset_index)]
    scale_params = dataset.scale.to_scale_params()
    if h5_path or inference_config.h5_path:
        selected_h5 = _resolve_path(inference_base_dir, h5_path or inference_config.h5_path)
    else:
        selected_h5 = discover_hdf5_files(_resolve_path(training_base_dir, dataset.h5_dir))[0]
    if checkpoint:
        selected_checkpoint = _resolve_path(inference_base_dir, checkpoint)
    else:
        selected_checkpoint = _resolve_path(
            training_base_dir,
            run_config.outputs.checkpoint_path if run_config.outputs is not None else run_config.data.checkpoint_path,
        )
    selected_output = _resolve_path(inference_base_dir, output_path or inference_config.output_path)
    selected_vtk_dir = (
        _resolve_path(inference_base_dir, inference_config.vtk_output_dir)
        if inference_config.vtk_output_dir is not None
        else selected_output.with_name(f"{selected_output.stem}_vtk")
    )

    timing = derive_timing_from_hdf5(selected_h5, scale_params, scan_velocity=dataset.scan_velocity)
    fallback_model_config = pdgcn_config_from_scale(
        scale_params,
        dt=timing["dt"],
        model_overrides=run_config.model,
    )
    device = torch.device(run_config.training.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint_payload = load_model_from_checkpoint(selected_checkpoint, fallback_model_config, device)

    num_frames, num_nodes = read_hdf5_temperature_shape(selected_h5)
    steps = int(inference_config.steps) if inference_config.steps is not None else int(num_frames)
    if steps > num_frames:
        raise ValueError(f"inference.steps={steps} exceeds available frames {num_frames}.")

    layer_spacing_star = float(inference_config.layer_spacing) / float(scale_params.L0)
    fdm_coefficient = compute_layer_fdm_coefficient(
        dt_star=getattr(model.config, "dt_star", 1.0),
        inverse_pe=getattr(model.config, "inverse_pe", 1.0),
        k_ratio=getattr(model.config, "k_ratio", 0.0),
        layer_spacing_star=layer_spacing_star,
    )

    loader = HDF5Loader(selected_h5, scale_params=scale_params)

    def graph_factory(frame_idx):
        raw = loader.load_graph_data(int(frame_idx), device=device)
        return build_graph(
            raw,
            scale_params,
            scan_velocity=timing["velocity_speed"],
            initial_temperature=torch.full(
                (raw.xyz.shape[0], 1),
                float(scale_params.T_amb),
                device=raw.xyz.device,
                dtype=raw.xyz.dtype,
            ),
        )

    metadata = {
        "checkpoint_path": str(selected_checkpoint),
        "source_h5": str(selected_h5),
        "config_path": str(config_path.resolve()),
        "training_config_path": str(training_config_path.resolve()),
        "num_layers": int(inference_config.num_layers),
        "layer_spacing": float(inference_config.layer_spacing),
        "layer_spacing_star": float(layer_spacing_star),
        "fdm_coefficient": float(fdm_coefficient),
        "top_heat_source_only": bool(inference_config.top_heat_source_only),
        "bottom_temperature_star": float(inference_config.bottom_temperature_star),
        "write_vtk": bool(inference_config.write_vtk),
        "vtk_interval": int(inference_config.vtk_interval),
        "vtk_output_dir": str(selected_vtk_dir) if bool(inference_config.write_vtk) else None,
        "hdf5_timing": timing,
        "scale_params": asdict(scale_params),
        "model_config": asdict(model.config),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
    }

    write_multilayer_hdf5(
        selected_output,
        model=model,
        graph_factory=graph_factory,
        steps=steps,
        scale_params=scale_params,
        num_layers=int(inference_config.num_layers),
        num_nodes=num_nodes,
        layer_spacing=float(inference_config.layer_spacing),
        metadata=metadata,
        warmup_steps=(
            int(inference_config.warmup_steps)
            if inference_config.warmup_steps is not None
            else int(run_config.training.warmup_steps)
        ),
        bottom_temperature_star=float(inference_config.bottom_temperature_star),
        top_heat_source_only=bool(inference_config.top_heat_source_only),
        allow_unstable_fdm=bool(inference_config.allow_unstable_fdm),
        write_vtk=bool(inference_config.write_vtk),
        vtk_interval=int(inference_config.vtk_interval),
        vtk_output_dir=selected_vtk_dir,
    )

    return {
        "output_path": str(selected_output),
        "checkpoint_path": str(selected_checkpoint),
        "h5_path": str(selected_h5),
        "steps": steps,
        "num_layers": int(inference_config.num_layers),
        "fdm_coefficient": float(fdm_coefficient),
        "vtk_output_dir": str(selected_vtk_dir) if bool(inference_config.write_vtk) else None,
    }


def load_inference_run_context(config_path):
    """Load split inference config or the legacy unified config."""

    config_path = Path(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("inference config JSON must contain an object at the top level.")

    inference_base_dir = config_path.resolve().parent
    if "training_config" not in payload:
        run_config = load_run_config(config_path)
        if run_config.inference is None:
            raise ValueError("Config must contain an 'inference' section for multilayer rollout.")
        return run_config, run_config.inference, inference_base_dir, inference_base_dir, config_path.resolve()

    unknown = sorted(set(payload) - {"training_config", "inference"})
    if unknown:
        raise ValueError(f"Unknown keys in inference config: {unknown}")
    training_config_value = payload.get("training_config")
    if not isinstance(training_config_value, str) or not training_config_value:
        raise ValueError("'training_config' must be a non-empty string path.")

    training_config_path = _resolve_path(inference_base_dir, training_config_value)
    run_config = load_run_config(training_config_path)
    inference_config = _build_inference_run_config(payload.get("inference"))
    return (
        run_config,
        inference_config,
        training_config_path.resolve().parent,
        inference_base_dir,
        training_config_path,
    )


def _build_inference_run_config(value) -> InferenceRunConfig:
    if value is None:
        raise ValueError("Missing required 'inference' section in inference config.")
    if not isinstance(value, dict):
        raise ValueError("'inference' section must be an object.")

    field_defs = fields(InferenceRunConfig)
    valid = {field.name for field in field_defs}
    unknown = sorted(set(value) - valid)
    if unknown:
        raise ValueError(f"Unknown keys in 'inference' section: {unknown}")
    missing = [
        field.name
        for field in field_defs
        if field.default is MISSING and field.default_factory is MISSING and field.name not in value
    ]
    if missing:
        raise ValueError(f"Missing required keys in 'inference' section: {missing}")
    return InferenceRunConfig(**dict(value))


def load_model_from_checkpoint(checkpoint_path, fallback_model_config: PDGCNConfig, device):
    checkpoint_payload = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint_payload.get("metadata", {})
    model_config_payload = metadata.get("model_config")
    model_config = PDGCNConfig(**model_config_payload) if model_config_payload is not None else fallback_model_config
    model = PDGCN(model_config).to(device)
    model.load_state_dict(checkpoint_payload["model"])
    model.eval()
    return model, checkpoint_payload


def read_hdf5_temperature_shape(h5_path):
    with h5py.File(h5_path, "r") as h5_file:
        shape = h5_file["dynamic/xyz"].shape
    return int(shape[0]), int(shape[1])


def write_multilayer_hdf5(
    output_path,
    *,
    model,
    graph_factory,
    steps: int,
    scale_params,
    num_layers: int,
    num_nodes: int,
    layer_spacing: float,
    metadata,
    warmup_steps: int,
    bottom_temperature_star: float,
    top_heat_source_only: bool,
    allow_unstable_fdm: bool,
    write_vtk: bool = True,
    vtk_interval: int = 20,
    vtk_output_dir=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vtk_interval = int(vtk_interval)
    if vtk_interval <= 0:
        raise ValueError(f"vtk_interval must be positive, got {vtk_interval}.")
    metadata_json = json.dumps(metadata, ensure_ascii=False)
    vtk_output_dir = Path(vtk_output_dir) if vtk_output_dir is not None else output_path.with_name(f"{output_path.stem}_vtk")
    graph_cache = {}

    def cached_graph_factory(frame_idx):
        graph = graph_factory(int(frame_idx))
        graph_cache[int(frame_idx)] = graph
        return graph

    with h5py.File(output_path, "w") as output_file:
        temperature_star_dataset = output_file.create_dataset(
            "temperature_star",
            shape=(int(steps), int(num_layers), int(num_nodes), 1),
            dtype="float32",
        )
        temperature_dataset = output_file.create_dataset(
            "temperature",
            shape=(int(steps), int(num_layers), int(num_nodes), 1),
            dtype="float32",
        )
        output_file.create_dataset("metadata", data=metadata_json)
        output_file.attrs["metadata"] = metadata_json

        def writer(step, temperature_star):
            temperature_star_dataset[int(step)] = temperature_star.numpy()
            temperature = temperature_from_dimensionless(temperature_star, scale_params)
            temperature_dataset[int(step)] = temperature.numpy()
            if bool(write_vtk) and _should_write_vtk_step(step, vtk_interval):
                _write_multilayer_step_vtk(
                    vtk_output_dir,
                    step=int(step),
                    graph=graph_cache[int(step)],
                    temperature=temperature,
                    temperature_star=temperature_star,
                )
            graph_cache.pop(int(step), None)

        rollout_multilayer_fdm(
            model,
            cached_graph_factory,
            int(steps),
            scale_params,
            num_layers=int(num_layers),
            layer_spacing=float(layer_spacing),
            return_dimensionless=True,
            return_all=False,
            writer=writer,
            warmup_steps=int(warmup_steps),
            bottom_temperature_star=float(bottom_temperature_star),
            top_heat_source_only=bool(top_heat_source_only),
            allow_unstable_fdm=bool(allow_unstable_fdm),
        )


def _should_write_vtk_step(step: int, vtk_interval: int) -> bool:
    return int(step) % int(vtk_interval) == 0


def _write_multilayer_step_vtk(vtk_output_dir, *, step: int, graph, temperature, temperature_star):
    vtk_output_dir = Path(vtk_output_dir)
    coords = graph.pos.detach().cpu().numpy() if getattr(graph, "pos", None) is not None else graph.x[:, 0:3].detach().cpu().numpy()
    edge_index = graph.edge_index.detach().cpu().numpy()
    temperature = temperature.detach().cpu().numpy()
    temperature_star = temperature_star.detach().cpu().numpy()
    num_layers = int(temperature.shape[0])
    for layer_index in range(num_layers):
        layer_values = {
            "temperature": temperature[layer_index].reshape(-1),
            "temperature_star": temperature_star[layer_index].reshape(-1),
            "layer_index": np.full((coords.shape[0],), float(layer_index), dtype=np.float32),
            "time_step": np.full((coords.shape[0],), float(step), dtype=np.float32),
        }
        write_polydata_vtk(
            vtk_output_dir / f"temperature_step_{step:06d}_layer_{layer_index:03d}.vtk",
            coords,
            edge_index=edge_index,
            point_data=layer_values,
            title=f"PDGCN step {step} layer {layer_index}",
        )


def _resolve_path(base_dir: Path, value) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
