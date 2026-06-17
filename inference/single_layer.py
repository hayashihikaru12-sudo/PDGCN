import json
import time
from dataclasses import MISSING, asdict, fields
from pathlib import Path

import h5py
import numpy as np
import torch

from data import HDF5FrameReader, HDF5Loader, build_graph, build_static_cache
from data.dimensionless import ScaleParams, temperature_from_dimensionless, temperature_to_dimensionless
from data.static_cache import META_FILE, STATIC_FILE
from models import PDGCN, PDGCNConfig
from pde import apply_dirichlet_boundary
from training.graph_utils import clone_graph_with_temperature, graph_explicit_source_delta
from training.run_config import load_run_config, pdgcn_config_from_scale
from training.static_topology import GpuFeatureBuilder, StaticGraphState
from training.train_entry import derive_timing_from_hdf5, discover_hdf5_files
from training.warmup import pseudo_time_relax_initial_temperature
from visualization import write_surface_vtu

from .config import SingleLayerInferenceRunConfig
from .io import (
    _read_prediction_metadata,
    _resolve_path,
    _should_write_cloud_step,
    _update_prediction_metadata,
    load_model_from_checkpoint,
    read_hdf5_temperature_shape,
)


def run_single_layer_inference_from_config(
    config_path,
    *,
    checkpoint=None,
    h5_path=None,
    output_path=None,
    vtu_output_dir=None,
    vtu_interval=None,
    mode=None,
):
    """Run single-layer PD-GCN inference from a split inference JSON config."""

    config_path = Path(config_path)
    (
        run_config,
        inference_config,
        training_base_dir,
        inference_base_dir,
        training_config_path,
    ) = load_single_layer_inference_run_context(config_path)
    if mode is not None:
        inference_config = SingleLayerInferenceRunConfig(
            **{**asdict(inference_config), "mode": str(mode)}
        )
    if vtu_interval is not None:
        inference_config = SingleLayerInferenceRunConfig(
            **{**asdict(inference_config), "vtu_interval": int(vtu_interval)}
        )

    if int(inference_config.dataset_index) >= len(run_config.datasets):
        raise IndexError(
            f"single_layer_inference.dataset_index={inference_config.dataset_index} exceeds "
            f"datasets length {len(run_config.datasets)}."
        )

    dataset = run_config.datasets[int(inference_config.dataset_index)]
    scale_params = dataset.scale.to_scale_params()
    selected_h5 = (
        _resolve_path(inference_base_dir, h5_path or inference_config.h5_path)
        if h5_path or inference_config.h5_path
        else discover_hdf5_files(_resolve_path(training_base_dir, dataset.h5_dir))[0]
    )
    selected_checkpoint = (
        _resolve_path(inference_base_dir, checkpoint)
        if checkpoint
        else _resolve_path(
            training_base_dir,
            run_config.outputs.checkpoint_path if run_config.outputs is not None else run_config.data.checkpoint_path,
        )
    )
    selected_output = _resolve_path(inference_base_dir, output_path or inference_config.output_path)
    selected_vtu_dir = (
        _resolve_path(inference_base_dir, vtu_output_dir or inference_config.vtu_output_dir)
        if vtu_output_dir or inference_config.vtu_output_dir
        else selected_output.with_name(f"{selected_output.stem}_vtu")
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
        raise ValueError(f"single_layer_inference.steps={steps} exceeds available frames {num_frames}.")

    cache_dir = _resolve_path(training_base_dir, dataset.cache_dir)
    _ensure_static_cache(selected_h5, cache_dir, scale_params, scan_velocity=dataset.scan_velocity)
    static_state = StaticGraphState.from_cache(cache_dir, device=device)
    warmup_steps = (
        int(inference_config.warmup_steps)
        if inference_config.warmup_steps is not None
        else int(run_config.training.warmup_steps)
    )
    mode_value = str(inference_config.mode).strip().lower()
    require_fem = mode_value == "teacher_forcing"

    metadata = {
        "checkpoint_path": str(selected_checkpoint),
        "source_h5": str(selected_h5),
        "config_path": str(config_path.resolve()),
        "training_config_path": str(training_config_path.resolve()),
        "mode": mode_value,
        "num_layers": 1,
        "thickness_solver": None,
        "write_vtu": bool(inference_config.write_vtu),
        "vtu_interval": int(inference_config.vtu_interval),
        "vtu_output_dir": str(selected_vtu_dir),
        "hdf5_timing": timing,
        "scale_params": asdict(scale_params),
        "model_config": asdict(model.config),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "fem_temperature_dataset": inference_config.fem_temperature_dataset,
        "fem_valid_mask_dataset": inference_config.fem_valid_mask_dataset,
    }

    with HDF5FrameReader(
        selected_h5,
        expected_num_nodes=static_state.num_nodes,
        scale_params=scale_params,
        scan_velocity=dataset.scan_velocity,
        require_fem_temperature=require_fem,
        fem_temperature_dataset=inference_config.fem_temperature_dataset,
        fem_valid_mask_dataset=inference_config.fem_valid_mask_dataset,
    ) as frame_reader:
        feature_builder = GpuFeatureBuilder(static_state, scale_params, model_config=model.config)
        timing_summary = write_single_layer_hdf5(
            selected_output,
            model=model,
            frame_reader=frame_reader,
            static_state=static_state,
            feature_builder=feature_builder,
            steps=steps,
            scale_params=scale_params,
            metadata=metadata,
            warmup_steps=warmup_steps,
            mode=mode_value,
        )

    render_summary = {"render_seconds": 0.0, "rendered_steps": [], "vtu_output_dir": str(selected_vtu_dir)}
    if bool(inference_config.write_vtu):
        render_summary = render_single_layer_surfaces_from_hdf5(
            selected_output,
            vtu_interval=int(inference_config.vtu_interval),
            vtu_output_dir=selected_vtu_dir,
        )
    total_seconds = float(timing_summary["inference_seconds"]) + float(render_summary["render_seconds"])
    _update_prediction_metadata(selected_output, {**render_summary, "total_seconds": total_seconds})

    return {
        "output_path": str(selected_output),
        "checkpoint_path": str(selected_checkpoint),
        "h5_path": str(selected_h5),
        "steps": steps,
        "mode": mode_value,
        "num_layers": 1,
        "vtu_interval": int(inference_config.vtu_interval),
        **timing_summary,
        **render_summary,
        "total_seconds": total_seconds,
    }


def load_single_layer_inference_run_context(config_path):
    config_path = Path(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("single-layer inference config JSON must contain an object at the top level.")
    unknown = sorted(set(payload) - {"training_config", "single_layer_inference"})
    if unknown:
        raise ValueError(f"Unknown keys in single-layer inference config: {unknown}")
    training_config_value = payload.get("training_config")
    if not isinstance(training_config_value, str) or not training_config_value:
        raise ValueError("'training_config' must be a non-empty string path.")

    inference_base_dir = config_path.resolve().parent
    training_config_path = _resolve_path(inference_base_dir, training_config_value)
    run_config = load_run_config(training_config_path)
    inference_config = _build_single_layer_inference_run_config(payload.get("single_layer_inference"))
    return (
        run_config,
        inference_config,
        training_config_path.resolve().parent,
        inference_base_dir,
        training_config_path,
    )


def _build_single_layer_inference_run_config(value) -> SingleLayerInferenceRunConfig:
    if value is None:
        raise ValueError("Missing required 'single_layer_inference' section in inference config.")
    if not isinstance(value, dict):
        raise ValueError("'single_layer_inference' section must be an object.")

    field_defs = fields(SingleLayerInferenceRunConfig)
    valid = {field.name for field in field_defs}
    unknown = sorted(set(value) - valid)
    if unknown:
        raise ValueError(f"Unknown keys in 'single_layer_inference' section: {unknown}")
    missing = [
        field.name
        for field in field_defs
        if field.default is MISSING and field.default_factory is MISSING and field.name not in value
    ]
    if missing:
        raise ValueError(f"Missing required keys in 'single_layer_inference' section: {missing}")
    return SingleLayerInferenceRunConfig(**dict(value))


def write_single_layer_hdf5(
    output_path,
    *,
    model,
    frame_reader: HDF5FrameReader,
    static_state: StaticGraphState,
    feature_builder: GpuFeatureBuilder,
    steps: int,
    scale_params: ScaleParams,
    metadata,
    warmup_steps: int,
    mode: str,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = str(mode).strip().lower()
    include_autoregressive = mode in {"autoregressive", "both"}
    include_teacher = mode in {"teacher_forcing", "both"} and frame_reader.has_fem_temperature
    if mode == "teacher_forcing" and not frame_reader.has_fem_temperature:
        raise KeyError(f"HDF5 file {frame_reader.h5_path} is missing FEM temperature for teacher_forcing mode.")
    if include_teacher and int(steps) < 2:
        raise ValueError("Teacher-forcing single-layer inference requires at least two frames.")

    metadata = dict(metadata)
    metadata["has_fem_temperature"] = bool(frame_reader.has_fem_temperature)
    metadata["teacher_forcing_enabled"] = bool(include_teacher)
    metadata["autoregressive_enabled"] = bool(include_autoregressive)
    step_inference_seconds = []
    timing_summary = {
        "inference_seconds": 0.0,
        "render_seconds": 0.0,
        "total_seconds": 0.0,
        "average_inference_seconds": 0.0,
        "max_inference_seconds": 0.0,
        "min_inference_seconds": 0.0,
        "rendered_steps": [],
    }

    with h5py.File(output_path, "w") as output_file:
        autoreg_temperature_star = None
        autoreg_temperature = None
        if include_autoregressive:
            autoreg_temperature_star = output_file.create_dataset(
                "temperature_star",
                shape=(int(steps), int(static_state.num_nodes), 1),
                dtype="float32",
            )
            autoreg_temperature = output_file.create_dataset(
                "temperature",
                shape=(int(steps), int(static_state.num_nodes), 1),
                dtype="float32",
            )

        start_total = time.perf_counter()
        if include_autoregressive:

            def writer(step, temperature_star):
                autoreg_temperature_star[int(step)] = temperature_star.numpy()
                temperature = temperature_from_dimensionless(temperature_star, scale_params)
                autoreg_temperature[int(step)] = temperature.numpy()

            rollout_single_layer_static(
                model,
                frame_reader,
                static_state,
                feature_builder,
                int(steps),
                scale_params,
                writer=writer,
                return_all=False,
                return_dimensionless=True,
                warmup_steps=int(warmup_steps),
                timing_recorder=step_inference_seconds.append,
            )
            if frame_reader.has_fem_temperature:
                _write_autoregressive_fem_datasets(output_file, frame_reader, scale_params, int(steps))

        if include_teacher:
            teacher_group = output_file.create_group("teacher_forcing")
            _write_teacher_forcing_group(
                teacher_group,
                model,
                frame_reader,
                static_state,
                feature_builder,
                int(steps),
                scale_params,
                timing_recorder=step_inference_seconds.append,
            )

        timing_summary["total_seconds"] = time.perf_counter() - start_total
        timing_summary["inference_seconds"] = timing_summary["total_seconds"]
        if step_inference_seconds:
            timing_summary["average_inference_seconds"] = float(np.mean(step_inference_seconds))
            timing_summary["max_inference_seconds"] = float(np.max(step_inference_seconds))
            timing_summary["min_inference_seconds"] = float(np.min(step_inference_seconds))
        metadata.update(timing_summary)
        metadata_json = json.dumps(metadata, ensure_ascii=False)
        output_file.create_dataset("metadata", data=metadata_json)
        output_file.attrs["metadata"] = metadata_json

    return timing_summary


@torch.no_grad()
def rollout_single_layer_static(
    model,
    frame_reader: HDF5FrameReader,
    static_state: StaticGraphState,
    feature_builder: GpuFeatureBuilder,
    steps: int,
    scale_params: ScaleParams,
    *,
    writer=None,
    return_all: bool = False,
    return_dimensionless: bool = False,
    warmup_steps: int = 0,
    timing_recorder=None,
):
    if int(steps) <= 0:
        raise ValueError(f"steps must be positive, got {steps}.")
    if int(steps) > frame_reader.num_frames:
        raise ValueError(f"steps={steps} exceeds available frames {frame_reader.num_frames}.")
    if int(warmup_steps) < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")

    model.to(static_state.device)
    was_training = model.training
    model.eval()
    setup_start = time.perf_counter()
    current_temperature = feature_builder.initial_temperature()
    if int(warmup_steps) > 0:
        node_base_cpu, global_cpu = frame_reader.read_frame(0)
        warmup_graph = feature_builder.build(node_base_cpu, global_cpu, current_temperature).clone()
        current_temperature = pseudo_time_relax_initial_temperature(model, warmup_graph, int(warmup_steps))
    setup_seconds = time.perf_counter() - setup_start
    outputs = []
    try:
        for frame_idx in range(int(steps)):
            step_start = time.perf_counter()
            node_base_cpu, global_cpu = frame_reader.read_frame(frame_idx)
            graph = feature_builder.build(node_base_cpu, global_cpu, current_temperature)
            delta_t_source = graph_explicit_source_delta(graph, model.config)
            source_temperature = apply_dirichlet_boundary(
                current_temperature + delta_t_source,
                static_state.boundary_nodes,
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            graph = clone_graph_with_temperature(graph, source_temperature, delta_t_source_star=delta_t_source)
            next_temperature = apply_dirichlet_boundary(
                source_temperature + model(graph),
                static_state.boundary_nodes,
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            output = (
                next_temperature
                if return_dimensionless
                else temperature_from_dimensionless(next_temperature, scale_params)
            )
            if writer is not None:
                writer(frame_idx, output.detach().cpu())
            if return_all:
                outputs.append(output.detach().cpu())
            current_temperature = next_temperature
            if timing_recorder is not None:
                elapsed = time.perf_counter() - step_start
                if frame_idx == 0:
                    elapsed += setup_seconds
                timing_recorder(max(0.0, elapsed))
    finally:
        if was_training:
            model.train()

    if return_all:
        return torch.stack(outputs, dim=0)
    return None


def render_single_layer_surfaces_from_hdf5(
    prediction_h5,
    *,
    vtu_interval=None,
    vtu_output_dir=None,
):
    prediction_h5 = Path(prediction_h5)
    metadata = _read_prediction_metadata(prediction_h5)
    scale_params = ScaleParams(**metadata["scale_params"])
    source_h5 = Path(metadata["source_h5"])
    if not source_h5.is_absolute():
        source_h5 = (prediction_h5.parent / source_h5).resolve()

    vtu_interval = int(vtu_interval if vtu_interval is not None else metadata.get("vtu_interval", 20))
    if vtu_interval <= 0:
        raise ValueError(f"vtu_interval must be positive, got {vtu_interval}.")
    if vtu_output_dir is not None:
        vtu_output_dir = Path(vtu_output_dir).resolve()
    else:
        vtu_output_dir = Path(metadata.get("vtu_output_dir") or prediction_h5.with_name(f"{prediction_h5.stem}_vtu"))
        if not vtu_output_dir.is_absolute():
            vtu_output_dir = (prediction_h5.parent / vtu_output_dir).resolve()

    hdf5_timing = metadata.get("hdf5_timing", {})
    scan_velocity = hdf5_timing.get("velocity_speed", scale_params.v0)
    loader = HDF5Loader(source_h5, scale_params=scale_params)
    rendered_steps = []
    start_render = time.perf_counter()
    with h5py.File(prediction_h5, "r") as output_file:
        if "temperature" in output_file:
            steps = int(output_file["temperature"].shape[0])
            for step in range(steps):
                if not _should_write_cloud_step(step, vtu_interval):
                    continue
                _write_single_layer_step_vtu(
                    output_file,
                    loader,
                    scale_params,
                    scan_velocity=scan_velocity,
                    vtu_output_dir=vtu_output_dir,
                    step=int(step),
                    teacher_only=False,
                )
                rendered_steps.append(int(step))
        elif "teacher_forcing" in output_file:
            frame_indices = np.asarray(output_file["teacher_forcing/frame_index"], dtype=np.int64)
            for teacher_index, step in enumerate(frame_indices):
                if not _should_write_cloud_step(int(step), vtu_interval):
                    continue
                _write_single_layer_step_vtu(
                    output_file,
                    loader,
                    scale_params,
                    scan_velocity=scan_velocity,
                    vtu_output_dir=vtu_output_dir,
                    step=int(step),
                    teacher_only=True,
                    teacher_index=int(teacher_index),
                )
                rendered_steps.append(int(step))

    return {
        "render_seconds": time.perf_counter() - start_render,
        "rendered_steps": rendered_steps,
        "vtu_output_dir": str(vtu_output_dir),
    }


def _write_single_layer_step_vtu(
    output_file,
    loader,
    scale_params,
    *,
    scan_velocity,
    vtu_output_dir,
    step: int,
    teacher_only: bool,
    teacher_index=None,
):
    raw = loader.load_graph_data(int(step), device=torch.device("cpu"))
    graph = build_graph(
        raw,
        scale_params,
        scan_velocity=scan_velocity,
        initial_temperature=torch.full(
            (raw.xyz.shape[0], 1),
            float(scale_params.T_amb),
            device=raw.xyz.device,
            dtype=raw.xyz.dtype,
        ),
    )
    coords = graph.pos.detach().cpu().numpy() * float(scale_params.L0)
    edge_index = graph.edge_index.detach().cpu().numpy()

    if teacher_only:
        group = output_file["teacher_forcing"]
        index = int(teacher_index)
        temperature = np.asarray(group["temperature"][index], dtype=np.float64).reshape(-1)
        temperature_star = np.asarray(group["temperature_star"][index], dtype=np.float64).reshape(-1)
    else:
        temperature = np.asarray(output_file["temperature"][step], dtype=np.float64).reshape(-1)
        temperature_star = np.asarray(output_file["temperature_star"][step], dtype=np.float64).reshape(-1)

    point_values = {
        "temperature": temperature,
        "temperature_star": temperature_star,
        "time_step": np.full((graph.num_nodes,), float(step), dtype=np.float32),
    }
    if "fem_temperature" in output_file and step < output_file["fem_temperature"].shape[0]:
        fem_temperature = np.asarray(output_file["fem_temperature"][step], dtype=np.float64).reshape(-1)
        point_values["fem_temperature"] = fem_temperature
        if "temperature_error" in output_file:
            point_values["temperature_error"] = np.asarray(
                output_file["temperature_error"][step],
                dtype=np.float64,
            ).reshape(-1)
            point_values["abs_temperature_error"] = np.abs(point_values["temperature_error"])
        if "fem_valid_mask" in output_file:
            point_values["fem_valid_mask"] = np.asarray(output_file["fem_valid_mask"][step], dtype=np.float64).reshape(-1)
    if "teacher_forcing" in output_file and int(step) > 0:
        group = output_file["teacher_forcing"]
        frame_index = np.asarray(group["frame_index"], dtype=np.int64)
        matches = np.nonzero(frame_index == int(step))[0]
        if matches.size > 0:
            index = int(matches[0])
            point_values["teacher_temperature"] = np.asarray(group["temperature"][index], dtype=np.float64).reshape(-1)
            point_values["teacher_temperature_star"] = np.asarray(
                group["temperature_star"][index],
                dtype=np.float64,
            ).reshape(-1)
            point_values["teacher_temperature_error"] = np.asarray(
                group["temperature_error"][index],
                dtype=np.float64,
            ).reshape(-1)

    write_surface_vtu(
        Path(vtu_output_dir) / f"temperature_step_{step:06d}.vtu",
        coords,
        point_data=point_values,
        edge_index=edge_index,
        title=f"PDGCN step {step} single layer",
    )


def _write_autoregressive_fem_datasets(output_file, frame_reader, scale_params, steps: int):
    fem_temperature = output_file.create_dataset(
        "fem_temperature",
        shape=(int(steps), int(frame_reader.num_nodes), 1),
        dtype="float32",
    )
    fem_valid_mask = output_file.create_dataset(
        "fem_valid_mask",
        shape=(int(steps), int(frame_reader.num_nodes), 1),
        dtype="float32",
    )
    temperature_error = output_file.create_dataset(
        "temperature_error",
        shape=(int(steps), int(frame_reader.num_nodes), 1),
        dtype="float32",
    )
    for frame_idx in range(int(steps)):
        fem = frame_reader.read_fem_temperature(frame_idx)
        fem_temperature[frame_idx] = fem.numpy()
        fem_valid_mask[frame_idx] = frame_reader.read_fem_valid_mask(frame_idx).numpy()
        predicted = torch.as_tensor(output_file["temperature"][frame_idx], dtype=torch.float32)
        temperature_error[frame_idx] = (predicted - fem).numpy()


@torch.no_grad()
def _write_teacher_forcing_group(
    group,
    model,
    frame_reader,
    static_state,
    feature_builder,
    steps: int,
    scale_params,
    *,
    timing_recorder=None,
):
    transition_count = int(steps) - 1
    temperature_star_dataset = group.create_dataset(
        "temperature_star",
        shape=(transition_count, int(static_state.num_nodes), 1),
        dtype="float32",
    )
    temperature_dataset = group.create_dataset(
        "temperature",
        shape=(transition_count, int(static_state.num_nodes), 1),
        dtype="float32",
    )
    fem_temperature_dataset = group.create_dataset(
        "fem_temperature",
        shape=(transition_count, int(static_state.num_nodes), 1),
        dtype="float32",
    )
    fem_valid_mask_dataset = group.create_dataset(
        "fem_valid_mask",
        shape=(transition_count, int(static_state.num_nodes), 1),
        dtype="float32",
    )
    temperature_error_dataset = group.create_dataset(
        "temperature_error",
        shape=(transition_count, int(static_state.num_nodes), 1),
        dtype="float32",
    )
    group.create_dataset("frame_index", data=np.arange(1, int(steps), dtype=np.int64))

    model.to(static_state.device)
    was_training = model.training
    model.eval()
    try:
        for frame_idx in range(transition_count):
            step_start = time.perf_counter()
            fem_current = _read_fem_temperature_star(frame_reader, frame_idx, feature_builder, scale_params)
            fem_next = _read_fem_temperature_star(frame_reader, frame_idx + 1, feature_builder, scale_params)
            node_base_cpu, global_cpu = frame_reader.read_frame(frame_idx)
            graph = feature_builder.build(node_base_cpu, global_cpu, fem_current)
            delta_t_source = graph_explicit_source_delta(graph, model.config)
            source_temperature = apply_dirichlet_boundary(
                fem_current + delta_t_source,
                static_state.boundary_nodes,
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            graph = clone_graph_with_temperature(graph, source_temperature, delta_t_source_star=delta_t_source)
            next_temperature = apply_dirichlet_boundary(
                source_temperature + model(graph),
                static_state.boundary_nodes,
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            temperature_star_dataset[frame_idx] = next_temperature.detach().cpu().numpy()
            temperature = temperature_from_dimensionless(next_temperature.detach(), scale_params).cpu()
            fem_temperature = temperature_from_dimensionless(fem_next.detach(), scale_params).cpu()
            temperature_dataset[frame_idx] = temperature.numpy()
            fem_temperature_dataset[frame_idx] = fem_temperature.numpy()
            fem_valid_mask_dataset[frame_idx] = frame_reader.read_fem_valid_mask(frame_idx + 1).numpy()
            temperature_error_dataset[frame_idx] = (temperature - fem_temperature).numpy()
            if timing_recorder is not None:
                timing_recorder(max(0.0, time.perf_counter() - step_start))
    finally:
        if was_training:
            model.train()


def _read_fem_temperature_star(frame_reader, frame_idx: int, feature_builder, scale_params):
    temperature = frame_reader.read_fem_temperature(frame_idx)
    return temperature_to_dimensionless(temperature, scale_params).to(
        device=feature_builder.device,
        dtype=feature_builder.dtype,
        non_blocking=feature_builder.device.type == "cuda",
    )


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
