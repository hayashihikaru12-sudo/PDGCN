import json
import time
from dataclasses import MISSING, asdict, fields
from pathlib import Path

import h5py
import numpy as np
import torch

from data import HDF5Loader, build_graph
from data.dimensionless import ScaleParams, temperature_from_dimensionless
from models import PDGCN, PDGCNConfig
from training.run_config import load_run_config, pdgcn_config_from_scale
from training.train_entry import derive_timing_from_hdf5, discover_hdf5_files
from visualization import write_topology_wedge_vtk

from .config import InferenceRunConfig
from .fdm import compute_layer_fdm_coefficient
from .multilayer import _build_multilayer_geometry, _resolve_layer_batch_size, rollout_multilayer_fdm


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
    cloud_interval = int(inference_config.cloud_interval)
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
        "layer_fiber_angles_deg": list(
            inference_config.layer_fiber_angles_deg
            if inference_config.layer_fiber_angles_deg is not None
            else [0.0] * int(inference_config.num_layers)
        ),
        "normal_offset_sign": int(inference_config.normal_offset_sign),
        "write_vtk": bool(inference_config.write_vtk),
        "cloud_interval": int(cloud_interval),
        "cloud_max_nodes_per_layer": (
            None
            if inference_config.cloud_max_nodes_per_layer is None
            else int(inference_config.cloud_max_nodes_per_layer)
        ),
        "layer_batch_size": None if inference_config.layer_batch_size is None else int(inference_config.layer_batch_size),
        "vtk_output_dir": str(selected_vtk_dir),
        "hdf5_timing": timing,
        "scale_params": asdict(scale_params),
        "model_config": asdict(model.config),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
    }

    timing_summary = write_multilayer_hdf5(
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
        layer_fiber_angles_deg=inference_config.layer_fiber_angles_deg,
        normal_offset_sign=int(inference_config.normal_offset_sign),
        layer_batch_size=inference_config.layer_batch_size,
    )
    return {
        "output_path": str(selected_output),
        "checkpoint_path": str(selected_checkpoint),
        "h5_path": str(selected_h5),
        "steps": steps,
        "num_layers": int(inference_config.num_layers),
        "fdm_coefficient": float(fdm_coefficient),
        "vtk_output_dir": str(selected_vtk_dir),
        "cloud_interval": int(cloud_interval),
        **timing_summary,
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
    layer_fiber_angles_deg=None,
    normal_offset_sign: int = -1,
    layer_batch_size=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    metadata["effective_layer_batch_size"] = int(
        _resolve_layer_batch_size(layer_batch_size, int(num_layers), next(model.parameters()).device)
    )
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
        start_total = time.perf_counter()

        def writer(step, temperature_star, graph_step=None):
            temperature_star_dataset[int(step)] = temperature_star.numpy()
            temperature = temperature_from_dimensionless(temperature_star, scale_params)
            temperature_dataset[int(step)] = temperature.numpy()
            return 0.0

        rollout_multilayer_fdm(
            model,
            graph_factory,
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
            layer_fiber_angles_deg=layer_fiber_angles_deg,
            normal_offset_sign=int(normal_offset_sign),
            layer_batch_size=layer_batch_size,
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


def _should_write_cloud_step(step: int, cloud_interval: int) -> bool:
    return int(step) % int(cloud_interval) == 0


def render_multilayer_clouds_from_hdf5(
    prediction_h5,
    *,
    cloud_interval=None,
    vtk_output_dir=None,
    max_nodes_per_layer=None,
):
    prediction_h5 = Path(prediction_h5)
    metadata = _read_prediction_metadata(prediction_h5)
    scale_params = ScaleParams(**metadata["scale_params"])
    source_h5 = Path(metadata["source_h5"])
    if not source_h5.is_absolute():
        source_h5 = (prediction_h5.parent / source_h5).resolve()

    cloud_interval = int(cloud_interval if cloud_interval is not None else metadata.get("cloud_interval", 20))
    if cloud_interval <= 0:
        raise ValueError(f"cloud_interval must be positive, got {cloud_interval}.")
    if max_nodes_per_layer is not None:
        raise ValueError(
            "max_nodes_per_layer is not supported for topology wedge rendering because node sampling breaks "
            "the Gmsh triangle connectivity. Omit --max-nodes-per-layer to render all nodes."
        )

    if vtk_output_dir is not None:
        vtk_output_dir = Path(vtk_output_dir).resolve()
    else:
        vtk_output_dir = Path(metadata.get("vtk_output_dir") or prediction_h5.with_name(f"{prediction_h5.stem}_vtk"))
        if not vtk_output_dir.is_absolute():
            vtk_output_dir = (prediction_h5.parent / vtk_output_dir).resolve()

    hdf5_timing = metadata.get("hdf5_timing", {})
    scan_velocity = hdf5_timing.get("velocity_speed", scale_params.v0)
    loader = HDF5Loader(source_h5, scale_params=scale_params)
    num_layers = int(metadata["num_layers"])
    num_nodes = _read_prediction_num_nodes(prediction_h5)
    layer_spacing_star = float(metadata.get("layer_spacing_star", float(metadata["layer_spacing"]) / scale_params.L0))
    layer_fiber_angles_deg = metadata.get("layer_fiber_angles_deg") or [0.0] * num_layers
    normal_offset_sign = int(metadata.get("normal_offset_sign", -1))

    rendered_steps = []
    start_render = time.perf_counter()
    with h5py.File(prediction_h5, "r") as output_file:
        temperature_dataset = output_file["temperature"]
        temperature_star_dataset = output_file["temperature_star"]
        steps = int(temperature_star_dataset.shape[0])
        for step in range(steps):
            if not _should_write_cloud_step(step, cloud_interval):
                continue
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
            if int(graph.num_nodes) != int(num_nodes):
                raise ValueError(
                    f"source graph node count {graph.num_nodes} does not match prediction node count {num_nodes}."
                )
            geometry = _build_multilayer_geometry(
                graph,
                num_layers,
                layer_spacing_star=layer_spacing_star,
                layer_fiber_angles_deg=layer_fiber_angles_deg,
                normal_offset_sign=normal_offset_sign,
            )
            _write_multilayer_step_vtk(
                vtk_output_dir,
                step=int(step),
                coords_star_layers=geometry["pos"].detach().cpu().numpy(),
                edge_index=graph.edge_index.detach().cpu().numpy(),
                temperature=temperature_dataset[step],
                temperature_star=temperature_star_dataset[step],
                scale_params=scale_params,
                max_nodes_per_layer=max_nodes_per_layer,
            )
            rendered_steps.append(int(step))

    return {
        "render_seconds": time.perf_counter() - start_render,
        "rendered_steps": rendered_steps,
        "vtk_output_dir": str(vtk_output_dir),
    }


def _write_multilayer_step_vtk(
    vtk_output_dir,
    *,
    step: int,
    coords_star_layers,
    edge_index,
    temperature,
    temperature_star,
    scale_params,
    max_nodes_per_layer=None,
):
    vtk_output_dir = Path(vtk_output_dir)
    coords_star_layers = np.asarray(coords_star_layers, dtype=np.float64)
    if coords_star_layers.ndim != 3 or coords_star_layers.shape[2] != 3:
        raise ValueError(f"coords_star_layers must have shape [layer, node, 3], got {coords_star_layers.shape}.")
    layer_count = int(coords_star_layers.shape[0])
    nodes_per_layer = int(coords_star_layers.shape[1])
    if max_nodes_per_layer is not None:
        raise ValueError(
            "max_nodes_per_layer is not supported for topology wedge rendering because node sampling breaks "
            "the Gmsh triangle connectivity."
        )
    sample_indices = np.arange(nodes_per_layer, dtype=np.int64)

    coords = (coords_star_layers[:, sample_indices, :].reshape(layer_count * len(sample_indices), 3)) * float(
        scale_params.L0
    )
    temperature = np.asarray(temperature, dtype=np.float64).reshape(layer_count, nodes_per_layer, 1)
    temperature_star = np.asarray(temperature_star, dtype=np.float64).reshape(layer_count, nodes_per_layer, 1)
    temperature = temperature[:, sample_indices, :].reshape(-1)
    temperature_star = temperature_star[:, sample_indices, :].reshape(-1)
    num_points = int(coords.shape[0])
    layer_size = int(len(sample_indices))
    layer_index = np.repeat(np.arange(layer_count, dtype=np.float32), layer_size)
    point_values = {
        "temperature": temperature,
        "temperature_star": temperature_star,
        "layer_index": layer_index,
        "time_step": np.full((num_points,), float(step), dtype=np.float32),
    }
    render_edge_index = edge_index if len(sample_indices) == int(nodes_per_layer) else _remap_edge_index(
        edge_index,
        sample_indices,
        nodes_per_layer,
    )
    write_topology_wedge_vtk(
        vtk_output_dir / f"temperature_step_{step:06d}.vtk",
        coords,
        point_data=point_values,
        layer_count=layer_count,
        nodes_per_layer=layer_size,
        edge_index=render_edge_index,
        title=f"PDGCN step {step} multilayer",
    )


def _sample_node_indices(nodes_per_layer: int, max_nodes_per_layer, *, coords=None):
    nodes_per_layer = int(nodes_per_layer)
    if max_nodes_per_layer is None or int(max_nodes_per_layer) >= nodes_per_layer:
        return np.arange(nodes_per_layer, dtype=np.int64)
    max_nodes_per_layer = int(max_nodes_per_layer)
    if max_nodes_per_layer < 3:
        raise ValueError(f"max_nodes_per_layer must be at least 3 when set, got {max_nodes_per_layer}.")
    if coords is None:
        return np.unique(np.linspace(0, nodes_per_layer - 1, max_nodes_per_layer, dtype=np.int64))
    return _spatially_sample_node_indices(coords, max_nodes_per_layer)


def _spatially_sample_node_indices(coords, max_nodes_per_layer: int):
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape [N, 3], got {coords.shape}.")
    count = int(coords.shape[0])
    max_nodes_per_layer = int(max_nodes_per_layer)
    if max_nodes_per_layer >= count:
        return np.arange(count, dtype=np.int64)

    ranges = np.ptp(coords, axis=0)
    axes = np.argsort(ranges)[-2:]
    projected = coords[:, np.sort(axes)]
    projected_min = np.min(projected, axis=0)
    projected_range = np.ptp(projected, axis=0)
    normalized = (projected - projected_min) / np.maximum(projected_range, 1e-12)
    grid = np.clip((normalized * 65535.0).astype(np.uint64), 0, 65535)
    morton = _part1by1(grid[:, 0]) | (_part1by1(grid[:, 1]) << np.uint64(1))
    order = np.argsort(morton, kind="mergesort")
    selected_positions = np.linspace(0, count - 1, max_nodes_per_layer, dtype=np.int64)
    selected = np.unique(order[selected_positions])
    if selected.size < max_nodes_per_layer:
        missing = max_nodes_per_layer - selected.size
        selected_set = set(int(index) for index in selected)
        supplement = [int(index) for index in order if int(index) not in selected_set][:missing]
        selected = np.concatenate([selected, np.asarray(supplement, dtype=np.int64)])
    return np.sort(selected.astype(np.int64))


def _part1by1(values):
    values = np.asarray(values, dtype=np.uint64) & np.uint64(0x0000FFFF)
    values = (values | (values << np.uint64(8))) & np.uint64(0x00FF00FF)
    values = (values | (values << np.uint64(4))) & np.uint64(0x0F0F0F0F)
    values = (values | (values << np.uint64(2))) & np.uint64(0x33333333)
    values = (values | (values << np.uint64(1))) & np.uint64(0x55555555)
    return values


def _remap_edge_index(edge_index, sample_indices, nodes_per_layer: int):
    if edge_index is None:
        return None
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.size == 0:
        return edges.reshape(2, 0)
    if edges.shape[0] != 2:
        edges = edges.T
    mapping = np.full((int(nodes_per_layer),), -1, dtype=np.int64)
    mapping[np.asarray(sample_indices, dtype=np.int64)] = np.arange(len(sample_indices), dtype=np.int64)
    keep = (mapping[edges[0]] >= 0) & (mapping[edges[1]] >= 0)
    return np.stack([mapping[edges[0, keep]], mapping[edges[1, keep]]], axis=0)


def _read_prediction_num_nodes(prediction_h5):
    with h5py.File(prediction_h5, "r") as output_file:
        return int(output_file["temperature_star"].shape[2])


def _read_prediction_metadata(prediction_h5):
    with h5py.File(prediction_h5, "r") as output_file:
        if "metadata" in output_file.attrs:
            metadata_json = output_file.attrs["metadata"]
        else:
            metadata_json = output_file["metadata"][()]
    if isinstance(metadata_json, bytes):
        metadata_json = metadata_json.decode("utf-8")
    return json.loads(str(metadata_json))


def _update_prediction_metadata(prediction_h5, values):
    prediction_h5 = Path(prediction_h5)
    with h5py.File(prediction_h5, "r+") as output_file:
        metadata = _read_prediction_metadata_from_open_file(output_file)
        metadata.update(values)
        metadata_json = json.dumps(metadata, ensure_ascii=False)
        if "metadata" in output_file:
            del output_file["metadata"]
        output_file.create_dataset("metadata", data=metadata_json)
        output_file.attrs["metadata"] = metadata_json


def _read_prediction_metadata_from_open_file(output_file):
    if "metadata" in output_file.attrs:
        metadata_json = output_file.attrs["metadata"]
    else:
        metadata_json = output_file["metadata"][()]
    if isinstance(metadata_json, bytes):
        metadata_json = metadata_json.decode("utf-8")
    return json.loads(str(metadata_json))


def _resolve_path(base_dir: Path, value) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
