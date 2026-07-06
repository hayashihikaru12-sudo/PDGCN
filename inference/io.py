import json
import shutil
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

from pde.fdm import compute_fin_cooling_gamma

from .config import InferenceRunConfig
from .fdm import compute_layer_fdm_coefficient
from .multilayer import _build_multilayer_geometry, _resolve_layer_batch_size, rollout_multilayer_fdm


DEFAULT_MULTILAYER_PREDICTION_GROUP_PATH = "prediction/pdgcn_multilayer"


def _as_non_negative_integer(value, name: str):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer, got {value}.")
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {value}.") from exc
    if int_value < 0 or int_value != value:
        raise ValueError(f"{name} must be a non-negative integer, got {value}.")
    return int_value


def run_multilayer_inference_from_config(
    config_path,
    *,
    checkpoint=None,
    h5_path=None,
    batch=None,
    h5_dir=None,
    output_path=None,
    output_dir=None,
    output_prefix=None,
    vtk_output_dir=None,
    cloud_interval=None,
):
    """Run multilayer PD-GCN + 1D FDM inference from an inference JSON config."""

    config_path = Path(config_path)
    run_config, inference_config, training_base_dir, inference_base_dir, training_config_path = (
        load_inference_run_context(config_path)
    )
    overrides = asdict(inference_config)
    if output_prefix is not None:
        overrides["output_prefix"] = str(output_prefix)
    if cloud_interval is not None:
        overrides["cloud_interval"] = int(cloud_interval)
    inference_config = InferenceRunConfig(**overrides)

    if int(inference_config.dataset_index) >= len(run_config.datasets):
        raise IndexError(
            f"inference.dataset_index={inference_config.dataset_index} exceeds "
            f"datasets length {len(run_config.datasets)}."
        )

    dataset = run_config.datasets[int(inference_config.dataset_index)]
    scale_params = dataset.scale.to_scale_params()
    selected_checkpoint = (
        _resolve_path(inference_base_dir, checkpoint)
        if checkpoint
        else _resolve_path(
            training_base_dir,
            run_config.outputs.checkpoint_path if run_config.outputs is not None else run_config.data.checkpoint_path,
        )
    )
    device = torch.device(run_config.training.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch_mode = bool(inference_config.batch_mode) if batch is None else bool(batch)

    if batch_mode:
        if h5_path or inference_config.h5_path:
            raise ValueError("multilayer batch mode uses h5_dir; do not set h5_path or --h5.")
        if output_path is not None:
            raise ValueError("multilayer batch mode uses output_dir; do not set output_path or --output.")
        if vtk_output_dir is not None:
            raise ValueError("multilayer batch mode uses per-file VTK directories; do not set --vtk-output-dir.")
        selected_h5_dir = (
            _resolve_path(inference_base_dir, h5_dir or inference_config.h5_dir)
            if h5_dir or inference_config.h5_dir
            else _resolve_path(training_base_dir, dataset.h5_dir)
        )
        selected_h5_paths = discover_hdf5_files(selected_h5_dir)
        selected_output_dir_value = output_dir or inference_config.output_dir
        if not selected_output_dir_value:
            raise ValueError("multilayer batch mode requires output_dir or --output-dir.")
        selected_output_dir = _resolve_path(inference_base_dir, selected_output_dir_value)
        selected_output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for selected_h5 in selected_h5_paths:
            selected_output = selected_output_dir / f"{inference_config.output_prefix}{selected_h5.name}"
            selected_vtk_dir = selected_output_dir / f"{inference_config.output_prefix}{selected_h5.stem}_vtk"
            try:
                item = _run_multilayer_inference_for_h5(
                    config_path=config_path,
                    training_config_path=training_config_path,
                    selected_h5=selected_h5,
                    selected_output=selected_output,
                    selected_vtk_dir=selected_vtk_dir,
                    selected_checkpoint=selected_checkpoint,
                    run_config=run_config,
                    inference_config=inference_config,
                    scale_params=scale_params,
                    scan_velocity=dataset.scan_velocity,
                    device=device,
                )
                item["status"] = "succeeded"
                results.append(item)
            except Exception as error:  # noqa: BLE001 - batch mode must summarize per-file failures.
                results.append(
                    {
                        "status": "failed",
                        "h5_path": str(selected_h5),
                        "output_path": str(selected_output),
                        "error": str(error),
                    }
                )
        succeeded = [item for item in results if item["status"] == "succeeded"]
        failed = [item for item in results if item["status"] == "failed"]
        return {
            "batch_mode": True,
            "checkpoint_path": str(selected_checkpoint),
            "h5_dir": str(selected_h5_dir),
            "output_dir": str(selected_output_dir),
            "output_prefix": str(inference_config.output_prefix),
            "prediction_group_path": str(inference_config.prediction_group_path).strip("/"),
            "processed_count": len(results),
            "succeeded_count": len(succeeded),
            "failed_count": len(failed),
            "results": results,
        }

    selected_h5 = (
        _resolve_path(inference_base_dir, h5_path or inference_config.h5_path)
        if h5_path or inference_config.h5_path
        else discover_hdf5_files(_resolve_path(training_base_dir, dataset.h5_dir))[0]
    )
    selected_output = _resolve_path(inference_base_dir, output_path or inference_config.output_path)
    selected_vtk_dir = (
        _resolve_path(inference_base_dir, vtk_output_dir or inference_config.vtk_output_dir)
        if vtk_output_dir or inference_config.vtk_output_dir
        else selected_output.with_name(f"{selected_output.stem}_vtk")
    )
    return _run_multilayer_inference_for_h5(
        config_path=config_path,
        training_config_path=training_config_path,
        selected_h5=selected_h5,
        selected_output=selected_output,
        selected_vtk_dir=selected_vtk_dir,
        selected_checkpoint=selected_checkpoint,
        run_config=run_config,
        inference_config=inference_config,
        scale_params=scale_params,
        scan_velocity=dataset.scan_velocity,
        device=device,
    )


def _run_multilayer_inference_for_h5(
    *,
    config_path,
    training_config_path,
    selected_h5,
    selected_output,
    selected_vtk_dir,
    selected_checkpoint,
    run_config,
    inference_config,
    scale_params,
    scan_velocity,
    device,
):
    timing = derive_timing_from_hdf5(selected_h5, scale_params, scan_velocity=scan_velocity)
    fallback_model_config = pdgcn_config_from_scale(
        scale_params,
        dt=timing["dt"],
        model_overrides=run_config.model,
    )
    model, checkpoint_payload = load_model_from_checkpoint(selected_checkpoint, fallback_model_config, device)

    num_frames, num_nodes = read_hdf5_temperature_shape(selected_h5)
    steps = int(inference_config.steps) if inference_config.steps is not None else int(num_frames)
    if steps > num_frames:
        raise ValueError(f"inference.steps={steps} exceeds available frames {num_frames}.")

    layer_spacing_star = float(inference_config.layer_spacing) / float(scale_params.L0)
    cloud_interval = int(inference_config.cloud_interval)
    model_k_ratio = float(getattr(model.config, "k_ratio", 0.0))
    fdm_k_ratio_scale = float(inference_config.fdm_k_ratio_scale)
    fdm_effective_k_ratio = model_k_ratio * fdm_k_ratio_scale
    fdm_coefficient = compute_layer_fdm_coefficient(
        dt_star=getattr(model.config, "dt_star", 1.0),
        inverse_pe=getattr(model.config, "inverse_pe", 1.0),
        k_ratio=fdm_effective_k_ratio,
        layer_spacing_star=layer_spacing_star,
    )
    fin_cooling = _resolve_fin_cooling_parameters(
        fdm_coefficient=fdm_coefficient,
        dt_star=getattr(model.config, "dt_star", 1.0),
        inverse_pe=getattr(model.config, "inverse_pe", 1.0),
        layer_spacing_star=layer_spacing_star,
        k_ratio=fdm_effective_k_ratio,
        num_layers=int(inference_config.num_layers),
        enabled=bool(inference_config.fin_cooling_enabled),
        mode=inference_config.fin_cooling_mode,
        r_char_star=inference_config.fin_cooling_r_char_star,
        direct_gamma_star=inference_config.fin_cooling_gamma_star,
        beta_h=float(inference_config.fin_cooling_beta_h),
        skip_top_layers=int(inference_config.fin_cooling_skip_top_layers),
        layer_profile=inference_config.fin_cooling_layer_profile,
        layer_profile_strength=float(inference_config.fin_cooling_layer_profile_strength),
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
            model_config=model.config,
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
        "fdm_k_ratio_scale": float(fdm_k_ratio_scale),
        "fdm_effective_k_ratio": float(fdm_effective_k_ratio),
        "fdm_layer_interface_scales": _metadata_optional_sequence(inference_config.fdm_layer_interface_scales),
        "fdm_top_surface_loss_gamma_dt": float(inference_config.fdm_top_surface_loss_gamma_dt),
        "fdm_top_surface_loss_velocity_exponent": float(
            inference_config.fdm_top_surface_loss_velocity_exponent
        ),
        "fdm_top_surface_loss_reference_velocity_star": float(
            inference_config.fdm_top_surface_loss_reference_velocity_star
        ),
        "model_k_ratio": float(model_k_ratio),
        "thickness_solver": "implicit_euler",
        "thickness_model": "transient_fin" if fin_cooling["enabled"] else "plain_fdm",
        "fin_cooling_enabled": bool(fin_cooling["enabled"]),
        "fin_cooling_mode": fin_cooling["mode"],
        "fin_cooling_r_char_star": fin_cooling["r_char_star"],
        "fin_cooling_beta_h": fin_cooling["beta_h"],
        "fin_cooling_equivalent_beta_h": fin_cooling["equivalent_beta_h"],
        "fin_cooling_skip_top_layers": int(fin_cooling["skip_top_layers"]),
        "fin_cooling_layer_profile": fin_cooling["layer_profile"],
        "fin_cooling_layer_profile_strength": fin_cooling["layer_profile_strength"],
        "fin_cooling_gamma_star": fin_cooling["gamma_star"],
        "fin_cooling_gamma_dt": fin_cooling["gamma_dt"],
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
        "delta_smoothing_alpha": float(inference_config.delta_smoothing_alpha),
        "delta_smoothing_steps": int(inference_config.delta_smoothing_steps),
        "use_pdgcn_inplane": bool(inference_config.use_pdgcn_inplane),
        "pdgcn_inplane_top_layer_only": bool(inference_config.pdgcn_inplane_top_layer_only),
        "use_alternating_order_average": bool(inference_config.use_alternating_order_average),
        "vtk_output_dir": str(selected_vtk_dir),
        "hdf5_timing": timing,
        "scale_params": asdict(scale_params),
        "model_config": asdict(model.config),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "prediction_group_path": str(inference_config.prediction_group_path).strip("/"),
    }

    timing_summary = write_multilayer_hdf5(
        selected_output,
        source_h5=selected_h5,
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
        allow_unstable_fdm=bool(inference_config.allow_unstable_fdm),
        layer_fiber_angles_deg=inference_config.layer_fiber_angles_deg,
        normal_offset_sign=int(inference_config.normal_offset_sign),
        layer_batch_size=inference_config.layer_batch_size,
        delta_smoothing_alpha=float(inference_config.delta_smoothing_alpha),
        delta_smoothing_steps=int(inference_config.delta_smoothing_steps),
        use_pdgcn_inplane=bool(inference_config.use_pdgcn_inplane),
        pdgcn_inplane_top_layer_only=bool(inference_config.pdgcn_inplane_top_layer_only),
        use_alternating_order_average=bool(inference_config.use_alternating_order_average),
        fdm_k_ratio_scale=float(fdm_k_ratio_scale),
        fdm_layer_interface_scales=inference_config.fdm_layer_interface_scales,
        fdm_top_surface_loss_gamma_dt=float(inference_config.fdm_top_surface_loss_gamma_dt),
        fdm_top_surface_loss_velocity_exponent=float(
            inference_config.fdm_top_surface_loss_velocity_exponent
        ),
        fdm_top_surface_loss_reference_velocity_star=float(
            inference_config.fdm_top_surface_loss_reference_velocity_star
        ),
        fin_cooling_gamma_star=fin_cooling["gamma_star"],
        fin_cooling_skip_top_layers=int(fin_cooling["skip_top_layers"]),
        prediction_group_path=str(inference_config.prediction_group_path).strip("/"),
    )
    render_summary = {
        "render_seconds": 0.0,
        "rendered_steps": [],
        "vtk_output_dir": str(selected_vtk_dir),
    }
    if bool(inference_config.write_vtk):
        render_summary = render_multilayer_clouds_from_hdf5(
            selected_output,
            cloud_interval=cloud_interval,
            vtk_output_dir=selected_vtk_dir,
            max_nodes_per_layer=None,
            prediction_group_path=str(inference_config.prediction_group_path).strip("/"),
        )
    total_seconds = float(timing_summary["inference_seconds"]) + float(render_summary["render_seconds"])
    final_timing = {**timing_summary, **render_summary, "total_seconds": total_seconds}
    _update_multilayer_prediction_after_render(
        selected_output,
        final_timing,
        prediction_group_path=str(inference_config.prediction_group_path).strip("/"),
    )
    return {
        "output_path": str(selected_output),
        "checkpoint_path": str(selected_checkpoint),
        "h5_path": str(selected_h5),
        "steps": steps,
        "num_layers": int(inference_config.num_layers),
        "fdm_coefficient": float(fdm_coefficient),
        "fdm_k_ratio_scale": float(fdm_k_ratio_scale),
        "fdm_effective_k_ratio": float(fdm_effective_k_ratio),
        "fdm_layer_interface_scales": _metadata_optional_sequence(inference_config.fdm_layer_interface_scales),
        "fdm_top_surface_loss_gamma_dt": float(inference_config.fdm_top_surface_loss_gamma_dt),
        "fdm_top_surface_loss_velocity_exponent": float(
            inference_config.fdm_top_surface_loss_velocity_exponent
        ),
        "fdm_top_surface_loss_reference_velocity_star": float(
            inference_config.fdm_top_surface_loss_reference_velocity_star
        ),
        "thickness_solver": "implicit_euler",
        "thickness_model": "transient_fin" if fin_cooling["enabled"] else "plain_fdm",
        "fin_cooling_mode": fin_cooling["mode"],
        "fin_cooling_r_char_star": fin_cooling["r_char_star"],
        "fin_cooling_beta_h": fin_cooling["beta_h"],
        "fin_cooling_equivalent_beta_h": fin_cooling["equivalent_beta_h"],
        "fin_cooling_skip_top_layers": int(fin_cooling["skip_top_layers"]),
        "fin_cooling_layer_profile": fin_cooling["layer_profile"],
        "fin_cooling_layer_profile_strength": fin_cooling["layer_profile_strength"],
        "fin_cooling_gamma_star": fin_cooling["gamma_star"],
        "vtk_output_dir": str(selected_vtk_dir),
        "cloud_interval": int(cloud_interval),
        "prediction_group_path": str(inference_config.prediction_group_path).strip("/"),
        **final_timing,
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


def _resolve_fin_cooling_parameters(
    *,
    fdm_coefficient: float,
    dt_star: float,
    num_layers: int,
    enabled: bool,
    beta_h: float,
    inverse_pe: float = 1.0,
    layer_spacing_star: float = None,
    k_ratio: float = None,
    skip_top_layers: int = 0,
    mode: str = "beta_h",
    r_char_star: float = None,
    direct_gamma_star=None,
    layer_profile: str = "uniform",
    layer_profile_strength: float = 0.0,
):
    """Resolve transient-fin cooling parameters from the desired curvature βH.

    The dimensionless cooling rate per time step is computed as::

        γ*·Δt* = (βH / M)² · C_n

    where *M = num_layers − 1* is the active layer count and *C_n* is the
    pre-computed 1D FDM coefficient.  This scaling ensures the fin term
    remains a *fixed fraction* of the through-thickness diffusion regardless
    of how small ``inverse_pe`` happens to be for the current material.

    When the transient-fin model is retained for multilayer inference, the
    calibrated path uses ``skip_top_layers=0`` so every active layer receives
    the equivalent cooling diagonal.
    """
    if float(beta_h) <= 0:
        raise ValueError(f"fin_cooling_beta_h must be positive, got {beta_h}.")
    if float(dt_star) <= 0:
        raise ValueError(f"dt_star must be positive, got {dt_star}.")
    skip_top_layers = _as_non_negative_integer(skip_top_layers, "fin_cooling_skip_top_layers")
    if bool(enabled) and skip_top_layers != 0:
        raise ValueError("fin_cooling_skip_top_layers must be 0 when fin_cooling_enabled=true.")
    mode = str(mode).strip().lower()
    if mode not in {"r_char", "beta_h", "direct"}:
        raise ValueError(f"fin_cooling_mode must be one of 'r_char', 'beta_h', or 'direct', got {mode!r}.")
    layer_profile = str(layer_profile).strip().lower()
    if layer_profile not in {"uniform", "linear", "quadratic"}:
        raise ValueError(
            "fin_cooling_layer_profile must be one of 'uniform', 'linear', or 'quadratic', "
            f"got {layer_profile!r}."
        )
    if float(layer_profile_strength) < 0:
        raise ValueError(
            "fin_cooling_layer_profile_strength must be non-negative, "
            f"got {layer_profile_strength}."
        )
    if not bool(enabled):
        return {
            "enabled": False,
            "mode": mode,
            "r_char_star": None if r_char_star is None else float(r_char_star),
            "beta_h": float(beta_h),
            "equivalent_beta_h": None,
            "skip_top_layers": int(skip_top_layers),
            "layer_profile": layer_profile,
            "layer_profile_strength": float(layer_profile_strength),
            "gamma_star": None,
            "gamma_dt": None,
        }

    active_layers = int(num_layers) - 1
    if active_layers < 1:
        raise ValueError(f"num_layers must be >= 2 for fin cooling, got {num_layers}.")

    if mode == "direct":
        if direct_gamma_star is None:
            raise ValueError("fin_cooling_gamma_star must be set when fin_cooling_mode='direct'.")
        gamma_star = _normalize_gamma_value(direct_gamma_star, active_layers=active_layers)
        base_gamma_dt = _multiply_gamma_value(gamma_star, float(dt_star))
        equivalent_beta_h = _equivalent_beta_h(base_gamma_dt, fdm_coefficient, active_layers)
    elif mode == "r_char":
        r_char_star = _resolve_r_char_star(
            r_char_star,
            beta_h=beta_h,
            active_layers=active_layers,
            layer_spacing_star=layer_spacing_star,
            k_ratio=k_ratio,
        )
        gamma_star = compute_fin_cooling_gamma(
            inverse_pe=float(inverse_pe),
            r_char_star=float(r_char_star),
        )
        base_gamma_dt = float(gamma_star) * float(dt_star)
        equivalent_beta_h = _equivalent_beta_h(base_gamma_dt, fdm_coefficient, active_layers)
    else:
        base_gamma_dt = (float(beta_h) / active_layers) ** 2 * float(fdm_coefficient)
        gamma_star = base_gamma_dt / float(dt_star)
        equivalent_beta_h = float(beta_h)

    gamma_star = _apply_fin_cooling_layer_profile(
        gamma_star,
        active_layers=active_layers,
        skip_top_layers=int(skip_top_layers),
        layer_profile=layer_profile,
        layer_profile_strength=float(layer_profile_strength),
    )
    gamma_dt = _multiply_gamma_value(gamma_star, float(dt_star))

    return {
        "enabled": True,
        "mode": mode,
        "r_char_star": None if mode != "r_char" else float(r_char_star),
        "beta_h": float(beta_h),
        "equivalent_beta_h": equivalent_beta_h,
        "skip_top_layers": int(skip_top_layers),
        "layer_profile": layer_profile,
        "layer_profile_strength": float(layer_profile_strength),
        "gamma_star": gamma_star,
        "gamma_dt": gamma_dt,
    }


def _resolve_r_char_star(
    r_char_star,
    *,
    beta_h: float,
    active_layers: int,
    layer_spacing_star,
    k_ratio,
):
    if r_char_star is not None:
        if float(r_char_star) <= 0:
            raise ValueError(f"fin_cooling_r_char_star must be positive, got {r_char_star}.")
        return float(r_char_star)
    if layer_spacing_star is None or k_ratio is None:
        raise ValueError(
            "fin_cooling_r_char_star must be set when layer_spacing_star or k_ratio is unavailable."
        )
    if float(layer_spacing_star) <= 0:
        raise ValueError(f"layer_spacing_star must be positive, got {layer_spacing_star}.")
    if float(k_ratio) <= 0:
        raise ValueError(f"k_ratio must be positive when deriving fin_cooling_r_char_star, got {k_ratio}.")
    total_thickness_star = float(active_layers) * float(layer_spacing_star)
    return total_thickness_star / (float(beta_h) * float(np.sqrt(float(k_ratio))))


def _normalize_gamma_value(value, *, active_layers: int):
    if isinstance(value, (list, tuple)):
        values = [float(item) for item in value]
        if len(values) != int(active_layers):
            raise ValueError(
                "fin_cooling_gamma_star sequence length must match the active layer count "
                f"{active_layers}, got {len(values)}."
            )
        if any(item < 0 for item in values):
            raise ValueError(f"fin_cooling_gamma_star must be non-negative, got {value}.")
        return values
    gamma = float(value)
    if gamma < 0:
        raise ValueError(f"fin_cooling_gamma_star must be non-negative, got {value}.")
    return gamma


def _multiply_gamma_value(value, factor: float):
    if isinstance(value, list):
        return [float(item) * float(factor) for item in value]
    return float(value) * float(factor)


def _metadata_optional_sequence(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return float(value)


def _equivalent_beta_h(gamma_dt, fdm_coefficient: float, active_layers: int):
    if isinstance(gamma_dt, list):
        nonzero = [float(value) for value in gamma_dt if float(value) > 0]
        representative = float(sum(nonzero) / len(nonzero)) if nonzero else 0.0
    else:
        representative = float(gamma_dt)
    if float(fdm_coefficient) <= 0 or representative < 0:
        return None
    return float(active_layers) * float(np.sqrt(representative / float(fdm_coefficient)))


def _apply_fin_cooling_layer_profile(
    gamma_star,
    *,
    active_layers: int,
    skip_top_layers: int,
    layer_profile: str,
    layer_profile_strength: float,
):
    if layer_profile == "uniform" or float(layer_profile_strength) == 0.0:
        return gamma_star
    if isinstance(gamma_star, list):
        base = gamma_star
    else:
        base = [float(gamma_star)] * int(active_layers)

    cooled_count = max(1, int(active_layers) - int(skip_top_layers))
    profiled = []
    for layer_index, value in enumerate(base):
        if layer_index < int(skip_top_layers):
            multiplier = 1.0
        else:
            depth = (layer_index - int(skip_top_layers) + 1) / cooled_count
            if layer_profile == "linear":
                shape = depth
            elif layer_profile == "quadratic":
                shape = depth * depth
            else:
                raise ValueError(
                    "fin_cooling_layer_profile must be one of 'uniform', 'linear', or 'quadratic', "
                    f"got {layer_profile!r}."
                )
            multiplier = 1.0 + float(layer_profile_strength) * shape
        profiled.append(float(value) * multiplier)
    return profiled


def write_multilayer_hdf5(
    output_path,
    *,
    source_h5,
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
    allow_unstable_fdm: bool,
    layer_fiber_angles_deg=None,
    normal_offset_sign: int = -1,
    layer_batch_size=None,
    delta_smoothing_alpha: float = 0.2,
    delta_smoothing_steps: int = 1,
    use_pdgcn_inplane: bool = True,
    pdgcn_inplane_top_layer_only: bool = False,
    use_alternating_order_average: bool = False,
    fdm_k_ratio_scale: float = 1.0,
    fdm_layer_interface_scales=None,
    fdm_top_surface_loss_gamma_dt: float = 0.0,
    fdm_top_surface_loss_velocity_exponent: float = 0.0,
    fdm_top_surface_loss_reference_velocity_star: float = 1.0,
    fin_cooling_gamma_star=None,
    fin_cooling_skip_top_layers: int = 0,
    prediction_group_path: str = DEFAULT_MULTILAYER_PREDICTION_GROUP_PATH,
):
    output_path = Path(output_path)
    temp_output = _copy_hdf5_to_prediction_output(source_h5, output_path, label="multilayer inference")
    metadata = dict(metadata)
    metadata["effective_layer_batch_size"] = int(
        _resolve_layer_batch_size(layer_batch_size, int(num_layers), next(model.parameters()).device)
    )
    metadata["prediction_group_path"] = str(prediction_group_path).strip("/")
    step_inference_seconds = []
    step_vtu_write_seconds = [0.0 for _ in range(max(0, int(steps) - 1))]
    timing_summary = {
        "inference_seconds": 0.0,
        "render_seconds": 0.0,
        "total_seconds": 0.0,
        "average_inference_seconds": 0.0,
        "max_inference_seconds": 0.0,
        "min_inference_seconds": 0.0,
        "rendered_steps": [],
    }

    try:
        with h5py.File(temp_output, "r+") as output_file:
            prediction_group_path = str(prediction_group_path).strip("/")
            if prediction_group_path in output_file:
                del output_file[prediction_group_path]
            output_group = output_file.create_group(prediction_group_path)
            multilayer_group = output_group.create_group("multilayer")
            temperature_star_dataset = output_group.create_dataset(
                "temperature_star",
                shape=(int(steps), int(num_layers), int(num_nodes), 1),
                dtype="float32",
            )
            temperature_dataset = output_group.create_dataset(
                "temperature",
                shape=(int(steps), int(num_layers), int(num_nodes), 1),
                dtype="float32",
            )
            top_temperature_dataset = output_group.create_dataset(
                "top_temperature",
                shape=(int(steps), int(num_nodes), 1),
                dtype="float32",
            )
            valid_mask_dataset = output_group.create_dataset(
                "valid_mask",
                data=np.ones((int(steps), int(num_layers), int(num_nodes), 1), dtype=np.uint8),
            )
            top_valid_mask_dataset = output_group.create_dataset(
                "top_valid_mask",
                data=np.ones((int(steps), int(num_nodes), 1), dtype=np.uint8),
            )
            del valid_mask_dataset, top_valid_mask_dataset
            time_values = _prediction_time_values(int(steps), metadata)
            output_group.create_dataset("time", data=time_values.astype(np.float64))
            _create_string_dataset(output_group, "temperature_layout", "time_layer_node_channel")
            _create_string_dataset(output_group, "temperature_unit", _temperature_unit_from_hdf5(output_file))
            coords_dataset = multilayer_group.create_dataset(
                "coordinates",
                shape=(int(steps), int(num_layers), int(num_nodes), 3),
                dtype="float32",
            )
            _create_string_dataset(multilayer_group, "coordinates_unit", "m")
            multilayer_group.create_dataset("bottom_temperature", data=float(_bottom_temperature(metadata, scale_params)))
            multilayer_group.create_dataset(
                "layer_fiber_angles_deg",
                data=np.asarray(metadata["layer_fiber_angles_deg"], dtype=np.float64),
            )
            multilayer_group.create_dataset("layer_spacing_m", data=float(layer_spacing))
            multilayer_group.create_dataset("normal_offset_sign", data=np.int64(normal_offset_sign))
            multilayer_group.create_dataset("num_layers", data=np.int64(num_layers))
            start_total = time.perf_counter()

            def writer(step, temperature_star, graph_step=None):
                temperature_star_dataset[int(step)] = temperature_star.numpy()
                temperature = temperature_from_dimensionless(temperature_star, scale_params).numpy()
                temperature_dataset[int(step)] = temperature
                top_temperature_dataset[int(step)] = temperature[0]
                if graph_step is not None:
                    geometry = _build_multilayer_geometry(
                        graph_step,
                        int(num_layers),
                        layer_spacing_star=float(metadata["layer_spacing_star"]),
                        layer_fiber_angles_deg=metadata["layer_fiber_angles_deg"],
                        normal_offset_sign=int(normal_offset_sign),
                    )
                    coords = geometry["pos"].detach().cpu().numpy() * float(scale_params.L0)
                    coords_dataset[int(step)] = coords.astype(np.float32, copy=False)
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
                allow_unstable_fdm=bool(allow_unstable_fdm),
                layer_fiber_angles_deg=layer_fiber_angles_deg,
                normal_offset_sign=int(normal_offset_sign),
                layer_batch_size=layer_batch_size,
                delta_smoothing_alpha=float(delta_smoothing_alpha),
                delta_smoothing_steps=int(delta_smoothing_steps),
                use_pdgcn_inplane=bool(use_pdgcn_inplane),
                pdgcn_inplane_top_layer_only=bool(pdgcn_inplane_top_layer_only),
                use_alternating_order_average=bool(use_alternating_order_average),
                fdm_k_ratio_scale=float(fdm_k_ratio_scale),
                fdm_layer_interface_scales=fdm_layer_interface_scales,
                fdm_top_surface_loss_gamma_dt=float(fdm_top_surface_loss_gamma_dt),
                fdm_top_surface_loss_velocity_exponent=float(fdm_top_surface_loss_velocity_exponent),
                fdm_top_surface_loss_reference_velocity_star=float(
                    fdm_top_surface_loss_reference_velocity_star
                ),
                fin_cooling_gamma_star=fin_cooling_gamma_star,
                fin_cooling_skip_top_layers=fin_cooling_skip_top_layers,
                timing_recorder=step_inference_seconds.append,
            )
            timing_summary["total_seconds"] = time.perf_counter() - start_total
            timing_summary["inference_seconds"] = timing_summary["total_seconds"]
            if step_inference_seconds:
                timing_summary["average_inference_seconds"] = float(np.mean(step_inference_seconds))
                timing_summary["max_inference_seconds"] = float(np.max(step_inference_seconds))
                timing_summary["min_inference_seconds"] = float(np.min(step_inference_seconds))
            metadata.update(timing_summary)
            _write_multilayer_prediction_timing(
                output_group,
                time_values=time_values,
                step_inference_seconds=step_inference_seconds,
                vtu_write_seconds=step_vtu_write_seconds,
                timing_summary=timing_summary,
                description="PDGCN multilayer wall-clock timings measured with time.perf_counter.",
            )
            metadata["timing"] = _timing_metadata_summary(timing_summary, step_inference_seconds)
            _write_prediction_metadata(output_group, metadata)
        temp_output.replace(output_path)
    except Exception:
        if temp_output.exists():
            temp_output.unlink()
        raise

    return timing_summary


def _copy_hdf5_to_prediction_output(source_h5, output_h5, *, label: str):
    source_h5 = Path(source_h5).resolve()
    output_h5 = Path(output_h5).resolve()
    if source_h5 == output_h5:
        raise ValueError(f"{label} output_path must differ from the source HDF5 path.")
    if not source_h5.exists():
        raise FileNotFoundError(f"HDF5 file not found: {source_h5}")
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_h5.with_name(f".{output_h5.name}.tmp")
    if temp_output.exists():
        temp_output.unlink()
    shutil.copy2(source_h5, temp_output)
    return temp_output


def _prediction_time_values(steps: int, metadata):
    hdf5_timing = metadata.get("hdf5_timing", {})
    dt = float(hdf5_timing.get("dt", 1.0))
    return np.arange(int(steps), dtype=np.float64) * dt


def _temperature_unit_from_hdf5(h5_file):
    if "fem/temperature_unit" in h5_file:
        value = h5_file["fem/temperature_unit"][()]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)
    return "degC"


def _bottom_temperature(metadata, scale_params):
    return float(scale_params.T_amb) + float(metadata.get("bottom_temperature_star", 0.0)) * float(scale_params.delta_T0)


def _create_string_dataset(group, name: str, value):
    group.create_dataset(name, data=str(value), dtype=h5py.string_dtype(encoding="utf-8"))


def _write_prediction_metadata(group, metadata):
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
    _create_string_dataset(group, "metadata_json", metadata_json)
    group.attrs["metadata_json"] = metadata_json


def _write_root_compatibility_metadata(output_file, metadata):
    metadata_json = json.dumps(metadata, ensure_ascii=False)
    if "metadata" in output_file:
        del output_file["metadata"]
    output_file.create_dataset("metadata", data=metadata_json, dtype=h5py.string_dtype(encoding="utf-8"))
    output_file.attrs["metadata"] = metadata_json


def _write_multilayer_prediction_timing(
    output_group,
    *,
    time_values,
    step_inference_seconds,
    vtu_write_seconds,
    timing_summary,
    description: str,
):
    if "timing" in output_group:
        del output_group["timing"]
    timing_group = output_group.create_group("timing")
    step_count = max(0, len(time_values) - 1)
    step_indices = np.arange(1, step_count + 1, dtype=np.int64)
    solve_seconds = _transition_step_seconds(step_inference_seconds, step_count)
    vtu_seconds = _transition_step_seconds(vtu_write_seconds, step_count)
    step_seconds = solve_seconds + vtu_seconds
    timing_group.create_dataset("step", data=step_indices)
    timing_group.create_dataset("frame_from", data=np.arange(0, step_count, dtype=np.int64))
    timing_group.create_dataset("frame_to", data=step_indices)
    timing_group.create_dataset("time_s", data=np.asarray(time_values[1:], dtype=np.float64))
    timing_group.create_dataset("solve_seconds", data=solve_seconds)
    timing_group.create_dataset("vtu_write_seconds", data=vtu_seconds)
    timing_group.create_dataset("step_seconds", data=step_seconds)
    timing_group.create_dataset("average_solve_seconds", data=float(np.mean(solve_seconds)) if step_count else 0.0)
    timing_group.create_dataset("average_step_seconds", data=float(np.mean(step_seconds)) if step_count else 0.0)
    timing_group.create_dataset("compute_seconds", data=float(timing_summary.get("inference_seconds", 0.0)))
    timing_group.create_dataset("step_total_seconds", data=float(np.sum(step_seconds)))
    timing_group.create_dataset("vtu_total_write_seconds", data=float(np.sum(vtu_seconds)))
    timing_group.create_dataset("total_seconds", data=float(timing_summary.get("total_seconds", 0.0)))
    _create_string_dataset(timing_group, "time_unit", "s")
    _create_string_dataset(timing_group, "description", description)


def _transition_step_seconds(values, step_count: int):
    values = np.asarray(list(values), dtype=np.float64)
    if step_count <= 0:
        return np.zeros((0,), dtype=np.float64)
    if values.size == 0:
        return np.zeros((step_count,), dtype=np.float64)
    if values.size >= step_count + 1:
        return values[1 : step_count + 1].astype(np.float64, copy=False)
    if values.size == step_count:
        return values.astype(np.float64, copy=False)
    padded = np.zeros((step_count,), dtype=np.float64)
    padded[: values.size] = values
    return padded


def _timing_metadata_summary(timing_summary, step_inference_seconds):
    return {
        "average_solve_seconds": float(timing_summary.get("average_inference_seconds", 0.0)),
        "compute_seconds": float(timing_summary.get("inference_seconds", 0.0)),
        "max_step_seconds": float(np.max(step_inference_seconds)) if step_inference_seconds else 0.0,
        "min_step_seconds": float(np.min(step_inference_seconds)) if step_inference_seconds else 0.0,
        "num_timed_steps": int(max(0, len(step_inference_seconds) - 1)),
        "time_unit": "s",
        "total_seconds": float(timing_summary.get("total_seconds", 0.0)),
        "vtu_total_write_seconds": float(timing_summary.get("render_seconds", 0.0)),
    }


def _update_multilayer_prediction_after_render(
    prediction_h5,
    timing_values,
    *,
    prediction_group_path: str = DEFAULT_MULTILAYER_PREDICTION_GROUP_PATH,
):
    prediction_h5 = Path(prediction_h5)
    with h5py.File(prediction_h5, "r+") as output_file:
        group = _resolve_multilayer_prediction_group(output_file, prediction_group_path=prediction_group_path)
        metadata = _read_multilayer_metadata_from_group(group)
        metadata.update(timing_values)
        metadata["timing"] = {
            **metadata.get("timing", {}),
            "compute_seconds": float(timing_values.get("inference_seconds", 0.0)),
            "total_seconds": float(timing_values.get("total_seconds", 0.0)),
            "vtu_total_write_seconds": float(timing_values.get("render_seconds", 0.0)),
        }
        time_values = np.asarray(group["time"], dtype=np.float64)
        timing_group = group["timing"]
        step_count = max(0, len(time_values) - 1)
        vtu_write_seconds = _vtu_write_seconds_for_rendered_steps(
            timing_values.get("rendered_steps", []),
            float(timing_values.get("render_seconds", 0.0)),
            step_count,
        )
        solve_seconds = np.asarray(timing_group["solve_seconds"], dtype=np.float64)
        summary = {
            "inference_seconds": float(timing_values.get("inference_seconds", 0.0)),
            "render_seconds": float(timing_values.get("render_seconds", 0.0)),
            "total_seconds": float(timing_values.get("total_seconds", 0.0)),
        }
        _write_multilayer_prediction_timing(
            group,
            time_values=time_values,
            step_inference_seconds=solve_seconds,
            vtu_write_seconds=vtu_write_seconds,
            timing_summary=summary,
            description="PDGCN multilayer wall-clock timings measured with time.perf_counter; step_seconds includes VTK write when enabled.",
        )
        if "metadata_json" in group:
            del group["metadata_json"]
        _write_prediction_metadata(group, metadata)


def _vtu_write_seconds_for_rendered_steps(rendered_steps, render_seconds: float, step_count: int):
    values = np.zeros((step_count,), dtype=np.float64)
    rendered_transitions = [int(step) for step in rendered_steps if int(step) > 0 and int(step) <= step_count]
    if rendered_transitions:
        per_step = float(render_seconds) / float(len(rendered_transitions))
        for step in rendered_transitions:
            values[step - 1] = per_step
    return values


def _should_write_cloud_step(step: int, cloud_interval: int) -> bool:
    return int(step) % int(cloud_interval) == 0


def render_multilayer_clouds_from_hdf5(
    prediction_h5,
    *,
    cloud_interval=None,
    vtk_output_dir=None,
    max_nodes_per_layer=None,
    prediction_group_path: str = DEFAULT_MULTILAYER_PREDICTION_GROUP_PATH,
):
    prediction_h5 = Path(prediction_h5)
    metadata = _read_prediction_metadata(prediction_h5, prediction_group_path=prediction_group_path)
    scale_params = ScaleParams(**metadata["scale_params"])

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

    num_layers = int(metadata["num_layers"])
    num_nodes = _read_prediction_num_nodes(prediction_h5)
    source_h5 = Path(metadata["source_h5"]) if "source_h5" in metadata else None
    if source_h5 is not None and not source_h5.is_absolute():
        source_h5 = (prediction_h5.parent / source_h5).resolve()
    hdf5_timing = metadata.get("hdf5_timing", {})
    scan_velocity = hdf5_timing.get("velocity_speed", scale_params.v0)
    loader = HDF5Loader(source_h5, scale_params=scale_params) if source_h5 is not None else None
    layer_spacing_star = float(metadata.get("layer_spacing_star", float(metadata["layer_spacing"]) / scale_params.L0))
    layer_fiber_angles_deg = metadata.get("layer_fiber_angles_deg") or [0.0] * num_layers
    normal_offset_sign = int(metadata.get("normal_offset_sign", -1))

    rendered_steps = []
    start_render = time.perf_counter()
    with h5py.File(prediction_h5, "r") as output_file:
        prediction_group = _resolve_multilayer_prediction_group(
            output_file,
            prediction_group_path=prediction_group_path,
        )
        temperature_dataset = prediction_group["temperature"]
        temperature_star_dataset = prediction_group["temperature_star"]
        coords_dataset = (
            prediction_group["multilayer/coordinates"] if "multilayer/coordinates" in prediction_group else None
        )
        steps = int(temperature_star_dataset.shape[0])
        for step in range(steps):
            if not _should_write_cloud_step(step, cloud_interval):
                continue
            if coords_dataset is not None:
                if int(coords_dataset.shape[2]) != int(num_nodes):
                    raise ValueError(
                        f"prediction coordinate node count {coords_dataset.shape[2]} does not match temperature node count {num_nodes}."
                    )
                coords_layers_m = coords_dataset[step]
                edge_index = output_file["edge_index"][()]
            else:
                if loader is None:
                    raise KeyError("Legacy multilayer prediction HDF5 requires metadata.source_h5 for rendering.")
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
                coords_layers_m = geometry["pos"].detach().cpu().numpy() * float(scale_params.L0)
                edge_index = graph.edge_index.detach().cpu().numpy()
            _write_multilayer_step_vtk(
                vtk_output_dir,
                step=int(step),
                coords_layers_m=coords_layers_m,
                edge_index=edge_index,
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
    coords_layers_m,
    edge_index,
    temperature,
    temperature_star,
    scale_params,
    max_nodes_per_layer=None,
):
    vtk_output_dir = Path(vtk_output_dir)
    coords_layers_m = np.asarray(coords_layers_m, dtype=np.float64)
    if coords_layers_m.ndim != 3 or coords_layers_m.shape[2] != 3:
        raise ValueError(f"coords_layers_m must have shape [layer, node, 3], got {coords_layers_m.shape}.")
    layer_count = int(coords_layers_m.shape[0])
    nodes_per_layer = int(coords_layers_m.shape[1])
    if max_nodes_per_layer is not None:
        raise ValueError(
            "max_nodes_per_layer is not supported for topology wedge rendering because node sampling breaks "
            "the Gmsh triangle connectivity."
        )
    sample_indices = np.arange(nodes_per_layer, dtype=np.int64)

    coords = coords_layers_m[:, sample_indices, :].reshape(layer_count * len(sample_indices), 3)
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


def _read_prediction_num_nodes(
    prediction_h5,
    *,
    prediction_group_path: str = DEFAULT_MULTILAYER_PREDICTION_GROUP_PATH,
):
    with h5py.File(prediction_h5, "r") as output_file:
        if prediction_group_path in output_file:
            return int(output_file[f"{prediction_group_path}/temperature_star"].shape[2])
        return int(output_file["temperature_star"].shape[2])


def _read_prediction_metadata(
    prediction_h5,
    *,
    prediction_group_path: str = DEFAULT_MULTILAYER_PREDICTION_GROUP_PATH,
):
    with h5py.File(prediction_h5, "r") as output_file:
        if prediction_group_path in output_file:
            return _read_multilayer_metadata_from_group(output_file[prediction_group_path])
        metadata_json = output_file.attrs["metadata"] if "metadata" in output_file.attrs else output_file["metadata"][()]
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


def _resolve_multilayer_prediction_group(
    output_file,
    *,
    prediction_group_path: str = DEFAULT_MULTILAYER_PREDICTION_GROUP_PATH,
):
    prediction_group_path = str(prediction_group_path or DEFAULT_MULTILAYER_PREDICTION_GROUP_PATH).strip("/")
    if prediction_group_path in output_file:
        return output_file[prediction_group_path]
    if "temperature" in output_file and "temperature_star" in output_file:
        return output_file
    raise KeyError(f"Prediction group '{prediction_group_path}' not found.")


def _read_multilayer_metadata_from_group(group):
    if "metadata_json" in group.attrs:
        metadata_json = group.attrs["metadata_json"]
    elif "metadata_json" in group:
        metadata_json = group["metadata_json"][()]
    elif "metadata" in group.attrs:
        metadata_json = group.attrs["metadata"]
    else:
        metadata_json = group["metadata"][()]
    if isinstance(metadata_json, bytes):
        metadata_json = metadata_json.decode("utf-8")
    return json.loads(str(metadata_json))


def _resolve_path(base_dir: Path, value) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
