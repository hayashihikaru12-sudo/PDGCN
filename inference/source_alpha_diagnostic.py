"""Diagnostic sweep for the effective surface-source multiplier alpha_Q."""

from __future__ import annotations

import csv
import html
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np
import torch

from data import HDF5Loader, build_graph
from training.train_entry import derive_timing_from_hdf5, discover_hdf5_files

from .io import (
    _prepare_multilayer_runtime,
    _resolve_path,
    load_inference_run_context,
    read_hdf5_temperature_shape,
)
from .multilayer import rollout_multilayer_fdm


@dataclass(frozen=True)
class SourceAlphaDiagnosticConfig:
    """Settings that are specific to the alpha_Q diagnostic sweep."""

    alpha_q_values: Sequence[float]
    output_dir: str = "../runs/pdgcn/source_alpha_diagnostic"
    h5_dir: Optional[str] = None
    fem_temperature_dataset: str = "fem/temperature"
    fem_valid_mask_dataset: Optional[str] = "fem/valid_mask"
    fem_frame_offset: int = 0
    max_cases: Optional[int] = None
    write_plots: bool = True

    def __post_init__(self):
        values = tuple(float(value) for value in self.alpha_q_values)
        if not values:
            raise ValueError("diagnostic.alpha_q_values must contain at least one value.")
        if any(not math.isfinite(value) or value < 1.0 or value > 2.0 for value in values):
            raise ValueError(
                "diagnostic.alpha_q_values must be finite and within [1, 2]. "
                "The current rollout hook represents alpha_Q as 1 + compensation_alpha."
            )
        if len(set(values)) != len(values):
            raise ValueError("diagnostic.alpha_q_values must not contain duplicates.")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("diagnostic.output_dir must be a non-empty string.")
        if self.h5_dir is not None and (not isinstance(self.h5_dir, str) or not self.h5_dir.strip()):
            raise ValueError("diagnostic.h5_dir must be null or a non-empty string.")
        if not isinstance(self.fem_temperature_dataset, str) or not self.fem_temperature_dataset.strip():
            raise ValueError("diagnostic.fem_temperature_dataset must be a non-empty string.")
        if self.fem_valid_mask_dataset is not None and (
            not isinstance(self.fem_valid_mask_dataset, str) or not self.fem_valid_mask_dataset.strip()
        ):
            raise ValueError("diagnostic.fem_valid_mask_dataset must be null or a non-empty string.")
        if isinstance(self.fem_frame_offset, bool) or int(self.fem_frame_offset) != self.fem_frame_offset:
            raise ValueError("diagnostic.fem_frame_offset must be an integer.")
        if int(self.fem_frame_offset) < 0:
            raise ValueError("diagnostic.fem_frame_offset must be non-negative.")
        if self.max_cases is not None and int(self.max_cases) <= 0:
            raise ValueError("diagnostic.max_cases must be null or a positive integer.")
        if not isinstance(self.write_plots, bool):
            raise ValueError("diagnostic.write_plots must be a boolean.")
        object.__setattr__(self, "alpha_q_values", values)


def load_source_alpha_diagnostic_config(config_path):
    config_path = Path(config_path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("diagnostic config JSON must contain an object at the top level.")
    unknown = sorted(set(payload) - {"base_inference_config", "diagnostic"})
    if unknown:
        raise ValueError(f"Unknown keys in diagnostic config: {unknown}")
    base_value = payload.get("base_inference_config")
    if not isinstance(base_value, str) or not base_value.strip():
        raise ValueError("base_inference_config must be a non-empty string path.")
    diagnostic_value = payload.get("diagnostic")
    if not isinstance(diagnostic_value, dict):
        raise ValueError("diagnostic must be an object.")
    valid_fields = set(SourceAlphaDiagnosticConfig.__dataclass_fields__)
    unknown_diagnostic = sorted(set(diagnostic_value) - valid_fields)
    if unknown_diagnostic:
        raise ValueError(f"Unknown keys in diagnostic section: {unknown_diagnostic}")
    if "alpha_q_values" not in diagnostic_value:
        raise ValueError("diagnostic.alpha_q_values is required.")
    diagnostic_config = SourceAlphaDiagnosticConfig(**diagnostic_value)
    base_config_path = _resolve_path(config_path.parent, base_value)
    return base_config_path, diagnostic_config


def run_source_alpha_diagnostic(config_path, *, checkpoint=None, h5_dir=None, output_dir=None):
    """Run every alpha_Q over every selected case and export compact diagnostics."""

    base_config_path, diagnostic_config = load_source_alpha_diagnostic_config(config_path)
    if h5_dir is not None:
        diagnostic_config = replace(diagnostic_config, h5_dir=str(h5_dir))
    if output_dir is not None:
        diagnostic_config = replace(diagnostic_config, output_dir=str(output_dir))

    run_config, inference_config, training_base_dir, inference_base_dir, training_config_path = (
        load_inference_run_context(base_config_path)
    )
    if int(inference_config.dataset_index) >= len(run_config.datasets):
        raise IndexError(
            f"inference.dataset_index={inference_config.dataset_index} exceeds "
            f"datasets length {len(run_config.datasets)}."
        )
    dataset = run_config.datasets[int(inference_config.dataset_index)]
    scale_params = dataset.scale.to_scale_params()
    selected_checkpoint = (
        _resolve_path(Path(config_path).resolve().parent, checkpoint)
        if checkpoint
        else _resolve_path(
            training_base_dir,
            run_config.outputs.checkpoint_path if run_config.outputs is not None else run_config.data.checkpoint_path,
        )
    )
    selected_h5_dir = (
        _resolve_path(Path(config_path).resolve().parent, diagnostic_config.h5_dir)
        if diagnostic_config.h5_dir is not None
        else (
            _resolve_path(inference_base_dir, inference_config.h5_dir)
            if inference_config.h5_dir is not None
            else _resolve_path(training_base_dir, dataset.h5_dir)
        )
    )
    selected_output_dir = _resolve_path(Path(config_path).resolve().parent, diagnostic_config.output_dir)
    selected_output_dir.mkdir(parents=True, exist_ok=True)

    selected_h5_paths = discover_hdf5_files(selected_h5_dir)
    if diagnostic_config.max_cases is not None:
        selected_h5_paths = selected_h5_paths[: int(diagnostic_config.max_cases)]
    if not selected_h5_paths:
        raise FileNotFoundError(f"No HDF5 files found in {selected_h5_dir}.")

    device = torch.device(run_config.training.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = _prepare_multilayer_runtime(
        selected_h5_paths[0],
        selected_checkpoint=selected_checkpoint,
        scale_params=scale_params,
        scan_velocity=dataset.scan_velocity,
        model_overrides=run_config.model,
        device=device,
    )
    warmup_steps = (
        int(inference_config.warmup_steps)
        if inference_config.warmup_steps is not None
        else int(run_config.training.warmup_steps)
    )

    case_records = []
    failures = []
    total_runs = len(diagnostic_config.alpha_q_values) * len(selected_h5_paths)
    completed_runs = 0
    for alpha_q in diagnostic_config.alpha_q_values:
        for selected_h5 in selected_h5_paths:
            try:
                prediction, fem_temperature, valid_mask = _run_one_case(
                    selected_h5,
                    model=model,
                    scale_params=scale_params,
                    scan_velocity=dataset.scan_velocity,
                    inference_config=inference_config,
                    warmup_steps=warmup_steps,
                    alpha_q=float(alpha_q),
                    diagnostic_config=diagnostic_config,
                )
                record = compute_case_temperature_metrics(prediction, fem_temperature, valid_mask)
                record.update(
                    {
                        "alpha_q": float(alpha_q),
                        "source_multiplier_alpha_q": float(alpha_q),
                        "case": selected_h5.name,
                        "h5_path": str(selected_h5),
                    }
                )
                case_records.append(record)
            except Exception as error:  # noqa: BLE001 - a sweep should retain all per-case failures.
                failures.append(
                    {
                        "alpha_q": float(alpha_q),
                        "case": selected_h5.name,
                        "h5_path": str(selected_h5),
                        "error": str(error),
                    }
                )
            completed_runs += 1
            print(
                f"[{completed_runs}/{total_runs}] alpha_Q={float(alpha_q):.4f} "
                f"case={selected_h5.name}"
            )

    if not case_records:
        raise RuntimeError(f"All diagnostic runs failed. First failures: {failures[:3]}")
    summary_records = summarize_alpha_metrics(case_records, num_layers=int(inference_config.num_layers))

    case_csv = selected_output_dir / "source_alpha_case_metrics.csv"
    summary_csv = selected_output_dir / "source_alpha_summary.csv"
    summary_json = selected_output_dir / "source_alpha_diagnostic.json"
    _write_csv(case_csv, case_records)
    _write_csv(summary_csv, summary_records)
    summary_payload = {
        "base_inference_config": str(base_config_path),
        "training_config": str(training_config_path),
        "checkpoint": str(selected_checkpoint),
        "h5_dir": str(selected_h5_dir),
        "output_dir": str(selected_output_dir),
        "diagnostic": asdict(diagnostic_config),
        "alpha_q_definition": "delta_T_source_corrected = alpha_Q * delta_T_source",
        "rollout_parameter_mapping": "source_multiplier_alpha_q = alpha_Q",
        "legacy_post_fdm_source_compensation_alpha": "forced to 0 during this diagnostic",
        "successful_runs": len(case_records),
        "failed_runs": len(failures),
        "failures": failures,
        "summary": summary_records,
    }
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_paths = []
    if diagnostic_config.write_plots:
        plot_paths = write_diagnostic_plots(selected_output_dir, case_records, summary_records)
    return {
        "output_dir": str(selected_output_dir),
        "case_metrics_csv": str(case_csv),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "plot_paths": [str(path) for path in plot_paths],
        "successful_runs": len(case_records),
        "failed_runs": len(failures),
        "summary": summary_records,
    }


def _run_one_case(
    selected_h5,
    *,
    model,
    scale_params,
    scan_velocity,
    inference_config,
    warmup_steps,
    alpha_q,
    diagnostic_config,
):
    timing = derive_timing_from_hdf5(selected_h5, scale_params, scan_velocity=scan_velocity)
    num_frames, _ = read_hdf5_temperature_shape(selected_h5)
    offset = int(diagnostic_config.fem_frame_offset)
    available_steps = num_frames - offset
    if available_steps <= 0:
        raise ValueError(f"fem_frame_offset={offset} leaves no comparable frames in {selected_h5}.")
    steps = int(inference_config.steps) if inference_config.steps is not None else available_steps
    if steps > available_steps:
        raise ValueError(
            f"inference.steps={steps} exceeds {available_steps} comparable frames after "
            f"fem_frame_offset={offset}."
        )

    loader = HDF5Loader(selected_h5, scale_params=scale_params)
    device = next(model.parameters()).device

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

    prediction = rollout_multilayer_fdm(
        model,
        graph_factory,
        steps,
        scale_params,
        num_layers=int(inference_config.num_layers),
        layer_spacing=float(inference_config.layer_spacing),
        return_dimensionless=False,
        return_all=True,
        warmup_steps=int(warmup_steps),
        bottom_temperature_star=float(inference_config.bottom_temperature_star),
        allow_unstable_fdm=bool(inference_config.allow_unstable_fdm),
        layer_fiber_angles_deg=inference_config.layer_fiber_angles_deg,
        normal_offset_sign=int(inference_config.normal_offset_sign),
        layer_batch_size=inference_config.layer_batch_size,
        delta_smoothing_alpha=float(inference_config.delta_smoothing_alpha),
        delta_smoothing_steps=int(inference_config.delta_smoothing_steps),
        use_pdgcn_inplane=bool(inference_config.use_pdgcn_inplane),
        pdgcn_inplane_top_layer_only=bool(inference_config.pdgcn_inplane_top_layer_only),
        source_multiplier_alpha_q=float(alpha_q),
        post_fdm_source_compensation_alpha=0.0,
    ).numpy()
    with h5py.File(selected_h5, "r") as h5_file:
        dataset_path = diagnostic_config.fem_temperature_dataset.strip("/")
        if dataset_path not in h5_file:
            raise KeyError(f"Required FEM dataset '{dataset_path}' not found in {selected_h5}.")
        fem_temperature = np.asarray(h5_file[dataset_path][offset : offset + steps], dtype=np.float64)
        fem_temperature = _convert_fem_temperature_to_deg_c(fem_temperature, h5_file)
        mask_path = diagnostic_config.fem_valid_mask_dataset
        if mask_path is not None and mask_path.strip("/") in h5_file:
            valid_mask = np.asarray(h5_file[mask_path.strip("/")][offset : offset + steps], dtype=bool)
        else:
            valid_mask = np.ones_like(fem_temperature, dtype=bool)
    return np.asarray(prediction, dtype=np.float64), fem_temperature, valid_mask


def _convert_fem_temperature_to_deg_c(temperature, h5_file):
    if "fem/temperature_unit" not in h5_file:
        return temperature
    unit = h5_file["fem/temperature_unit"][()]
    if isinstance(unit, bytes):
        unit = unit.decode("utf-8")
    normalized = str(unit).strip().lower()
    if normalized == "k":
        return temperature - 273.15
    if normalized in {"c", "degc", "°c"}:
        return temperature
    raise ValueError(f"Unsupported fem/temperature_unit: {unit!r}.")


def compute_case_temperature_metrics(prediction, fem_temperature, valid_mask=None):
    prediction = np.asarray(prediction, dtype=np.float64)
    fem_temperature = np.asarray(fem_temperature, dtype=np.float64)
    if prediction.shape != fem_temperature.shape:
        raise ValueError(
            f"prediction and FEM shapes must match, got {prediction.shape} and {fem_temperature.shape}."
        )
    if prediction.ndim != 4 or prediction.shape[-1] != 1:
        raise ValueError(f"temperature arrays must have shape [T, L, N, 1], got {prediction.shape}.")
    if valid_mask is None:
        valid = np.ones_like(fem_temperature, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != fem_temperature.shape:
            raise ValueError(f"valid_mask shape must match temperature shape, got {valid.shape}.")
    valid &= np.isfinite(prediction) & np.isfinite(fem_temperature)
    if not np.any(valid):
        raise ValueError("No valid finite FEM comparison points are available.")

    error = prediction - fem_temperature
    record = _error_metrics(error[valid], prefix="field")
    pred_valid = prediction[valid]
    fem_valid = fem_temperature[valid]
    record.update(
        {
            "global_pred_peak": float(np.max(pred_valid)),
            "global_fem_peak": float(np.max(fem_valid)),
            "global_peak_bias": float(np.max(pred_valid) - np.max(fem_valid)),
            "num_layers": int(prediction.shape[1]),
        }
    )
    for layer in range(prediction.shape[1]):
        layer_valid = valid[:, layer, :, :]
        if not np.any(layer_valid):
            continue
        layer_prediction = prediction[:, layer, :, :]
        layer_fem = fem_temperature[:, layer, :, :]
        layer_error = layer_prediction - layer_fem
        prefix = f"layer_{layer + 1}"
        record.update(_error_metrics(layer_error[layer_valid], prefix=prefix))
        pred_peak = float(np.max(layer_prediction[layer_valid]))
        fem_peak = float(np.max(layer_fem[layer_valid]))
        record[f"{prefix}_pred_peak"] = pred_peak
        record[f"{prefix}_fem_peak"] = fem_peak
        record[f"{prefix}_peak_bias"] = pred_peak - fem_peak
    return record


def _error_metrics(error, *, prefix):
    error = np.asarray(error, dtype=np.float64)
    return {
        f"{prefix}_valid_count": int(error.size),
        f"{prefix}_rmse": float(np.sqrt(np.mean(np.square(error)))),
        f"{prefix}_mae": float(np.mean(np.abs(error))),
        f"{prefix}_max_abs_error": float(np.max(np.abs(error))),
    }


def summarize_alpha_metrics(case_records, *, num_layers):
    alpha_values = sorted({float(record["alpha_q"]) for record in case_records})
    summaries = []
    for alpha_q in alpha_values:
        records = [record for record in case_records if float(record["alpha_q"]) == alpha_q]
        fem_peaks = np.asarray([record["global_fem_peak"] for record in records], dtype=np.float64)
        pred_peaks = np.asarray([record["global_pred_peak"] for record in records], dtype=np.float64)
        slope, intercept, r2 = _linear_fit(fem_peaks, pred_peaks)
        peak_bias = pred_peaks - fem_peaks
        summary = {
            "alpha_q": alpha_q,
            "source_multiplier_alpha_q": alpha_q,
            "case_count": len(records),
            "global_peak_fit_slope": slope,
            "global_peak_fit_intercept": intercept,
            "global_peak_fit_r2": r2,
            "global_peak_bias_mean": float(np.mean(peak_bias)),
            "global_peak_bias_rmse": float(np.sqrt(np.mean(np.square(peak_bias)))),
            "field_rmse": _pooled_rmse(records, "field"),
            "field_mae": _pooled_mae(records, "field"),
            "field_max_abs_error": float(max(record["field_max_abs_error"] for record in records)),
            "mean_case_field_rmse": float(np.mean([record["field_rmse"] for record in records])),
        }
        for layer in range(int(num_layers)):
            prefix = f"layer_{layer + 1}"
            selected = [record for record in records if f"{prefix}_peak_bias" in record]
            if not selected:
                continue
            biases = np.asarray([record[f"{prefix}_peak_bias"] for record in selected], dtype=np.float64)
            summary[f"{prefix}_peak_bias_mean"] = float(np.mean(biases))
            summary[f"{prefix}_peak_bias_std"] = float(np.std(biases, ddof=1)) if len(biases) > 1 else 0.0
            summary[f"{prefix}_peak_bias_sem"] = (
                float(np.std(biases, ddof=1) / np.sqrt(len(biases))) if len(biases) > 1 else 0.0
            )
            summary[f"{prefix}_peak_bias_rmse"] = float(np.sqrt(np.mean(np.square(biases))))
            summary[f"{prefix}_rmse"] = _pooled_rmse(selected, prefix)
            summary[f"{prefix}_mae"] = _pooled_mae(selected, prefix)
        summaries.append(summary)
    return summaries


def _linear_fit(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.ptp(x) <= 1e-12:
        return float("nan"), float("nan"), float("nan")
    # Use the closed-form simple-regression solution instead of np.polyfit.
    # Some Windows NumPy builds used by this project have an unstable LAPACK
    # binding for the least-squares call behind polyfit.
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = np.sum(np.square(x_centered))
    slope = np.sum(x_centered * y_centered) / denominator
    intercept = np.mean(y) - slope * np.mean(x)
    fitted = slope * x + intercept
    total = np.sum(np.square(y - np.mean(y)))
    r2 = 1.0 - np.sum(np.square(y - fitted)) / total if total > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def _pooled_rmse(records, prefix):
    count = sum(int(record[f"{prefix}_valid_count"]) for record in records)
    sum_squared = sum(
        float(record[f"{prefix}_rmse"]) ** 2 * int(record[f"{prefix}_valid_count"])
        for record in records
    )
    return float(np.sqrt(sum_squared / count))


def _pooled_mae(records, prefix):
    count = sum(int(record[f"{prefix}_valid_count"]) for record in records)
    sum_absolute = sum(
        float(record[f"{prefix}_mae"]) * int(record[f"{prefix}_valid_count"])
        for record in records
    )
    return float(sum_absolute / count)


def _write_csv(path, records):
    fieldnames = []
    seen = set()
    for record in records:
        for key in record:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with Path(path).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_diagnostic_plots(output_dir, case_records, summary_records):
    output_dir = Path(output_dir)
    alpha = [float(record["alpha_q"]) for record in summary_records]
    summary_panels = [
        {
            "title": "Slope diagnostic",
            "xlabel": "alpha_Q",
            "ylabel": "Global peak fit slope",
            "series": [
                _svg_series(alpha, [record["global_peak_fit_slope"] for record in summary_records], "slope"),
                _svg_horizontal_series(alpha, 1.0, "target", "#666666", dashed=True),
            ],
        },
        {
            "title": "Intercept diagnostic",
            "xlabel": "alpha_Q",
            "ylabel": "Intercept (deg C)",
            "series": [
                _svg_series(alpha, [record["global_peak_fit_intercept"] for record in summary_records], "intercept"),
                _svg_horizontal_series(alpha, 0.0, "target", "#666666", dashed=True),
            ],
        },
        {
            "title": "Top-layer peak bias",
            "xlabel": "alpha_Q",
            "ylabel": "Prediction - FEM (deg C)",
            "series": [
                _svg_series(
                    alpha,
                    [record.get("layer_1_peak_bias_mean", float("nan")) for record in summary_records],
                    "layer 1",
                    "#1f77b4",
                    [record.get("layer_1_peak_bias_sem", 0.0) for record in summary_records],
                ),
                _svg_series(
                    alpha,
                    [record.get("layer_2_peak_bias_mean", float("nan")) for record in summary_records],
                    "layer 2",
                    "#ff7f0e",
                    [record.get("layer_2_peak_bias_sem", 0.0) for record in summary_records],
                ),
                _svg_horizontal_series(alpha, 0.0, "target", "#666666", dashed=True),
            ],
        },
        {
            "title": "Full-field error",
            "xlabel": "alpha_Q",
            "ylabel": "Pooled field RMSE (deg C)",
            "series": [_svg_series(alpha, [record["field_rmse"] for record in summary_records], "RMSE")],
        },
    ]
    summary_path = output_dir / "source_alpha_summary.svg"
    _write_svg_panel_grid(summary_path, summary_panels, columns=2)

    columns = min(3, len(summary_records))
    agreement_panels = []
    for summary in summary_records:
        selected = [record for record in case_records if float(record["alpha_q"]) == float(summary["alpha_q"])]
        fem = [float(record["global_fem_peak"]) for record in selected]
        pred = [float(record["global_pred_peak"]) for record in selected]
        low = float(min(min(fem), min(pred)))
        high = float(max(max(fem), max(pred)))
        slope = summary["global_peak_fit_slope"]
        intercept = summary["global_peak_fit_intercept"]
        series = [
            _svg_series(fem, pred, "cases", "#1f77b4", points_only=True),
            _svg_series([low, high], [low, high], "1:1", "#666666"),
        ]
        if np.isfinite(slope) and np.isfinite(intercept):
            series.append(
                _svg_series(
                    [low, high],
                    [slope * low + intercept, slope * high + intercept],
                    "fit",
                    "#d95f02",
                )
            )
        agreement_panels.append(
            {
                "title": f"alpha_Q={summary['alpha_q']:.3f}, a={slope:.3f}, b={intercept:.1f}",
                "xlabel": "FEM global peak (deg C)",
                "ylabel": "Prediction global peak (deg C)",
                "series": series,
                "equal_axes": True,
            }
        )
    agreement_path = output_dir / "source_alpha_global_peak_agreement.svg"
    _write_svg_panel_grid(agreement_path, agreement_panels, columns=columns)
    return [summary_path, agreement_path]


def _svg_series(x, y, label, color="#1f77b4", yerr=None, points_only=False, dashed=False):
    return {
        "x": [float(value) for value in x],
        "y": [float(value) for value in y],
        "yerr": None if yerr is None else [float(value) for value in yerr],
        "label": str(label),
        "color": str(color),
        "points_only": bool(points_only),
        "dashed": bool(dashed),
    }


def _svg_horizontal_series(x, value, label, color, dashed=False):
    finite_x = [float(item) for item in x if np.isfinite(item)]
    if not finite_x:
        finite_x = [0.0, 1.0]
    if len(finite_x) == 1:
        finite_x = [finite_x[0] - 0.5, finite_x[0] + 0.5]
    return _svg_series([min(finite_x), max(finite_x)], [value, value], label, color, dashed=dashed)


def _write_svg_panel_grid(path, panels, *, columns):
    panel_width = 520
    panel_height = 390
    columns = max(1, int(columns))
    rows = max(1, int(math.ceil(len(panels) / columns)))
    width = panel_width * columns
    height = panel_height * rows
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1.2}.grid{stroke:#dddddd;stroke-width:1}.tick{font-size:11px}.label{font-size:13px}.title{font-size:16px;font-weight:600}.legend{font-size:11px}</style>',
    ]
    for index, panel in enumerate(panels):
        origin_x = (index % columns) * panel_width
        origin_y = (index // columns) * panel_height
        elements.extend(_render_svg_panel(panel, origin_x, origin_y, panel_width, panel_height))
    elements.append("</svg>")
    Path(path).write_text("\n".join(elements), encoding="utf-8")


def _render_svg_panel(panel, origin_x, origin_y, panel_width, panel_height):
    left, right, top, bottom = 78, 22, 48, 62
    x0, y0 = origin_x + left, origin_y + top
    plot_width = panel_width - left - right
    plot_height = panel_height - top - bottom
    finite_points = [
        (x, y)
        for series in panel["series"]
        for x, y in zip(series["x"], series["y"])
        if np.isfinite(x) and np.isfinite(y)
    ]
    if not finite_points:
        finite_points = [(0.0, 0.0), (1.0, 1.0)]
    x_values = [point[0] for point in finite_points]
    y_values = [point[1] for point in finite_points]
    if panel.get("equal_axes"):
        common_min = min(min(x_values), min(y_values))
        common_max = max(max(x_values), max(y_values))
        x_min, x_max = common_min, common_max
        y_min, y_max = common_min, common_max
    else:
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
    x_min, x_max = _padded_limits(x_min, x_max)
    y_min, y_max = _padded_limits(y_min, y_max)

    def sx(value):
        return x0 + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value):
        return y0 + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    elements = [
        f'<text x="{origin_x + panel_width / 2:.1f}" y="{origin_y + 24}" text-anchor="middle" class="title">{html.escape(panel["title"])}</text>'
    ]
    for tick_index in range(5):
        fraction = tick_index / 4
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_min + fraction * (y_max - y_min)
        x_position = sx(x_value)
        y_position = sy(y_value)
        elements.extend(
            [
                f'<line x1="{x_position:.2f}" y1="{y0}" x2="{x_position:.2f}" y2="{y0 + plot_height}" class="grid"/>',
                f'<line x1="{x0}" y1="{y_position:.2f}" x2="{x0 + plot_width}" y2="{y_position:.2f}" class="grid"/>',
                f'<text x="{x_position:.2f}" y="{y0 + plot_height + 19}" text-anchor="middle" class="tick">{x_value:.3g}</text>',
                f'<text x="{x0 - 8}" y="{y_position + 4:.2f}" text-anchor="end" class="tick">{y_value:.3g}</text>',
            ]
        )
    elements.extend(
        [
            f'<line x1="{x0}" y1="{y0 + plot_height}" x2="{x0 + plot_width}" y2="{y0 + plot_height}" class="axis"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_height}" class="axis"/>',
            f'<text x="{x0 + plot_width / 2:.2f}" y="{origin_y + panel_height - 13}" text-anchor="middle" class="label">{html.escape(panel["xlabel"])}</text>',
            f'<text x="{origin_x + 17}" y="{y0 + plot_height / 2:.2f}" text-anchor="middle" transform="rotate(-90 {origin_x + 17} {y0 + plot_height / 2:.2f})" class="label">{html.escape(panel["ylabel"])}</text>',
        ]
    )
    legend_x = x0 + 8
    legend_y = y0 + 15
    for series_index, series in enumerate(panel["series"]):
        points = [
            (sx(x), sy(y), index)
            for index, (x, y) in enumerate(zip(series["x"], series["y"]))
            if np.isfinite(x) and np.isfinite(y)
        ]
        dash = ' stroke-dasharray="6,4"' if series["dashed"] else ""
        if len(points) >= 2 and not series["points_only"]:
            coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
            elements.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{series["color"]}" stroke-width="2"{dash}/>'
            )
        for x_position, y_position, source_index in points:
            if series["yerr"] is not None and source_index < len(series["yerr"]):
                error = series["yerr"][source_index]
                if np.isfinite(error) and error > 0:
                    upper = sy(series["y"][source_index] + error)
                    lower = sy(series["y"][source_index] - error)
                    elements.extend(
                        [
                            f'<line x1="{x_position:.2f}" y1="{upper:.2f}" x2="{x_position:.2f}" y2="{lower:.2f}" stroke="{series["color"]}"/>',
                            f'<line x1="{x_position - 4:.2f}" y1="{upper:.2f}" x2="{x_position + 4:.2f}" y2="{upper:.2f}" stroke="{series["color"]}"/>',
                            f'<line x1="{x_position - 4:.2f}" y1="{lower:.2f}" x2="{x_position + 4:.2f}" y2="{lower:.2f}" stroke="{series["color"]}"/>',
                        ]
                    )
            if series["points_only"] or not series["dashed"]:
                elements.append(
                    f'<circle cx="{x_position:.2f}" cy="{y_position:.2f}" r="3.2" fill="{series["color"]}"/>'
                )
        current_legend_y = legend_y + 16 * series_index
        elements.extend(
            [
                f'<line x1="{legend_x}" y1="{current_legend_y}" x2="{legend_x + 18}" y2="{current_legend_y}" stroke="{series["color"]}" stroke-width="2"{dash}/>',
                f'<text x="{legend_x + 23}" y="{current_legend_y + 4}" class="legend">{html.escape(series["label"])}</text>',
            ]
        )
    return elements


def _padded_limits(low, high):
    low = float(low)
    high = float(high)
    if abs(high - low) <= 1e-12:
        padding = max(0.5, abs(low) * 0.05)
    else:
        padding = 0.08 * (high - low)
    return low - padding, high + padding
