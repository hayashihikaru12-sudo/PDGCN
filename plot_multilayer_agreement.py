from __future__ import annotations

import csv
import re
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm


DATA_DIR = Path("multilayer_batch")
OUT_DIR = Path("figures")
OUT_BASENAME = "multilayer_fem_inference_agreement"
PRED_DATASET = "prediction/pdgcn_multilayer/temperature"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def parse_case_label(path: Path) -> str:
    name = path.stem
    q = re.search(r"Q([0-9p]+)", name)
    v = re.search(r"V([0-9p]+)", name)
    f = re.search(r"F(\d+)", name)
    rot = re.search(r"rot(\d+deg)", name)
    parts = []
    if q:
        parts.append("Q=" + q.group(1).replace("p", "."))
    if v:
        parts.append("V=" + v.group(1).replace("p", "."))
    if f:
        parts.append("F=" + f.group(1))
    if rot:
        parts.append(rot.group(1))
    return ", ".join(parts)


def valid_values(fem: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = mask.astype(bool) & np.isfinite(fem) & np.isfinite(pred)
    return fem[valid], pred[valid]


def metrics(fem: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    y, yhat = valid_values(fem, pred, mask)
    diff = yhat - y
    ss_res = np.sum(diff**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if y.size > 1 and np.std(y) > 0 and np.std(yhat) > 0:
        r = float(np.corrcoef(y, yhat)[0, 1])
    else:
        r = np.nan
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "bias": float(np.mean(diff)),
        "r": r,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "n": int(y.size),
    }


def load_case(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        fem = f["fem/temperature"][...].squeeze(-1).astype(float)
        pred = f[PRED_DATASET][...].squeeze(-1).astype(float)
        mask = f["fem/valid_mask"][...].squeeze(-1).astype(bool)
        coords = f["multilayer/coordinates"][...].astype(float)
        time = f["fem/time"][...].astype(float)
    return {"fem": fem, "pred": pred, "mask": mask, "coords": coords, "time": time}


def collect_batch_metrics(
    paths: list[Path],
) -> tuple[list[dict[str, float | str | int]], np.ndarray, np.ndarray]:
    rows: list[dict[str, float | str | int]] = []
    per_layer_mae = []
    per_layer_peak_bias = []
    for path in paths:
        data = load_case(path)
        fem = data["fem"]
        pred = data["pred"]
        mask = data["mask"]
        overall = metrics(fem, pred, mask)
        final = metrics(fem[-1], pred[-1], mask[-1])
        top = metrics(fem[:, 0], pred[:, 0], mask[:, 0])
        fem_valid = np.where(mask, fem, np.nan)
        pred_valid = np.where(mask, pred, np.nan)
        row: dict[str, float | str | int] = {
            "case": path.name,
            "frames": fem.shape[0],
            "layers": fem.shape[1],
            "nodes": fem.shape[2],
            "mae_all_C": overall["mae"],
            "rmse_all_C": overall["rmse"],
            "bias_all_C": overall["bias"],
            "r_all": overall["r"],
            "r2_all": overall["r2"],
            "mae_final_C": final["mae"],
            "rmse_final_C": final["rmse"],
            "bias_final_C": final["bias"],
            "r_final": final["r"],
            "r2_final": final["r2"],
            "global_t_peak_fem_C": float(np.nanmax(fem_valid)),
            "global_t_peak_prediction_C": float(np.nanmax(pred_valid)),
            "global_t_peak_signed_error_C": float(
                np.nanmax(pred_valid) - np.nanmax(fem_valid)
            ),
            "global_t_peak_abs_error_C": float(
                abs(np.nanmax(pred_valid) - np.nanmax(fem_valid))
            ),
            "top_layer_r2_all": top["r2"],
        }
        layer_mae = []
        layer_peak_bias = []
        for layer_idx in range(fem.shape[1]):
            m = metrics(fem[:, layer_idx], pred[:, layer_idx], mask[:, layer_idx])
            fem_peak = float(np.nanmax(fem_valid[:, layer_idx]))
            pred_peak = float(np.nanmax(pred_valid[:, layer_idx]))
            peak_bias = pred_peak - fem_peak
            row[f"mae_layer_{layer_idx + 1}_C"] = m["mae"]
            row[f"t_peak_fem_layer_{layer_idx + 1}_C"] = fem_peak
            row[f"t_peak_prediction_layer_{layer_idx + 1}_C"] = pred_peak
            row[f"t_peak_signed_error_layer_{layer_idx + 1}_C"] = peak_bias
            row[f"t_peak_abs_error_layer_{layer_idx + 1}_C"] = abs(peak_bias)
            layer_mae.append(m["mae"])
            layer_peak_bias.append(peak_bias)
        rows.append(row)
        per_layer_mae.append(layer_mae)
        per_layer_peak_bias.append(layer_peak_bias)
    return (
        rows,
        np.asarray(per_layer_mae, dtype=float),
        np.asarray(per_layer_peak_bias, dtype=float),
    )


def save_metrics_csv(rows: list[dict[str, float | str | int]], output: Path) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_triangulation(x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> mtri.Triangulation:
    tri = mtri.Triangulation(x, y)
    invalid_tri = ~np.all(valid[tri.triangles], axis=1)
    if np.any(invalid_tri):
        tri.set_mask(invalid_tri)
    return tri


def draw_field(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    *,
    cmap: str,
    levels: np.ndarray,
    norm=None,
) -> None:
    tri = make_triangulation(x, y, valid)
    z = np.ma.masked_where(~valid, values)
    ax.tricontourf(tri, z, levels=levels, cmap=cmap, norm=norm, extend="both")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    paths = sorted(DATA_DIR.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No .h5 files found in {DATA_DIR}")

    OUT_DIR.mkdir(exist_ok=True)
    rows, per_layer_mae, per_layer_peak_bias = collect_batch_metrics(paths)
    metrics_csv = OUT_DIR / f"{OUT_BASENAME}_metrics.csv"
    save_metrics_csv(rows, metrics_csv)

    median_mae = np.median([float(r["mae_all_C"]) for r in rows])
    rep_idx = int(np.argmin([abs(float(r["mae_all_C"]) - median_mae) for r in rows]))
    rep_path = paths[rep_idx]
    rep = load_case(rep_path)
    fem = rep["fem"]
    pred = rep["pred"]
    mask = rep["mask"]
    coords = rep["coords"]
    time = rep["time"]

    t_idx = fem.shape[0] - 1
    layer_indices = [0, fem.shape[1] // 2, fem.shape[1] - 1]
    layer_names = ["Top layer", f"Layer {layer_indices[1] + 1}", "Bottom layer"]

    field_stack = np.concatenate(
        [fem[t_idx, layer_indices].ravel(), pred[t_idx, layer_indices].ravel()]
    )
    temp_min = np.nanpercentile(field_stack, 1)
    temp_max = np.nanpercentile(field_stack, 99.5)
    temp_levels = np.linspace(temp_min, temp_max, 18)
    diff_stack = (pred[t_idx, layer_indices] - fem[t_idx, layer_indices]).ravel()
    err_lim = max(1.0, float(np.nanpercentile(np.abs(diff_stack), 99)))
    err_levels = np.linspace(-err_lim, err_lim, 17)
    err_norm = TwoSlopeNorm(vmin=-err_lim, vcenter=0.0, vmax=err_lim)

    fig = plt.figure(figsize=(7.8, 6.35), constrained_layout=False)
    outer = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[3.1, 1.55],
        wspace=0.2,
        left=0.055,
        right=0.985,
        top=0.93,
        bottom=0.075,
    )
    field_grid = gridspec.GridSpecFromSubplotSpec(
        3, 3, subplot_spec=outer[0], hspace=0.05, wspace=0.05
    )
    stat_grid = gridspec.GridSpecFromSubplotSpec(
        4,
        1,
        subplot_spec=outer[1],
        height_ratios=[0.72, 1.0, 1.0, 1.28],
        hspace=0.56,
    )

    image_axes = []
    for row_i, layer_idx in enumerate(layer_indices):
        valid = mask[t_idx, layer_idx]
        x = coords[t_idx, layer_idx, :, 0] * 1000.0
        y = coords[t_idx, layer_idx, :, 1] * 1000.0
        panels = [
            (fem[t_idx, layer_idx], "FEM", "inferno", temp_levels, None),
            (pred[t_idx, layer_idx], "Inference", "inferno", temp_levels, None),
            (
                pred[t_idx, layer_idx] - fem[t_idx, layer_idx],
                "Inference - FEM",
                "coolwarm",
                err_levels,
                err_norm,
            ),
        ]
        for col_i, (values, title, cmap, levels, norm) in enumerate(panels):
            ax = fig.add_subplot(field_grid[row_i, col_i])
            draw_field(ax, x, y, values, valid, cmap=cmap, levels=levels, norm=norm)
            if row_i == 0:
                ax.set_title(title, pad=4, fontsize=8)
            if col_i == 0:
                ax.text(
                    -0.06,
                    0.5,
                    layer_names[row_i],
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=7,
                    color="#333333",
                )
            image_axes.append(ax)

    cax_temp = fig.add_axes([0.102, 0.045, 0.255, 0.012])
    cbar_temp = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=temp_levels[0], vmax=temp_levels[-1]),
            cmap="inferno",
        ),
        cax=cax_temp,
        orientation="horizontal",
    )
    cbar_temp.set_label("Temperature (deg C)", labelpad=1)
    cbar_temp.ax.tick_params(length=2, pad=1)

    cax_err = fig.add_axes([0.428, 0.045, 0.18, 0.012])
    cbar_err = fig.colorbar(
        mpl.cm.ScalarMappable(norm=err_norm, cmap="coolwarm"),
        cax=cax_err,
        orientation="horizontal",
    )
    cbar_err.set_label("Error (deg C)", labelpad=1)
    cbar_err.ax.tick_params(length=2, pad=1)

    ax_text = fig.add_subplot(stat_grid[0])
    ax_text.axis("off")
    overall = metrics(fem, pred, mask)
    final = metrics(fem[t_idx], pred[t_idx], mask[t_idx])
    ax_text.text(
        0,
        1.0,
        "Representative case",
        transform=ax_text.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
    )
    ax_text.text(
        0,
        0.72,
        parse_case_label(rep_path),
        transform=ax_text.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="#333333",
    )
    ax_text.text(
        0,
        0.45,
        f"Final frame: t={time[t_idx]:.2f} s\n"
        f"MAE={final['mae']:.2f} deg C, RMSE={final['rmse']:.2f} deg C\n"
        f"All frames: r={overall['r']:.3f}, R2={overall['r2']:.3f}\n"
        f"Batch: {len(paths)} cases; compared against physical inference temperature.",
        transform=ax_text.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        linespacing=1.35,
        color="#333333",
    )

    ax_box = fig.add_subplot(stat_grid[1])
    positions = np.arange(1, per_layer_mae.shape[1] + 1)
    box = ax_box.boxplot(
        [per_layer_mae[:, i] for i in range(per_layer_mae.shape[1])],
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.0},
        whiskerprops={"color": "#666666", "linewidth": 0.8},
        capprops={"color": "#666666", "linewidth": 0.8},
        boxprops={"color": "#666666", "linewidth": 0.8},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#d7e6ef")
    ax_box.plot(
        positions,
        np.median(per_layer_mae, axis=0),
        color="#2b6f8a",
        linewidth=1.0,
        marker="o",
        markersize=2.2,
    )
    ax_box.set_title("Batch layer-wise MAE", fontsize=8, pad=4)
    ax_box.set_xlabel("Layer index")
    ax_box.set_ylabel("MAE (deg C)")
    ax_box.set_xlim(0.4, per_layer_mae.shape[1] + 0.6)
    ax_box.grid(axis="y", color="#e8e8e8", linewidth=0.6)

    ax_peak_box = fig.add_subplot(stat_grid[2])
    peak_box = ax_peak_box.boxplot(
        [per_layer_peak_bias[:, i] for i in range(per_layer_peak_bias.shape[1])],
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.0},
        whiskerprops={"color": "#666666", "linewidth": 0.8},
        capprops={"color": "#666666", "linewidth": 0.8},
        boxprops={"color": "#666666", "linewidth": 0.8},
    )
    for patch in peak_box["boxes"]:
        patch.set_facecolor("#f1dccb")
    ax_peak_box.plot(
        positions,
        np.median(per_layer_peak_bias, axis=0),
        color="#c4471a",
        linewidth=1.0,
        marker="o",
        markersize=2.2,
    )
    peak_lim = max(1.0, float(np.nanpercentile(np.abs(per_layer_peak_bias), 98)))
    ax_peak_box.axhline(0.0, color="#555555", linewidth=0.8, zorder=0)
    ax_peak_box.set_ylim(-1.08 * peak_lim, 1.08 * peak_lim)
    ax_peak_box.set_title("Batch layer-wise signed T_peak bias", fontsize=8, pad=4)
    ax_peak_box.set_xlabel("Layer index")
    ax_peak_box.set_ylabel("Inference - FEM (deg C)")
    ax_peak_box.set_xlim(0.4, per_layer_peak_bias.shape[1] + 0.6)
    ax_peak_box.grid(axis="y", color="#e8e8e8", linewidth=0.6)

    ax_scatter = fig.add_subplot(stat_grid[3])
    peak_fem = np.asarray([float(r["global_t_peak_fem_C"]) for r in rows])
    peak_pred = np.asarray([float(r["global_t_peak_prediction_C"]) for r in rows])
    top_r2 = np.asarray([float(r["top_layer_r2_all"]) for r in rows])
    fit_slope, fit_intercept = np.polyfit(peak_fem, peak_pred, 1)
    margin = 0.04 * (max(peak_fem.max(), peak_pred.max()) - min(peak_fem.min(), peak_pred.min()))
    low = min(peak_fem.min(), peak_pred.min()) - margin
    high = max(peak_fem.max(), peak_pred.max()) + margin
    ax_scatter.plot([low, high], [low, high], color="#5f5f5f", linewidth=1.0, label="1:1")
    ax_scatter.plot(
        [low, high],
        [fit_slope * low + fit_intercept, fit_slope * high + fit_intercept],
        color="#c4471a",
        linewidth=1.2,
        label=f"fit: y={fit_slope:.2f}x{fit_intercept:+.1f}",
    )
    scatter = ax_scatter.scatter(
        peak_fem,
        peak_pred,
        c=top_r2,
        s=24,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.25,
        zorder=3,
    )
    ax_scatter.set_xlim(low, high)
    ax_scatter.set_ylim(low, high)
    ax_scatter.set_title("Global T_peak agreement", fontsize=8, pad=4)
    ax_scatter.set_xlabel("FEM global T_peak (deg C)")
    ax_scatter.set_ylabel("Inference global T_peak (deg C)")
    ax_scatter.grid(color="#e8e8e8", linewidth=0.6)
    ax_scatter.legend(loc="upper left", handlelength=1.7, borderaxespad=0.3)
    cbar = fig.colorbar(scatter, ax=ax_scatter, pad=0.02, fraction=0.08)
    cbar.set_label("Top-layer R2")
    cbar.ax.tick_params(length=2, pad=1)
    ax_scatter.text(
        0.98,
        0.04,
        f"n={len(paths)} cases",
        transform=ax_scatter.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
    )

    fig.suptitle(
        "Multilayer FEM and inference temperature fields agree across layers",
        x=0.055,
        y=0.982,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )

    out = OUT_DIR / OUT_BASENAME
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=240, bbox_inches="tight")
    print(f"Saved {out.with_suffix('.svg')}")
    print(f"Saved {out.with_suffix('.pdf')}")
    print(f"Saved {out.with_suffix('.tiff')}")
    print(f"Saved {out.with_suffix('.png')}")
    print(f"Saved {metrics_csv}")


if __name__ == "__main__":
    main()
