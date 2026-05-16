import argparse
import os
import sys
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []


def _add_conda_dll_directories():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env_prefix = Path(sys.executable).resolve().parent
    candidates = (env_prefix / "Library" / "bin", env_prefix / "DLLs")
    for directory in candidates:
        if not directory.exists():
            continue
        if hasattr(os, "add_dll_directory"):
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")


_add_conda_dll_directories()

import h5py
import numpy as np


LOSS_FIELDS = ("loss_total", "loss_pde", "loss_outflow", "loss_beta")
TEMP_FIELDS = ("temperature_mean", "temperature_max", "temperature_min", "temperature_var")
PROJECTION_AXES = {
    "xy": (0, 1, "x*", "y*"),
    "xz": (0, 2, "x*", "z*"),
    "yz": (1, 2, "y*", "z*"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render PD-GCN monitor figures from monitor_data.h5.")
    parser.add_argument("--monitor-data", required=True, help="Path to metrics/monitor_data.h5.")
    parser.add_argument("--output-dir", default=None, help="Figure output directory. Defaults to HDF5 attrs or run/figures.")
    parser.add_argument("--grid-resolution", type=int, default=1024, help="Raster aggregation grid resolution.")
    parser.add_argument("--projection", choices=sorted(PROJECTION_AXES), default="xy", help="2D projection for snapshots.")
    parser.add_argument("--temperature-cmap", default="viridis", help="Matplotlib colormap for temperature.")
    args = parser.parse_args(argv)

    monitor_path = Path(args.monitor_data)
    if int(args.grid_resolution) <= 0:
        raise ValueError(f"grid-resolution must be positive, got {args.grid_resolution}.")
    output_dir = Path(args.output_dir) if args.output_dir else None

    render_monitor_figures(
        monitor_path,
        output_dir=output_dir,
        grid_resolution=int(args.grid_resolution),
        projection=args.projection,
        temperature_cmap=args.temperature_cmap,
    )
    return 0


def render_monitor_figures(
    monitor_path,
    *,
    output_dir=None,
    grid_resolution: int = 1024,
    projection: str = "xy",
    temperature_cmap: str = "viridis",
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monitor_path = Path(monitor_path)
    if not monitor_path.exists():
        raise FileNotFoundError(f"monitor data not found: {monitor_path}")
    if projection not in PROJECTION_AXES:
        raise ValueError(f"Unsupported projection: {projection}.")

    with h5py.File(monitor_path, "r") as h5_file:
        figure_dir = Path(output_dir) if output_dir is not None else _default_output_dir(monitor_path, h5_file)
        figure_dir.mkdir(parents=True, exist_ok=True)

        if "epoch_metrics" in h5_file:
            epoch_metrics = h5_file["epoch_metrics"]
            if len(epoch_metrics["epoch"]) > 0:
                _plot_loss_curve(plt, epoch_metrics, figure_dir / "loss_curve.png")
                _plot_temperature_stats(plt, epoch_metrics, figure_dir / "temperature_stats.png")

        if "slice_metrics" in h5_file:
            slice_metrics = h5_file["slice_metrics"]
            if len(slice_metrics["epoch"]) > 0:
                _plot_slice_loss_curve(plt, slice_metrics, figure_dir / "first_slice_loss_curve.png")

        if "epoch_snapshots" in h5_file:
            for name in sorted(h5_file["epoch_snapshots"].keys()):
                group = h5_file["epoch_snapshots"][name]
                epoch_number = int(group.attrs.get("epoch", 0)) + 1
                frame_index = int(group.attrs.get("frame_index", 0))
                _plot_snapshot(
                    plt,
                    group,
                    value_key="residual",
                    output_path=figure_dir / f"residual_epoch_{epoch_number:04d}_frame_{frame_index:04d}.png",
                    title=f"PDE residual epoch {epoch_number}, frame {frame_index}",
                    colorbar_label="PDE residual",
                    cmap="RdBu_r",
                    centered=True,
                    grid_resolution=grid_resolution,
                    projection=projection,
                )
                _plot_snapshot(
                    plt,
                    group,
                    value_key="temperature",
                    output_path=figure_dir / f"temperature_epoch_{epoch_number:04d}_frame_{frame_index:04d}.png",
                    title=f"Temperature epoch {epoch_number}, frame {frame_index}",
                    colorbar_label="Temperature",
                    cmap=temperature_cmap,
                    centered=False,
                    grid_resolution=grid_resolution,
                    projection=projection,
                )

        if "slice_snapshots" in h5_file:
            first_slice_dir = figure_dir / "first_slice"
            for name in sorted(h5_file["slice_snapshots"].keys()):
                group = h5_file["slice_snapshots"][name]
                epoch_number = int(group.attrs.get("epoch", 0)) + 1
                slice_number = int(group.attrs.get("slice_index", 0)) + 1
                _plot_snapshot(
                    plt,
                    group,
                    value_key="residual",
                    output_path=first_slice_dir / f"residual_epoch_{epoch_number:04d}_after_slice_{slice_number:03d}.png",
                    title=f"First slice residual epoch {epoch_number}, after slice {slice_number}",
                    colorbar_label="PDE residual",
                    cmap="RdBu_r",
                    centered=True,
                    grid_resolution=grid_resolution,
                    projection=projection,
                )
                _plot_snapshot(
                    plt,
                    group,
                    value_key="temperature",
                    output_path=first_slice_dir / f"temperature_epoch_{epoch_number:04d}_after_slice_{slice_number:03d}.png",
                    title=f"First slice temperature epoch {epoch_number}, after slice {slice_number}",
                    colorbar_label="Temperature",
                    cmap=temperature_cmap,
                    centered=False,
                    grid_resolution=grid_resolution,
                    projection=projection,
                )

    return figure_dir


def _default_output_dir(monitor_path: Path, h5_file) -> Path:
    attr_value = h5_file.attrs.get("figures_dir")
    if attr_value:
        return Path(str(attr_value))
    if monitor_path.parent.name == "metrics":
        return monitor_path.parent.parent / "figures"
    return monitor_path.parent / "figures"


def _plot_loss_curve(plt, metrics_group, output_path: Path):
    epochs = _read_metric(metrics_group, "epoch") + 1
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=160)
    for field in LOSS_FIELDS:
        if field in metrics_group:
            values = _read_metric(metrics_group, field)
            if np.isfinite(values).any():
                ax.plot(epochs, values, marker="o", linewidth=1.8, markersize=3, label=field)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss components")
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def _plot_slice_loss_curve(plt, metrics_group, output_path: Path):
    steps = np.arange(1, len(metrics_group["epoch"]) + 1)
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=160)
    for field in LOSS_FIELDS:
        if field in metrics_group:
            values = _read_metric(metrics_group, field)
            if np.isfinite(values).any():
                ax.plot(steps, values, marker="o", linewidth=1.8, markersize=3, label=field)
    ax.set_xlabel("Slice evaluation")
    ax.set_ylabel("Loss")
    ax.set_title("First-slice evaluation loss")
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def _plot_temperature_stats(plt, metrics_group, output_path: Path):
    epochs = _read_metric(metrics_group, "epoch") + 1
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=160)
    labels = {
        "temperature_mean": "mean",
        "temperature_max": "max",
        "temperature_min": "min",
        "temperature_var": "var",
    }
    for field in TEMP_FIELDS:
        if field in metrics_group:
            values = _read_metric(metrics_group, field)
            if np.isfinite(values).any():
                ax.plot(epochs, values, marker="o", linewidth=1.8, markersize=3, label=labels[field])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Temperature")
    ax.set_title("Temperature statistics")
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def _plot_snapshot(
    plt,
    snapshot_group,
    *,
    value_key: str,
    output_path: Path,
    title: str,
    colorbar_label: str,
    cmap: str,
    centered: bool,
    grid_resolution: int,
    projection: str,
):
    from matplotlib.colors import TwoSlopeNorm

    n_nodes = int(snapshot_group[value_key].shape[0])
    axis_x, axis_y, label_x, label_y = PROJECTION_AXES[projection]
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=180)

    if n_nodes <= 50000:
        coords = np.asarray(snapshot_group["coords"], dtype=np.float32)
        values = np.asarray(snapshot_group[value_key], dtype=np.float32).reshape(-1)
        x = coords[:, axis_x]
        y = coords[:, axis_y]
        vmin, vmax = _robust_limits(values, centered=centered)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax) if centered else None
        artist = ax.scatter(
            x,
            y,
            c=values,
            s=5,
            cmap=cmap,
            norm=norm,
            vmin=None if centered else vmin,
            vmax=None if centered else vmax,
            linewidths=0,
            rasterized=True,
        )
    else:
        grid, extent = _aggregate_snapshot(
            snapshot_group["coords"],
            snapshot_group[value_key],
            grid_resolution=grid_resolution,
            axis_x=axis_x,
            axis_y=axis_y,
            stream=n_nodes > 300000,
        )
        vmin, vmax = _robust_limits(grid, centered=centered)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax) if centered else None
        artist = ax.imshow(
            grid,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            vmin=None if centered else vmin,
            vmax=None if centered else vmax,
            interpolation="nearest",
            aspect="auto",
        )

    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    colorbar = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    _save_figure(fig, output_path)


def _aggregate_snapshot(coords_dataset, values_dataset, *, grid_resolution: int, axis_x: int, axis_y: int, stream: bool):
    if stream:
        x_min, x_max, y_min, y_max = _stream_bounds(coords_dataset, axis_x=axis_x, axis_y=axis_y)
        chunks = _iter_dataset_chunks(coords_dataset, values_dataset)
    else:
        coords = np.asarray(coords_dataset, dtype=np.float32)
        values = np.asarray(values_dataset, dtype=np.float32).reshape(-1)
        x_min, x_max = _expanded_range(coords[:, axis_x])
        y_min, y_max = _expanded_range(coords[:, axis_y])
        chunks = ((coords, values),)

    sum_grid = np.zeros((grid_resolution, grid_resolution), dtype=np.float64)
    count_grid = np.zeros((grid_resolution, grid_resolution), dtype=np.int64)
    for coords, values in chunks:
        x = coords[:, axis_x]
        y = coords[:, axis_y]
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
        if not np.any(valid):
            continue
        ix = np.floor((x[valid] - x_min) / (x_max - x_min) * (grid_resolution - 1)).astype(np.int64)
        iy = np.floor((y[valid] - y_min) / (y_max - y_min) * (grid_resolution - 1)).astype(np.int64)
        ix = np.clip(ix, 0, grid_resolution - 1)
        iy = np.clip(iy, 0, grid_resolution - 1)
        np.add.at(sum_grid, (iy, ix), values[valid])
        np.add.at(count_grid, (iy, ix), 1)

    grid = np.full_like(sum_grid, np.nan, dtype=np.float64)
    filled = count_grid > 0
    grid[filled] = sum_grid[filled] / count_grid[filled]
    return grid, (x_min, x_max, y_min, y_max)


def _stream_bounds(coords_dataset, *, axis_x: int, axis_y: int):
    x_min = y_min = np.inf
    x_max = y_max = -np.inf
    for start in range(0, coords_dataset.shape[0], _chunk_size(coords_dataset)):
        stop = min(start + _chunk_size(coords_dataset), coords_dataset.shape[0])
        coords = np.asarray(coords_dataset[start:stop], dtype=np.float32)
        x = coords[:, axis_x]
        y = coords[:, axis_y]
        x_min = min(x_min, float(np.nanmin(x)))
        x_max = max(x_max, float(np.nanmax(x)))
        y_min = min(y_min, float(np.nanmin(y)))
        y_max = max(y_max, float(np.nanmax(y)))
    x_min, x_max = _expand_pair(x_min, x_max)
    y_min, y_max = _expand_pair(y_min, y_max)
    return x_min, x_max, y_min, y_max


def _iter_dataset_chunks(coords_dataset, values_dataset):
    chunk_size = _chunk_size(coords_dataset)
    for start in range(0, coords_dataset.shape[0], chunk_size):
        stop = min(start + chunk_size, coords_dataset.shape[0])
        yield (
            np.asarray(coords_dataset[start:stop], dtype=np.float32),
            np.asarray(values_dataset[start:stop], dtype=np.float32).reshape(-1),
        )


def _chunk_size(dataset) -> int:
    if dataset.chunks:
        return int(dataset.chunks[0])
    return min(int(dataset.shape[0]), 65536)


def _robust_limits(values, *, centered: bool):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0) if centered else (0.0, 1.0)
    low, high = np.nanpercentile(finite, [2.0, 98.0])
    if centered:
        bound = max(abs(float(low)), abs(float(high)))
        if bound == 0.0:
            bound = 1.0
        return -bound, bound
    return _expand_pair(float(low), float(high))


def _expanded_range(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    return _expand_pair(float(np.min(finite)), float(np.max(finite)))


def _expand_pair(low: float, high: float):
    if not np.isfinite(low) or not np.isfinite(high):
        return 0.0, 1.0
    if low == high:
        margin = abs(low) * 0.05 + 1.0
        return low - margin, high + margin
    margin = (high - low) * 0.02
    return low - margin, high + margin


def _read_metric(metrics_group, name: str):
    return np.asarray(metrics_group[name], dtype=np.float64)


def _save_figure(fig, output_path: Path):
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.12, top=0.9)
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
