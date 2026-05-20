import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from visualization import write_polydata_vtk


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MONITOR_DATA = REPO_ROOT / "runs" / "pdgcn" / "metrics" / "monitor_data.h5"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "vtk"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export PD-GCN monitor snapshots to ParaView VTK files.")
    parser.add_argument(
        "--monitor-data",
        default=str(DEFAULT_MONITOR_DATA),
        help=f"Path to metrics/monitor_data.h5. Defaults to {DEFAULT_MONITOR_DATA}.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"VTK output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument("--grid-resolution", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--projection", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--temperature-cmap", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    export_monitor_vtk(
        args.monitor_data,
        output_dir=Path(args.output_dir),
    )
    return 0


def export_monitor_vtk(monitor_path, *, output_dir=None):
    monitor_path = Path(monitor_path)
    if not monitor_path.exists():
        raise FileNotFoundError(f"monitor data not found: {monitor_path}")

    with h5py.File(monitor_path, "r") as h5_file:
        vtk_dir = Path(output_dir) if output_dir is not None else _default_output_dir(monitor_path, h5_file)
        vtk_dir.mkdir(parents=True, exist_ok=True)

        if "epoch_snapshots" in h5_file:
            for name in sorted(h5_file["epoch_snapshots"].keys()):
                group = h5_file["epoch_snapshots"][name]
                epoch_number = int(group.attrs.get("epoch", 0)) + 1
                frame_index = int(group.attrs.get("frame_index", 0))
                _write_snapshot_vtk(
                    group,
                    vtk_dir / f"epoch_temperature_residual_epoch_{epoch_number:04d}_frame_{frame_index:04d}.vtk",
                    title=f"PDGCN epoch {epoch_number} frame {frame_index}",
                )

        if "slice_snapshots" in h5_file:
            for name in sorted(h5_file["slice_snapshots"].keys()):
                group = h5_file["slice_snapshots"][name]
                epoch_number = int(group.attrs.get("epoch", 0)) + 1
                slice_number = int(group.attrs.get("slice_index", 0)) + 1
                _write_snapshot_vtk(
                    group,
                    vtk_dir / f"first_slice_temperature_residual_epoch_{epoch_number:04d}_after_slice_{slice_number:03d}.vtk",
                    title=f"PDGCN first slice epoch {epoch_number} after slice {slice_number}",
                )

    return vtk_dir


def _default_output_dir(monitor_path: Path, h5_file) -> Path:
    attr_value = h5_file.attrs.get("figures_dir")
    if attr_value:
        return Path(str(attr_value)) / "vtk"
    if monitor_path.parent.name == "metrics":
        return monitor_path.parent.parent / "vtk"
    return monitor_path.parent / "vtk"


def _write_snapshot_vtk(snapshot_group, output_path: Path, *, title: str):
    coords = np.asarray(snapshot_group["coords"], dtype=np.float32)
    point_data = {
        "temperature": np.asarray(snapshot_group["temperature"], dtype=np.float32).reshape(-1),
        "residual": np.asarray(snapshot_group["residual"], dtype=np.float32).reshape(-1),
    }
    edge_index = np.asarray(snapshot_group["edge_index"], dtype=np.int64) if "edge_index" in snapshot_group else None
    write_polydata_vtk(
        output_path,
        coords,
        edge_index=edge_index,
        point_data=point_data,
        title=title,
    )


if __name__ == "__main__":
    raise SystemExit(main())
