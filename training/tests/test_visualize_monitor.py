import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.visualize_monitor import main as visualize_main


class VisualizeMonitorTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(
            prefix="_tmp_visualize_monitor_",
            dir=Path(__file__).resolve().parent,
        )
        self.root = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_visualize_monitor_exports_vtk_without_history_json(self):
        monitor_path = self.root / "metrics" / "monitor_data.h5"
        output_dir = self.root / "vtk"
        _make_monitor_h5(monitor_path, self.root / "figures", node_count=60000)

        result = visualize_main(
            [
                "--monitor-data",
                str(monitor_path),
                "--output-dir",
                str(output_dir),
            ]
        )

        self.assertEqual(result, 0)
        for relative_path in (
            "epoch_temperature_residual_epoch_0001_frame_0007.vtk",
            "first_slice_temperature_residual_epoch_0001_after_slice_001.vtk",
        ):
            path = output_dir / relative_path
            self.assertTrue(path.exists(), relative_path)
            self.assertGreater(path.stat().st_size, 0, relative_path)
            text = path.read_text(encoding="ascii")
            self.assertIn("SCALARS temperature float 1", text)
            self.assertIn("SCALARS residual float 1", text)

    def test_visualize_monitor_exports_point_cloud_when_edge_index_is_missing(self):
        monitor_path = self.root / "large" / "metrics" / "monitor_data.h5"
        output_dir = self.root / "large" / "vtk"
        _make_monitor_h5(
            monitor_path,
            self.root / "large" / "figures",
            node_count=300001,
            include_slice=False,
            include_edges=False,
        )

        result = visualize_main(
            [
                "--monitor-data",
                str(monitor_path),
                "--output-dir",
                str(output_dir),
            ]
        )

        self.assertEqual(result, 0)
        output_path = output_dir / "epoch_temperature_residual_epoch_0001_frame_0007.vtk"
        self.assertTrue(output_path.exists())
        self.assertIn("VERTICES 300001 600002", output_path.read_text(encoding="ascii"))


def _make_monitor_h5(path: Path, output_dir: Path, *, node_count: int, include_slice: bool = True, include_edges: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = np.zeros((node_count, 3), dtype=np.float32)
    coords[:, 0] = np.linspace(0.0, 1.0, node_count, dtype=np.float32)
    coords[:, 1] = np.mod(np.arange(node_count, dtype=np.float32), 997.0) / 997.0
    coords[:, 2] = coords[:, 0] * 0.25
    residual = np.sin(coords[:, 0] * np.pi * 4.0).astype(np.float32)
    temperature = (300.0 + 20.0 * coords[:, 0] + 5.0 * coords[:, 1]).astype(np.float32)

    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["schema_version"] = "1.0"
        h5_file.attrs["figures_dir"] = str(output_dir)
        epoch_metrics = h5_file.create_group("epoch_metrics")
        epoch_metrics.create_dataset("epoch", data=np.array([0, 1], dtype=np.int64))
        for name, values in {
            "loss_total": [2.0, 1.0],
            "loss_pde": [1.5, 0.8],
            "loss_outflow": [0.4, 0.15],
            "loss_beta": [0.1, 0.05],
            "temperature_mean": [300.0, 305.0],
            "temperature_max": [310.0, 315.0],
            "temperature_min": [295.0, 298.0],
            "temperature_var": [2.0, 1.5],
        }.items():
            epoch_metrics.create_dataset(name, data=np.asarray(values, dtype=np.float64))

        slice_metrics = h5_file.create_group("slice_metrics")
        slice_metrics.create_dataset("epoch", data=np.array([0], dtype=np.int64))
        slice_metrics.create_dataset("slice_index", data=np.array([0], dtype=np.int64))
        for name in ("loss_total", "loss_pde", "loss_outflow", "loss_beta"):
            slice_metrics.create_dataset(name, data=np.array([1.0], dtype=np.float64))
        for name in ("temperature_mean", "temperature_max", "temperature_min", "temperature_var"):
            slice_metrics.create_dataset(name, data=np.array([300.0], dtype=np.float64))

        epoch_snapshots = h5_file.create_group("epoch_snapshots")
        _write_snapshot(
            epoch_snapshots,
            "epoch_0001",
            coords,
            residual,
            temperature,
            frame_index=7,
            include_edges=include_edges,
        )
        h5_file.create_group("slice_snapshots")
        if include_slice:
            _write_snapshot(
                h5_file["slice_snapshots"],
                "epoch_0001_slice_0001",
                coords,
                residual,
                temperature,
                frame_index=7,
                include_edges=include_edges,
            )


def _write_snapshot(parent, name, coords, residual, temperature, *, frame_index: int, include_edges: bool = True):
    group = parent.create_group(name)
    group.attrs["epoch"] = 0
    group.attrs["slice_index"] = 0
    group.attrs["kind"] = "epoch"
    group.attrs["frame_index"] = int(frame_index)
    group.create_dataset("coords", data=coords, chunks=(min(len(coords), 65536), 3), compression="gzip")
    group.create_dataset("residual", data=residual, chunks=(min(len(residual), 65536),), compression="gzip")
    group.create_dataset("temperature", data=temperature, chunks=(min(len(temperature), 65536),), compression="gzip")
    if include_edges:
        edge_index = np.vstack(
            [
                np.arange(max(len(coords) - 1, 0), dtype=np.int64),
                np.arange(1, len(coords), dtype=np.int64),
            ]
        )
        group.create_dataset("edge_index", data=edge_index, chunks=(2, min(edge_index.shape[1], 65536)), compression="gzip")


if __name__ == "__main__":
    unittest.main()
