import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from data import ScaleParams
from training import load_run_config, pdgcn_config_from_scale
from training.run_config import derive_dt_star
from training.train_entry import derive_timing_from_hdf5, discover_hdf5_files, run_training_from_config


def make_h5(path: Path):
    """创建一份用于端到端入口测试的小型固定拓扑 HDF5。"""

    xyz = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [0.1, 1.0, 0.0], [1.1, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    fiber = np.tile(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32), (2, 4, 1))
    q = np.array([[[0.0], [1.0], [0.5], [0.0]], [[0.0], [0.8], [0.4], [0.0]]], dtype=np.float32)
    edge_index = np.array([[0, 1, 2, 0], [1, 3, 3, 2]], dtype=np.int64)

    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["velocity_speed"] = 2.0
        dynamic = h5_file.create_group("dynamic")
        dynamic.create_dataset("xyz", data=xyz)
        dynamic.create_dataset("fiber", data=fiber)
        dynamic.create_dataset("Q", data=q)
        h5_file.create_dataset("edge_index", data=edge_index)
        boundary = h5_file.create_group("boundary_nodes")
        boundary.create_dataset("upwind", data=np.array([0], dtype=np.int64))
        boundary.create_dataset("downwind", data=np.array([3], dtype=np.int64))
        boundary.create_dataset("side", data=np.array([2], dtype=np.int64))
        path_group = h5_file.create_group("path")
        path_group.create_dataset("heat_center_step_distance", data=np.float64(0.5))
        path_group.create_dataset("slice_path_length", data=np.float64(0.5))


class RunConfigTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("training/tests/_tmp_run_config")
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_pdgcn_config_from_scale_derives_physics_coefficients(self):
        scale = ScaleParams(
            L0=2.0,
            v0=4.0,
            T_amb=300.0,
            delta_T0=10.0,
            Q0=5.0,
            K0=8.0,
            rho=2.0,
            Cp=1.0,
        )

        config = pdgcn_config_from_scale(
            scale,
            dt=0.5,
            model_overrides={
                "hidden_size": 8,
                "inverse_pe": 999.0,
                "pi_q": 999.0,
                "dt_star": 999.0,
                "thermal_loss_beta": 0.25,
                "thermal_loss_base_temperature_star": 0.0,
                "residual_time_scheme": "backward",
            },
        )

        self.assertEqual(config.hidden_size, 8)
        self.assertAlmostEqual(config.inverse_pe, 0.5)
        self.assertAlmostEqual(config.pi_q, 0.125)
        self.assertAlmostEqual(config.dt_star, 1.0)
        self.assertAlmostEqual(config.thermal_loss_beta, 0.25)
        self.assertAlmostEqual(config.thermal_loss_base_temperature_star, 0.0)
        self.assertEqual(config.residual_time_scheme, "backward")

    def test_pdgcn_config_from_scale_requires_physics_inputs_and_dt(self):
        scale = ScaleParams(L0=2.0, v0=4.0, T_amb=300.0, delta_T0=10.0, Q0=5.0)

        with self.assertRaises(ValueError):
            pdgcn_config_from_scale(scale, dt=0.5)
        with self.assertRaises(ValueError):
            derive_dt_star(scale, 0.0)

    def test_load_classified_run_config_groups_dataset_and_hyperparameters(self):
        config_path = self.root / "classified.json"
        payload = {
            "outputs": {
                "checkpoint_path": "checkpoint.pt",
                "history_path": "history.json",
            },
            "datasets": [
                {
                    "name": "case_a",
                    "h5_dir": "h5",
                    "cache_dir": "cache/case_a",
                    "scale": {
                        "L0": 2.0,
                        "v0": 2.0,
                        "T_amb": 300.0,
                        "delta_T0": 10.0,
                        "Q0": 2.0,
                        "K0": 8.0,
                        "rho": 2.0,
                        "Cp": 1.0,
                    },
                }
            ],
            "hyperparameters": {
                "model": {
                    "hidden_size": 8,
                    "message_passing_num": 1,
                },
                "physics_loss": {
                    "lambda_outflow": 0.0,
                    "thermal_loss_beta": 0.25,
                    "residual_time_scheme": "backward",
                },
                "training": {
                    "lr": 0.001,
                    "epochs": 3,
                    "tbptt_window": 2,
                    "warmup_steps": 1,
                    "device": "cpu",
                },
            },
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        config = load_run_config(config_path)

        self.assertEqual(config.schema, "classified")
        self.assertEqual(config.outputs.checkpoint_path, "checkpoint.pt")
        self.assertEqual(len(config.datasets), 1)
        self.assertEqual(config.datasets[0].name, "case_a")
        self.assertEqual(config.data.h5_dir, "h5")
        self.assertEqual(config.data.cache_dir, "cache/case_a")
        self.assertEqual(config.data.checkpoint_path, "checkpoint.pt")
        self.assertEqual(config.scale.L0, 2.0)
        self.assertEqual(config.model["hidden_size"], 8)
        self.assertEqual(config.model["lambda_outflow"], 0.0)
        self.assertEqual(config.model["residual_time_scheme"], "backward")
        self.assertEqual(config.training.device, "cpu")

    def test_load_config_rejects_manual_dt_in_scale(self):
        config_path = self.root / "manual_dt.json"
        payload = {
            "outputs": {
                "checkpoint_path": "checkpoint.pt",
                "history_path": "history.json",
            },
            "datasets": [
                {
                    "h5_dir": "h5",
                    "cache_dir": "cache",
                    "scale": {
                        "L0": 2.0,
                        "v0": 2.0,
                        "T_amb": 300.0,
                        "delta_T0": 10.0,
                        "Q0": 2.0,
                        "K0": 8.0,
                        "rho": 2.0,
                        "Cp": 1.0,
                        "dt": 0.5,
                    },
                }
            ],
            "hyperparameters": {
                "model": {},
                "physics_loss": {},
                "training": {"lr": 0.001, "epochs": 1, "tbptt_window": 1},
            },
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Unknown keys.*dt"):
            load_run_config(config_path)

    def test_derive_timing_from_hdf5_scalar_step_distance(self):
        h5_path = self.root / "input.h5"
        make_h5(h5_path)
        scale = ScaleParams(L0=2.0, v0=2.0, T_amb=300.0, delta_T0=10.0, Q0=2.0)

        timing = derive_timing_from_hdf5(h5_path, scale)

        self.assertAlmostEqual(timing["step_distance"], 0.5)
        self.assertAlmostEqual(timing["velocity_speed"], 2.0)
        self.assertAlmostEqual(timing["dt"], 0.25)
        self.assertAlmostEqual(timing["dt_star"], 0.25)
        self.assertAlmostEqual(timing["slice_path_length"], 0.5)

    def test_derive_timing_from_hdf5_accepts_constant_step_array(self):
        h5_path = self.root / "input_array.h5"
        make_h5(h5_path)
        with h5py.File(h5_path, "a") as h5_file:
            del h5_file["path/heat_center_step_distance"]
            h5_file["path"].create_dataset("heat_center_step_distance", data=np.array([0.5], dtype=np.float64))
        scale = ScaleParams(L0=2.0, v0=2.0, T_amb=300.0, delta_T0=10.0, Q0=2.0)

        timing = derive_timing_from_hdf5(h5_path, scale)

        self.assertAlmostEqual(timing["dt"], 0.25)

    def test_derive_timing_from_hdf5_rejects_nonuniform_step_array(self):
        h5_path = self.root / "input_nonuniform.h5"
        make_h5(h5_path)
        with h5py.File(h5_path, "a") as h5_file:
            del h5_file["dynamic"]
            dynamic = h5_file.create_group("dynamic")
            dynamic.create_dataset(
                "xyz",
                data=np.array(
                    [
                        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                        [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [0.1, 1.0, 0.0], [1.1, 1.0, 0.0]],
                        [[0.2, 0.0, 0.0], [1.2, 0.0, 0.0], [0.2, 1.0, 0.0], [1.2, 1.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
            )
            dynamic.create_dataset(
                "fiber",
                data=np.tile(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32), (3, 4, 1)),
            )
            dynamic.create_dataset(
                "Q",
                data=np.array(
                    [
                        [[0.0], [1.0], [0.5], [0.0]],
                        [[0.0], [0.8], [0.4], [0.0]],
                        [[0.0], [0.6], [0.3], [0.0]],
                    ],
                    dtype=np.float32,
                ),
            )
            del h5_file["path/heat_center_step_distance"]
            del h5_file["path/slice_path_length"]
            h5_file["path"].create_dataset(
                "heat_center_step_distance",
                data=np.array([0.5, 0.6], dtype=np.float64),
            )
        scale = ScaleParams(L0=2.0, v0=2.0, T_amb=300.0, delta_T0=10.0, Q0=2.0)

        with self.assertRaisesRegex(ValueError, "must be constant"):
            derive_timing_from_hdf5(h5_path, scale)

    def test_derive_timing_from_hdf5_requires_path_timing(self):
        h5_path = self.root / "input_missing_timing.h5"
        make_h5(h5_path)
        with h5py.File(h5_path, "a") as h5_file:
            del h5_file["path/heat_center_step_distance"]
        scale = ScaleParams(L0=2.0, v0=2.0, T_amb=300.0, delta_T0=10.0, Q0=2.0)

        with self.assertRaisesRegex(ValueError, "heat_center_step_distance"):
            derive_timing_from_hdf5(h5_path, scale)

    def test_derive_timing_from_hdf5_rejects_length_mismatch(self):
        h5_path = self.root / "input_bad_length.h5"
        make_h5(h5_path)
        with h5py.File(h5_path, "a") as h5_file:
            h5_file["path/slice_path_length"][()] = np.float64(1.0)
        scale = ScaleParams(L0=2.0, v0=2.0, T_amb=300.0, delta_T0=10.0, Q0=2.0)

        with self.assertRaisesRegex(ValueError, "slice_path_length"):
            derive_timing_from_hdf5(h5_path, scale)

    def test_derive_timing_from_hdf5_rejects_scan_velocity_mismatch(self):
        h5_path = self.root / "input_velocity_mismatch.h5"
        make_h5(h5_path)
        scale = ScaleParams(L0=2.0, v0=2.0, T_amb=300.0, delta_T0=10.0, Q0=2.0)

        with self.assertRaisesRegex(ValueError, "scan_velocity"):
            derive_timing_from_hdf5(h5_path, scale, scan_velocity=3.0)

    def test_run_training_from_config_writes_checkpoint_history_and_metadata(self):
        h5_dir = self.root / "h5"
        h5_dir.mkdir()
        h5_path = h5_dir / "input.h5"
        config_path = self.root / "config.json"
        make_h5(h5_path)
        payload = {
            "data": {
                "h5_dir": "h5",
                "cache_dir": "cache",
                "checkpoint_path": "checkpoint.pt",
                "history_path": "history.json",
            },
            "scale": {
                "L0": 2.0,
                "v0": 2.0,
                "T_amb": 300.0,
                "delta_T0": 10.0,
                "Q0": 2.0,
                "K0": 8.0,
                "rho": 2.0,
                "Cp": 1.0,
            },
            "model": {
                "hidden_size": 8,
                "message_passing_num": 1,
                "inverse_pe": 999.0,
                "pi_q": 999.0,
                "dt_star": 999.0,
                "lambda_outflow": 0.0,
                "thermal_loss_beta": 0.25,
                "thermal_loss_base_temperature_star": 0.0,
                "residual_time_scheme": "backward",
            },
            "training": {
                "lr": 0.001,
                "epochs": 3,
                "tbptt_window": 2,
                "warmup_steps": 1,
                "loss_threshold": 1e20,
                "device": "cpu",
            },
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        result = run_training_from_config(config_path)
        checkpoint_path = Path(result["checkpoint_path"])
        history_path = Path(result["history_path"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self.assertTrue(checkpoint_path.exists())
        self.assertTrue(history_path.exists())
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(checkpoint["epoch"], 0)
        self.assertTrue(result["history"][-1]["stopped_early"])
        metadata = checkpoint["metadata"]
        self.assertAlmostEqual(metadata["model_config"]["inverse_pe"], 1.0)
        self.assertAlmostEqual(metadata["model_config"]["pi_q"], 0.1)
        self.assertAlmostEqual(metadata["model_config"]["dt_star"], 0.25)
        self.assertAlmostEqual(metadata["hdf5_timing"]["dt"], 0.25)
        self.assertAlmostEqual(metadata["hdf5_timing"]["dt_star"], 0.25)
        self.assertAlmostEqual(metadata["hdf5_timing"]["step_distance"], 0.5)
        self.assertAlmostEqual(metadata["hdf5_timing"]["velocity_speed"], 2.0)
        self.assertAlmostEqual(metadata["model_config"]["thermal_loss_beta"], 0.25)
        self.assertAlmostEqual(metadata["model_config"]["thermal_loss_base_temperature_star"], 0.0)
        self.assertEqual(metadata["model_config"]["residual_time_scheme"], "backward")
        self.assertEqual(metadata["train_config"]["loss_threshold"], 1e20)
        self.assertEqual(len(metadata["h5_files"]), 1)

    def test_discover_hdf5_files_uses_natural_filename_order(self):
        h5_dir = self.root / "h5_order"
        h5_dir.mkdir()
        for name in ("slice10.h5", "slice2.h5", "slice1.h5"):
            make_h5(h5_dir / name)

        ordered = [path.name for path in discover_hdf5_files(h5_dir)]

        self.assertEqual(ordered, ["slice1.h5", "slice2.h5", "slice10.h5"])

    def test_train_entry_can_run_as_script_path(self):
        completed = subprocess.run(
            [sys.executable, "training/train_entry.py", "--help"],
            cwd=Path.cwd(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--config", completed.stdout)


if __name__ == "__main__":
    unittest.main()
