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
from training import pdgcn_config_from_scale
from training.run_config import derive_dt_star
from training.train_entry import run_training_from_config


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

    def test_run_training_from_config_writes_checkpoint_history_and_metadata(self):
        h5_path = self.root / "input.h5"
        config_path = self.root / "config.json"
        make_h5(h5_path)
        payload = {
            "data": {
                "h5_path": "input.h5",
                "cache_dir": "cache",
                "checkpoint_path": "checkpoint.pt",
                "history_path": "history.json",
                "overwrite_cache": True,
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
                "dt": 0.5,
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
        self.assertAlmostEqual(metadata["model_config"]["dt_star"], 0.5)
        self.assertAlmostEqual(metadata["model_config"]["thermal_loss_beta"], 0.25)
        self.assertAlmostEqual(metadata["model_config"]["thermal_loss_base_temperature_star"], 0.0)
        self.assertEqual(metadata["model_config"]["residual_time_scheme"], "backward")
        self.assertEqual(metadata["train_config"]["loss_threshold"], 1e20)

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
