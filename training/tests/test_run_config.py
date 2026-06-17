import json
import shutil
import subprocess
import sys
import unittest
from dataclasses import asdict
from unittest import mock
from pathlib import Path

import h5py
import numpy as np
import torch

from data import ScaleParams
from inference.io import load_inference_run_context, run_multilayer_inference_from_config
from models import PDGCN, PDGCNConfig
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
    normal = np.tile(np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32), (2, 4, 1))
    q = np.array([[[0.0], [1.0], [0.5], [0.0]], [[0.0], [0.8], [0.4], [0.0]]], dtype=np.float32)
    edge_index = np.array([[0, 1, 2, 0], [1, 3, 3, 2]], dtype=np.int64)

    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["velocity_speed"] = 2.0
        h5_file.attrs["velocity_direction_local"] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        dynamic = h5_file.create_group("dynamic")
        dynamic.create_dataset("xyz", data=xyz)
        dynamic.create_dataset("fiber", data=fiber)
        dynamic.create_dataset("normal", data=normal)
        dynamic.create_dataset("Q", data=q)
        h5_file.create_dataset("edge_index", data=edge_index)
        boundary = h5_file.create_group("boundary_nodes")
        boundary.create_dataset("upwind", data=np.array([0], dtype=np.int64))
        boundary.create_dataset("downwind", data=np.array([3], dtype=np.int64))
        boundary.create_dataset("side", data=np.array([2], dtype=np.int64))
        path_group = h5_file.create_group("path")
        path_group.create_dataset("heat_center_step_distance", data=np.float64(0.5))
        path_group.create_dataset("slice_path_length", data=np.float64(0.5))


def add_fem_temperature(path: Path):
    fem_temperature = np.array(
        [
            [[[300.0], [302.0], [300.0], [304.0]]],
            [[[300.0], [303.5], [300.0], [305.5]]],
        ],
        dtype=np.float32,
    ).reshape(2, 4, 1)
    with h5py.File(path, "a") as h5_file:
        fem = h5_file.require_group("fem")
        fem.create_dataset("temperature", data=fem_temperature)
        fem.create_dataset("temperature_unit", data="degC")
        fem.create_dataset("time", data=np.array([0.0, 0.25], dtype=np.float64))
        fem.create_dataset("valid_mask", data=np.ones((2, 4, 1), dtype=np.uint8))


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
                "gradient_regularization": 0.001,
                "thermal_loss_beta": 0.25,
                "thermal_loss_base_temperature_star": 0.0,
                "residual_time_scheme": "backward",
            },
        )

        self.assertEqual(config.hidden_size, 8)
        self.assertAlmostEqual(config.inverse_pe, 0.5)
        self.assertAlmostEqual(config.source_coefficient, 0.125)
        self.assertAlmostEqual(config.pi_q, 0.125)
        self.assertAlmostEqual(config.dt_star, 1.0)
        self.assertAlmostEqual(config.gradient_regularization, 0.001)
        self.assertAlmostEqual(config.thermal_loss_beta, 0.25)
        self.assertAlmostEqual(config.thermal_loss_base_temperature_star, 0.0)
        self.assertEqual(config.residual_time_scheme, "backward")

    def test_pdgcn_config_derives_node_input_size_from_heat_source_feature_flags(self):
        default_config = PDGCNConfig()
        self.assertEqual(default_config.node_input_size, 7)
        self.assertEqual(default_config.encoder_node_input_size, 8)

        config = PDGCNConfig(
            include_q_in_features=True,
            include_delta_t_source_in_features=True,
        )
        self.assertEqual(config.node_input_size, 9)
        self.assertEqual(config.encoder_node_input_size, 10)

        delta_only = PDGCNConfig(include_delta_t_source_in_features=True)
        self.assertEqual(delta_only.node_input_size, 8)

        with self.assertRaisesRegex(ValueError, "node_input_size"):
            PDGCNConfig(node_input_size=7, include_q_in_features=True)

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
                        "heat_source_effective_thickness": 0.001,
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
                    "gradient_regularization": 0.001,
                    "thermal_loss_beta": 0.25,
                    "residual_time_scheme": "backward",
                },
                "training": {
                    "lr": 0.001,
                    "lr_scheduler": "warmup_cosine",
                    "min_lr": 1e-5,
                    "lr_warmup_epochs": 2,
                    "epochs": 3,
                    "tbptt_window": 2,
                    "warmup_steps": 1,
                    "device": "cpu",
                },
            },
            "inference": {
                "num_layers": 4,
                "layer_spacing": 0.00015,
                "output_path": "../runs/pdgcn/prediction.h5",
                "steps": 2,
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
        self.assertEqual(config.training.lr_scheduler, "warmup_cosine")
        self.assertAlmostEqual(config.training.min_lr, 1e-5)
        self.assertEqual(config.training.lr_warmup_epochs, 2)
        self.assertFalse(config.supervision.enabled)
        self.assertEqual(config.inference.num_layers, 4)
        self.assertAlmostEqual(config.inference.layer_spacing, 0.00015)
        self.assertEqual(config.inference.steps, 2)

    def test_load_config_uses_default_monitoring(self):
        config_path = self.root / "monitoring_default.json"
        payload = {
            "data": {
                "h5_dir": "h5",
                "cache_dir": "cache",
                "checkpoint_path": "checkpoint.pt",
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
                "heat_source_effective_thickness": 0.001,
            },
            "model": {},
            "training": {"lr": 0.001, "epochs": 1, "tbptt_window": 1},
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        config = load_run_config(config_path)

        self.assertTrue(config.monitoring.enabled)
        self.assertEqual(config.monitoring.interval_epochs, 10)
        self.assertIsNone(config.monitoring.temperature_frame_index)
        self.assertFalse(config.supervision.enabled)

    def test_load_config_accepts_legacy_supervision(self):
        config_path = self.root / "supervision_legacy.json"
        payload = {
            "data": {
                "h5_dir": "h5",
                "cache_dir": "cache",
                "checkpoint_path": "checkpoint.pt",
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
                "heat_source_effective_thickness": 0.001,
            },
            "model": {},
            "training": {"lr": 0.001, "epochs": 1, "tbptt_window": 1},
            "supervision": {
                "enabled": True,
                "temperature_dataset": "fem/temperature",
                "valid_mask_dataset": "fem/valid_mask",
                "lambda_temperature": 0.5,
                "lambda_rollout_temperature": 0.75,
                "rollout_window": 3,
                "mode": "mixed",
            },
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        config = load_run_config(config_path)

        self.assertTrue(config.supervision.enabled)
        self.assertEqual(config.supervision.temperature_dataset, "fem/temperature")
        self.assertEqual(config.supervision.valid_mask_dataset, "fem/valid_mask")
        self.assertAlmostEqual(config.supervision.lambda_temperature, 0.5)
        self.assertAlmostEqual(config.supervision.lambda_rollout_temperature, 0.75)
        self.assertEqual(config.supervision.rollout_window, 3)
        self.assertEqual(config.supervision.mode, "mixed")

    def test_load_config_accepts_supervision_modes(self):
        config_path = self.root / "supervision_modes.json"
        base_payload = {
            "data": {
                "h5_dir": "h5",
                "cache_dir": "cache",
                "checkpoint_path": "checkpoint.pt",
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
                "heat_source_effective_thickness": 0.001,
            },
            "model": {},
            "training": {"lr": 0.001, "epochs": 1, "tbptt_window": 1},
        }
        for mode in ("teacher_forcing", "rollout", "mixed"):
            payload = dict(base_payload)
            payload["supervision"] = {"enabled": True, "mode": mode}
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            config = load_run_config(config_path)

            self.assertEqual(config.supervision.mode, mode)

    def test_load_config_accepts_classified_supervision(self):
        config_path = self.root / "supervision_classified.json"
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
                        "heat_source_effective_thickness": 0.001,
                    },
                }
            ],
            "hyperparameters": {
                "model": {},
                "physics_loss": {},
                "training": {"lr": 0.001, "epochs": 1, "tbptt_window": 1},
            },
            "supervision": {"enabled": True, "lambda_temperature": 0.25},
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        config = load_run_config(config_path)

        self.assertTrue(config.supervision.enabled)
        self.assertAlmostEqual(config.supervision.lambda_temperature, 0.25)

    def test_load_config_rejects_invalid_supervision(self):
        base_payload = {
            "data": {
                "h5_dir": "h5",
                "cache_dir": "cache",
                "checkpoint_path": "checkpoint.pt",
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
                "heat_source_effective_thickness": 0.001,
            },
            "model": {},
            "training": {"lr": 0.001, "epochs": 1, "tbptt_window": 1},
        }
        bad_mode = dict(base_payload)
        bad_mode["supervision"] = {"enabled": True, "mode": "bad_mode"}
        bad_mode_path = self.root / "supervision_bad_mode.json"
        bad_mode_path.write_text(json.dumps(bad_mode), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "teacher_forcing.*rollout.*mixed"):
            load_run_config(bad_mode_path)

        bad_lambda = dict(base_payload)
        bad_lambda["supervision"] = {"lambda_temperature": -1.0}
        bad_lambda_path = self.root / "supervision_bad_lambda.json"
        bad_lambda_path.write_text(json.dumps(bad_lambda), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lambda_temperature"):
            load_run_config(bad_lambda_path)

        bad_rollout_lambda = dict(base_payload)
        bad_rollout_lambda["supervision"] = {"lambda_rollout_temperature": -1.0}
        bad_rollout_lambda_path = self.root / "supervision_bad_rollout_lambda.json"
        bad_rollout_lambda_path.write_text(json.dumps(bad_rollout_lambda), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lambda_rollout_temperature"):
            load_run_config(bad_rollout_lambda_path)

        bad_rollout_window = dict(base_payload)
        bad_rollout_window["supervision"] = {"rollout_window": 0}
        bad_rollout_window_path = self.root / "supervision_bad_rollout_window.json"
        bad_rollout_window_path.write_text(json.dumps(bad_rollout_window), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "rollout_window"):
            load_run_config(bad_rollout_window_path)

    def test_load_config_accepts_monitoring_interval(self):
        config_path = self.root / "monitoring_interval.json"
        payload = {
            "data": {
                "h5_dir": "h5",
                "cache_dir": "cache",
                "checkpoint_path": "checkpoint.pt",
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
                "heat_source_effective_thickness": 0.001,
            },
            "model": {},
            "training": {"lr": 0.001, "epochs": 1, "tbptt_window": 1},
            "monitoring": {"interval_epochs": 2, "temperature_frame_index": 1},
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        config = load_run_config(config_path)

        self.assertEqual(config.monitoring.interval_epochs, 2)
        self.assertEqual(config.monitoring.temperature_frame_index, 1)

    def test_load_config_rejects_non_positive_monitoring_interval(self):
        config_path = self.root / "monitoring_bad_interval.json"
        payload = {
            "data": {
                "h5_dir": "h5",
                "cache_dir": "cache",
                "checkpoint_path": "checkpoint.pt",
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
                "heat_source_effective_thickness": 0.001,
            },
            "model": {},
            "training": {"lr": 0.001, "epochs": 1, "tbptt_window": 1},
            "monitoring": {"interval_epochs": 0},
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "monitoring.interval_epochs"):
            load_run_config(config_path)

    def test_multilayer_inference_entry_writes_hdf5(self):
        h5_dir = self.root / "h5"
        h5_dir.mkdir()
        h5_path = h5_dir / "input.h5"
        make_h5(h5_path)
        checkpoint_path = self.root / "checkpoint.pt"
        output_path = self.root / "prediction.h5"
        model_config = PDGCNConfig(
            hidden_size=8,
            message_passing_num=1,
            inverse_pe=0.0,
            source_coefficient=0.0,
            pi_q=0.0,
            k_ratio=0.05,
            dt_star=1.0,
        )
        model = PDGCN(model_config)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": None,
                "epoch": 0,
                "metadata": {"model_config": model_config.__dict__},
            },
            checkpoint_path,
        )
        training_config_path = self.root / "train.json"
        inference_config_path = self.root / "infer.json"
        training_payload = {
            "outputs": {
                "checkpoint_path": str(checkpoint_path.resolve()),
                "history_path": "history.json",
            },
            "datasets": [
                {
                    "name": "case_a",
                    "h5_dir": str(h5_dir.resolve()),
                    "cache_dir": "cache/case_a",
                    "scale": {
                        "L0": 0.002,
                        "v0": 0.002,
                        "T_amb": 300.0,
                        "delta_T0": 10.0,
                        "Q0": 2.0e6,
                        "K0": 8.0,
                        "rho": 2.0,
                        "Cp": 1.0,
                        "heat_source_effective_thickness": 0.001,
                    },
                }
            ],
            "hyperparameters": {
                "model": {"hidden_size": 8, "message_passing_num": 1},
                "physics_loss": {"lambda_outflow": 0.0},
                "training": {"lr": 0.001, "epochs": 1, "tbptt_window": 1, "warmup_steps": 0, "device": "cpu"},
            },
        }
        inference_payload = {
            "training_config": "train.json",
            "inference": {
                "num_layers": 3,
                "layer_spacing": 0.001,
                "output_path": str(output_path.resolve()),
                "steps": 2,
                "warmup_steps": 0,
                "cloud_interval": 1,
                "delta_smoothing_alpha": 0.3,
                "delta_smoothing_steps": 2,
            },
        }
        training_config_path.write_text(json.dumps(training_payload), encoding="utf-8")
        inference_config_path.write_text(json.dumps(inference_payload), encoding="utf-8")

        result = run_multilayer_inference_from_config(inference_config_path)

        self.assertEqual(result["output_path"], str(output_path.resolve()))
        with h5py.File(output_path, "r") as h5_file:
            self.assertEqual(tuple(h5_file["temperature"].shape), (2, 3, 4, 1))
            self.assertEqual(tuple(h5_file["temperature_star"].shape), (2, 3, 4, 1))
            self.assertIn("metadata", h5_file)
            self.assertIn("metadata", h5_file.attrs)
            metadata = json.loads(h5_file.attrs["metadata"])
            self.assertEqual(metadata["cloud_interval"], 1)
            self.assertAlmostEqual(metadata["delta_smoothing_alpha"], 0.3)
            self.assertEqual(metadata["delta_smoothing_steps"], 2)
            self.assertEqual(metadata["training_config_path"], str(training_config_path.resolve()))
            self.assertIn("inference_seconds", metadata)
            self.assertIn("average_inference_seconds", metadata)
            self.assertIn("max_inference_seconds", metadata)
            self.assertIn("min_inference_seconds", metadata)
            self.assertIn("render_seconds", metadata)
            self.assertEqual(metadata["render_seconds"], 0.0)
            self.assertEqual(metadata["rendered_steps"], [])
            self.assertLessEqual(metadata["min_inference_seconds"], metadata["average_inference_seconds"])
            self.assertLessEqual(metadata["average_inference_seconds"], metadata["max_inference_seconds"])
        vtk_dir = output_path.with_name(f"{output_path.stem}_vtk")
        vtk_files = sorted(vtk_dir.glob("temperature_step_*.vtk"))
        self.assertEqual(vtk_files, [])

    def test_load_inference_run_context_accepts_legacy_unified_config(self):
        config_path = self.root / "legacy_infer.json"
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
                        "heat_source_effective_thickness": 0.001,
                    },
                }
            ],
            "hyperparameters": {
                "model": {"hidden_size": 8, "message_passing_num": 1},
                "physics_loss": {"lambda_outflow": 0.0},
                "training": {"lr": 0.001, "epochs": 1, "tbptt_window": 1, "warmup_steps": 0, "device": "cpu"},
            },
            "inference": {
                "num_layers": 3,
                "layer_spacing": 0.001,
            },
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        run_config, inference_config, training_base_dir, inference_base_dir, training_config_path = (
            load_inference_run_context(config_path)
        )

        self.assertEqual(run_config.schema, "classified")
        self.assertEqual(inference_config.num_layers, 3)
        self.assertEqual(training_base_dir, config_path.resolve().parent)
        self.assertEqual(inference_base_dir, config_path.resolve().parent)
        self.assertEqual(training_config_path, config_path.resolve())

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
                        "heat_source_effective_thickness": 0.001,
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

    def test_load_config_requires_heat_source_effective_thickness(self):
        config_path = self.root / "missing_thickness.json"
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
                        "L0": 1.0,
                        "v0": 1.0,
                        "T_amb": 300.0,
                        "delta_T0": 10.0,
                        "Q0": 1.0,
                        "K0": 1.0,
                        "rho": 1.0,
                        "Cp": 1.0,
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

        with self.assertRaisesRegex(ValueError, "heat_source_effective_thickness"):
            load_run_config(config_path)

    def test_derive_timing_from_hdf5_scalar_step_distance(self):
        h5_path = self.root / "input.h5"
        make_h5(h5_path)
        scale = ScaleParams(L0=2.0, v0=2.0, T_amb=300.0, delta_T0=10.0, Q0=2.0)

        timing = derive_timing_from_hdf5(h5_path, scale)

        self.assertAlmostEqual(timing["step_distance"], 0.0005)
        self.assertAlmostEqual(timing["velocity_speed"], 0.002)
        self.assertAlmostEqual(timing["dt"], 0.25)
        self.assertAlmostEqual(timing["dt_star"], 0.25)
        self.assertAlmostEqual(timing["slice_path_length"], 0.0005)
        self.assertAlmostEqual(timing["native_step_distance_mm"], 0.5)
        self.assertAlmostEqual(timing["native_velocity_speed_mm_per_s"], 2.0)
        self.assertAlmostEqual(timing["native_slice_path_length_mm"], 0.5)

    def test_derive_timing_from_hdf5_converts_mm_timing_to_si(self):
        h5_path = self.root / "input_si_timing.h5"
        make_h5(h5_path)
        with h5py.File(h5_path, "a") as h5_file:
            h5_file.attrs["velocity_speed"] = 1000.0
            h5_file["path/heat_center_step_distance"][()] = np.float64(500.0)
            h5_file["path/slice_path_length"][()] = np.float64(500.0)
        scale = ScaleParams(L0=1.0, v0=2.0, T_amb=300.0, delta_T0=10.0, Q0=1.0)

        timing = derive_timing_from_hdf5(h5_path, scale)

        self.assertAlmostEqual(timing["step_distance"], 0.5)
        self.assertAlmostEqual(timing["velocity_speed"], 1.0)
        self.assertAlmostEqual(timing["dt"], 0.5)
        self.assertAlmostEqual(timing["dt_star"], 1.0)

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
                "L0": 0.002,
                "v0": 0.002,
                "T_amb": 300.0,
                "delta_T0": 10.0,
                "Q0": 2.0e6,
                "K0": 8.0e-6,
                "rho": 2.0,
                "Cp": 1.0,
                "heat_source_effective_thickness": 0.001,
            },
            "model": {
                "hidden_size": 8,
                "message_passing_num": 1,
                "inverse_pe": 999.0,
                "pi_q": 999.0,
                "dt_star": 999.0,
                "lambda_outflow": 0.0,
                "gradient_regularization": 0.001,
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
        self.assertAlmostEqual(metadata["model_config"]["source_coefficient"], 1.0e8)
        self.assertAlmostEqual(metadata["model_config"]["pi_q"], 1.0e8)
        self.assertAlmostEqual(metadata["model_config"]["dt_star"], 0.25)
        self.assertAlmostEqual(metadata["hdf5_timing"]["dt"], 0.25)
        self.assertAlmostEqual(metadata["hdf5_timing"]["dt_star"], 0.25)
        self.assertAlmostEqual(metadata["hdf5_timing"]["step_distance"], 0.0005)
        self.assertAlmostEqual(metadata["hdf5_timing"]["velocity_speed"], 0.002)
        self.assertAlmostEqual(metadata["model_config"]["gradient_regularization"], 0.001)
        self.assertAlmostEqual(metadata["model_config"]["thermal_loss_beta"], 0.25)
        self.assertAlmostEqual(metadata["model_config"]["thermal_loss_base_temperature_star"], 0.0)
        self.assertEqual(metadata["model_config"]["residual_time_scheme"], "backward")
        self.assertEqual(metadata["train_config"]["loss_threshold"], 1e20)
        self.assertEqual(len(metadata["h5_files"]), 1)
        history_payload = json.loads(history_path.read_text(encoding="utf-8"))
        self.assertNotIn("slice_records", history_payload)
        self.assertIn("loss_beta", history_payload["history"][0])
        self.assertIn("loss_smooth", history_payload["history"][0])
        figures_dir = history_path.parent / "figures"
        self.assertFalse((figures_dir / "loss_curve.png").exists())
        monitor_path = history_path.parent / "metrics" / "monitor_data.h5"
        self.assertEqual(Path(result["monitor_data_path"]), monitor_path)
        self.assertTrue(monitor_path.exists())
        with h5py.File(monitor_path, "r") as monitor_h5:
            self.assertEqual(monitor_h5.attrs["schema_version"], "1.0")
            self.assertIn("epoch_metrics", monitor_h5)
            self.assertIn("slice_metrics", monitor_h5)
            self.assertIn("epoch_snapshots/epoch_0001", monitor_h5)
            self.assertIn("slice_snapshots", monitor_h5)
            self.assertEqual(monitor_h5["epoch_metrics/epoch"].shape, (1,))
            self.assertEqual(monitor_h5["epoch_metrics/loss_beta"].shape, (1,))
            self.assertEqual(monitor_h5["epoch_metrics/loss_smooth"].shape, (1,))
            self.assertEqual(monitor_h5["slice_metrics/epoch"].shape, (0,))
            self.assertEqual(len(monitor_h5["slice_snapshots"].keys()), 0)
            self.assertEqual(monitor_h5["epoch_snapshots/epoch_0001/coords"].shape, (4, 3))
            self.assertEqual(monitor_h5["epoch_snapshots/epoch_0001/temperature"].shape, (4,))
            self.assertEqual(monitor_h5["epoch_snapshots/epoch_0001/edge_index"].shape[0], 2)

        history_path.unlink()
        from training.visualize_monitor import main as visualize_main

        self.assertEqual(
            visualize_main(
                [
                    "--monitor-data",
                    str(monitor_path),
                    "--output-dir",
                    str(figures_dir / "vtk"),
                    "--grid-resolution",
                    "64",
                ]
            ),
            0,
        )
        vtk_path = figures_dir / "vtk" / "epoch_temperature_residual_epoch_0001_frame_0001.vtk"
        self.assertTrue(vtk_path.exists())
        vtk_text = vtk_path.read_text(encoding="ascii")
        self.assertIn("LINES", vtk_text)
        self.assertIn("SCALARS temperature float 1", vtk_text)
        self.assertIn("SCALARS residual float 1", vtk_text)
        self.assertFalse((figures_dir / "loss_curve.png").exists())
        self.assertFalse((figures_dir / "temperature_stats.png").exists())

    def test_run_training_from_config_with_supervision(self):
        h5_dir = self.root / "h5_supervised"
        h5_dir.mkdir()
        h5_path = h5_dir / "input.h5"
        make_h5(h5_path)
        add_fem_temperature(h5_path)
        config_path = self.root / "supervised_config.json"
        payload = {
            "data": {
                "h5_dir": str(h5_dir.resolve()),
                "cache_dir": "cache_supervised",
                "checkpoint_path": "supervised_checkpoint.pt",
                "history_path": "supervised_history.json",
            },
            "scale": {
                "L0": 0.002,
                "v0": 0.002,
                "T_amb": 300.0,
                "delta_T0": 10.0,
                "Q0": 2.0e6,
                "K0": 8.0e-6,
                "rho": 2.0,
                "Cp": 1.0,
                "heat_source_effective_thickness": 0.001,
            },
            "model": {
                "hidden_size": 8,
                "message_passing_num": 1,
                "lambda_outflow": 0.0,
                "gradient_regularization": 0.0,
            },
            "training": {
                "lr": 0.001,
                "epochs": 1,
                "tbptt_window": 1,
                "warmup_steps": 3,
                "device": "cpu",
            },
            "supervision": {"enabled": True, "lambda_temperature": 1.0},
            "monitoring": {"enabled": False},
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        result = run_training_from_config(config_path)

        self.assertEqual(len(result["history"]), 1)
        record = result["history"][0]
        self.assertEqual(len(record["window_losses"]), 1)
        self.assertIn("loss_temperature", record)
        self.assertIn("loss_supervised", record)
        self.assertTrue(np.isfinite(record["loss_temperature"]))
        checkpoint = torch.load(result["checkpoint_path"], map_location="cpu")
        self.assertTrue(checkpoint["metadata"]["run_config"]["supervision"]["enabled"])

    def test_run_training_from_config_resumes_from_checkpoint(self):
        h5_dir = self.root / "h5_resume"
        h5_dir.mkdir()
        h5_path = h5_dir / "input.h5"
        make_h5(h5_path)
        resume_path = self.root / "resume_checkpoint.pt"
        output_path = self.root / "resume_output.pt"
        history_path = self.root / "resume_history.json"
        scale = ScaleParams(
            L0=0.002,
            v0=0.002,
            T_amb=300.0,
            delta_T0=10.0,
            Q0=2.0e6,
            K0=8.0e-6,
            rho=2.0,
            Cp=1.0,
            heat_source_effective_thickness=0.001,
        )
        timing = derive_timing_from_hdf5(h5_path, scale)
        model_config = pdgcn_config_from_scale(
            scale,
            dt=timing["dt"],
            model_overrides={"hidden_size": 8, "message_passing_num": 1, "lambda_outflow": 0.0},
        )
        resume_model = PDGCN(model_config)
        for parameter in resume_model.parameters():
            torch.nn.init.constant_(parameter, 0.123)
        resume_optimizer = torch.optim.Adam(resume_model.parameters(), lr=0.02)
        torch.save(
            {
                "model": resume_model.state_dict(),
                "optimizer": resume_optimizer.state_dict(),
                "epoch": 4,
                "metadata": {
                    "history": [
                        {
                            "epoch": 4,
                            "loss": 2.0,
                            "loss_total": 2.0,
                            "loss_pde": 1.5,
                            "loss_outflow": 0.5,
                            "loss_beta": 0.0,
                            "loss_smooth": 0.0,
                            "temperature_mean": 300.0,
                            "temperature_max": 301.0,
                            "temperature_min": 299.0,
                            "temperature_var": 1.0,
                            "window_losses": [2.0],
                            "file_window_counts": [1],
                        }
                    ]
                },
            },
            resume_path,
        )
        config_path = self.root / "resume_config.json"
        payload = {
            "data": {
                "h5_dir": str(h5_dir.resolve()),
                "cache_dir": "cache_resume",
                "checkpoint_path": str(output_path.resolve()),
                "history_path": str(history_path.resolve()),
            },
            "scale": asdict(scale),
            "model": {"hidden_size": 8, "message_passing_num": 1, "lambda_outflow": 0.0},
            "training": {
                "lr": 0.001,
                "epochs": 1,
                "tbptt_window": 1,
                "warmup_steps": 0,
                "device": "cpu",
                "resume_from_checkpoint": True,
                "resume_checkpoint_path": str(resume_path.resolve()),
            },
            "monitoring": {"enabled": False},
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        expected_state = {name: value.clone() for name, value in resume_model.state_dict().items()}

        def resumed_training(model, *args, **kwargs):
            self.assertEqual(kwargs["start_epoch"], 5)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.allclose(value.cpu(), expected_state[name]))
            return [
                {
                    "epoch": 5,
                    "loss": 1.0,
                    "loss_total": 1.0,
                    "loss_pde": 0.8,
                    "loss_outflow": 0.2,
                    "loss_beta": 0.0,
                    "loss_smooth": 0.0,
                    "temperature_mean": 300.0,
                    "temperature_max": 301.0,
                    "temperature_min": 299.0,
                    "temperature_var": 1.0,
                    "window_losses": [1.0],
                    "file_window_counts": [1],
                }
            ]

        with mock.patch("training.train_entry.train_static_topology_sequences", side_effect=resumed_training):
            result = run_training_from_config(config_path)

        self.assertEqual([record["epoch"] for record in result["history"]], [4, 5])
        checkpoint = torch.load(output_path, map_location="cpu")
        self.assertEqual(checkpoint["metadata"]["resume"]["loaded_epoch"], 4)
        self.assertEqual(checkpoint["metadata"]["resume"]["next_epoch"], 5)
        self.assertEqual([record["epoch"] for record in checkpoint["metadata"]["history"]], [4, 5])

    def test_run_training_from_config_saves_checkpoint_after_completed_epoch_on_interrupt(self):
        h5_dir = self.root / "h5_interrupt"
        h5_dir.mkdir()
        h5_path = h5_dir / "input.h5"
        make_h5(h5_path)
        checkpoint_path = self.root / "interrupt_checkpoint.pt"
        history_path = self.root / "interrupt_history.json"
        config_path = self.root / "interrupt_config.json"
        payload = {
            "data": {
                "h5_dir": str(h5_dir.resolve()),
                "cache_dir": "cache_interrupt",
                "checkpoint_path": str(checkpoint_path.resolve()),
                "history_path": str(history_path.resolve()),
            },
            "scale": {
                "L0": 0.002,
                "v0": 0.002,
                "T_amb": 300.0,
                "delta_T0": 10.0,
                "Q0": 2.0e6,
                "K0": 8.0e-6,
                "rho": 2.0,
                "Cp": 1.0,
                "heat_source_effective_thickness": 0.001,
            },
            "model": {
                "hidden_size": 8,
                "message_passing_num": 1,
                "lambda_outflow": 0.0,
            },
            "training": {
                "lr": 0.001,
                "epochs": 3,
                "tbptt_window": 1,
                "warmup_steps": 0,
                "device": "cpu",
            },
            "monitoring": {"enabled": False},
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        def interrupted_training(*args, **kwargs):
            kwargs["epoch_callback"](
                {
                    "epoch": 0,
                    "loss": 1.0,
                    "loss_total": 1.0,
                    "loss_pde": 0.8,
                    "loss_outflow": 0.2,
                    "loss_beta": 0.0,
                    "loss_smooth": 0.0,
                    "temperature_mean": 300.0,
                    "temperature_max": 301.0,
                    "temperature_min": 299.0,
                    "temperature_var": 1.0,
                    "window_losses": [1.0],
                    "file_window_counts": [1],
                }
            )
            raise KeyboardInterrupt()

        with mock.patch("training.train_entry.train_static_topology_sequences", side_effect=interrupted_training):
            with self.assertRaises(KeyboardInterrupt):
                run_training_from_config(config_path)

        self.assertTrue(checkpoint_path.exists())
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.assertEqual(checkpoint["epoch"], 0)
        self.assertEqual(checkpoint["metadata"]["history"][-1]["epoch"], 0)
        history_payload = json.loads(history_path.read_text(encoding="utf-8"))
        self.assertEqual(history_payload["history"][-1]["epoch"], 0)
        self.assertIn("metadata", history_payload)

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
