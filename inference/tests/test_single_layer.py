import json
import shutil
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from inference.single_layer_infer_entry import main as single_layer_entry_main
from inference.single_layer import (
    _build_qv_tag,
    _build_single_layer_vtu_name,
    _format_filename_scalar,
    run_single_layer_inference_from_config,
)
from models import PDGCN, PDGCNConfig


class SingleLayerVtuFilenameTests(unittest.TestCase):
    def test_format_filename_scalar_case_style(self):
        # 小数点 -> p，整数无小数点，与源 case_*.h5 文件名风格一致
        self.assertEqual(_format_filename_scalar(0.6666666865348816), "0p666667")
        self.assertEqual(_format_filename_scalar(25.0), "25")
        self.assertEqual(_format_filename_scalar(10.0), "10")
        self.assertEqual(_format_filename_scalar(0.5), "0p5")
        # 0 仍按整数渲染
        self.assertEqual(_format_filename_scalar(0.0), "0")

    def test_build_single_layer_vtu_name_with_and_without_tag(self):
        self.assertEqual(
            _build_single_layer_vtu_name("INF", 20, "Q0p666667_V25"),
            "INF_temperature_step_Q0p666667_V25_000020.vtu",
        )
        self.assertEqual(
            _build_single_layer_vtu_name("FEM", 0, ""),
            "FEM_temperature_step_000000.vtu",
        )

    def test_build_qv_tag_reads_attrs_and_degrades_when_missing(self):
        root = Path("inference/tests/_tmp_qv_tag")
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        h5_path = root / "case.h5"
        with h5py.File(h5_path, "w") as h5_file:
            h5_file.attrs["heat_source_qmax"] = np.float32(0.6666667)
            h5_file.attrs["velocity_speed"] = 25.0
            with h5py.File(h5_path, "r") as h5_file:
                self.assertEqual(_build_qv_tag(h5_file), "Q0p666667_V25")

        h5_no_q = root / "no_q.h5"
        with h5py.File(h5_no_q, "w") as h5_file:
            h5_file.attrs["velocity_speed"] = 10.0
        with h5py.File(h5_no_q, "r") as h5_file:
            # heat_source_qmax 缺失时只保留 V token，不报错
            self.assertEqual(_build_qv_tag(h5_file), "V10")

        h5_empty = root / "empty.h5"
        with h5py.File(h5_empty, "w") as _:
            pass
        with h5py.File(h5_empty, "r") as h5_file:
            # 两个 attr 都缺失时返回空串
            self.assertEqual(_build_qv_tag(h5_file), "")
        shutil.rmtree(root)


class SingleLayerInferenceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("inference/tests/_tmp_single_layer")
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_single_layer_entry_writes_hdf5_and_vtu(self):
        h5_dir = self.root / "h5"
        h5_dir.mkdir()
        h5_path = h5_dir / "input.h5"
        checkpoint_path = self.root / "checkpoint.pt"
        output_path = self.root / "single_prediction.h5"
        self._write_source_h5(h5_path)
        self._write_checkpoint(checkpoint_path)
        train_config_path = self.root / "train.json"
        infer_config_path = self.root / "single_infer.json"
        train_config_path.write_text(
            json.dumps(
                {
                    "outputs": {
                        "checkpoint_path": str(checkpoint_path.resolve()),
                        "history_path": "history.json",
                    },
                    "datasets": [
                        {
                            "name": "case_a",
                            "h5_dir": str(h5_dir.resolve()),
                            "cache_dir": str((self.root / "cache").resolve()),
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
            ),
            encoding="utf-8",
        )
        infer_config_path.write_text(
            json.dumps(
                {
                    "training_config": "train.json",
                    "single_layer_inference": {
                        "output_path": str(output_path.resolve()),
                        "steps": 2,
                        "warmup_steps": 0,
                        "mode": "both",
                        "write_vtu": True,
                        "vtu_interval": 1,
                    },
                }
            ),
            encoding="utf-8",
        )

        result = run_single_layer_inference_from_config(infer_config_path)

        self.assertEqual(result["output_path"], str(output_path.resolve()))
        self.assertEqual(result["prediction_group_path"], "prediction/pdgcn_single_layer")
        with h5py.File(h5_path, "r") as h5_file:
            self.assertIn("dynamic", h5_file)
            self.assertIn("fem", h5_file)
            self.assertNotIn("prediction", h5_file)
        with h5py.File(output_path, "r") as h5_file:
            self.assertIn("dynamic", h5_file)
            self.assertIn("fem", h5_file)
            group = h5_file["prediction/pdgcn_single_layer"]
            self.assertEqual(sorted(group.keys()), ["temperature", "time", "timing"])
            self.assertEqual(tuple(group["temperature"].shape), (2, 4, 1))
            self.assertEqual(group["temperature"].dtype, np.dtype("float32"))
            self.assertEqual(tuple(group["time"].shape), (2,))
            self.assertIn("solve_seconds", group["timing"])
            self.assertIn("total_seconds", group["timing"])
        vtu_dir = output_path.with_name(f"{output_path.stem}_vtu")
        tag = "Q0p666667_V2"
        first_vtu = vtu_dir / f"INF_temperature_step_{tag}_000000.vtu"
        second_vtu = vtu_dir / f"INF_temperature_step_{tag}_000001.vtu"
        self.assertTrue(first_vtu.exists())
        self.assertTrue(second_vtu.exists())
        # 源 HDF5 含 fem/temperature，应同步生成 FEM_* 对比 vtu
        first_fem_vtu = vtu_dir / f"FEM_temperature_step_{tag}_000000.vtu"
        second_fem_vtu = vtu_dir / f"FEM_temperature_step_{tag}_000001.vtu"
        self.assertTrue(first_fem_vtu.exists())
        self.assertTrue(second_fem_vtu.exists())
        text = second_vtu.read_text(encoding="utf-8")
        self.assertIn('Name="temperature"', text)
        self.assertNotIn('Name="temperature_star"', text)
        self.assertNotIn('Name="fem_temperature"', text)
        self.assertNotIn('Name="teacher_temperature_error"', text)
        fem_text = second_fem_vtu.read_text(encoding="utf-8")
        self.assertIn('Name="temperature"', fem_text)
        self.assertIn('Name="fem_valid_mask"', fem_text)

        result_second = run_single_layer_inference_from_config(infer_config_path)
        self.assertEqual(result_second["output_path"], str(output_path.resolve()))
        with h5py.File(output_path, "r") as h5_file:
            self.assertIn("prediction/pdgcn_single_layer/temperature", h5_file)

    def test_single_layer_batch_writes_prefixed_outputs_and_summarizes_failures(self):
        h5_dir = self.root / "h5_batch"
        output_dir = self.root / "batch_outputs"
        h5_dir.mkdir()
        good_one = h5_dir / "case1.h5"
        good_ten = h5_dir / "case10.h5"
        bad_zero = h5_dir / "case0_bad.h5"
        checkpoint_path = self.root / "checkpoint.pt"
        self._write_source_h5(good_one)
        self._write_source_h5(good_ten)
        with h5py.File(bad_zero, "w") as h5_file:
            h5_file.create_dataset("not_a_slice", data=np.array([1], dtype=np.int64))
        self._write_checkpoint(checkpoint_path)
        train_config_path = self.root / "train_batch.json"
        infer_config_path = self.root / "single_batch_infer.json"
        train_config_path.write_text(
            json.dumps(
                {
                    "outputs": {
                        "checkpoint_path": str(checkpoint_path.resolve()),
                        "history_path": "history.json",
                    },
                    "datasets": [
                        {
                            "name": "case_batch",
                            "h5_dir": str(h5_dir.resolve()),
                            "cache_dir": str((self.root / "batch_cache").resolve()),
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
            ),
            encoding="utf-8",
        )
        infer_config_path.write_text(
            json.dumps(
                {
                    "training_config": "train_batch.json",
                    "single_layer_inference": {
                        "batch_mode": True,
                        "h5_dir": str(h5_dir.resolve()),
                        "output_dir": str(output_dir.resolve()),
                        "steps": 2,
                        "warmup_steps": 0,
                        "mode": "both",
                        "write_vtu": True,
                        "vtu_interval": 1,
                    },
                }
            ),
            encoding="utf-8",
        )

        result = run_single_layer_inference_from_config(infer_config_path)

        self.assertTrue(result["batch_mode"])
        self.assertEqual(result["processed_count"], 3)
        self.assertEqual(result["succeeded_count"], 2)
        self.assertEqual(result["failed_count"], 1)
        self.assertTrue((output_dir / "pre_case1.h5").exists())
        self.assertTrue((output_dir / "pre_case10.h5").exists())
        self.assertFalse((output_dir / "pre_case0_bad.h5").exists())
        tag = "Q0p666667_V2"
        self.assertTrue((output_dir / "pre_case1_vtu" / f"INF_temperature_step_{tag}_000000.vtu").exists())
        self.assertTrue((output_dir / "pre_case10_vtu" / f"INF_temperature_step_{tag}_000001.vtu").exists())
        # 源 HDF5 含 fem/temperature，批量模式应同步生成 FEM_* vtu
        self.assertTrue((output_dir / "pre_case1_vtu" / f"FEM_temperature_step_{tag}_000000.vtu").exists())
        self.assertTrue((output_dir / "pre_case10_vtu" / f"FEM_temperature_step_{tag}_000001.vtu").exists())
        with h5py.File(output_dir / "pre_case1.h5", "r") as h5_file:
            self.assertIn("dynamic", h5_file)
            group = h5_file["prediction/pdgcn_single_layer"]
            self.assertEqual(sorted(group.keys()), ["temperature", "time", "timing"])
            self.assertEqual(tuple(group["temperature"].shape), (2, 4, 1))

    def test_single_layer_entry_respects_config_batch_mode_without_cli_flag(self):
        h5_dir = self.root / "h5_entry_batch"
        output_dir = self.root / "entry_batch_outputs"
        h5_dir.mkdir()
        h5_path = h5_dir / "case1.h5"
        checkpoint_path = self.root / "checkpoint.pt"
        self._write_source_h5(h5_path)
        self._write_checkpoint(checkpoint_path)
        train_config_path = self.root / "train_entry_batch.json"
        infer_config_path = self.root / "single_entry_batch_infer.json"
        train_config_path.write_text(
            json.dumps(
                {
                    "outputs": {
                        "checkpoint_path": str(checkpoint_path.resolve()),
                        "history_path": "history.json",
                    },
                    "datasets": [
                        {
                            "name": "case_entry_batch",
                            "h5_dir": str((self.root / "unused_dataset_dir").resolve()),
                            "cache_dir": str((self.root / "entry_batch_cache").resolve()),
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
            ),
            encoding="utf-8",
        )
        infer_config_path.write_text(
            json.dumps(
                {
                    "training_config": "train_entry_batch.json",
                    "single_layer_inference": {
                        "batch_mode": True,
                        "h5_path": None,
                        "h5_dir": str(h5_dir.resolve()),
                        "output_dir": str(output_dir.resolve()),
                        "steps": 2,
                        "warmup_steps": 0,
                        "mode": "both",
                        "write_vtu": False,
                        "vtu_interval": 1,
                    },
                }
            ),
            encoding="utf-8",
        )

        exit_code = single_layer_entry_main(["--config", str(infer_config_path)])

        self.assertEqual(exit_code, 0)
        self.assertTrue((output_dir / "pre_case1.h5").exists())
        with h5py.File(output_dir / "pre_case1.h5", "r") as h5_file:
            self.assertIn("prediction/pdgcn_single_layer/temperature", h5_file)

    def test_single_layer_skips_fem_vtu_when_fem_missing(self):
        h5_dir = self.root / "h5_no_fem"
        output_dir = self.root / "no_fem_outputs"
        h5_dir.mkdir()
        h5_path = h5_dir / "case1.h5"
        checkpoint_path = self.root / "checkpoint.pt"
        self._write_source_h5(h5_path, include_fem=False)
        self._write_checkpoint(checkpoint_path)
        train_config_path = self.root / "train_no_fem.json"
        infer_config_path = self.root / "single_no_fem_infer.json"
        train_config_path.write_text(
            json.dumps(
                {
                    "outputs": {
                        "checkpoint_path": str(checkpoint_path.resolve()),
                        "history_path": "history.json",
                    },
                    "datasets": [
                        {
                            "name": "case_no_fem",
                            "h5_dir": str(h5_dir.resolve()),
                            "cache_dir": str((self.root / "no_fem_cache").resolve()),
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
            ),
            encoding="utf-8",
        )
        infer_config_path.write_text(
            json.dumps(
                {
                    "training_config": "train_no_fem.json",
                    "single_layer_inference": {
                        "batch_mode": True,
                        "h5_dir": str(h5_dir.resolve()),
                        "output_dir": str(output_dir.resolve()),
                        "steps": 2,
                        "warmup_steps": 0,
                        "mode": "both",
                        "write_vtu": True,
                        "vtu_interval": 1,
                        "write_fem_vtu": True,
                    },
                }
            ),
            encoding="utf-8",
        )

        result = run_single_layer_inference_from_config(infer_config_path)

        self.assertTrue(result["batch_mode"])
        self.assertEqual(result["succeeded_count"], 1)
        vtu_dir = output_dir / "pre_case1_vtu"
        tag = "Q0p666667_V2"
        # 无 fem/temperature 时仍正常生成 INF_* vtu，但不生成 FEM_* vtu、不报错
        self.assertTrue((vtu_dir / f"INF_temperature_step_{tag}_000000.vtu").exists())
        self.assertTrue((vtu_dir / f"INF_temperature_step_{tag}_000001.vtu").exists())
        self.assertFalse((vtu_dir / f"FEM_temperature_step_{tag}_000000.vtu").exists())
        self.assertFalse((vtu_dir / f"FEM_temperature_step_{tag}_000001.vtu").exists())

    def _write_checkpoint(self, path):
        model_config = PDGCNConfig(
            hidden_size=8,
            message_passing_num=1,
            inverse_pe=0.0,
            source_coefficient=0.0,
            pi_q=0.0,
            k_ratio=0.0,
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
            path,
        )

    def _write_source_h5(self, path, *, include_fem=True):
        xyz = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [0.1, 1.0, 0.0], [1.1, 1.0, 0.0]],
            ],
            dtype=np.float32,
        )
        fiber = np.tile(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32), (2, 4, 1))
        normal = np.tile(np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32), (2, 4, 1))
        q = np.zeros((2, 4, 1), dtype=np.float32)
        fem_temperature = np.array(
            [
                [[300.0], [301.0], [302.0], [303.0]],
                [[300.5], [301.5], [302.5], [303.5]],
            ],
            dtype=np.float32,
        )
        edge_index = np.array([[0, 1, 3, 0, 2, 3], [1, 3, 0, 2, 3, 0]], dtype=np.int64)

        with h5py.File(path, "w") as h5_file:
            h5_file.attrs["velocity_speed"] = 2.0
            h5_file.attrs["heat_source_qmax"] = np.float32(0.6666667)
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
            if include_fem:
                fem = h5_file.create_group("fem")
                fem.create_dataset("temperature", data=fem_temperature)
                fem.create_dataset("temperature_unit", data="degC")
                fem.create_dataset("valid_mask", data=np.ones((2, 4, 1), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
