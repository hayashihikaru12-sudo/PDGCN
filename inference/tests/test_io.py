import json
import shutil
import unittest
from pathlib import Path

import h5py
import numpy as np

from data import ScaleParams
from inference.io import (
    _remap_edge_index,
    _resolve_fin_cooling_parameters,
    _sample_node_indices,
    render_multilayer_clouds_from_hdf5,
)


class InferenceIOTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("inference/tests/_tmp_io")
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_sample_node_indices_caps_large_layers(self):
        indices = _sample_node_indices(10, 4)

        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 9)
        self.assertLessEqual(len(indices), 4)

    def test_sample_node_indices_can_use_spatial_distribution(self):
        coords = np.array([[float(i % 10), float(i // 10), 0.0] for i in range(100)], dtype=np.float64)

        indices = _sample_node_indices(100, 10, coords=coords)
        sampled = coords[indices]

        self.assertEqual(len(indices), 10)
        self.assertLess(float(sampled[:, 0].min()), 2.0)
        self.assertGreater(float(sampled[:, 0].max()), 7.0)
        self.assertLess(float(sampled[:, 1].min()), 2.0)
        self.assertGreater(float(sampled[:, 1].max()), 7.0)

    def test_remap_edge_index_filters_unsampled_nodes(self):
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)

        remapped = _remap_edge_index(edge_index, np.array([0, 1, 3], dtype=np.int64), 5)

        self.assertEqual(remapped.tolist(), [[0], [1]])

    def test_render_multilayer_clouds_from_hdf5_ignores_metadata_node_cap(self):
        source_h5 = self.root / "source.h5"
        prediction_h5 = self.root / "prediction.h5"
        self._write_source_h5(source_h5)
        self._write_prediction_h5(prediction_h5, source_h5)

        result = render_multilayer_clouds_from_hdf5(
            prediction_h5,
            cloud_interval=1,
            vtk_output_dir=self.root / "vtk",
        )

        vtk_path = self.root / "vtk" / "temperature_step_000000.vtk"
        self.assertTrue(vtk_path.exists())
        self.assertEqual(result["rendered_steps"], [0, 1])
        text = vtk_path.read_text(encoding="ascii")
        self.assertIn("POINTS 8 float", text)
        self.assertIn("DATASET UNSTRUCTURED_GRID", text)
        self.assertIn("CELLS 2 14", text)
        self.assertIn("CELL_TYPES", text)
        self.assertIn("\n13\n", text)

    def test_render_multilayer_clouds_from_hdf5_rejects_explicit_node_cap(self):
        source_h5 = self.root / "source.h5"
        prediction_h5 = self.root / "prediction.h5"
        self._write_source_h5(source_h5)
        self._write_prediction_h5(prediction_h5, source_h5)

        with self.assertRaisesRegex(ValueError, "max_nodes_per_layer is not supported"):
            render_multilayer_clouds_from_hdf5(
                prediction_h5,
                cloud_interval=1,
                vtk_output_dir=self.root / "vtk",
                max_nodes_per_layer=3,
            )

    def test_fin_cooling_gamma_dt_scales_with_beta_h_squared(self):
        resolved = _resolve_fin_cooling_parameters(
            fdm_coefficient=9.0,  # C_n = (M/βH)² · γ_dt  → arbitrary
            dt_star=1.0,
            num_layers=10,
            enabled=True,
            beta_h=3.0,
        )
        self.assertTrue(resolved["enabled"])
        # M = 9, gamma_dt = (3/9)² * 9 = 1.0
        self.assertAlmostEqual(resolved["gamma_dt"], 1.0)
        self.assertAlmostEqual(resolved["gamma_star"], 1.0)

        resolved2 = _resolve_fin_cooling_parameters(
            fdm_coefficient=9.0,
            dt_star=1.0,
            num_layers=10,
            enabled=True,
            beta_h=6.0,
        )
        # (6/9)² = 4× larger than βH=3
        self.assertAlmostEqual(resolved2["gamma_dt"], 4.0)

    def test_fin_cooling_gamma_dt_scales_with_active_layers(self):
        resolved = _resolve_fin_cooling_parameters(
            fdm_coefficient=4.0,
            dt_star=1.0,
            num_layers=5,  # M = 4
            enabled=True,
            beta_h=2.0,
        )
        # (2/4)² * 4 = 0.25 * 4 = 1.0
        self.assertAlmostEqual(resolved["gamma_dt"], 1.0)

    def test_fin_cooling_gamma_dt_proportional_to_fdm_coefficient(self):
        resolved = _resolve_fin_cooling_parameters(
            fdm_coefficient=18.0,
            dt_star=1.0,
            num_layers=10,
            enabled=True,
            beta_h=3.0,
        )
        # (3/9)² * 18 = 1/9 * 18 = 2.0
        self.assertAlmostEqual(resolved["gamma_dt"], 2.0)

    def test_fin_cooling_r_char_mode_is_independent_of_fdm_coefficient(self):
        resolved = _resolve_fin_cooling_parameters(
            fdm_coefficient=18.0,
            dt_star=0.5,
            inverse_pe=0.02,
            num_layers=10,
            enabled=True,
            mode="r_char",
            r_char_star=0.1,
            beta_h=3.0,
        )

        self.assertEqual(resolved["mode"], "r_char")
        self.assertAlmostEqual(resolved["gamma_star"], 2.0)
        self.assertAlmostEqual(resolved["gamma_dt"], 1.0)
        self.assertAlmostEqual(resolved["equivalent_beta_h"], 9.0 * np.sqrt(1.0 / 18.0))

    def test_fin_cooling_r_char_mode_can_derive_compatible_length_scale(self):
        resolved = _resolve_fin_cooling_parameters(
            fdm_coefficient=1.0,
            dt_star=1.0,
            inverse_pe=1.0,
            layer_spacing_star=1.0,
            k_ratio=1.0,
            num_layers=10,
            enabled=True,
            mode="r_char",
            r_char_star=None,
            beta_h=3.0,
        )

        self.assertAlmostEqual(resolved["r_char_star"], 3.0)
        self.assertAlmostEqual(resolved["gamma_star"], 1.0 / 9.0)
        self.assertAlmostEqual(resolved["gamma_dt"], 1.0 / 9.0)
        self.assertAlmostEqual(resolved["equivalent_beta_h"], 3.0)

    def test_fin_cooling_direct_mode_accepts_layerwise_profile(self):
        resolved = _resolve_fin_cooling_parameters(
            fdm_coefficient=1.0,
            dt_star=2.0,
            num_layers=5,
            enabled=True,
            mode="direct",
            direct_gamma_star=[1.0, 1.0, 1.0, 1.0],
            beta_h=3.0,
            skip_top_layers=0,
            layer_profile="linear",
            layer_profile_strength=1.0,
        )

        self.assertEqual(resolved["mode"], "direct")
        self.assertEqual(resolved["gamma_star"], [1.25, 1.5, 1.75, 2.0])
        self.assertEqual(resolved["gamma_dt"], [2.5, 3.0, 3.5, 4.0])

    def test_fin_cooling_can_be_disabled(self):
        resolved = _resolve_fin_cooling_parameters(
            fdm_coefficient=1.0,
            dt_star=1.0,
            num_layers=10,
            enabled=False,
            beta_h=3.0,
        )
        self.assertFalse(resolved["enabled"])
        self.assertIsNone(resolved["gamma_star"])
        self.assertIsNone(resolved["gamma_dt"])
        self.assertEqual(resolved["mode"], "beta_h")
        self.assertAlmostEqual(resolved["beta_h"], 3.0)
        self.assertEqual(resolved["skip_top_layers"], 0)

    def test_fin_cooling_rejects_nonzero_skip_top_layers_when_enabled(self):
        with self.assertRaisesRegex(ValueError, "fin_cooling_skip_top_layers"):
            _resolve_fin_cooling_parameters(
                fdm_coefficient=1.0,
                dt_star=1.0,
                num_layers=10,
                enabled=True,
                beta_h=3.0,
                skip_top_layers=4,
            )

    def test_disabled_fin_cooling_accepts_legacy_skip_top_layers(self):
        resolved = _resolve_fin_cooling_parameters(
            fdm_coefficient=1.0,
            dt_star=1.0,
            num_layers=10,
            enabled=False,
            beta_h=3.0,
            skip_top_layers=4,
        )

        self.assertEqual(resolved["skip_top_layers"], 4)

    def test_fin_cooling_rejects_invalid_skip_top_layers(self):
        for value in (-1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "fin_cooling_skip_top_layers"):
                    _resolve_fin_cooling_parameters(
                        fdm_coefficient=1.0,
                        dt_star=1.0,
                        num_layers=10,
                        enabled=True,
                        beta_h=3.0,
                        skip_top_layers=value,
                    )

    def test_fin_cooling_rejects_invalid_beta_h(self):
        with self.assertRaisesRegex(ValueError, "fin_cooling_beta_h"):
            _resolve_fin_cooling_parameters(
                fdm_coefficient=1.0,
                dt_star=1.0,
                num_layers=10,
                enabled=True,
                beta_h=0.0,
            )
        with self.assertRaisesRegex(ValueError, "fin_cooling_beta_h"):
            _resolve_fin_cooling_parameters(
                fdm_coefficient=1.0,
                dt_star=1.0,
                num_layers=10,
                enabled=True,
                beta_h=-1.0,
            )

    def test_fin_cooling_rejects_invalid_new_mode_parameters(self):
        with self.assertRaisesRegex(ValueError, "fin_cooling_mode"):
            _resolve_fin_cooling_parameters(
                fdm_coefficient=1.0,
                dt_star=1.0,
                num_layers=10,
                enabled=True,
                mode="old",
                beta_h=3.0,
            )
        with self.assertRaisesRegex(ValueError, "fin_cooling_r_char_star"):
            _resolve_fin_cooling_parameters(
                fdm_coefficient=1.0,
                dt_star=1.0,
                num_layers=10,
                enabled=True,
                mode="r_char",
                r_char_star=0.0,
                beta_h=3.0,
            )
        with self.assertRaisesRegex(ValueError, "fin_cooling_gamma_star"):
            _resolve_fin_cooling_parameters(
                fdm_coefficient=1.0,
                dt_star=1.0,
                num_layers=10,
                enabled=True,
                mode="direct",
                beta_h=3.0,
            )
        with self.assertRaisesRegex(ValueError, "active layer count"):
            _resolve_fin_cooling_parameters(
                fdm_coefficient=1.0,
                dt_star=1.0,
                num_layers=10,
                enabled=True,
                mode="direct",
                direct_gamma_star=[1.0, 2.0],
                beta_h=3.0,
            )
        with self.assertRaisesRegex(ValueError, "fin_cooling_layer_profile"):
            _resolve_fin_cooling_parameters(
                fdm_coefficient=1.0,
                dt_star=1.0,
                num_layers=10,
                enabled=True,
                mode="r_char",
                beta_h=3.0,
                layer_profile="steep",
            )

    def test_fin_cooling_rejects_single_layer(self):
        with self.assertRaisesRegex(ValueError, "num_layers"):
            _resolve_fin_cooling_parameters(
                fdm_coefficient=1.0,
                dt_star=1.0,
                num_layers=1,
                enabled=True,
                beta_h=3.0,
            )

    def _write_source_h5(self, path):
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
        edge_index = np.array([[0, 1, 2, 3, 0], [1, 3, 3, 0, 2]], dtype=np.int64)

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

    def _write_prediction_h5(self, path, source_h5):
        metadata = {
            "source_h5": str(source_h5.resolve()),
            "num_layers": 2,
            "layer_spacing": 0.1,
            "layer_spacing_star": 0.1,
            "layer_fiber_angles_deg": [0.0, 90.0],
            "normal_offset_sign": -1,
            "cloud_interval": 1,
            "cloud_max_nodes_per_layer": 3,
            "vtk_output_dir": str((self.root / "vtk").resolve()),
            "hdf5_timing": {"velocity_speed": 2.0},
            "scale_params": {
                "L0": 1.0,
                "v0": 1.0,
                "T_amb": 300.0,
                "delta_T0": 10.0,
                "Q0": 1.0,
                "K0": None,
                "rho": None,
                "Cp": None,
                "heat_source_effective_thickness": 0.001,
                "eps": 1e-12,
            },
        }
        with h5py.File(path, "w") as h5_file:
            h5_file.create_dataset("temperature_star", data=np.ones((2, 2, 4, 1), dtype=np.float32))
            h5_file.create_dataset("temperature", data=np.full((2, 2, 4, 1), 310.0, dtype=np.float32))
            metadata_json = json.dumps(metadata)
            h5_file.create_dataset("metadata", data=metadata_json)
            h5_file.attrs["metadata"] = metadata_json


if __name__ == "__main__":
    unittest.main()
