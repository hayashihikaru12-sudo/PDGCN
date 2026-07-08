import unittest

from inference.config import InferenceRunConfig, SingleLayerInferenceRunConfig


class InferenceConfigTests(unittest.TestCase):
    def test_requires_multiple_layers(self):
        with self.assertRaisesRegex(ValueError, "num_layers"):
            InferenceRunConfig(num_layers=1, layer_spacing=0.1)

    def test_rejects_non_positive_layer_spacing(self):
        with self.assertRaisesRegex(ValueError, "layer_spacing"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.0)

    def test_rejects_non_positive_cloud_interval(self):
        with self.assertRaisesRegex(ValueError, "cloud_interval"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, cloud_interval=0)

    def test_rejects_non_positive_layer_batch_size(self):
        with self.assertRaisesRegex(ValueError, "layer_batch_size"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, layer_batch_size=0)

    def test_accepts_delta_smoothing_parameters(self):
        config = InferenceRunConfig(
            num_layers=2,
            layer_spacing=0.1,
            delta_smoothing_alpha=0.3,
            delta_smoothing_steps=2,
        )

        self.assertAlmostEqual(config.delta_smoothing_alpha, 0.3)
        self.assertEqual(config.delta_smoothing_steps, 2)

    def test_accepts_disabled_pdgcn_inplane_flag(self):
        config = InferenceRunConfig(num_layers=2, layer_spacing=0.1, use_pdgcn_inplane=False)

        self.assertFalse(config.use_pdgcn_inplane)

    def test_accepts_top_layer_only_pdgcn_inplane_flag(self):
        config = InferenceRunConfig(
            num_layers=2,
            layer_spacing=0.1,
            pdgcn_inplane_top_layer_only=True,
        )

        self.assertTrue(config.pdgcn_inplane_top_layer_only)

    def test_accepts_thickness_coupling_modes(self):
        default_config = InferenceRunConfig(num_layers=2, layer_spacing=0.1)
        legacy_config = InferenceRunConfig(
            num_layers=2,
            layer_spacing=0.1,
            thickness_coupling_mode="temperature_fdm",
        )

        self.assertEqual(default_config.thickness_coupling_mode, "source_distribution")
        self.assertEqual(legacy_config.thickness_coupling_mode, "temperature_fdm")

    def test_accepts_disabled_internal_pseudo_source_flag(self):
        config = InferenceRunConfig(
            num_layers=3,
            layer_spacing=0.1,
            enable_internal_pseudo_source_features=False,
        )

        self.assertFalse(config.enable_internal_pseudo_source_features)

    def test_accepts_multilayer_batch_config(self):
        config = InferenceRunConfig(
            num_layers=2,
            layer_spacing=0.1,
            batch_mode=True,
            h5_dir="inputs",
            output_dir="outputs",
            output_prefix="pre_",
        )

        self.assertTrue(config.batch_mode)
        self.assertEqual(config.h5_dir, "inputs")
        self.assertEqual(config.output_dir, "outputs")
        self.assertEqual(config.output_prefix, "pre_")

    def test_rejects_invalid_delta_smoothing_alpha(self):
        with self.assertRaisesRegex(ValueError, "delta_smoothing_alpha"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, delta_smoothing_alpha=-0.1)
        with self.assertRaisesRegex(ValueError, "delta_smoothing_alpha"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, delta_smoothing_alpha=1.1)

    def test_rejects_negative_delta_smoothing_steps(self):
        with self.assertRaisesRegex(ValueError, "delta_smoothing_steps"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, delta_smoothing_steps=-1)
        with self.assertRaisesRegex(ValueError, "delta_smoothing_steps"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, delta_smoothing_steps=1.5)

    def test_rejects_too_small_cloud_max_nodes_per_layer(self):
        with self.assertRaisesRegex(ValueError, "cloud_max_nodes_per_layer"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, cloud_max_nodes_per_layer=2)

    def test_rejects_layer_angle_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "layer_fiber_angles_deg length"):
            InferenceRunConfig(num_layers=3, layer_spacing=0.1, layer_fiber_angles_deg=[0.0, 45.0])

    def test_rejects_nonzero_base_layer_angle(self):
        with self.assertRaisesRegex(ValueError, "layer_fiber_angles_deg\\[0\\]"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, layer_fiber_angles_deg=[10.0, 45.0])

    def test_rejects_invalid_normal_offset_sign(self):
        with self.assertRaisesRegex(ValueError, "normal_offset_sign"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, normal_offset_sign=0)

    def test_rejects_bad_multilayer_batch_paths(self):
        with self.assertRaisesRegex(ValueError, "batch_mode"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, batch_mode="true")
        with self.assertRaisesRegex(ValueError, "write_vtk"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, write_vtk="true")
        with self.assertRaisesRegex(ValueError, "thickness_coupling_mode"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, thickness_coupling_mode="bad")
        with self.assertRaisesRegex(ValueError, "enable_internal_pseudo_source_features"):
            InferenceRunConfig(
                num_layers=2,
                layer_spacing=0.1,
                enable_internal_pseudo_source_features="true",
            )
        with self.assertRaisesRegex(ValueError, "h5_dir"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, h5_dir="")
        with self.assertRaisesRegex(ValueError, "output_dir"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, output_dir="")
        with self.assertRaisesRegex(ValueError, "output_prefix"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, output_prefix="")

    def test_single_layer_config_accepts_modes(self):
        config = SingleLayerInferenceRunConfig(
            mode="both",
            vtu_interval=2,
            batch_mode=True,
            h5_dir="inputs",
            output_dir="outputs",
            output_prefix="pre_",
            prediction_group_path="prediction/pdgcn_single_layer",
        )

        self.assertEqual(config.mode, "both")
        self.assertEqual(config.vtu_interval, 2)
        self.assertTrue(config.batch_mode)
        self.assertEqual(config.h5_dir, "inputs")
        self.assertEqual(config.output_dir, "outputs")
        self.assertEqual(config.output_prefix, "pre_")
        self.assertEqual(config.prediction_group_path, "prediction/pdgcn_single_layer")

    def test_single_layer_config_rejects_bad_mode(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            SingleLayerInferenceRunConfig(mode="rollout")

    def test_single_layer_config_rejects_bad_vtu_interval(self):
        with self.assertRaisesRegex(ValueError, "vtu_interval"):
            SingleLayerInferenceRunConfig(vtu_interval=0)

    def test_single_layer_config_rejects_bad_batch_paths(self):
        with self.assertRaisesRegex(ValueError, "h5_dir"):
            SingleLayerInferenceRunConfig(h5_dir="")
        with self.assertRaisesRegex(ValueError, "output_dir"):
            SingleLayerInferenceRunConfig(output_dir="")
        with self.assertRaisesRegex(ValueError, "output_prefix"):
            SingleLayerInferenceRunConfig(output_prefix="")
        with self.assertRaisesRegex(ValueError, "prediction_group_path"):
            SingleLayerInferenceRunConfig(prediction_group_path="prediction//bad")


if __name__ == "__main__":
    unittest.main()
