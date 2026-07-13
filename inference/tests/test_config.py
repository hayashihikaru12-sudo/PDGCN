import unittest

from inference.config import InferenceRunConfig, SingleLayerInferenceRunConfig
from inference.io import _build_inference_run_config


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

    def test_accepts_post_fdm_source_compensation_alpha(self):
        config = InferenceRunConfig(
            num_layers=2,
            layer_spacing=0.1,
            post_fdm_source_compensation_alpha=0.3,
        )

        self.assertAlmostEqual(config.post_fdm_source_compensation_alpha, 0.3)

    def test_rejects_invalid_post_fdm_source_compensation_alpha(self):
        with self.assertRaisesRegex(ValueError, "post_fdm_source_compensation_alpha"):
            InferenceRunConfig(
                num_layers=2,
                layer_spacing=0.1,
                post_fdm_source_compensation_alpha=-0.1,
            )
        with self.assertRaisesRegex(ValueError, "post_fdm_source_compensation_alpha"):
            InferenceRunConfig(
                num_layers=2,
                layer_spacing=0.1,
                post_fdm_source_compensation_alpha=1.1,
            )

    def test_rejects_bottom_layer_for_enabled_output_compensation(self):
        with self.assertRaisesRegex(ValueError, "non-bottom layer"):
            InferenceRunConfig(
                num_layers=3,
                layer_spacing=0.1,
                post_fdm_output_layer_compensations=(
                    {"layer": 3, "temperature": 20.0},
                ),
            )

    def test_accepts_per_layer_output_compensations(self):
        config = InferenceRunConfig(
            num_layers=10,
            layer_spacing=0.1,
            post_fdm_output_layer_compensations=(
                {"layer": 1, "temperature": 125.0},
                {"layer": 2, "temperature": 102.0},
                {"layer": 3, "temperature": 20.0},
            ),
        )

        self.assertEqual(len(config.post_fdm_output_layer_compensations), 3)

    def test_rejects_duplicate_per_layer_output_compensations(self):
        with self.assertRaisesRegex(ValueError, "duplicate layer"):
            InferenceRunConfig(
                num_layers=10,
                layer_spacing=0.1,
                post_fdm_output_layer_compensations=(
                    {"layer": 1, "temperature": 125.0},
                    {"layer": 1, "temperature": 102.0},
                ),
            )

    def test_rejects_removed_uniform_output_compensation_fields(self):
        with self.assertRaisesRegex(ValueError, "Unknown keys.*post_fdm_output_compensation_temperature"):
            _build_inference_run_config(
                {
                    "num_layers": 10,
                    "layer_spacing": 0.1,
                    "post_fdm_output_compensation_temperature": 102.0,
                }
            )

    def test_rejects_null_per_layer_output_compensations(self):
        with self.assertRaisesRegex(ValueError, "sequence of objects"):
            InferenceRunConfig(
                num_layers=10,
                layer_spacing=0.1,
                post_fdm_output_layer_compensations=None,
            )

    def test_accepts_and_validates_q_region_percent(self):
        config = InferenceRunConfig(
            num_layers=4,
            layer_spacing=0.1,
            post_fdm_output_q_region_percent=25.0,
        )
        self.assertAlmostEqual(config.post_fdm_output_q_region_percent, 25.0)
        for value in (0.0, 100.1):
            with self.assertRaisesRegex(ValueError, "q_region_percent"):
                InferenceRunConfig(
                    num_layers=4,
                    layer_spacing=0.1,
                    post_fdm_output_q_region_percent=value,
                )

    def test_accepts_and_validates_q_transition_percent(self):
        config = InferenceRunConfig(
            num_layers=4,
            layer_spacing=0.1,
            post_fdm_output_q_region_percent=10.0,
            post_fdm_output_q_transition_percent=5.0,
        )
        self.assertAlmostEqual(config.post_fdm_output_q_transition_percent, 5.0)
        for value in (-0.1, 91.0):
            with self.assertRaisesRegex(ValueError, "q_transition_percent"):
                InferenceRunConfig(
                    num_layers=4,
                    layer_spacing=0.1,
                    post_fdm_output_q_region_percent=10.0,
                    post_fdm_output_q_transition_percent=value,
                )

    def test_accepts_multilayer_batch_config(self):
        config = InferenceRunConfig(
            num_layers=2,
            layer_spacing=0.1,
            batch_mode=True,
            h5_dir="inputs",
            output_dir="outputs",
            output_prefix="pre_",
            prediction_group_path="prediction/pdgcn_multilayer",
        )

        self.assertTrue(config.batch_mode)
        self.assertEqual(config.h5_dir, "inputs")
        self.assertEqual(config.output_dir, "outputs")
        self.assertEqual(config.output_prefix, "pre_")
        self.assertEqual(config.prediction_group_path, "prediction/pdgcn_multilayer")

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
        with self.assertRaisesRegex(ValueError, "h5_dir"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, h5_dir="")
        with self.assertRaisesRegex(ValueError, "output_dir"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, output_dir="")
        with self.assertRaisesRegex(ValueError, "output_prefix"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, output_prefix="")
        with self.assertRaisesRegex(ValueError, "prediction_group_path"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, prediction_group_path="prediction//bad")

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
