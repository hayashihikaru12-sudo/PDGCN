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

    def test_accepts_alternating_order_average_flag(self):
        config = InferenceRunConfig(
            num_layers=2,
            layer_spacing=0.1,
            use_alternating_order_average=True,
        )

        self.assertTrue(config.use_alternating_order_average)

    def test_accepts_fin_cooling_defaults_and_overrides(self):
        default_config = InferenceRunConfig(num_layers=2, layer_spacing=0.1)
        self.assertTrue(default_config.fin_cooling_enabled)
        self.assertAlmostEqual(default_config.fdm_k_ratio_scale, 1.0)
        self.assertIsNone(default_config.fdm_layer_interface_scales)
        self.assertAlmostEqual(default_config.fdm_top_surface_loss_gamma_dt, 0.0)
        self.assertAlmostEqual(default_config.fdm_top_surface_loss_velocity_exponent, 0.0)
        self.assertAlmostEqual(default_config.fdm_top_surface_loss_reference_velocity_star, 1.0)
        self.assertEqual(default_config.fin_cooling_mode, "r_char")
        self.assertIsNone(default_config.fin_cooling_r_char_star)
        self.assertIsNone(default_config.fin_cooling_gamma_star)
        self.assertAlmostEqual(default_config.fin_cooling_beta_h, 3.0)
        self.assertEqual(default_config.fin_cooling_skip_top_layers, 0)
        self.assertEqual(default_config.fin_cooling_layer_profile, "uniform")
        self.assertAlmostEqual(default_config.fin_cooling_layer_profile_strength, 0.0)

        config = InferenceRunConfig(
            num_layers=2,
            layer_spacing=0.1,
            fin_cooling_enabled=False,
            fin_cooling_mode="direct",
            fin_cooling_gamma_star=0.25,
            fdm_k_ratio_scale=0.85,
            fdm_layer_interface_scales=[1.0],
            fdm_top_surface_loss_gamma_dt=0.1,
            fdm_top_surface_loss_velocity_exponent=1.5,
            fdm_top_surface_loss_reference_velocity_star=0.8,
            fin_cooling_beta_h=5.0,
            fin_cooling_skip_top_layers=1,
            fin_cooling_layer_profile="linear",
            fin_cooling_layer_profile_strength=0.5,
        )

        self.assertFalse(config.fin_cooling_enabled)
        self.assertEqual(config.fin_cooling_mode, "direct")
        self.assertAlmostEqual(config.fin_cooling_gamma_star, 0.25)
        self.assertAlmostEqual(config.fdm_k_ratio_scale, 0.85)
        self.assertEqual(config.fdm_layer_interface_scales, [1.0])
        self.assertAlmostEqual(config.fdm_top_surface_loss_gamma_dt, 0.1)
        self.assertAlmostEqual(config.fdm_top_surface_loss_velocity_exponent, 1.5)
        self.assertAlmostEqual(config.fdm_top_surface_loss_reference_velocity_star, 0.8)
        self.assertAlmostEqual(config.fin_cooling_beta_h, 5.0)
        self.assertEqual(config.fin_cooling_skip_top_layers, 1)
        self.assertEqual(config.fin_cooling_layer_profile, "linear")
        self.assertAlmostEqual(config.fin_cooling_layer_profile_strength, 0.5)

    def test_rejects_invalid_fin_cooling_config(self):
        with self.assertRaisesRegex(ValueError, "fin_cooling_enabled"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, fin_cooling_enabled="true")
        with self.assertRaisesRegex(ValueError, "fdm_k_ratio_scale"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, fdm_k_ratio_scale=0.0)
        with self.assertRaisesRegex(ValueError, "fdm_layer_interface_scales"):
            InferenceRunConfig(num_layers=3, layer_spacing=0.1, fdm_layer_interface_scales=[1.0])
        with self.assertRaisesRegex(ValueError, "fdm_layer_interface_scales"):
            InferenceRunConfig(num_layers=3, layer_spacing=0.1, fdm_layer_interface_scales=-1.0)
        with self.assertRaisesRegex(ValueError, "fdm_top_surface_loss_gamma_dt"):
            InferenceRunConfig(num_layers=3, layer_spacing=0.1, fdm_top_surface_loss_gamma_dt=-0.1)
        with self.assertRaisesRegex(ValueError, "fdm_top_surface_loss_velocity_exponent"):
            InferenceRunConfig(num_layers=3, layer_spacing=0.1, fdm_top_surface_loss_velocity_exponent=-0.1)
        with self.assertRaisesRegex(ValueError, "fdm_top_surface_loss_reference_velocity_star"):
            InferenceRunConfig(num_layers=3, layer_spacing=0.1, fdm_top_surface_loss_reference_velocity_star=0.0)
        with self.assertRaisesRegex(ValueError, "fin_cooling_mode"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, fin_cooling_mode="legacy")
        with self.assertRaisesRegex(ValueError, "fin_cooling_r_char_star"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, fin_cooling_r_char_star=0.0)
        with self.assertRaisesRegex(ValueError, "fin_cooling_gamma_star"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, fin_cooling_mode="direct")
        with self.assertRaisesRegex(ValueError, "fin_cooling_gamma_star"):
            InferenceRunConfig(
                num_layers=2,
                layer_spacing=0.1,
                fin_cooling_mode="direct",
                fin_cooling_gamma_star=-1.0,
            )
        with self.assertRaisesRegex(ValueError, "fin_cooling_beta_h"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, fin_cooling_beta_h=0.0)
        with self.assertRaisesRegex(ValueError, "fin_cooling_beta_h"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, fin_cooling_beta_h=-1.0)
        for value in (-1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "fin_cooling_skip_top_layers"):
                    InferenceRunConfig(
                        num_layers=2,
                        layer_spacing=0.1,
                        fin_cooling_skip_top_layers=value,
                    )
        with self.assertRaisesRegex(ValueError, "fin_cooling_skip_top_layers"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, fin_cooling_skip_top_layers=1)
        with self.assertRaisesRegex(ValueError, "fin_cooling_layer_profile"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, fin_cooling_layer_profile="steep")
        with self.assertRaisesRegex(ValueError, "fin_cooling_layer_profile_strength"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, fin_cooling_layer_profile_strength=-0.1)

    def test_rejects_non_boolean_alternating_order_average_flag(self):
        with self.assertRaisesRegex(ValueError, "use_alternating_order_average"):
            InferenceRunConfig(
                num_layers=2,
                layer_spacing=0.1,
                use_alternating_order_average="true",
            )

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

    def test_accepts_multilayer_batch_fields(self):
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

    def test_rejects_bad_multilayer_batch_fields(self):
        with self.assertRaisesRegex(ValueError, "h5_dir"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, h5_dir="")
        with self.assertRaisesRegex(ValueError, "output_dir"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, output_dir="")
        with self.assertRaisesRegex(ValueError, "output_prefix"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, output_prefix="")
        with self.assertRaisesRegex(ValueError, "prediction_group_path"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, prediction_group_path="prediction//bad")

    def test_rejects_layer_angle_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "layer_fiber_angles_deg length"):
            InferenceRunConfig(num_layers=3, layer_spacing=0.1, layer_fiber_angles_deg=[0.0, 45.0])

    def test_rejects_nonzero_base_layer_angle(self):
        with self.assertRaisesRegex(ValueError, "layer_fiber_angles_deg\\[0\\]"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, layer_fiber_angles_deg=[10.0, 45.0])

    def test_rejects_invalid_normal_offset_sign(self):
        with self.assertRaisesRegex(ValueError, "normal_offset_sign"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, normal_offset_sign=0)

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
