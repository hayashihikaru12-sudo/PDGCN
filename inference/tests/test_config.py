import unittest

from inference.config import InferenceRunConfig


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


if __name__ == "__main__":
    unittest.main()
