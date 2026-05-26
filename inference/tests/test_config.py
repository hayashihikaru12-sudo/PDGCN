import unittest

from inference.config import InferenceRunConfig


class InferenceConfigTests(unittest.TestCase):
    def test_requires_multiple_layers(self):
        with self.assertRaisesRegex(ValueError, "num_layers"):
            InferenceRunConfig(num_layers=1, layer_spacing=0.1)

    def test_rejects_non_positive_layer_spacing(self):
        with self.assertRaisesRegex(ValueError, "layer_spacing"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.0)

    def test_rejects_non_positive_vtk_interval(self):
        with self.assertRaisesRegex(ValueError, "vtk_interval"):
            InferenceRunConfig(num_layers=2, layer_spacing=0.1, vtk_interval=0)


if __name__ == "__main__":
    unittest.main()
