import unittest

import torch

from inference.fdm import compute_layer_fdm_coefficient, compute_layer_fdm_delta, compute_layer_implicit_fdm_step


class InferenceFDMTests(unittest.TestCase):
    def test_coefficient_uses_k_ratio(self):
        coefficient = compute_layer_fdm_coefficient(
            dt_star=2.0,
            inverse_pe=0.5,
            k_ratio=0.25,
            layer_spacing_star=0.5,
        )

        self.assertAlmostEqual(coefficient, 1.0)

    def test_delta_keeps_bottom_boundary_increment_zero(self):
        temperature = torch.tensor([[[2.0]], [[1.0]], [[0.0]]])

        delta = compute_layer_fdm_delta(
            temperature,
            dt_star=1.0,
            inverse_pe=1.0,
            k_ratio=0.1,
            layer_spacing_star=1.0,
        )

        self.assertTrue(torch.allclose(delta[-1], torch.zeros_like(delta[-1])))

    def test_implicit_step_keeps_bottom_boundary_fixed(self):
        temperature = torch.tensor([[[2.0]], [[1.0]], [[0.0]]])

        updated = compute_layer_implicit_fdm_step(
            temperature,
            dt_star=1.0,
            inverse_pe=1.0,
            k_ratio=0.1,
            layer_spacing_star=1.0,
            bottom_temperature_star=-0.25,
        )

        self.assertTrue(torch.allclose(updated[-1], torch.full_like(updated[-1], -0.25)))

    def test_implicit_step_damps_large_unstable_explicit_coefficient(self):
        temperature = torch.tensor([[[10.0]], [[0.0]], [[0.0]]])

        updated = compute_layer_implicit_fdm_step(
            temperature,
            dt_star=10.0,
            inverse_pe=1.0,
            k_ratio=1.0,
            layer_spacing_star=1.0,
        )

        self.assertGreaterEqual(float(updated.min()), 0.0)
        self.assertLess(float(updated.max()), 10.0)


if __name__ == "__main__":
    unittest.main()
