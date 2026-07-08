import unittest

import torch

from pde import (
    compute_layer_fdm_coefficient,
    compute_layer_fdm_delta,
    compute_layer_implicit_fdm_step,
    compute_layer_implicit_source_distribution,
)


class FDMTests(unittest.TestCase):
    def test_compute_layer_fdm_delta_uses_k_ratio_and_boundary_forms(self):
        temperature = torch.tensor(
            [
                [[2.0], [4.0]],
                [[1.0], [1.0]],
                [[0.0], [0.0]],
            ]
        )

        delta = compute_layer_fdm_delta(
            temperature,
            dt_star=2.0,
            inverse_pe=0.5,
            k_ratio=0.2,
            layer_spacing_star=1.0,
        )

        coefficient = 0.2
        expected = torch.tensor(
            [
                [[coefficient * (1.0 - 2.0)], [coefficient * (1.0 - 4.0)]],
                [[coefficient * (2.0 - 2.0 * 1.0 + 0.0)], [coefficient * (4.0 - 2.0 * 1.0 + 0.0)]],
                [[0.0], [0.0]],
            ]
        )
        self.assertTrue(torch.allclose(delta, expected))

    def test_compute_layer_fdm_coefficient_rejects_invalid_spacing(self):
        with self.assertRaisesRegex(ValueError, "layer_spacing_star"):
            compute_layer_fdm_coefficient(dt_star=1.0, inverse_pe=1.0, k_ratio=1.0, layer_spacing_star=0.0)

    def test_compute_layer_implicit_fdm_step_is_exported_from_pde(self):
        temperature = torch.tensor([[[2.0]], [[0.0]]])

        updated = compute_layer_implicit_fdm_step(
            temperature,
            dt_star=1.0,
            inverse_pe=1.0,
            k_ratio=1.0,
            layer_spacing_star=1.0,
        )

        self.assertTrue(torch.allclose(updated, torch.tensor([[[1.0]], [[0.0]]])))

    def test_compute_layer_implicit_source_distribution_spreads_top_source(self):
        source = torch.tensor([[[1.0]], [[0.0]], [[0.0]]])

        distributed = compute_layer_implicit_source_distribution(
            source,
            dt_star=1.0,
            inverse_pe=1.0,
            k_ratio=0.1,
            layer_spacing_star=1.0,
        )

        expected = torch.tensor([[[1.2 / 1.31]], [[0.1 / 1.31]], [[0.0]]])
        self.assertTrue(torch.allclose(distributed, expected, atol=1e-6))
        self.assertGreater(float(distributed[1, 0, 0]), 0.0)
        self.assertTrue(torch.allclose(distributed[-1], torch.zeros_like(distributed[-1])))
        self.assertLessEqual(float(distributed[:-1].sum()), float(source[:-1].sum()))

    def test_compute_layer_implicit_source_distribution_noops_without_thickness_diffusion(self):
        source = torch.tensor([[[1.0]], [[0.0]], [[0.0]]])

        distributed = compute_layer_implicit_source_distribution(
            source,
            dt_star=1.0,
            inverse_pe=0.0,
            k_ratio=1.0,
            layer_spacing_star=1.0,
        )

        self.assertTrue(torch.allclose(distributed, source))


if __name__ == "__main__":
    unittest.main()
