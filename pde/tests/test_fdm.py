import unittest

import torch

from pde import (
    compute_layer_fdm_coefficient,
    compute_layer_fdm_delta,
    compute_layer_implicit_fdm_step,
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

    def test_implicit_fdm_step_defaults_to_plain_fdm_when_fin_gamma_is_none(self):
        temperature = torch.tensor([[[2.0]], [[0.0]]])

        plain = compute_layer_implicit_fdm_step(
            temperature,
            dt_star=1.0,
            inverse_pe=1.0,
            k_ratio=1.0,
            layer_spacing_star=1.0,
        )
        explicit_none = compute_layer_implicit_fdm_step(
            temperature,
            dt_star=1.0,
            inverse_pe=1.0,
            k_ratio=1.0,
            layer_spacing_star=1.0,
            fin_cooling_gamma_star=None,
        )

        self.assertTrue(torch.allclose(explicit_none, plain))

    def test_implicit_fdm_step_applies_fin_cooling_diagonal(self):
        temperature = torch.tensor([[[2.0]], [[0.0]]])

        updated = compute_layer_implicit_fdm_step(
            temperature,
            dt_star=1.0,
            inverse_pe=1.0,
            k_ratio=1.0,
            layer_spacing_star=1.0,
            fin_cooling_gamma_star=1.0,
            fin_cooling_skip_top_layers=0,
        )

        self.assertTrue(torch.allclose(updated, torch.tensor([[[2.0 / 3.0]], [[0.0]]])))

    def test_fin_cooling_rejects_nonzero_skip_top_layers_when_gamma_is_set(self):
        with self.assertRaisesRegex(ValueError, "fin_cooling_skip_top_layers"):
            compute_layer_implicit_fdm_step(
                torch.tensor([[[2.0]], [[2.0]], [[0.0]]]),
                dt_star=1.0,
                inverse_pe=1.0,
                k_ratio=1.0,
                layer_spacing_star=1.0,
                fin_cooling_gamma_star=1.0,
                fin_cooling_skip_top_layers=1,
            )

    def test_fin_cooling_accepts_layerwise_gamma(self):
        temperature = torch.tensor([[[3.0]], [[2.0]], [[1.0]], [[0.0]]])

        uniform = compute_layer_implicit_fdm_step(
            temperature,
            dt_star=1.0,
            inverse_pe=1.0,
            k_ratio=1.0,
            layer_spacing_star=1.0,
            fin_cooling_gamma_star=1.0,
            fin_cooling_skip_top_layers=0,
        )
        layerwise = compute_layer_implicit_fdm_step(
            temperature,
            dt_star=1.0,
            inverse_pe=1.0,
            k_ratio=1.0,
            layer_spacing_star=1.0,
            fin_cooling_gamma_star=[0.0, 1.0, 4.0],
            fin_cooling_skip_top_layers=0,
        )

        self.assertGreater(float(layerwise[0, 0, 0]), float(uniform[0, 0, 0]))
        self.assertLess(float(layerwise[2, 0, 0]), float(uniform[2, 0, 0]))

    def test_fin_cooling_rejects_wrong_layerwise_gamma_length(self):
        with self.assertRaisesRegex(ValueError, "active layer count"):
            compute_layer_implicit_fdm_step(
                torch.tensor([[[3.0]], [[2.0]], [[1.0]], [[0.0]]]),
                dt_star=1.0,
                inverse_pe=1.0,
                k_ratio=1.0,
                layer_spacing_star=1.0,
                fin_cooling_gamma_star=[1.0, 2.0],
                fin_cooling_skip_top_layers=0,
            )

    def test_implicit_fdm_step_accepts_layer_interface_scales(self):
        temperature = torch.tensor([[[3.0]], [[2.0]], [[0.0]]])

        isolated_top = compute_layer_implicit_fdm_step(
            temperature,
            dt_star=1.0,
            inverse_pe=1.0,
            k_ratio=1.0,
            layer_spacing_star=1.0,
            layer_interface_scales=[0.0, 1.0],
        )

        self.assertTrue(torch.allclose(isolated_top, torch.tensor([[[3.0]], [[1.0]], [[0.0]]])))

    def test_implicit_fdm_step_rejects_invalid_layer_interface_scales(self):
        with self.assertRaisesRegex(ValueError, "layer_interface_scales"):
            compute_layer_implicit_fdm_step(
                torch.tensor([[[3.0]], [[2.0]], [[0.0]]]),
                dt_star=1.0,
                inverse_pe=1.0,
                k_ratio=1.0,
                layer_spacing_star=1.0,
                layer_interface_scales=[1.0],
            )
        with self.assertRaisesRegex(ValueError, "layer_interface_scales"):
            compute_layer_implicit_fdm_step(
                torch.tensor([[[3.0]], [[2.0]], [[0.0]]]),
                dt_star=1.0,
                inverse_pe=1.0,
                k_ratio=1.0,
                layer_spacing_star=1.0,
                layer_interface_scales=-1.0,
            )

    def test_fin_cooling_rejects_invalid_skip_top_layers(self):
        for value in (-1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "fin_cooling_skip_top_layers"):
                    compute_layer_implicit_fdm_step(
                        torch.tensor([[[2.0]], [[0.0]]]),
                        dt_star=1.0,
                        inverse_pe=1.0,
                        k_ratio=1.0,
                        layer_spacing_star=1.0,
                        fin_cooling_gamma_star=1.0,
                        fin_cooling_skip_top_layers=value,
                    )

    def test_fin_cooling_rejects_negative_gamma(self):
        with self.assertRaisesRegex(ValueError, "fin_cooling_gamma_star"):
            compute_layer_implicit_fdm_step(
                torch.tensor([[[2.0]], [[0.0]]]),
                dt_star=1.0,
                inverse_pe=1.0,
                k_ratio=1.0,
                layer_spacing_star=1.0,
                fin_cooling_gamma_star=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
