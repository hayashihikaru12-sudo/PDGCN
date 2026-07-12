import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from inference.source_alpha_diagnostic import (
    SourceAlphaDiagnosticConfig,
    compute_case_temperature_metrics,
    summarize_alpha_metrics,
    write_diagnostic_plots,
)


class SourceAlphaDiagnosticTests(unittest.TestCase):
    def test_alpha_q_is_mapped_from_one_to_two(self):
        config = SourceAlphaDiagnosticConfig(alpha_q_values=[1.0, 1.2])

        self.assertEqual(config.alpha_q_values, (1.0, 1.2))

    def test_rejects_alpha_q_below_existing_source_baseline(self):
        with self.assertRaisesRegex(ValueError, r"within \[1, 2\]"):
            SourceAlphaDiagnosticConfig(alpha_q_values=[0.9])

    def test_case_metrics_use_prediction_minus_fem_peak_bias(self):
        fem = np.array(
            [
                [[[[100.0]], [[110.0]]], [[[90.0]], [[95.0]]]],
                [[[[120.0]], [[130.0]]], [[[100.0]], [[105.0]]]],
            ]
        ).reshape(2, 2, 2, 1)
        prediction = fem + 10.0

        metrics = compute_case_temperature_metrics(prediction, fem)

        self.assertAlmostEqual(metrics["global_peak_bias"], 10.0)
        self.assertAlmostEqual(metrics["layer_1_peak_bias"], 10.0)
        self.assertAlmostEqual(metrics["layer_2_peak_bias"], 10.0)
        self.assertAlmostEqual(metrics["field_rmse"], 10.0)

    def test_summary_recovers_peak_fit_and_pooled_rmse(self):
        records = []
        for alpha_q, offset in ((1.0, -10.0), (1.2, 0.0)):
            for case_index, fem_peak in enumerate((100.0, 200.0, 300.0)):
                records.append(
                    {
                        "alpha_q": alpha_q,
                        "global_fem_peak": fem_peak,
                        "global_pred_peak": fem_peak + offset,
                        "field_valid_count": 4,
                        "field_rmse": abs(offset),
                        "field_mae": abs(offset),
                        "field_max_abs_error": abs(offset),
                        "layer_1_valid_count": 2,
                        "layer_1_rmse": abs(offset),
                        "layer_1_mae": abs(offset),
                        "layer_1_peak_bias": offset,
                    }
                )

        summary = summarize_alpha_metrics(records, num_layers=1)

        self.assertAlmostEqual(summary[0]["global_peak_fit_slope"], 1.0)
        self.assertAlmostEqual(summary[0]["global_peak_fit_intercept"], -10.0)
        self.assertAlmostEqual(summary[1]["global_peak_fit_intercept"], 0.0)
        self.assertAlmostEqual(summary[1]["field_rmse"], 0.0)

    def test_writes_dependency_free_svg_plots(self):
        case_records = [
            {"alpha_q": 1.0, "global_fem_peak": 100.0, "global_pred_peak": 90.0},
            {"alpha_q": 1.0, "global_fem_peak": 200.0, "global_pred_peak": 190.0},
        ]
        summary_records = [
            {
                "alpha_q": 1.0,
                "global_peak_fit_slope": 1.0,
                "global_peak_fit_intercept": -10.0,
                "field_rmse": 8.0,
                "layer_1_peak_bias_mean": -10.0,
                "layer_1_peak_bias_sem": 1.0,
                "layer_2_peak_bias_mean": -5.0,
                "layer_2_peak_bias_sem": 0.5,
            }
        ]

        with TemporaryDirectory() as temp_dir:
            paths = write_diagnostic_plots(Path(temp_dir), case_records, summary_records)

            self.assertEqual(len(paths), 2)
            self.assertTrue(all(path.exists() for path in paths))
            self.assertTrue(all("<svg" in path.read_text(encoding="utf-8") for path in paths))


if __name__ == "__main__":
    unittest.main()
