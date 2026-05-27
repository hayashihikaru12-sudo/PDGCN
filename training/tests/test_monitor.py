import tempfile
import unittest
from pathlib import Path

import h5py

from training.monitor import LossMonitor, TrainingProcessMonitor


class FakeClock:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        if not self.values:
            raise AssertionError("FakeClock exhausted.")
        return self.values.pop(0)


class MonitorTimingTests(unittest.TestCase):
    def test_loss_monitor_prints_epoch_elapsed_and_eta(self):
        messages = []
        monitor = LossMonitor(
            total_epochs=3,
            print_fn=messages.append,
            clock=FakeClock([0.0, 5.0, 11.0]),
        )

        monitor({"epoch": 0, "loss": 2.0})
        monitor({"epoch": 1, "loss": 1.0})

        self.assertIn("Epoch 1/3", messages[0])
        self.assertIn("loss=2", messages[0])
        self.assertIn("epoch_time=00:00:05", messages[0])
        self.assertIn("elapsed=00:00:05", messages[0])
        self.assertIn("eta=00:00:10", messages[0])
        self.assertIn("epoch_time=00:00:06", messages[1])
        self.assertIn("elapsed=00:00:11", messages[1])
        self.assertIn("eta=00:00:06", messages[1])

    def test_training_process_monitor_prints_timing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            messages = []
            monitor = TrainingProcessMonitor(
                total_epochs=2,
                history_path=Path(tmpdir) / "history.json",
                metrics_path=Path(tmpdir) / "monitor_data.h5",
                print_fn=messages.append,
                clock=FakeClock([0.0, 3.0]),
            )

            monitor({"epoch": 0, "loss_total": 4.0})

            self.assertIn("Epoch 1/2", messages[0])
            self.assertIn("loss=4", messages[0])
            self.assertIn("epoch_time=00:00:03", messages[0])
            self.assertIn("elapsed=00:00:03", messages[0])
            self.assertIn("eta=00:00:03", messages[0])
            with h5py.File(Path(tmpdir) / "monitor_data.h5", "r") as h5_file:
                self.assertIn("loss_smooth", h5_file["epoch_metrics"])
                self.assertIn("loss_smooth", h5_file["slice_metrics"])


if __name__ == "__main__":
    unittest.main()
