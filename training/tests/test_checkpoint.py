import unittest
from pathlib import Path

import torch
import torch.nn as nn

from training import load_checkpoint, save_checkpoint


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)


class CheckpointTests(unittest.TestCase):
    def test_save_and_load_checkpoint_restores_parameters(self):
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        path = Path("training/tests/_tmp_checkpoint.pt")
        if path.exists():
            path.unlink()
        try:
            save_checkpoint(model, optimizer, path, epoch=3, metadata={"scale_params": {"T_amb": 300.0}})

            loaded = TinyModel()
            loaded_optimizer = torch.optim.Adam(loaded.parameters(), lr=1e-3)
            checkpoint = load_checkpoint(loaded, loaded_optimizer, path)
        finally:
            if path.exists():
                path.unlink()

        for left, right in zip(model.parameters(), loaded.parameters()):
            self.assertTrue(torch.allclose(left, right))
        self.assertEqual(checkpoint["epoch"], 3)
        self.assertEqual(checkpoint["metadata"]["scale_params"]["T_amb"], 300.0)


if __name__ == "__main__":
    unittest.main()
