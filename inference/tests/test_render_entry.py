import json
import shutil
import unittest
from pathlib import Path

from inference.render_entry import _prediction_path_from_config


class RenderEntryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("inference/tests/_tmp_render_entry")
        if self.root.exists():
            shutil.rmtree(self.root)
        self.config_dir = self.root / "configs"
        self.config_dir.mkdir(parents=True)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_prediction_path_defaults_to_inference_output_path(self):
        config_path = self.config_dir / "pdgcn_infer.example.json"
        config_path.write_text(
            json.dumps({"inference": {"output_path": "../runs/pdgcn/prediction.h5"}}),
            encoding="utf-8",
        )

        resolved = _prediction_path_from_config(config_path)

        self.assertEqual(resolved, (self.root / "runs" / "pdgcn" / "prediction.h5").resolve())


if __name__ == "__main__":
    unittest.main()
