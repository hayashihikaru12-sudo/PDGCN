import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    script_dir = Path(__file__).resolve().parent
    sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != script_dir]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from inference.infer_entry import main
from inference.io import run_multilayer_inference_from_config

__all__ = ["main", "run_multilayer_inference_from_config"]


if __name__ == "__main__":
    raise SystemExit(main())
