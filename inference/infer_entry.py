import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from inference.io import run_multilayer_inference_from_config


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pdgcn_infer.example.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run multilayer PD-GCN + 1D FDM inference.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to a PD-GCN JSON config.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path override.")
    parser.add_argument("--h5", default=None, help="Optional HDF5 input file override.")
    parser.add_argument("--output", default=None, help="Optional output HDF5 path override.")
    args = parser.parse_args(argv)
    result = run_multilayer_inference_from_config(
        args.config,
        checkpoint=args.checkpoint,
        h5_path=args.h5,
        output_path=args.output,
    )
    print(f"output: {result['output_path']}")
    print(f"checkpoint: {result['checkpoint_path']}")
    print(f"h5: {result['h5_path']}")
    print(f"steps: {result['steps']}")
    print(f"num_layers: {result['num_layers']}")
    print(f"fdm_coefficient: {result['fdm_coefficient']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
