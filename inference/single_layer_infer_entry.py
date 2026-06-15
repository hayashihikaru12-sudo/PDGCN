import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from inference.single_layer import run_single_layer_inference_from_config


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pdgcn_single_layer_infer.example.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run single-layer PD-GCN inference and VTU diagnostics.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to a single-layer inference JSON config.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path override.")
    parser.add_argument("--h5", default=None, help="Optional HDF5 input file override.")
    parser.add_argument("--output", default=None, help="Optional output HDF5 path override.")
    parser.add_argument("--vtu-output-dir", default=None, help="Optional VTU output directory override.")
    parser.add_argument("--vtu-interval", type=int, default=None, help="Optional VTU output interval override.")
    parser.add_argument(
        "--mode",
        choices=("autoregressive", "teacher_forcing", "both"),
        default=None,
        help="Optional inference mode override.",
    )
    args = parser.parse_args(argv)
    result = run_single_layer_inference_from_config(
        args.config,
        checkpoint=args.checkpoint,
        h5_path=args.h5,
        output_path=args.output,
        vtu_output_dir=args.vtu_output_dir,
        vtu_interval=args.vtu_interval,
        mode=args.mode,
    )
    print(f"output: {result['output_path']}")
    print(f"checkpoint: {result['checkpoint_path']}")
    print(f"h5: {result['h5_path']}")
    print(f"steps: {result['steps']}")
    print(f"mode: {result['mode']}")
    print(f"vtu_output_dir: {result['vtu_output_dir']}")
    print(f"vtu_interval: {result['vtu_interval']}")
    print(f"inference_seconds: {result['inference_seconds']:.6f}")
    print(f"render_seconds: {result['render_seconds']:.6f}")
    print(f"total_seconds: {result['total_seconds']:.6f}")
    print(f"rendered_steps: {result['rendered_steps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
