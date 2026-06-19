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
    parser.add_argument(
        "--batch",
        dest="batch",
        action="store_true",
        default=None,
        help="Override config and run all HDF5 files in the selected input directory.",
    )
    parser.add_argument(
        "--no-batch",
        dest="batch",
        action="store_false",
        help="Override config and force single-file inference.",
    )
    parser.add_argument("--h5-dir", default=None, help="Optional batch input HDF5 directory override.")
    parser.add_argument("--output", default=None, help="Optional output HDF5 path override.")
    parser.add_argument("--output-dir", default=None, help="Optional batch output directory override.")
    parser.add_argument("--output-prefix", default=None, help="Optional batch output filename prefix.")
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
        batch=args.batch,
        h5_dir=args.h5_dir,
        output_path=args.output,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        vtu_output_dir=args.vtu_output_dir,
        vtu_interval=args.vtu_interval,
        mode=args.mode,
    )
    if result.get("batch_mode"):
        print(f"batch_mode: {result['batch_mode']}")
        print(f"output_dir: {result['output_dir']}")
        print(f"checkpoint: {result['checkpoint_path']}")
        print(f"processed: {result['processed_count']}")
        print(f"succeeded: {result['succeeded_count']}")
        print(f"failed: {result['failed_count']}")
        for item in result["results"]:
            status = item["status"]
            print(f"{status}: {item['h5_path']} -> {item.get('output_path', '')}")
            if status == "failed":
                print(f"  error: {item['error']}")
        return 1 if result["failed_count"] else 0
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
