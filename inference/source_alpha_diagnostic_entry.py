import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from inference.source_alpha_diagnostic import run_source_alpha_diagnostic


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "source_alpha_diagnostic.example.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sweep alpha_Q through full multilayer rollout and compare with FEM.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to diagnostic JSON config.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override.")
    parser.add_argument("--h5-dir", default=None, help="Optional multilayer FEM HDF5 directory override.")
    parser.add_argument("--output-dir", default=None, help="Optional diagnostic output directory override.")
    args = parser.parse_args(argv)
    result = run_source_alpha_diagnostic(
        args.config,
        checkpoint=args.checkpoint,
        h5_dir=args.h5_dir,
        output_dir=args.output_dir,
    )
    print(f"output_dir: {result['output_dir']}")
    print(f"successful_runs: {result['successful_runs']}")
    print(f"failed_runs: {result['failed_runs']}")
    print(f"case_metrics_csv: {result['case_metrics_csv']}")
    print(f"summary_csv: {result['summary_csv']}")
    print(f"summary_json: {result['summary_json']}")
    for path in result["plot_paths"]:
        print(f"plot: {path}")
    return 1 if result["failed_runs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

