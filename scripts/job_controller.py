"""Route JSON jobs to transfer learning or base/full model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import base_full_trainer, tester
from scripts.helper import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR


def _read_transfer_learning(config_path: Path) -> bool:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The top-level JSON value must be an object")
    transfer_learning = config.get("transfer_learning", True)
    if not isinstance(transfer_learning, bool):
        raise ValueError("'transfer_learning' must be a JSON boolean")
    return transfer_learning


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="directory containing job configuration JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="shared directory in which run_<index> folders are created",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    config_paths = sorted(input_dir.glob("*.json"))
    if not config_paths:
        print(f"No JSON input files found in {input_dir}", file=sys.stderr)
        return 1

    failed = 0
    for config_path in config_paths:
        try:
            transfer_learning = _read_transfer_learning(config_path)
            if transfer_learning:
                runner = tester.run_config
                destination = "tester.py"
            else:
                runner = base_full_trainer.run_config
                destination = "base_full_trainer.py"

            print(f"\nRouting {config_path.name} to {destination}")
            result_path = runner(config_path, output_dir)
            print(f"Results written to {result_path}")
        except Exception as exc:
            failed += 1
            print(f"Job failed for {config_path}: {exc}", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
