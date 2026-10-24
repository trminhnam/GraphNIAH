"""Merge model inference pools after inference finishes; existing target entries win."""

import argparse
from pathlib import Path

from src.llm.inference_pool import merge_inference_pools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument(
        "--copy-back",
        action="store_true",
        help="After merging, overwrite all source pools with the target; create missing sources.",
    )
    args = parser.parse_args()
    try:
        summary = merge_inference_pools(
            args.sources, args.target, copy_back=args.copy_back
        )
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.exit(1, f"Merge failed: {error}\n")
    print(f"Merged {len(args.sources)} source pool(s) into {args.target}")
    print(
        f"  read={summary['read']} added={summary['added']} "
        f"skipped={summary['skipped']} target_total={summary['target_total']}"
    )
    if args.copy_back:
        print(f"Copied merged pool back to {summary['copied_back']} model pool(s).")


if __name__ == "__main__":
    main()
