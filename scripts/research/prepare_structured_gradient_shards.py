"""Create private BBH split files before starting any optimization process."""

from __future__ import annotations

import argparse
from pathlib import Path

from promptwitness.structured_data import write_split_shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--holdout-dir", type=Path, required=True)
    args = parser.parse_args()
    write_split_shards(args.data_dir, args.manifest, args.shard_dir, args.holdout_dir)
    print(
        f"Prepared private fit/validation shards in {args.shard_dir}; holdout in {args.holdout_dir}"
    )


if __name__ == "__main__":
    main()
