"""Preserve and re-key one zero-request startup failure, never delete its costs."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--archive-run", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if not args.archive_run.startswith(args.run + ":startup-failed"):
        raise ValueError("archive identity must retain the original phase")
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    if evidence["ledger_run"] != args.run or evidence["model_requests_attempted"] != 0:
        raise ValueError("explicit zero-request failure evidence is required")
    with sqlite3.connect(args.ledger) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT elapsed,cold_start FROM runs WHERE model=?", (args.run,)
        ).fetchone()
        if (
            row is None
            or row[1] is not None
            or row[0] is None
            or not math.isfinite(row[0])
            or row[0] < 0
            or connection.execute(
                "SELECT COUNT(*) FROM attempts WHERE model=?", (args.run,)
            ).fetchone()[0]
            != 0
        ):
            raise ValueError("only a finished, zero-request pre-load failure is recoverable")
        if connection.execute("SELECT 1 FROM runs WHERE model=?", (args.archive_run,)).fetchone():
            raise ValueError("failure archive already exists")
        connection.execute("UPDATE runs SET model=? WHERE model=?", (args.archive_run, args.run))
        print(json.dumps({"preserved_run": args.archive_run, "elapsed_seconds": row[0]}))


if __name__ == "__main__":
    main()
