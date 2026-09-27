"""Enforce statement AND branch coverage of the new incremental core."""

import argparse
import json
from pathlib import Path


def check(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    files = {
        key: value["summary"]
        for key, value in report["files"].items()
        if "/promptwitness/incremental/" in "/" + key.replace("\\", "/")
    }
    if len(files) != 12:
        raise ValueError("complete 12-file core inventory required")
    statements = sum(row["num_statements"] for row in files.values())
    covered = sum(row["covered_lines"] for row in files.values())
    branches = sum(row["num_branches"] for row in files.values())
    covered_branches = sum(row["covered_branches"] for row in files.values())
    if (
        not statements
        or not branches
        or 10 * covered < 9 * statements
        or 10 * covered_branches < 9 * branches
    ):
        raise ValueError("core statement or branch coverage is below 90 percent")
    return {
        "statements": statements,
        "covered_statements": covered,
        "branches": branches,
        "covered_branches": covered_branches,
        "statement_percent": covered * 100 / statements,
        "branch_percent": covered_branches * 100 / branches,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.report)))
