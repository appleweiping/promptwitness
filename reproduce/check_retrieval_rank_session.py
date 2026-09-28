"""Authored-only persistent ranker witness; never a CLIP or benchmark result."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from reproduce.retrieval_direct_clip_ranker import load_input_only
from reproduce.retrieval_rank_role import serve


class AuthoredRanker:
    def __init__(
        self, *, input_dir, dataset, image_paths, checkpoint, image_weight, ledger, attempt_prefix
    ):
        store = input_dir.parents[1]
        for leaf in ("fit/gold", "search/gold", "selection/gold", "final/gold"):
            try:
                (store / leaf / "sentinel.txt").read_text(encoding="utf-8")
            except PermissionError:
                pass
            else:
                raise ValueError("ranker unexpectedly read a gold leaf")
        self.queries, self.galleries = load_input_only(input_dir, dataset)
        self.ledger, self.attempt_prefix = ledger, attempt_prefix

    def rank_one(self, query_id):
        if query_id not in self.queries:
            raise ValueError("query is outside authored input")
        _, _, category = self.queries[query_id]
        pool = self.galleries[category]

        def authored_rank():
            if self.attempt_prefix == "authored-stall":
                time.sleep(3)
            return ("target", *(item for item in pool if item != "target"))

        return self.ledger.run(
            f"{self.attempt_prefix}:query:{query_id}",
            "search",
            "rank_callback",
            authored_rank,
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_retrieval_rank_session.py STORE")
    serve(Path(sys.argv[1]), AuthoredRanker)
