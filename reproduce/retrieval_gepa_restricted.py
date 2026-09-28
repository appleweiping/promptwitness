"""Authored GEPA qualification data and real retrieval scorer subprocesses.

This is deliberately not a benchmark loader. The trusted fixture builder sees
its own labels; the native optimizer still runs in that trusted controller
process. Only scoring calls cross the existing Landlock role boundary.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from reproduce.process_access import LEAVES
from reproduce.retrieval_role_scoring import score_rankings_restricted
from reproduce.retrieval_work_ledger import RetrievalWorkLedger


def qualification_report_header(format_name: str) -> dict:
    """A failed run starts with no claim that a worker was verified."""
    return {
        "format": format_name,
        "status": "FAILED_RETAINED",
        "scientific_result": False,
        "official_images_or_annotations_accessed": False,
        "real_generator_calls": 0,
        "gpu_hours": 0,
        "native_original_engine": True,
        "score_worker_is_real_landlock_child": False,
        "optimizer_process_is_gold_isolated": False,
        "real_clip_exercised": False,
        "fresh_run_only": True,
    }


def create_authored_store(
    root: Path,
    *,
    search_ids: tuple[str, ...],
    selection_ids: tuple[str, ...],
    pool: tuple[str, ...],
) -> None:
    """Create disjoint self-authored search/selection populations in a new store."""
    if (
        not search_ids
        or not selection_ids
        or set(search_ids) & set(selection_ids)
        or len(set(search_ids)) != len(search_ids)
        or len(set(selection_ids)) != len(selection_ids)
        or len(set(pool)) != len(pool)
        or "reference" not in pool
        or "target" not in pool
    ):
        raise ValueError("disjoint authored populations and unique gallery required")
    root.mkdir(parents=True, exist_ok=False)
    for leaf in LEAVES:
        (root / leaf).mkdir(parents=True)
    for stage, identifiers in (("search", search_ids), ("selection", selection_ids)):
        inputs = root / stage / "inputs"
        gold = root / stage / "gold"
        (inputs / "cirr-gallery.json").write_text(json.dumps({"cirr": pool}), encoding="utf-8")
        (inputs / "cirr.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "id": identifier,
                        "reference_id": "reference",
                        "modification": "authored modification",
                        "category": "",
                    }
                )
                + "\n"
                for identifier in identifiers
            ),
            encoding="utf-8",
        )
        (gold / "cirr.jsonl").write_text(
            "".join(
                json.dumps({"id": identifier, "target_id": "target", "subset": pool}) + "\n"
                for identifier in identifiers
            ),
            encoding="utf-8",
        )


class RestrictedGEPAQualifierScoring:
    """Meter native/reference scorer calls; audit already has its own ledger."""

    def __init__(
        self,
        *,
        store: Path,
        scratch_root: Path,
        ledger: RetrievalWorkLedger,
        search_ids: tuple[str, ...],
        selection_ids: tuple[str, ...],
    ) -> None:
        if not scratch_root.is_dir() or not store.is_dir():
            raise ValueError("existing private store and scratch are required")
        self.store = store
        self.scratch_root = scratch_root
        self.ledger = ledger
        self.search_ids = frozenset(search_ids)
        self.selection_ids = frozenset(selection_ids)
        self.worker_pids: list[int] = []
        self.reference_units = 0
        self.native_search_units = 0
        self.native_selection_units = 0
        self.audit_units = 0
        self._attempt = 0

    def _require_population(self, stage: str, rankings: dict[str, tuple[str, ...]]) -> None:
        if not rankings or any(not isinstance(ranking, tuple) for ranking in rankings.values()):
            raise ValueError("complete immutable rankings required")
        identifiers = set(rankings)
        if stage == "search" and identifiers <= self.search_ids:
            return
        if stage == "selection" and identifiers == self.selection_ids:
            return
        raise ValueError("scoring stage must match search subset or full selection")

    def score(
        self,
        stage: str,
        rankings: dict[str, tuple[str, ...]],
        *,
        purpose: str,
    ) -> dict:
        """Score one native batch or a complete reference through the real worker."""
        if purpose not in {"native", "reference"}:
            raise ValueError("native or reference scoring purpose required")
        self._require_population(stage, rankings)
        if purpose == "reference" and (stage != "search" or set(rankings) != self.search_ids):
            raise ValueError("reference requires the complete search population")
        self._attempt += 1
        scratch = Path(tempfile.mkdtemp(prefix="gepa-score-", dir=self.scratch_root))
        report = self.ledger.run(
            f"{purpose}:{stage}:{self._attempt}",
            stage,
            "restricted_score",
            lambda: score_rankings_restricted(self.store, scratch, "cirr", stage, rankings),
        )
        self._record_worker(report, rankings)
        if purpose == "reference":
            self.reference_units += len(rankings)
        elif stage == "search":
            self.native_search_units += len(rankings)
        else:
            self.native_selection_units += len(rankings)
        return report

    def audit_score(
        self,
        store: Path,
        scratch: Path,
        dataset: str,
        stage: str,
        rankings: dict[str, tuple[str, ...]],
    ) -> dict:
        """Observe, not replace, the audit bridge's actual restricted worker."""
        if store != self.store or dataset != "cirr" or stage != "search":
            raise ValueError("audit scorer identity changed")
        self._require_population(stage, rankings)
        report = score_rankings_restricted(store, scratch, dataset, stage, rankings)
        self._record_worker(report, rankings)
        self.audit_units += len(rankings)
        return report

    def _record_worker(self, report: dict, rankings: dict[str, tuple[str, ...]]) -> None:
        pid = report.get("worker_pid")
        if type(pid) is not int or pid <= 0 or pid == os.getpid():
            raise ValueError("a separate restricted scorer process is required")
        if set(report.get("observations", {})) != set(rankings):
            raise ValueError("restricted scorer did not cover the requested queries")
        self.worker_pids.append(pid)


def scenario_progress_receipt(
    scoring: RestrictedGEPAQualifierScoring,
    native_ledger: RetrievalWorkLedger,
    audit_ledger: RetrievalWorkLedger,
    *,
    audit_rank_units: int,
) -> dict:
    """Snapshot confirmed completions even when a later step fails.

    Failed or unresolved attempts remain in the ledgers; these successful-unit
    counters must not be interpreted as a zero cost for those attempts.
    """
    return {
        "reference_actual_restricted_units": scoring.reference_units,
        "native_search_scored_units": scoring.native_search_units,
        "native_selection_scored_units": scoring.native_selection_units,
        "audit_scored_units": scoring.audit_units,
        "audit_rank_units": audit_rank_units,
        "scorer_worker_launches": len(scoring.worker_pids),
        "separate_scorer_worker_pids": len(set(scoring.worker_pids)),
        "native_score_work": native_ledger.summary(),
        "audit_work": audit_ledger.summary(),
    }
