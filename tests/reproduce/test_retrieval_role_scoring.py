"""Authored ranking/label isolation; no benchmark data or model call."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reproduce import retrieval_role_scoring as scorer
from reproduce.process_access import LEAVES, AccessBoundaryError
from reproduce.retrieval_role_scoring import FORMAT, score_rankings_restricted, score_worker


def store(tmp_path, dataset="cirr"):
    root = tmp_path / "store"
    for leaf in LEAVES:
        (root / leaf).mkdir(parents=True)
    pool = ("reference", "target", *(f"other-{n}" for n in range(9)))
    for stage in ("fit", "search", "selection"):
        inputs = root / stage / "inputs"
        gold = root / stage / "gold"
        rows = [
            {
                "id": "q1",
                "reference_id": "reference",
                "modification": "make it blue",
                "category": "",
            }
        ]
        labels = [{"id": "q1", "target_id": "target", "subset": ["reference", "target"]}]
        if dataset == "fashioniq":
            rows[0]["category"] = "dress"
            labels[0].pop("subset")
            gallery = {category: pool for category in ("dress", "shirt", "toptee")}
        else:
            gallery = {"cirr": pool}
        (inputs / f"{dataset}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (gold / f"{dataset}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8"
        )
        (inputs / f"{dataset}-gallery.json").write_text(json.dumps(gallery), encoding="utf-8")
    return root, pool


def test_unsupported_or_final_stage_never_launches(tmp_path):
    data, pool = store(tmp_path)
    with pytest.raises(AccessBoundaryError, match="unadmitted"):
        score_rankings_restricted(data, tmp_path, "cirr", "final", {"q1": pool})
    with pytest.raises(AccessBoundaryError, match="unsupported"):
        score_rankings_restricted(data, tmp_path, "circo", "search", {"q1": pool})


def test_malformed_rank_mapping_fails_before_launch(tmp_path):
    data, _ = store(tmp_path)
    with pytest.raises(ValueError, match="rank mapping"):
        score_rankings_restricted(data, tmp_path, "cirr", "search", {})


@pytest.mark.parametrize("dataset", ["cirr", "fashioniq"])
def test_authored_worker_uses_gold_only_inside_scorer(tmp_path, monkeypatch, dataset):
    data, pool = store(tmp_path, dataset)
    monkeypatch.setenv("PW_ACCESS_ROLE", "retrieval_search_scorer")
    monkeypatch.setenv("PW_ACCESS_STAGE", "search")
    monkeypatch.setenv("PW_LANDLOCK_ABI", "1")
    report = score_worker(
        data,
        {"format": FORMAT, "dataset": dataset, "stage": "search", "rankings": {"q1": list(pool)}},
    )
    assert report["observations"]["q1"]["primary_hit"] == 1
    assert "target_id" not in report["observations"]["q1"]
    with pytest.raises((ValueError, AccessBoundaryError)):
        score_worker(
            data,
            {
                "format": FORMAT,
                "dataset": dataset,
                "stage": "search",
                "rankings": {"q1": list(pool)},
                "target_id": "target",
            },
        )


def test_dispatch_uses_only_retrieval_scorer_role_and_no_gold_in_pipe(tmp_path, monkeypatch):
    data, pool = store(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    observed = {}

    def launch(role, stage, root, work, entrypoint, arguments, **kwargs):
        observed.update(
            {
                "role": role,
                "stage": stage,
                "message": kwargs["message"],
                "root": root,
                "arguments": arguments,
            }
        )
        report = {
            "format": FORMAT,
            "dataset": "cirr",
            "stage": "search",
            "role": role,
            "worker_pid": os.getpid() + 1,
            "landlock_abi": 1,
            "observations": {"q1": {"primary_hit": 1, "recalls": {}, "subset_recalls": {}}},
        }
        return subprocess.CompletedProcess([], 0, json.dumps(report), "")

    monkeypatch.setattr(scorer, "launch_role", launch)
    monkeypatch.chdir(tmp_path)
    score_rankings_restricted(Path("store"), scratch, "cirr", "search", {"q1": pool})
    assert (observed["role"], observed["stage"]) == ("retrieval_search_scorer", "search")
    assert observed["root"] == data.resolve()
    assert observed["arguments"] == [str(data.resolve())]
    assert set(observed["message"]) == {"format", "dataset", "stage", "rankings"}
    assert "target_id" not in json.dumps(observed["message"])


def test_authored_search_subset_but_fit_and_selection_require_complete_population(
    tmp_path, monkeypatch
):
    data, pool = store(tmp_path)
    for stage in ("fit", "search", "selection"):
        inputs = data / stage / "inputs/cirr.jsonl"
        gold = data / stage / "gold/cirr.jsonl"
        with inputs.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "id": "q2",
                        "reference_id": "reference",
                        "modification": "make it green",
                        "category": "",
                    }
                )
                + "\n"
            )
        with gold.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"id": "q2", "target_id": "target", "subset": ["reference", "target"]})
                + "\n"
            )
    monkeypatch.setenv("PW_ACCESS_ROLE", "retrieval_search_scorer")
    monkeypatch.setenv("PW_ACCESS_STAGE", "search")
    monkeypatch.setenv("PW_LANDLOCK_ABI", "1")
    request = {
        "format": FORMAT,
        "dataset": "cirr",
        "stage": "search",
        "rankings": {"q1": list(pool)},
    }
    assert set(score_worker(data, request)["observations"]) == {"q1"}
    for stage in ("fit", "selection"):
        monkeypatch.setenv("PW_ACCESS_ROLE", f"retrieval_{stage}_scorer")
        monkeypatch.setenv("PW_ACCESS_STAGE", stage)
        request["stage"] = stage
        with pytest.raises(ValueError, match="population"):
            score_worker(data, request)


@pytest.mark.parametrize("dataset", ["cirr", "fashioniq"])
def test_real_linux_restricted_scorer_reads_gold_without_message_labels(tmp_path, dataset):
    if sys.platform != "linux":
        pytest.skip("actual Landlock scorer execution requires Linux")
    data, pool = store(tmp_path, dataset)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    report = score_rankings_restricted(data, scratch, dataset, "search", {"q1": pool})
    assert report["role"] == "retrieval_search_scorer"
    assert report["worker_pid"] != 0 and report["landlock_abi"] >= 1
    assert report["observations"]["q1"]["primary_hit"] == 1
    assert "target_id" not in report["observations"]["q1"]
    assert report["observations"]["q1"]["recalls"]["5" if dataset == "cirr" else "10"] == 1


def test_real_linux_bad_full_ranking_is_failure_not_zero(tmp_path):
    if sys.platform != "linux":
        pytest.skip("actual Landlock scorer execution requires Linux")
    data, pool = store(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(ValueError, match="failed with exit"):
        score_rankings_restricted(data, scratch, "cirr", "search", {"q1": pool[:-1]})
    stderr = (scratch / "scorer-stderr.log").read_text(encoding="utf-8")
    assert "ValueError" in stderr and "ranking" in stderr
