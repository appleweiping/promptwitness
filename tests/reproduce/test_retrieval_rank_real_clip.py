"""Authored, dependency-free checks for the real-CLIP qualification driver.

The success test injects a fake image library and role session. It does not
claim model, Landlock or dataset qualification; those need the separate run.
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

from reproduce import check_retrieval_rank_real_clip as qualification
from reproduce.retrieval_work_ledger import RetrievalWorkLedger


def test_missing_checkpoint_retains_failure_without_model_or_image_import(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output = tmp_path / "new-output"
    with pytest.raises(FileNotFoundError):
        qualification.check(tmp_path / "missing.pt", output)
    report = json.loads((output / "qualification.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAILED_RETAINED"
    assert report["error_type"] == "FileNotFoundError"
    assert report["scientific_result"] is False


@pytest.mark.parametrize("failure", ["none", "direct_score", "description_score"])
def test_authored_qualification_driver_wires_two_process_roles(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    checkpoint = tmp_path / "owned.pt"
    checkpoint.write_bytes(b"authored-not-a-real-weight")
    monkeypatch.setattr(qualification, "verify_checkpoint", lambda _path: "authored-sha")

    class FakeImage:
        def save(self, path):
            path.write_bytes(b"authored-image")

    image = types.ModuleType("PIL.Image")
    image.new = lambda *_args: FakeImage()
    draw = types.ModuleType("PIL.ImageDraw")
    draw.Draw = lambda _image: types.SimpleNamespace(rectangle=lambda *_args, **_kw: None)
    package = types.ModuleType("PIL")
    package.Image = image
    package.ImageDraw = draw
    monkeypatch.setitem(sys.modules, "PIL", package)
    monkeypatch.setitem(sys.modules, "PIL.Image", image)
    monkeypatch.setitem(sys.modules, "PIL.ImageDraw", draw)

    class FakeSession:
        def __init__(self, **kwargs):
            self.scratch = kwargs["scratch"]
            self.ready = {"query_count": 2, "gallery_sizes": {"cirr": 3}, "pid": os.getpid() + 1}
            ledger = RetrievalWorkLedger(self.scratch / "ranker-work.sqlite")
            try:
                ledger.run("load", "shared", "encoder_load", lambda: None)
                count = [0]

                def forward():
                    count[0] += 1

                for index in range(3):
                    ledger.run(
                        f"image:{index}",
                        "shared",
                        "image_encode",
                        forward,
                        forward_counter=lambda: count[0],
                    )
                for index in range(2):
                    ledger.run(
                        f"text:{index}",
                        "search",
                        "text_encode",
                        forward,
                        forward_counter=lambda: count[0],
                    )
                for index in range(2):
                    ledger.run(f"rank:{index}", "search", "rank_callback", lambda: None)
            finally:
                ledger.close()

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def rank_one(self, query_id):
            if query_id == "authored-q-red":
                return ("authored-blue", "authored-red", "authored-grey")
            return ("authored-red", "authored-blue", "authored-grey")

        def rank_description(self, query_id, request_id, target_description):
            assert (query_id, request_id, target_description) == (
                "authored-q-red",
                "authored-description-red",
                "a blue square on a white background",
            )
            ledger = RetrievalWorkLedger(self.scratch / "ranker-work.sqlite")
            try:
                count = [0]

                def forward():
                    count[0] += 1

                ledger.run(
                    "text:description",
                    "search",
                    "text_encode",
                    forward,
                    forward_counter=lambda: count[0],
                )
                ledger.run("rank:description", "search", "rank_callback", lambda: None)
            finally:
                ledger.close()
            return ("authored-blue", "authored-red", "authored-grey")

    def fake_score(store, scratch, dataset, stage, rankings):
        assert store.is_dir() and scratch.is_dir()
        assert (store / "search/inputs/ViT-L-14.pt").samefile(checkpoint)
        assert dataset == "cirr" and stage == "search"
        assert set(rankings) in ({"authored-q-red", "authored-q-blue"}, {"authored-q-red"})
        if (failure == "direct_score" and len(rankings) == 2) or (
            failure == "description_score" and len(rankings) == 1
        ):
            raise RuntimeError("authored scorer failure")
        return {
            "worker_pid": os.getpid() + 2,
            "observations": {identifier: {"primary_hit": 1} for identifier in rankings},
        }

    monkeypatch.setattr(qualification, "RestrictedRankerSession", FakeSession)
    monkeypatch.setattr(qualification, "score_rankings_restricted", fake_score)
    output = tmp_path / "new-output"
    if failure != "none":
        with pytest.raises(RuntimeError, match="authored scorer failure"):
            qualification.check(checkpoint, output)
        report = json.loads((output / "qualification.json").read_text(encoding="utf-8"))
        assert report["status"] == "FAILED_RETAINED"
        assert report["error_type"] == "RuntimeError"
        assert report["outer_operation_ledger"]["attempts"] == (
            4 if failure == "direct_score" else 5
        )
        assert report["outer_operation_ledger"]["failed"] == 1
        assert report["full_rankings"] == {
            "authored-q-red": ["authored-blue", "authored-red", "authored-grey"],
            "authored-q-blue": ["authored-red", "authored-blue", "authored-grey"],
        }
        if failure == "direct_score":
            assert "observations" not in report
        else:
            assert "observations" in report
            assert report["supplied_description_full_ranking"] == [
                "authored-blue",
                "authored-red",
                "authored-grey",
            ]
            assert "supplied_description_observation" not in report
        return
    report = qualification.check(checkpoint, output)
    assert report["status"] == "PASS_AUTHORED_REAL_CLIP_DESCRIPTION_RESTRICTED_SEARCH_AND_SCORER"
    assert report["inner_operation_ledger"]["known_forward_calls"] == 6
    assert report["outer_operation_ledger"]["attempts"] == 5
    assert report["ranker_pid"] != report["scorer_pid"]
    assert report["ranker_pid"] != report["description_scorer_pid"]
    assert report["supplied_description_full_ranking"] == [
        "authored-blue",
        "authored-red",
        "authored-grey",
    ]
    assert report["supplied_description_from_real_generator"] is False
    assert json.loads((output / "qualification.json").read_text(encoding="utf-8")) == report
