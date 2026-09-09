"""Hand-counted oracles for the original, explicitly synthetic interview example."""

import importlib.util
import json
import socket
from pathlib import Path

import pytest

from promptwitness.interview_journal import InterviewJournal
from promptwitness.interview_models import contract_json
from promptwitness.interview_state import interview_report, schedule
from promptwitness.providers import OpenAICompatibleProvider

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "interview_demo.py"


@pytest.fixture(scope="module")
def demo_module():
    specification = importlib.util.spec_from_file_location("original_interview_demo", EXAMPLE)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def completed_demo(tmp_path_factory, demo_module):
    directory = tmp_path_factory.mktemp("interview-demo") / "new-output"

    def no_external_effect(*_args, **_kwargs):
        raise AssertionError("the synthetic example must never use a live model or network")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket, "create_connection", no_external_effect)
        patch.setattr(socket, "socket", no_external_effect)
        patch.setattr(OpenAICompatibleProvider, "complete", no_external_effect)
        result = demo_module.run_demo(directory)
    return directory, result


def test_authored_trace_counts_and_separate_denominators(completed_demo):
    directory, report = completed_demo
    assert report["synthetic"] is True
    assert report["model_calls"] == report["network_calls"] == 0
    assert report["scripted_provider_calls"] == 9
    assert report["scripted_stage_counts"] == {"question": 5, "analysis": 4}
    assert report["database_reopens"] == 2
    assert report["first"]["revision"] == report["first"]["event_count"] == 18
    assert report["first"]["event_kinds"] == [
        "created",
        "question_reserved",
        "question_committed",
        "answer_committed",
        "analysis_reserved",
        "analysis_committed",
        "proposal_decided",
        "question_reserved",
        "question_committed",
        "answer_committed",
        "analysis_reserved",
        "analysis_committed",
        "question_reserved",
        "question_committed",
        "answer_committed",
        "analysis_reserved",
        "analysis_committed",
        "finished",
    ]
    assert report["first"]["question_topics"] == ["preparation", "checks", "accessibility"]
    assert report["first"]["reason"] == "agenda_completed"
    assert report["first"]["questions"] == report["first"]["answers"] == 3
    assert report["first"]["local_memories"] == 3
    assert report["first"]["coverage"]["required"] == {
        "total": 2,
        "covered": 2,
        "partial": 0,
        "unanswered": 0,
        "assessed_coverage": 1.0,
    }
    assert report["first"]["coverage"]["emergent"] == {
        "total": 1,
        "covered": 1,
        "partial": 0,
        "unanswered": 0,
        "assessed_coverage": 1.0,
    }
    assert report["other_participant"]["revision"] == 7
    assert report["other_participant"]["local_memories"] == 1
    assert report["second"]["revision"] == 4
    assert report["second"]["reason"] == "participant_stopped"
    assert report["second"]["answers"] == report["second"]["local_memories"] == 0
    assert report["second"]["imported_memories"] == 3
    assert report["second"]["coverage"]["required"] == {
        "total": 2,
        "covered": 0,
        "partial": 0,
        "unanswered": 2,
        "assessed_coverage": 0.0,
    }
    assert report["checkpoints"] == {
        "closed_after_answer_revision": 4,
        "pending_review_revision": 6,
        "accepted_proposal_revision": 7,
        "required_complete_revision": 12,
        "required_complete_emergent_covered": 0,
        "second_recall_revision": 3,
        "second_recalled_memories": 3,
        "second_required_covered_after_recall": 0,
    }
    assert json.loads((directory / "summary.json").read_text(encoding="utf-8")) == report


def test_reopened_journal_contains_exact_authored_text_spans_and_review_boundary(completed_demo):
    directory, _ = completed_demo
    with InterviewJournal(directory / "interviews.sqlite") as journal:
        first = journal.read("mira-first")
        expected = "I mapped the lantern walkway 🏮.\r\nThe quiet lane keeps {{template}} literal."
        assert first.answers[0].text == expected
        references = [entry.memory.evidence[0].reference for entry in first.memories]
        assert references[0].quote == expected
        assert references[0].start == 0 and references[0].end == len(expected)
        assert references[0].answer_id == "answer-route"
        assert {entry.revision for entry in first.memories} == {6, 12, 17}
        review = journal.read("mira-first", revision=6)
        assert schedule(review).kind == "await_review"
        assert len(review.questions) == 1
        assert review.proposals[0].status == "pending"
        assert review.proposals[0].proposal.evidence[0].reference.quote == expected
        assert interview_report(review)["assessed_coverage"]["emergent"]["total"] == 0
        accepted = journal.read("mira-first", revision=7)
        assert accepted.proposals[0].status == "accepted"
        assert interview_report(accepted)["assessed_coverage"]["emergent"]["total"] == 1
        required_done = journal.read("mira-first", revision=12)
        assert required_done.finished_reason is None
        assert schedule(required_done).target.topic_id == "accessibility"
        assert len(required_done.questions) == 2


def test_second_interview_recalls_only_committed_same_participant_without_coverage(completed_demo):
    directory, report = completed_demo
    with InterviewJournal(directory / "interviews.sqlite") as journal:
        source = journal.read("mira-first")
        other = journal.read("noel-first")
        returned = journal.read("mira-return", revision=3)
        assert returned.assessments == ()
        assert returned.memory_snapshot.heads == (source.head,)
        assert {answer.participant_id for answer in returned.memory_snapshot.answers} == {
            "synthetic-mira"
        }
        cited = set(returned.questions[0].memory_ids)
        assert len(cited) == 3
        assert cited == {entry.memory.memory_id for entry in source.memories}
        assert cited.isdisjoint(entry.memory.memory_id for entry in other.memories)
        assert (
            len(journal.events("mira-first"))
            + len(journal.events("noel-first"))
            + len(journal.events("mira-return"))
            == 29
        )
        with pytest.raises(KeyError):
            journal.read("forbidden-mix")
    assert report["cross_participant_selection_rejected"] is True
    assert report["rejected_creation_left_no_event"] is True


def test_aggregate_report_does_not_publish_answers_or_raw_envelopes(completed_demo):
    _, report = completed_demo
    encoded = contract_json(report)
    for private_fixture_text in (
        "{{template}}",
        "I mapped",
        "Mira's notes",
        "finish_reason",
        "source-quote",
    ):
        assert private_fixture_text not in encoded
    for section in ("first", "other_participant", "second"):
        for field in ("head_sha256", "export_sha256"):
            assert len(report[section][field]) == 64
            int(report[section][field], 16)
    assert "not interview-quality evidence" in report["evidence_scope"]


def test_existing_artifacts_are_never_reused_or_overwritten(completed_demo, demo_module):
    directory, _ = completed_demo
    database = directory / "interviews.sqlite"
    summary = directory / "summary.json"
    before = database.read_bytes(), summary.read_bytes()
    with pytest.raises(FileExistsError):
        demo_module.run_demo(directory)
    assert (database.read_bytes(), summary.read_bytes()) == before


def test_cli_existing_directory_fails_without_claiming_success(tmp_path, demo_module, capsys):
    directory = tmp_path / "existing"
    directory.mkdir()
    marker = directory / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    assert demo_module.main(["--output-dir", str(directory)]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "did not complete" in output.err
    assert marker.read_text(encoding="utf-8") == "keep"


def test_cli_default_routes_fresh_child_through_its_temporary_context(
    tmp_path, demo_module, monkeypatch, capsys
):
    parent = tmp_path / "temporary-parent"
    parent.mkdir()
    passed = []

    class TemporaryFixture:
        def __init__(self, *, prefix):
            assert prefix == "promptwitness-interview-demo-"

        def __enter__(self):
            return str(parent)

        def __exit__(self, *_args):
            passed.append("context-closed")

    def run(directory):
        assert directory == parent / "demo"
        assert not directory.exists()
        passed.append("fresh-child")
        return {"synthetic": True, "model_calls": 0}

    monkeypatch.setattr(demo_module, "TemporaryDirectory", TemporaryFixture)
    monkeypatch.setattr(demo_module, "run_demo", run)
    assert demo_module.main([]) == 0
    assert passed == ["fresh-child", "context-closed"]
    assert json.loads(capsys.readouterr().out) == {"synthetic": True, "model_calls": 0}


def test_stdout_failure_does_not_claim_completed_demo_failed(
    tmp_path, demo_module, monkeypatch, capsys
):
    class BrokenStdout:
        def write(self, _value):
            raise BrokenPipeError("closed reader")

    monkeypatch.setattr(demo_module, "run_demo", lambda _directory: {"synthetic": True})
    with monkeypatch.context() as patch:
        patch.setattr(demo_module.sys, "stdout", BrokenStdout())
        assert demo_module.main(["--output-dir", str(tmp_path / "new")]) == 2
    message = capsys.readouterr().err
    assert "completed, but stdout publication failed" in message
    assert "did not complete" not in message
