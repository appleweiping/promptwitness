"""Independent runner boundary checks; scripted fixtures, no paid model calls."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace

import pytest
from interview_fixtures import completion_envelope

import promptwitness.interviews as execution
from promptwitness.interview_journal import InterviewJournal, InterviewJournalConflict
from promptwitness.interview_models import (
    MAX_CONTRACT_BYTES,
    InterviewBudgets,
    InterviewContractError,
    InterviewCriterion,
    InterviewPlan,
    InterviewTopic,
    ParticipantAnswer,
    RetrievalPolicy,
    contract_digest,
    contract_json,
)
from promptwitness.interview_stages import InterviewQuestion, QuestionResult, StageTarget
from promptwitness.interview_state import InterviewCommand, schedule
from promptwitness.interviews import InterviewRunner, OpenAIInterviewProvider, create_interview
from promptwitness.providers import OpenAICompatibleProvider


class FixtureProvider:
    def __init__(self, callback=None):
        self.calls = []
        self.callback = callback
        self.identity = {
            "provider": "independent-runner-fixture",
            "transport": "test-only/no-network",
            "model": "scripted-test",
            "endpoint_sha256": contract_digest("no-network"),
            "headers_sha256": contract_digest({}),
            "api_key_env": None,
            "timeout": 1,
        }

    def __call__(self, request):
        self.calls.append(request)
        if self.callback is not None:
            return self.callback(request)
        return completion_envelope(QuestionResult(request, "Please describe this experience."))


def study_plan(**budget_changes):
    return InterviewPlan(
        "independent-fixture",
        "1",
        "A bounded engineering interview fixture.",
        (
            InterviewTopic(
                "experience", "Experience", (InterviewCriterion("detail", "Describe a detail."),)
            ),
        ),
        budgets=InterviewBudgets(**budget_changes),
        retrieval=RetrievalPolicy(
            context_bytes=min(1024, budget_changes.get("context_bytes", 262144))
        ),
    )


def start(journal, provider, *, name="main", plan=None):
    return create_interview(
        journal,
        name,
        "person",
        plan or study_plan(),
        command_id="create",
        analyst_identity=provider.identity,
        questioner_identity=provider.identity,
    )


def runner(journal, provider):
    return InterviewRunner(journal, analyst=provider, questioner=provider)


def test_mutating_caller_transport_after_body_check_cannot_retarget_actual_send(
    tmp_path, monkeypatch
):
    configured = OpenAICompatibleProvider(
        "http://127.0.0.1:1/original",
        model="original-model",
        headers={"X-Tenant": "original-tenant"},
        timeout=2,
    )
    provider = OpenAIInterviewProvider(configured)
    with InterviewJournal(tmp_path / "review.db", create=True) as journal:
        initial = start(journal, provider)
        request = runner(journal, provider)._request(initial, "body-check")
    original_prepare = execution.prepare_chat_body
    sends = []

    def prepare_then_mutate(*args, **kwargs):
        body = original_prepare(*args, **kwargs)
        configured.endpoint = "http://127.0.0.1:2/retargeted"
        configured.model = "changed-model"
        configured.headers["X-Tenant"] = "changed-tenant"
        configured.timeout = 299
        return body

    @contextmanager
    def intercept(endpoint, timeout, headers, body):
        sends.append((endpoint, timeout, dict(headers), body))
        yield object()

    monkeypatch.setattr(execution, "prepare_chat_body", prepare_then_mutate)
    monkeypatch.setattr("promptwitness.providers.exchange", intercept)
    monkeypatch.setattr(
        "promptwitness.providers.response_chunks", lambda response: iter((b'{"ok":true}',))
    )
    assert provider(request) == {"ok": True}
    assert len(sends) == 1
    endpoint, timeout, headers, actual = sends[0]
    assert endpoint == "http://127.0.0.1:1/original" and timeout == 2
    assert headers["X-Tenant"] == "original-tenant"
    assert actual == request.wire_body
    assert json.loads(actual)["model"] == "original-model"
    assert configured.model == "changed-model"  # The mutation really happened.


def test_provider_callback_sees_committed_reservation_and_can_write_on_another_connection(tmp_path):
    path = tmp_path / "review.db"
    observed = []
    provider = FixtureProvider()

    def inspect_while_provider_runs(request):
        with InterviewJournal(path) as second:
            reserved = second.read("main")
            assert reserved.pending is not None
            assert reserved.pending.request.digest == request.digest
            assert reserved.revision == request.source_revision + 1
            # A real second writer succeeds while the first runner is inside
            # its provider callback. It would fail if the writer lock remained.
            peer = start(second, provider, name="independent-peer")
            observed.append(peer.interview_id)
            nested = runner(second, provider).run_until_input("main", operation_id="nested-resume")
            assert nested.digest == reserved.digest
        return completion_envelope(QuestionResult(request, "What happened?"))

    provider.callback = inspect_while_provider_runs
    with InterviewJournal(path, create=True) as first:
        start(first, provider)
        result = runner(first, provider).run_until_input("main", operation_id="outer")
        assert schedule(result).kind == "await_answer"
        assert first.read("independent-peer").revision == 1
    assert observed == ["independent-peer"]
    assert len(provider.calls) == 1


def test_stop_committed_during_provider_callback_fences_the_late_valid_response(tmp_path):
    path = tmp_path / "review.db"

    def stop_during_call(request):
        with InterviewJournal(path) as other:
            pending = other.read("main")
            other.execute(
                InterviewCommand(
                    "main",
                    "stop-now",
                    pending.revision,
                    pending.digest,
                    "finished",
                    {"reason": "participant_stopped"},
                )
            )
        return completion_envelope(QuestionResult(request, "A late question must not publish."))

    provider = FixtureProvider(stop_during_call)
    with InterviewJournal(path, create=True) as journal:
        start(journal, provider)
        with pytest.raises(InterviewJournalConflict):
            runner(journal, provider).run_until_input("main", operation_id="racing-stop")
        finished = journal.read("main")
        assert finished.finished_reason == "participant_stopped"
        assert finished.questions == ()
        assert [event.kind for event in journal.events("main")] == [
            "created",
            "question_reserved",
            "finished",
        ]
        assert (
            runner(journal, provider).run_until_input("main", operation_id="resume").digest
            == finished.digest
        )
    assert len(provider.calls) == 1


def test_mutated_completion_after_initial_parse_is_rejected_by_journal_reparse(
    tmp_path, monkeypatch
):
    provider = FixtureProvider()
    original_parser = execution.parse_question_completion

    def parse_then_change_status(request, response):
        result = original_parser(request, response)
        response["choices"][0]["finish_reason"] = "length"
        return result

    monkeypatch.setattr(execution, "parse_question_completion", parse_then_change_status)
    with InterviewJournal(tmp_path / "review.db", create=True) as journal:
        start(journal, provider)
        with pytest.raises(InterviewContractError):
            runner(journal, provider).run_until_input("main", operation_id="mutated-response")
        pending = journal.read("main")
        assert pending.pending.status == "pending" and not pending.questions
        assert [event.kind for event in journal.events("main")] == ["created", "question_reserved"]
        assert (
            runner(journal, provider).run_until_input("main", operation_id="resume").digest
            == pending.digest
        )
    assert len(provider.calls) == 1


def test_historical_reservation_receipt_returns_latest_state_without_second_call(tmp_path):
    provider = FixtureProvider()
    with InterviewJournal(tmp_path / "review.db", create=True) as journal:
        original = start(journal, provider)
        engine = runner(journal, provider)
        request = engine._request(original, "once")
        command = InterviewCommand(
            "main",
            "reserve-once",
            original.revision,
            original.digest,
            "question_reserved",
            {"request": request.to_dict()},
        )
        first = journal.submit(command)
        completed = engine._invoke(first, request, "once")
        historical = journal.submit(command)
        assert historical.applied is False and historical.state.pending is not None
        current = engine._invoke(historical, request, "once")
        assert current.digest == completed.digest
        assert current.pending is None and len(current.questions) == 1
    assert len(provider.calls) == 1


@pytest.mark.parametrize("failure", ["provider_error", "invalid_response"])
def test_reused_retry_operation_id_after_failed_retry_does_not_repeat_cost(tmp_path, failure):
    path = tmp_path / "review.db"

    def fail(request):
        if failure == "provider_error":
            raise RuntimeError("private diagnostic should not be journaled")
        response = completion_envelope(QuestionResult(request, "A question"))
        response["choices"][0]["finish_reason"] = "length"
        return response

    provider = FixtureProvider(fail)
    with InterviewJournal(path, create=True) as journal:
        start(journal, provider)
        first = runner(journal, provider).run_until_input("main", operation_id="first")
        assert first.pending.error_code == failure
        second = runner(journal, provider).retry_pending("main", operation_id="explicit-once")
        assert second.pending.error_code == failure
    with InterviewJournal(path) as journal:
        before = journal.read("main")
        with pytest.raises(InterviewJournalConflict):
            runner(journal, provider).retry_pending("main", operation_id="explicit-once")
        assert journal.read("main").digest == before.digest
        assert (
            runner(journal, provider).run_until_input("main", operation_id="ordinary-resume").digest
            == before.digest
        )
        assert "private diagnostic" not in contract_json(journal.export("main"))
    assert len(provider.calls) == 2


def test_context_budget_failure_precedes_reservation_and_provider_effects(tmp_path):
    provider = FixtureProvider()
    with InterviewJournal(tmp_path / "review.db", create=True) as journal:
        initial = start(journal, provider, plan=study_plan(context_bytes=1))
        with pytest.raises(InterviewContractError, match="fixed complete"):
            runner(journal, provider).run_until_input("main", operation_id="too-small")
        assert journal.read("main").digest == initial.digest
        assert len(journal.events("main")) == 1
    assert provider.calls == []


def test_window_contains_exact_last_64_complete_question_answer_pairs_without_text_truncation(
    tmp_path,
):
    """Exercise only window preparation on a typed view, not a fabricated journal."""
    provider = FixtureProvider()
    with InterviewJournal(tmp_path / "review.db", create=True) as journal:
        original = start(
            journal,
            provider,
            plan=study_plan(participant_turns=100, provider_calls=200, followups_per_topic=100),
        )
        target = StageTarget("experience", "detail", "initial", None, 1, original.digest)
        questions = tuple(
            InterviewQuestion(
                f"q{index}",
                "main",
                "person",
                3 + index * 2,
                target,
                f"Question {index}?",
                (),
                "d" * 64,
            )
            for index in range(67)
        )
        answers = tuple(
            ParticipantAnswer(
                "main",
                "person",
                f"a{index}",
                question.question_id,
                f"{index}: 😀 e\u0301\r\n{{{{literal}}}}" + "z" * 20,
                question.revision + 1,
            )
            for index, question in enumerate(questions)
        )
        view = replace(
            original,
            revision=150,
            questions=questions,
            answers=answers,
            analyzed_answer_ids=tuple(item.answer_id for item in answers[:-1]),
        )
        request = runner(journal, provider)._request(view, "window")
        assert request.stage == "analysis" and request.context.analysis_answer_id == "a66"
        assert request.context.question_context == questions[3:]
        assert request.context.source_answers == answers[3:]
        wire_input = json.loads(json.loads(request.wire_body)["messages"][1]["content"])
        included = wire_input["context"]["source_answers"]
        assert [item["answer_id"] for item in included] == [f"a{index}" for index in range(3, 67)]
        assert [item["text"].encode("utf-8") for item in included] == [
            item.text.encode("utf-8") for item in answers[3:]
        ]
        assert journal.read("main").revision == 1  # Pure preparation made no writes.
    assert provider.calls == []


@pytest.mark.parametrize(
    "failure",
    [
        "bare_text",
        "normalized_result",
        "missing_finish",
        "wrong_binding",
        "oversized_text",
        "raw_envelope_overflow",
    ],
)
def test_invalid_provider_outcome_cannot_publish_and_resume_cannot_reinvoke(tmp_path, failure):
    def invalid(request):
        result = QuestionResult(request, "A fixture question.")
        response = completion_envelope(result)
        if failure == "bare_text":
            return response["choices"][0]["message"]["content"]
        if failure == "normalized_result":
            return result.to_dict()
        if failure == "missing_finish":
            response["choices"][0].pop("finish_reason")
        elif failure == "wrong_binding":
            body = json.loads(response["choices"][0]["message"]["content"])
            body["binding_digest"] = "f" * 64
            response["choices"][0]["message"]["content"] = json.dumps(body)
        elif failure == "oversized_text":
            response["choices"][0]["message"]["content"] = "x" * (1024 * 1024 + 1)
        elif failure == "raw_envelope_overflow":
            response["fixture_padding"] = "x" * MAX_CONTRACT_BYTES
        return response

    provider = FixtureProvider(invalid)
    with InterviewJournal(tmp_path / "review.db", create=True) as journal:
        start(journal, provider)
        failed = runner(journal, provider).run_until_input("main", operation_id="bad-response")
        assert failed.pending.error_code == "invalid_response"
        assert not failed.questions and not failed.assessments and not failed.memories
        assert [event.kind for event in journal.events("main")] == [
            "created",
            "question_reserved",
            "stage_failed",
        ]
        resumed = runner(journal, provider).run_until_input("main", operation_id="resume")
        assert resumed.digest == failed.digest
    assert len(provider.calls) == 1


def test_completion_payload_overflow_leaves_reservation_without_publishing_or_repeating(tmp_path):
    """A valid raw response need not fit its larger durable event envelope."""

    def too_large_to_commit(request):
        response = completion_envelope(QuestionResult(request, "A bounded question."))
        response["fixture_padding"] = ""
        target = MAX_CONTRACT_BYTES - 128
        response["fixture_padding"] = "x" * (target - len(contract_json(response).encode("utf-8")))
        assert len(contract_json(response).encode("utf-8")) == target
        return response

    provider = FixtureProvider(too_large_to_commit)
    with InterviewJournal(tmp_path / "review.db", create=True) as journal:
        start(journal, provider)
        with pytest.raises(InterviewContractError, match="byte limit"):
            runner(journal, provider).run_until_input("main", operation_id="envelope-overflow")
        pending = journal.read("main")
        assert pending.pending is not None and not pending.questions
        assert pending.pending.status == "pending"
        assert [event.kind for event in journal.events("main")] == ["created", "question_reserved"]
        assert (
            runner(journal, provider).run_until_input("main", operation_id="resume").digest
            == pending.digest
        )
    assert len(provider.calls) == 1
