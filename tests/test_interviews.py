"""Scripted and literal-loopback workflow oracles, not interview-quality scores."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from threading import Barrier, Thread

import pytest
from interview_fixtures import completion_envelope

from promptwitness.interview_journal import InterviewJournal, InterviewJournalConflict
from promptwitness.interview_models import (
    BoundEvidence,
    CriterionAssessment,
    InterviewContractError,
    InterviewCriterion,
    InterviewPlan,
    InterviewTopic,
    MemoryRecord,
    contract_digest,
    contract_json,
)
from promptwitness.interview_stages import AnalysisResult, QuestionResult
from promptwitness.interview_state import InterviewCommand, interview_report, schedule
from promptwitness.interviews import (
    InterviewRunner,
    OpenAIInterviewProvider,
    create_interview,
)
from promptwitness.providers import OpenAICompatibleProvider


def plan():
    return InterviewPlan(
        "workshop",
        "v1",
        "A workshop interview",
        (
            InterviewTopic(
                "sensor",
                "Sensor task",
                (InterviewCriterion("construction", "Sensor construction"),),
            ),
            InterviewTopic(
                "evaluation",
                "Evaluation",
                (InterviewCriterion("measurement", "Measurement result"),),
            ),
        ),
    )


class ScriptedProvider:
    def __init__(self):
        self.calls = []
        self.behavior = "normal"
        self.identity = {
            "provider": "scripted-interview/v1",
            "transport": "in-process-test/v1",
            "model": "fixture-only",
            "endpoint_sha256": contract_digest("no-network"),
            "headers_sha256": contract_digest({}),
            "api_key_env": None,
            "timeout": 1,
        }

    def __call__(self, request):
        self.calls.append(request)
        if self.behavior == "crash":
            raise KeyboardInterrupt("simulated response loss")
        if self.behavior == "error":
            raise RuntimeError("sensitive provider exception must not be stored")
        response = response_for(request)
        if self.behavior == "truncated":
            response["choices"][0]["finish_reason"] = "length"
        return response


def response_for(request):
    if request.stage == "question":
        result = QuestionResult(
            request,
            f"Please describe {request.target.criterion_id}.",
            request.selected_memory_ids,
        )
    else:
        answer = request.context.analysis_answer
        evidence = BoundEvidence.from_answer(answer, 0, len(answer.text))
        link = request.target.link
        result = AnalysisResult(
            request,
            (
                CriterionAssessment(
                    answer.participant_id, answer.interview_id, link, "covered", (evidence,)
                ),
            ),
            (
                MemoryRecord(
                    answer.participant_id, (evidence,), (link,), "Model-proposed workshop note."
                ),
            ),
        )
    return completion_envelope(result)


def create(journal, provider, *, interview_id="one", prior_heads=()):
    return create_interview(
        journal,
        interview_id,
        "participant",
        plan(),
        command_id="create",
        analyst_identity=provider.identity,
        questioner_identity=provider.identity,
        prior_heads=prior_heads,
    )


def answer(journal, state, text, *, answer_id="answer"):
    return journal.execute(
        InterviewCommand(
            state.interview_id,
            "answer-" + answer_id,
            state.revision,
            state.digest,
            "answer_committed",
            {"question_id": schedule(state).question_id, "answer_id": answer_id, "text": text},
        )
    )


def runner(journal, provider):
    return InterviewRunner(journal, analyst=provider, questioner=provider)


def test_full_two_topic_interview_reopen_and_next_interview_recall(tmp_path):
    path = tmp_path / "interviews.db"
    provider = ScriptedProvider()
    with InterviewJournal(path, create=True) as journal:
        create(journal, provider)
        first = runner(journal, provider).run_until_input("one", operation_id="first")
        assert schedule(first).kind == "await_answer" and len(provider.calls) == 1
        answer(journal, first, "I built a solar sensor 🛰️.")
    with InterviewJournal(path) as journal:
        second = runner(journal, provider).run_until_input("one", operation_id="second")
        assert [request.stage for request in provider.calls] == ["question", "analysis", "question"]
        assert schedule(second).kind == "await_answer"
        assert second.questions[-1].target.topic_id == "evaluation"
        assert len(second.memories) == 1
        answer(journal, second, "Its output tracked daylight.", answer_id="measurement")
        finished = runner(journal, provider).run_until_input("one", operation_id="finish")
        assert finished.finished_reason == "agenda_completed"
        assert len(provider.calls) == 4 and len(finished.memories) == 2
        report = interview_report(finished)
        assert "assessed" in contract_json(report)
        assert all(
            "completion" in event.payload
            for event in journal.events("one")
            if event.kind.endswith("_committed") and event.kind != "answer_committed"
        )
        create(journal, provider, interview_id="two", prior_heads=(finished.head,))
        recalled = runner(journal, provider).run_until_input("two", operation_id="recall")
        assert recalled.questions[-1].memory_ids
        assert recalled.assessments == ()
        assert all(
            item.participant_id == "participant" for item in recalled.memory_snapshot.answers
        )
        assert len(provider.calls) == 5


@pytest.mark.parametrize(
    "behavior,code", [("error", "provider_error"), ("truncated", "invalid_response")]
)
def test_failure_does_not_publish_question_or_repeat_on_resume(tmp_path, behavior, code):
    provider = ScriptedProvider()
    provider.behavior = behavior
    with InterviewJournal(tmp_path / "journal.db", create=True) as journal:
        create(journal, provider)
        failed = runner(journal, provider).run_until_input("one", operation_id="first")
        assert failed.pending.error_code == code and not failed.questions
        assert len(provider.calls) == 1
        same = runner(journal, provider).run_until_input("one", operation_id="resume")
        assert same == failed and len(provider.calls) == 1
        assert "sensitive provider exception" not in contract_json(journal.export("one"))
        provider.behavior = "normal"
        recovered = runner(journal, provider).retry_pending("one", operation_id="explicit-retry")
        assert schedule(recovered).kind == "await_answer" and len(provider.calls) == 2
        assert provider.calls[0].digest != provider.calls[1].digest


def test_interruption_after_reservation_requires_new_explicit_retry_id(tmp_path):
    path = tmp_path / "journal.db"
    provider = ScriptedProvider()
    provider.behavior = "crash"
    with InterviewJournal(path, create=True) as journal:
        create(journal, provider)
        with pytest.raises(KeyboardInterrupt):
            runner(journal, provider).run_until_input("one", operation_id="initial")
        assert journal.read("one").pending.status == "pending"
    with InterviewJournal(path) as journal:
        state = runner(journal, provider).run_until_input("one", operation_id="resume")
        assert state.pending.status == "pending" and len(provider.calls) == 1
        with pytest.raises(KeyboardInterrupt):
            runner(journal, provider).retry_pending("one", operation_id="retry-once")
        assert len(provider.calls) == 2
        with pytest.raises(InterviewJournalConflict):
            runner(journal, provider).retry_pending("one", operation_id="retry-once")
        assert len(provider.calls) == 2
        provider.behavior = "normal"
        recovered = runner(journal, provider).retry_pending("one", operation_id="retry-twice")
        assert schedule(recovered).kind == "await_answer" and len(provider.calls) == 3


def test_concurrent_same_operation_only_new_receipt_owner_invokes_provider(tmp_path):
    path = tmp_path / "journal.db"
    provider = ScriptedProvider()
    with InterviewJournal(path, create=True) as journal:
        create(journal, provider)
    barrier = Barrier(2)

    class RacingRunner(InterviewRunner):
        def _request(self, state, operation_id, *, retry=False):
            request = super()._request(state, operation_id, retry=retry)
            barrier.wait(timeout=10)
            return request

    def invoke(_index):
        with InterviewJournal(path) as journal:
            return RacingRunner(journal, analyst=provider, questioner=provider).run_until_input(
                "one", operation_id="shared-operation"
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        states = list(executor.map(invoke, range(2)))
    assert len(provider.calls) == 1
    assert {schedule(state).kind for state in states} <= {"await_answer", "await_completion"}
    with InterviewJournal(path) as journal:
        assert len(journal.read("one").questions) == 1


def test_changed_provider_identity_rejected_before_reserving_or_calling(tmp_path):
    provider = ScriptedProvider()
    with InterviewJournal(tmp_path / "journal.db", create=True) as journal:
        initial = create(journal, provider)
        provider.identity = {**provider.identity, "model": "different"}
        with pytest.raises(InterviewContractError):
            runner(journal, provider).run_until_input("one", operation_id="changed")
        assert journal.read("one") == initial and provider.calls == []


def test_real_loopback_transport_matches_reserved_wire_and_replay_envelope(tmp_path):
    requests = Queue()
    observed = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            expected = requests.get(timeout=5)
            observed.append((raw, expected.wire_body))
            payload = json.dumps(response_for(expected), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class ObservedProvider(OpenAIInterviewProvider):
        def __call__(self, request):
            requests.put(request)
            return super().__call__(request)

    try:
        transport = OpenAICompatibleProvider(
            f"http://127.0.0.1:{server.server_port}/chat/completions",
            model="loopback-fixture",
            api_key_env=None,
            timeout=5,
        )
        provider = ObservedProvider(transport)
        with InterviewJournal(tmp_path / "journal.db", create=True) as journal:
            create(journal, provider)
            question = runner(journal, provider).run_until_input(
                "one", operation_id="live-question"
            )
            answer(journal, question, "I assembled a sensor {{not_a_template}}.")
            next_question = runner(journal, provider).run_until_input(
                "one", operation_id="live-analysis"
            )
            assert schedule(next_question).kind == "await_answer"
            assert len(next_question.memories) == 1
            assert journal.read("one") == next_question
            completed = [event for event in journal.events("one") if "completion" in event.payload]
            assert len(completed) == 3
            assert all(
                event.payload["completion"]["choices"][0]["finish_reason"] == "stop"
                for event in completed
            )
        assert len(observed) == 3 and all(actual == expected for actual, expected in observed)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_transport_adapter_rejects_changed_identity_without_http(tmp_path):
    provider = ScriptedProvider()
    with InterviewJournal(tmp_path / "journal.db", create=True) as journal:
        initial = create(journal, provider)
        request = runner(journal, provider)._request(initial, "fixture")
        live = OpenAIInterviewProvider(
            OpenAICompatibleProvider("http://127.0.0.1:1/test", model="other")
        )
        with pytest.raises(InterviewContractError):
            live(request)
        with pytest.raises(InterviewContractError):
            live(copy.deepcopy(request.to_dict()))
        with pytest.raises(TypeError):
            OpenAIInterviewProvider(None)
