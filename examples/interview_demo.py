"""An original, entirely scripted multi-interview engineering demonstration.

No model, credentials, network request or real participant is involved. The
authored responses below exercise contracts and durability, not interview skill.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from promptwitness.interview_journal import (
    InterviewJournal,
    InterviewJournalConflict,
    verify_interview_export,
)
from promptwitness.interview_memory import SourceHead
from promptwitness.interview_models import (
    InterviewBudgets,
    InterviewCriterion,
    InterviewPlan,
    InterviewTopic,
    contract_digest,
    contract_json,
)
from promptwitness.interview_stages import InterviewStageRequest
from promptwitness.interview_state import (
    InterviewCommand,
    InterviewState,
    interview_report,
    schedule,
)
from promptwitness.interviews import InterviewRunner, create_interview

FIRST = "mira-first"
SECOND = "mira-return"
OTHER = "noel-first"
MIRA = "synthetic-mira"
NOEL = "synthetic-noel"
ANSWERS = {
    "route": "I mapped the lantern walkway 🏮.\r\nThe quiet lane keeps {{template}} literal.",
    "lamp_check": "I checked each lantern walkway lamp with a paper checklist.",
    "quiet_lane": "I marked the lantern walkway quiet lane with soft blue arrows.",
}
OTHER_ANSWER = "My separate lantern walkway route uses green flags, not Mira's notes."
QUESTIONS = {
    "route": "How did you prepare the lantern walkway route? 🏮",
    "lamp_check": "What lamp check did you record for the lantern walkway?",
    "quiet_lane": "How did you mark the lantern walkway quiet lane?",
}


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def demo_plan(*, other: bool = False) -> InterviewPlan:
    topics = (
        InterviewTopic(
            "preparation",
            "Lantern walkway preparation",
            (InterviewCriterion("route", "Describe the lantern walkway route."),),
        ),
        InterviewTopic(
            "checks",
            "Lantern walkway checks",
            (InterviewCriterion("lamp_check", "Describe a lantern walkway lamp check."),),
        ),
    )
    return InterviewPlan(
        "separate-synthetic-plan" if other else "lantern-walkway-demo",
        "v1",
        "Exercise source-linked interview storage with authored synthetic responses.",
        topics[:1] if other else topics,
        budgets=InterviewBudgets(
            participant_turns=3,
            provider_calls=6,
            followups_per_topic=0,
            emergent_topics=0 if other else 1,
        ),
    )


class ScriptedLanternProvider:
    """Return native response JSON written for this fixture, never inferred by a model."""

    def __init__(self) -> None:
        self.calls: list[InterviewStageRequest] = []
        self.identity = {
            "provider": "original-synthetic-lantern-demo/v1",
            "transport": "in-process-no-network/v1",
            "model": "no-model-scripted-responses",
            "endpoint_sha256": contract_digest("no-network-endpoint"),
            "headers_sha256": contract_digest({}),
            "api_key_env": None,
            "timeout": 1,
        }

    def __call__(self, request: InterviewStageRequest) -> dict[str, Any]:
        self.calls.append(request)
        body: dict[str, Any]
        if request.stage == "question":
            body = {
                "format": "promptwitness.question-response/v1",
                "binding_digest": request.binding_digest,
                "target": request.target.link.to_dict(),
                "text": QUESTIONS[request.target.criterion_id],
                "memory_ids": list(request.selected_memory_ids),
            }
        else:
            source = request.context.analysis_answer
            _check(source is not None, "analysis must have an actual committed source answer")
            if source is None:  # Type narrowing only; _check is not disabled by python -O.
                raise AssertionError("missing source")
            proposal: dict[str, Any] | None = None
            if request.context.interview_id == FIRST and request.target.criterion_id == "route":
                proposal = {
                    "topic": {
                        "id": "accessibility",
                        "description": "Lantern walkway quiet lane",
                        "criteria": [
                            {"id": "quiet_lane", "description": "Describe quiet-lane marks."}
                        ],
                    },
                    "parent_topic_id": "preparation",
                    "evidence_ids": ["source-quote"],
                }
            body = {
                "format": "promptwitness.analysis-response/v1",
                "binding_digest": request.binding_digest,
                "evidence": [
                    {
                        "id": "source-quote",
                        "answer_id": source.answer_id,
                        "start": 0,
                        "end": len(source.text),
                        "quote": source.text,
                    }
                ],
                "assessments": [
                    {
                        **request.target.link.to_dict(),
                        "status": "covered",
                        "evidence_ids": ["source-quote"],
                        "rationale": "Authored demo decision, not a factual accuracy label.",
                    }
                ],
                "memories": [
                    {
                        "links": [request.target.link.to_dict()],
                        "evidence_ids": ["source-quote"],
                        "summary": {
                            "kind": "model_proposed",
                            "text": "Synthetic lantern walkway note; no model generated this text.",
                        },
                    }
                ],
                "proposal": proposal,
            }
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(body, ensure_ascii=False, allow_nan=False),
                    },
                }
            ]
        }


def _create(
    journal: InterviewJournal,
    provider: ScriptedLanternProvider,
    interview_id: str,
    participant: str,
    *,
    other: bool = False,
    prior_heads: tuple[SourceHead, ...] = (),
) -> InterviewState:
    return create_interview(
        journal,
        interview_id,
        participant,
        demo_plan(other=other),
        command_id="create-demo-interview",
        analyst_identity=provider.identity,
        questioner_identity=provider.identity,
        prior_heads=prior_heads,
    )


def _command(state: InterviewState, kind: str, payload: dict[str, Any]) -> InterviewCommand:
    return InterviewCommand(
        state.interview_id,
        f"demo-{kind}-{state.revision + 1}",
        state.revision,
        state.digest,
        kind,
        payload,
    )


def _answer(journal: InterviewJournal, state: InterviewState, text: str) -> InterviewState:
    _check(schedule(state).kind == "await_answer", "answer must follow a published question")
    return journal.execute(
        _command(
            state,
            "answer_committed",
            {
                "question_id": schedule(state).question_id,
                "answer_id": "answer-" + state.questions[-1].target.criterion_id,
                "text": text,
            },
        )
    )


def _runner(journal: InterviewJournal, provider: ScriptedLanternProvider) -> InterviewRunner:
    return InterviewRunner(journal, analyst=provider, questioner=provider)


def _summary(journal: InterviewJournal, state: InterviewState) -> dict[str, Any]:
    exported = journal.export(state.interview_id)
    _check(
        verify_interview_export(exported, expected_head=state.head) == state,
        "export must replay against the independently retained head",
    )
    report = interview_report(state)
    for entry in state.memories:
        for evidence in entry.memory.evidence:
            evidence.reference.validate(evidence.answer, participant_id=state.participant_id)
    return {
        "revision": state.revision,
        "event_count": len(exported["events"]),
        "event_kinds": [event["kind"] for event in exported["events"]],
        "reason": state.finished_reason,
        "questions": len(state.questions),
        "answers": len(state.answers),
        "local_memories": len(state.memories),
        "imported_memories": len(state.memory_snapshot.entries),
        "coverage": report["assessed_coverage"],
        "question_topics": [question.target.topic_id for question in state.questions],
        "head_sha256": state.digest,
        "export_sha256": contract_digest(exported),
    }


def run_demo(output_directory: str | Path) -> dict[str, Any]:
    """Create a new directory exclusively; an existing directory is never reused."""
    directory = Path(output_directory)
    directory.mkdir(parents=False, exist_ok=False)
    database = directory / "interviews.sqlite"
    provider = ScriptedLanternProvider()
    with InterviewJournal(database, create=True) as journal:
        _create(journal, provider, FIRST, MIRA)
        first_question = _runner(journal, provider).run_until_input(FIRST, operation_id="first-q")
        saved = _answer(journal, first_question, ANSWERS["route"])
        saved_head = saved.head
        _check(saved.revision == 4, "first close must occur immediately after the real answer")

    with InterviewJournal(database) as journal:
        _check(
            journal.read(FIRST, expected_head=saved_head) == saved, "reopen changed committed input"
        )
        review = _runner(journal, provider).run_until_input(FIRST, operation_id="analyze-route")
        _check(
            review.revision == 6 and schedule(review).kind == "await_review", "review gate missing"
        )
        _check(len(provider.calls) == 2, "proposal must block publication of another question")
        _check(len(review.proposals) == 1, "one cited emergent proposal is required")
        proposal = review.proposals[0].proposal
        _check(proposal.evidence[0].answer == saved.answers[0], "proposal citation changed source")
        accepted = journal.execute(
            _command(
                review,
                "proposal_decided",
                {"proposal_id": proposal.proposal_id, "decision": "accepted"},
            )
        )
        second_question = _runner(journal, provider).run_until_input(FIRST, operation_id="checks-q")
        _check(second_question.questions[-1].target.topic_id == "checks", "required order changed")
        _answer(journal, second_question, ANSWERS["lamp_check"])
        emergent_question = _runner(journal, provider).run_until_input(
            FIRST, operation_id="checks-a"
        )
        required_complete = journal.read(FIRST, revision=12)
        middle_coverage = interview_report(required_complete)["assessed_coverage"]
        _check(
            middle_coverage["required"]["covered"] == 2
            and middle_coverage["emergent"]["covered"] == 0
            and schedule(required_complete).kind == "question",
            "required coverage must not prematurely stop eligible emergent work",
        )
        _check(
            emergent_question.questions[-1].target.topic_id == "accessibility",
            "emergent work missing",
        )
        _answer(journal, emergent_question, ANSWERS["quiet_lane"])
        first_finished = _runner(journal, provider).run_until_input(
            FIRST, operation_id="finish-first"
        )
        _check(
            first_finished.revision == 18, "first interview differs from the authored event trace"
        )
        _check(
            first_finished.finished_reason == "agenda_completed", "first agenda remains incomplete"
        )
        first_summary = _summary(journal, first_finished)

        _create(journal, provider, OTHER, NOEL, other=True)
        other_question = _runner(journal, provider).run_until_input(OTHER, operation_id="other-q")
        _answer(journal, other_question, OTHER_ANSWER)
        other_finished = _runner(journal, provider).run_until_input(OTHER, operation_id="other-a")
        _check(
            other_finished.revision == 7 and len(other_finished.memories) == 1,
            "other source missing",
        )
        other_summary = _summary(journal, other_finished)

    with InterviewJournal(database) as journal:
        _check(
            journal.read(FIRST, expected_head=first_finished.head) == first_finished,
            "first head changed",
        )
        _check(
            journal.read(OTHER, expected_head=other_finished.head) == other_finished,
            "other head changed",
        )
        try:
            _create(
                journal,
                provider,
                "forbidden-mix",
                MIRA,
                prior_heads=(first_finished.head, other_finished.head),
            )
        except InterviewJournalConflict:
            pass
        else:
            raise AssertionError("cross-participant source selection must be rejected")
        try:
            journal.read("forbidden-mix")
        except KeyError:
            pass
        else:
            raise AssertionError("rejected cross-participant creation left a journal event")
        next_created = _create(journal, provider, SECOND, MIRA, prior_heads=(first_finished.head,))
        _check(
            len(next_created.memory_snapshot.entries) == 3,
            "second interview must import three memories",
        )
        _check(next_created.assessments == (), "memory import must not import coverage")
        returned = _runner(journal, provider).run_until_input(SECOND, operation_id="return-q")
        recalled = returned.questions[-1].memory_ids
        _check(
            len(recalled) == 3, "the authored common vocabulary should recall all three prior notes"
        )
        _check(
            set(recalled) == {entry.memory.memory_id for entry in first_finished.memories}
            and set(recalled).isdisjoint(
                entry.memory.memory_id for entry in other_finished.memories
            ),
            "recall contains missing or cross-participant notes",
        )
        second_coverage = interview_report(returned)["assessed_coverage"]
        _check(
            second_coverage["required"]["covered"] == 0, "recall incorrectly covered the new agenda"
        )
        stopped = journal.execute(_command(returned, "finished", {"reason": "participant_stopped"}))
        second_summary = _summary(journal, stopped)

    calls = Counter(request.context.interview_id for request in provider.calls)
    stages = Counter(request.stage for request in provider.calls)
    _check(dict(calls) == {FIRST: 6, OTHER: 2, SECOND: 1}, "unexpected scripted call count")
    result = {
        "format": "promptwitness.synthetic-interview-demo/v1",
        "synthetic": True,
        "model_calls": 0,
        "network_calls": 0,
        "evidence_scope": "authored offline engineering fixture; not interview-quality evidence",
        "scripted_provider_calls": len(provider.calls),
        "scripted_stage_counts": dict(stages),
        "database_reopens": 2,
        "first": first_summary,
        "other_participant": other_summary,
        "second": second_summary,
        "checkpoints": {
            "closed_after_answer_revision": saved.revision,
            "pending_review_revision": review.revision,
            "accepted_proposal_revision": accepted.revision,
            "required_complete_revision": required_complete.revision,
            "required_complete_emergent_covered": middle_coverage["emergent"]["covered"],
            "second_recall_revision": returned.revision,
            "second_recalled_memories": len(recalled),
            "second_required_covered_after_recall": second_coverage["required"]["covered"],
        },
        "cross_participant_selection_rejected": True,
        "rejected_creation_left_no_event": True,
        "literal_unicode_crlf_template_preserved": saved.answers[0].text == ANSWERS["route"],
    }
    # Exclusive creation, even though the containing directory was freshly made.
    with (directory / "summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, help="new directory for private SQLite and aggregate JSON"
    )
    args = parser.parse_args(argv)
    try:
        if args.output_dir is None:
            with TemporaryDirectory(prefix="promptwitness-interview-demo-") as parent:
                result = run_demo(Path(parent) / "demo")
        else:
            result = run_demo(args.output_dir)
    except (OSError, ValueError, AssertionError) as exc:
        print(f"Synthetic interview demo did not complete ({type(exc).__name__}).", file=sys.stderr)
        return 2
    try:
        print(contract_json(result))
    except OSError:
        print("Synthetic interview demo completed, but stdout publication failed.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
