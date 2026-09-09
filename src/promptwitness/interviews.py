"""Sequential evidence-linked interviews with explicit uncertain-work recovery.

Provider invocation is deliberately outside every journal transaction. Scripted
providers are useful engineering fixtures, not evidence of interview quality.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Protocol

from .interview_journal import InterviewJournal, InterviewReceipt
from .interview_memory import SourceHead
from .interview_models import (
    InterviewContractError,
    InterviewPlan,
    _identifier,
    contract_digest,
    contract_json,
)
from .interview_stages import (
    RENDERER_VERSION,
    TEMPLATE_HASHES,
    InterviewStageContext,
    InterviewStageRequest,
    build_analysis_request,
    build_question_request,
    parse_analysis_completion,
    parse_question_completion,
)
from .interview_state import (
    SCHEDULER_VERSION,
    InterviewCommand,
    InterviewState,
    current_memory_snapshot,
    schedule,
)
from .providers import OpenAICompatibleProvider, prepare_chat_body
from .sessions import OpenAISessionProvider

TRANSCRIPT_QUESTION_WINDOW = 64


class InterviewProvider(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def __call__(self, request: InterviewStageRequest) -> Any: ...


class OpenAIInterviewProvider:
    """Invoke a configured transport only for its exact prepared request bytes."""

    def __init__(self, provider: OpenAICompatibleProvider) -> None:
        if not isinstance(provider, OpenAICompatibleProvider):
            raise TypeError("interview provider requires an OpenAICompatibleProvider")
        self.provider = provider

    @property
    def identity(self) -> Mapping[str, Any]:
        return OpenAISessionProvider(self.provider).identity

    def __call__(self, request: InterviewStageRequest) -> Any:
        if not isinstance(request, InterviewStageRequest):
            raise InterviewContractError("provider requires a validated stage request")
        # Snapshot caller-owned mutable configuration into a fresh base transport.
        # Check and send this same snapshot, not a provider another thread could
        # retarget between checking its identity and preparing the actual body.
        transport = OpenAICompatibleProvider(
            self.provider.endpoint,
            model=self.provider.model,
            api_key_env=self.provider.api_key_env,
            timeout=self.provider.timeout,
            headers=dict(self.provider.headers),
        )
        identity = OpenAISessionProvider(transport).identity
        if contract_json(request.provider_identity) != contract_json(identity):
            raise InterviewContractError("configured provider differs from the reserved identity")
        if (
            prepare_chat_body(
                request.messages, generation=request.generation, model=transport.model
            )
            != request.wire_body
        ):
            raise InterviewContractError("configured provider body differs from the reserved bytes")
        return transport.complete(request.messages, generation=request.generation)


def create_interview(
    journal: InterviewJournal,
    interview_id: str,
    participant_id: str,
    plan: InterviewPlan,
    *,
    command_id: str,
    analyst_identity: Mapping[str, Any],
    questioner_identity: Mapping[str, Any],
    prior_heads: Sequence[SourceHead] = (),
) -> InterviewState:
    """Pin the agenda, role configurations and explicitly selected same-database history."""
    if not isinstance(plan, InterviewPlan):
        raise InterviewContractError("create_interview requires an InterviewPlan")
    snapshot = journal.memory_snapshot(participant_id, prior_heads)
    command = InterviewCommand(
        interview_id,
        command_id,
        0,
        None,
        "created",
        {
            "participant_id": participant_id,
            "plan": plan.to_dict(),
            "provider_identities": {"analyst": analyst_identity, "questioner": questioner_identity},
            "template_hashes": dict(TEMPLATE_HASHES),
            "renderer_version": RENDERER_VERSION,
            "scheduler_version": SCHEDULER_VERSION,
            "memory_snapshot": snapshot.to_dict(),
        },
    )
    return journal.execute(command)


def _command(
    state: InterviewState, operation_id: str, kind: str, payload: Mapping[str, Any]
) -> InterviewCommand:
    _identifier(operation_id, "operation ID")
    # An explicit retry ID is single-use across source heads. Repeating the same
    # CLI/API retry after an interruption must not authorize another paid call.
    scope = (
        {"interview_id": state.interview_id}
        if kind == "retry_authorized"
        else {"head": state.head.to_dict()}
    )
    command_id = "cmd-" + contract_digest({"operation_id": operation_id, **scope, "kind": kind})
    return InterviewCommand(
        state.interview_id, command_id, state.revision, state.digest, kind, payload
    )


class InterviewRunner:
    """Run analyst then questioner until input/review/recovery/termination is needed.

    The explicit operation ID makes overlapping identical invocations share the
    same reservation command. Only its newly-applying transaction invokes work;
    a historical receipt or existing pending stage never authorizes another call.
    An application may still lose a response after the provider executes: explicit
    retry is a user decision with possible repeated external cost, not exactly-once.
    """

    def __init__(
        self,
        journal: InterviewJournal,
        *,
        analyst: InterviewProvider,
        questioner: InterviewProvider,
    ) -> None:
        if not isinstance(journal, InterviewJournal):
            raise TypeError("runner requires an InterviewJournal")
        self.journal = journal
        self.analyst = analyst
        self.questioner = questioner

    def _provider(self, state: InterviewState, stage: str) -> InterviewProvider:
        role = "analyst" if stage == "analysis" else "questioner"
        provider = self.analyst if stage == "analysis" else self.questioner
        if not callable(provider) or contract_json(provider.identity) != contract_json(
            state.provider_identities[role]
        ):
            raise InterviewContractError(
                "configured provider differs from the pinned role identity"
            )
        return provider

    def _request(
        self, state: InterviewState, operation_id: str, *, retry: bool = False
    ) -> InterviewStageRequest:
        ready = replace(state, pending=None) if retry else state
        action = schedule(ready)
        if action.kind not in ("analysis", "question") or action.target is None:
            raise InterviewContractError("interview has no eligible provider stage")
        provider = self._provider(state, action.kind)
        identity = contract_digest(
            {
                "operation_id": operation_id,
                "head": state.head.to_dict(),
                "stage": action.kind,
                "retry": retry,
            }
        )
        questions = state.questions[-TRANSCRIPT_QUESTION_WINDOW:]
        question_ids = {item.question_id for item in questions}
        answers = tuple(item for item in state.answers if item.question_id in question_ids)
        context = InterviewStageContext(
            state.interview_id,
            state.participant_id,
            state.revision,
            state.digest,
            "req-" + identity,
            "attempt-" + identity,
            state.plan,
            action.target,
            state.accepted_emergent,
            answers,
            questions,
            action.answer_id,
            (action.target.link,),
        )
        if action.kind == "analysis":
            return build_analysis_request(context, provider.identity)
        topics = (*state.plan.topics, *state.accepted_emergent)
        topic = next(item for item in topics if item.topic_id == action.target.topic_id)
        criterion = next(
            item for item in topic.criteria if item.criterion_id == action.target.criterion_id
        )
        query = topic.description + " " + criterion.description
        return build_question_request(
            context, provider.identity, snapshot=current_memory_snapshot(state), query=query
        )

    def _invoke(
        self, receipt: InterviewReceipt, request: InterviewStageRequest, operation_id: str
    ) -> InterviewState:
        if not receipt.applied:
            return self.journal.read(receipt.state.interview_id)
        state = receipt.state
        binding = {
            "request_id": request.request_id,
            "attempt_id": request.attempt_id,
            "request_digest": request.digest,
        }
        # Do not catch BaseException: process interruption leaves a durable
        # uncertain reservation and must never silently repeat external work.
        try:
            response = self._provider(state, request.stage)(request)
        except Exception:
            return self.journal.execute(
                _command(
                    state, operation_id, "stage_failed", {**binding, "error_code": "provider_error"}
                )
            )
        try:
            result = (
                parse_analysis_completion(request, response)
                if request.stage == "analysis"
                else parse_question_completion(request, response)
            )
        except ValueError:
            return self.journal.execute(
                _command(
                    state,
                    operation_id,
                    "stage_failed",
                    {**binding, "error_code": "invalid_response"},
                )
            )
        payload = {**binding, "result": result.to_dict(), "completion": response}
        if request.stage == "question":
            payload["question_id"] = "q-" + request.digest
        # A commit failure leaves the reservation intact; it is not another
        # provider failure, and this runner makes no automatic retry.
        return self.journal.execute(
            _command(state, operation_id, request.stage + "_committed", payload)
        )

    def run_until_input(self, interview_id: str, *, operation_id: str) -> InterviewState:
        _identifier(operation_id, "operation ID")
        for _ in range(4):
            state = self.journal.read(interview_id)
            action = schedule(state)
            if action.kind == "finish":
                return self.journal.execute(
                    _command(state, operation_id, "finished", {"reason": action.reason})
                )
            if action.kind not in ("analysis", "question"):
                return state
            request = self._request(state, operation_id)
            receipt = self.journal.submit(
                _command(
                    state, operation_id, request.stage + "_reserved", {"request": request.to_dict()}
                )
            )
            updated = self._invoke(receipt, request, operation_id)
            if not receipt.applied or schedule(updated).kind not in (
                "analysis",
                "question",
                "finish",
            ):
                return updated
        raise InterviewContractError("sequential interview exceeded its stage transition bound")

    def retry_pending(self, interview_id: str, *, operation_id: str) -> InterviewState:
        """Explicitly replace one uncertain/failed attempt; its old result is fenced."""
        _identifier(operation_id, "operation ID")
        state = self.journal.read(interview_id)
        if state.pending is None:
            raise InterviewContractError("interview has no pending stage to retry")
        request = self._request(state, operation_id, retry=True)
        receipt = self.journal.submit(
            _command(state, operation_id, "retry_authorized", {"request": request.to_dict()})
        )
        updated = self._invoke(receipt, request, operation_id)
        if receipt.applied and schedule(updated).kind in ("analysis", "question", "finish"):
            return self.run_until_input(interview_id, operation_id=operation_id)
        return updated
