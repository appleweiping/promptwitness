"""Fixed-prefix controller route with actual scoring and separate cost ledgers.

The injected executor must execute the frozen request, not predict a score.
ONLINE_PINNED is an externally established assumption, not granted by this
driver. Authored transport qualification is not online or native-engine proof.
"""

from __future__ import annotations

from promptwitness.incremental.contracts import check_contract
from promptwitness.incremental.gate import GateStatus, evaluate_candidate
from promptwitness.incremental.identity import ExecutionIdentity
from promptwitness.incremental.journal import AuditJournal
from promptwitness.incremental.optimizers import complete_survivor_scores
from promptwitness.incremental.sampling import AuditPlan, digest, make_plan
from promptwitness.parser import parse_prompt
from reproduce.pipeline_controller import copied
from reproduce.role_pipeline import rows, validate
from reproduce.strict_scoring import ScoringError


class IncrementalPipeline:
    """Single-writer route. Failed/unresolved generations never silently replay.

    Resources are supplied from the authorized backend/configuration. No new
    stage quota is imposed here; the existing resource ledger owns admission.
    GPU allocation must be recorded by the backend for its whole cold/idle/exit
    lifetime, not inferred from these CPU/scoring calls or per-response walltime.
    """

    def __init__(self, controller, ledger, *, resource_stage, input_cap, output_cap):
        self.controller, self.ledger = controller, ledger
        self.resources = {
            "stage": resource_stage,
            "input_cap": input_cap,
            "output_cap": output_cap,
        }
        # Use the unchanged scientific run budget, explicitly in the run freeze.
        if controller.spec["configuration"].get("max_evaluation_episodes") != 2048:
            raise ScoringError("original 2048 episode budget must be in literal run configuration")
        identity = ExecutionIdentity(**controller.spec["execution"])
        self.identity = identity
        reference = controller._one("reference_complete")["reference"]
        self.journal = AuditJournal(
            controller.directory / "audit.sqlite",
            run_id=controller.spec["run_id"],
            reference_digest=digest(reference),
            execution_digest=identity.sha256,
            reference_episodes=len(reference),
            max_episodes=2048,
        )

    def close(self):
        self.journal.close()

    def freeze(self, identifier, *, replicates, execution_scope, use_risk=False, fixture_seed=None):
        controller = self.controller
        controller._absent("search_ended")
        frozen = controller._candidate(identifier)
        if identifier == controller.spec["seed_prompt"]["id"]:
            raise ScoringError("seed already has complete original reference")
        if (
            set(replicates) != set(controller.spec["search_ids"])
            or any(not isinstance(v, str) or not v for v in replicates.values())
            or type(use_risk) is not bool
        ):
            raise ScoringError("full fixed random-unit mapping and explicit risk mode required")
        existing = [
            e
            for e in controller.events
            if e["kind"] == "audit_frozen" and e["identifier"] == identifier
        ]
        settings = {
            "replicates": copied(replicates),
            "execution_scope": execution_scope,
            "use_risk": use_risk,
            "fixture_seed": fixture_seed,
            "resources": self.resources,
        }
        if existing:
            if len(existing) != 1 or any(existing[0][k] != v for k, v in settings.items()):
                raise ScoringError("audit execution/random units/resources changed on recovery")
            plan = AuditPlan.from_dict(existing[0]["plan"])
        else:
            if controller._search_requested(identifier) or controller._candidate_scores(identifier):
                raise ScoringError("audit freeze must precede current candidate generation")
            risk = None
            if use_risk:
                predictions = [
                    e["report"]
                    for e in controller.events
                    if e["kind"] == "result" and e["tag"] == "predict:" + identifier
                ]
                if len(predictions) != 1:
                    raise ScoringError("one frozen predictor result required for risk strata")
                risk = {
                    u: p[
                        "regression_probability"
                        if frozen["reference"][u]
                        else "improvement_probability"
                    ]
                    for u, p in predictions[0]["predictions"].items()
                }
                if any(v is None for v in risk.values()):
                    raise ScoringError(
                        "UNFITTED or missing required transition head for M1 risk strata"
                    )
            plan = make_plan(
                frozen["reference"],
                candidate_digest=digest(frozen["candidate"]),
                execution_digest=self.identity.sha256,
                risk=risk,
                fixture_seed=fixture_seed,
            )
            controller._record(
                "audit_frozen", identifier=identifier, plan=plan.to_dict(), **settings
            )
        self.journal.freeze(plan)
        return plan

    def _audit(self, identifier):
        self.controller._absent("search_ended")
        events = [
            e
            for e in self.controller.events
            if e["kind"] == "audit_frozen" and e["identifier"] == identifier
        ]
        if len(events) != 1 or events[0]["resources"] != self.resources:
            raise ScoringError("missing original audit/resources freeze")
        audit = events[0]
        plan = AuditPlan.from_dict(audit["plan"])
        if self.journal.load_plan(plan.candidate_digest) != plan:
            raise ScoringError("original controller/journal plans disagree")
        return audit, plan

    def _query(self, identifier, unit, execute):
        controller = self.controller
        audit, _ = self._audit(identifier)
        if any(
            e["kind"] == "generation_request"
            and e["identifier"] == identifier
            and e["unit"] == unit
            for e in controller.events
        ):
            raise ScoringError("original generation attempt cannot be replayed")
        request = copied(
            {
                "candidate": controller._candidate(identifier)["candidate"],
                "execution": controller.spec["execution"],
                "configuration": controller.spec["configuration"],
                "family": controller.spec["family"],
                "model": controller.spec["model"],
                "unit": rows(controller.store, "search/inputs", controller.spec["family"])[unit],
                "replicate": audit["replicates"][unit],
            }
        )
        attempt = len(controller.events)
        call_id = f"{controller.spec['run_id']}:search:{attempt}"
        controller._record(
            "generation_request", identifier=identifier, unit=unit, call_id=call_id, request=request
        )
        reserved = False
        response = None
        try:
            self.ledger.reserve_call(
                call_id,
                self.resources["stage"],
                self.identity.request_key(request, unit_id=unit, replicate_id=request["replicate"]),
                input_cap=self.resources["input_cap"],
                output_cap=self.resources["output_cap"],
            )
            reserved = True
            response = copied(execute(copied(request)))
            validate(
                controller._message("score", request["candidate"], responses=[response]),
                "search_scorer",
                "search",
            )
            if response["id"] != unit or response["replicate"] != request["replicate"]:
                raise ScoringError("generation differs from frozen actual random unit")
            # Model-call success is separate from scorer success; retain actual
            # incurred tokens even if the subsequent official scorer fails.
            tokens = (response["input_tokens"], response["output_tokens"])
            if None in tokens:
                raise ScoringError("completed physical call requires both actual token counts")
        except Exception as exc:
            if response is None:
                physical = getattr(exc, "physical_response", None)
                response = copied(physical) if isinstance(physical, dict) else None
            known = isinstance(response, dict) and all(
                type(response.get(k)) is int and response[k] >= 0
                for k in ("input_tokens", "output_tokens")
            )
            try:
                if reserved:
                    self.ledger.settle_call(
                        call_id,
                        succeeded=False,
                        input_tokens=response["input_tokens"] if known else None,
                        output_tokens=response["output_tokens"] if known else None,
                    )
            finally:
                controller._record(
                    "generation_failed",
                    call_id=call_id,
                    error=type(exc).__name__,
                    response=response,
                )
            raise
        try:
            self.ledger.settle_call(
                call_id, succeeded=True, input_tokens=tokens[0], output_tokens=tokens[1]
            )
        except Exception as exc:
            controller._record(
                "generation_failed", call_id=call_id, error=type(exc).__name__, response=response
            )
            raise
        controller._record("generation_result", call_id=call_id, response=response)
        return controller.score_unit(identifier, response)["score"]

    def evaluate(self, identifier, execute):
        audit, plan = self._audit(identifier)
        frozen = self.controller._candidate(identifier)
        result = evaluate_candidate(
            plan,
            self.journal,
            lambda unit: self._query(identifier, unit, execute),
            contract=check_contract(
                parse_prompt(frozen["parent"]), parse_prompt(frozen["candidate"])
            ),
            execution_scope=audit["execution_scope"],
        )
        self.controller._record("gate_result", identifier=identifier, result=result.to_dict())
        return result

    def complete_survivor(self, identifier, execute):
        _, plan = self._audit(identifier)
        results = [
            e["result"]
            for e in self.controller.events
            if e["kind"] == "gate_result" and e["identifier"] == identifier
        ]
        if not results or results[-1]["status"] != GateStatus.ELIGIBLE.value:
            raise ScoringError("only an eligible gate result may complete a native survivor")
        if self.journal.has_unresolved(plan):
            raise ScoringError("unresolved logical episode; do not replay")

        def query(unit):
            attempt = self.journal.reserve(plan, unit)
            try:
                score = self._query(identifier, unit, execute)
            except Exception as exc:
                self.journal.settle(plan, unit, attempt, score=None, error_type=type(exc).__name__)
                raise
            self.journal.settle(plan, unit, attempt, score=score)
            return score

        vector = complete_survivor_scores(
            tuple(self.controller.spec["search_ids"]), self.journal.observations(plan), query
        )
        self.controller._record(
            "survivor_complete",
            identifier=identifier,
            unit_ids=list(vector.unit_ids),
            scores=list(vector.scores),
        )
        return vector
