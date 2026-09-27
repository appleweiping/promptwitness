"""Connect actual metered inference to the existing restricted stage controller.

Complete reference/selection vectors are generated from frozen input-only
requests, then scored against original annotations by the existing workers.
No native selector, online guarantee, M1 fit, or scientific admission is added.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from promptwitness.incremental.identity import ExecutionIdentity
from promptwitness.incremental.sampling import digest
from reproduce.bfcl_stage import staged_inventory as bfcl_inventory
from reproduce.incremental_pipeline import IncrementalPipeline
from reproduce.online_resources import STAGE
from reproduce.pipeline_controller import copied
from reproduce.role_pipeline import rows, validate
from reproduce.strict_scoring import ScoringError
from reproduce.text_scorer_stage import staged_inventory as text_inventory
from reproduce.torch_runtime import SETTINGS, TASK_CAPS, task_wire


def execution_bindings(store: Path, family: str, runtimes: dict[str, Path]) -> dict:
    """Bind the existing ExecutionIdentity to actual training/scorer bytes.

    The trusted controller hashes training annotations without interpreting
    them. Only the scorer opens them semantically; no final leaf is inspected.
    This is the existing identity/cache invariant, not a security signature.
    """
    if family not in TASK_CAPS:
        raise ScoringError("unsupported original task family")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    data = {
        f"{pool}/{leaf}": sha(store / pool / leaf / (family + ".jsonl"))
        for pool in ("fit", "search", "selection")
        for leaf in ("inputs", "gold")
    }
    if family == "bfcl":
        inventory = bfcl_inventory(runtimes["bfcl"])
        runtime, names = runtimes["bfcl"], inventory["files"]
    else:
        inventory = text_inventory(runtimes["text"])
        runtime = runtimes["text"]
        names = inventory["code_files"] + inventory["resource_files"]
    wrapper_names = (
        "strict_scoring.py",
        "role_pipeline.py",
        "bfcl_native.py",
        "bfcl_stage.py",
        "text_scorer_stage.py",
        "check_strict_scorers.py",
    )
    scorer = {
        "inventory": inventory,
        "runtime": {name: sha(runtime / name) for name in names},
        "wrappers": {name: sha(Path(__file__).with_name(name)) for name in wrapper_names},
    }
    return {
        "data_digest": digest(data),
        "scorer_digest": digest(scorer),
        "tool_environment_digest": digest(
            {
                "interface": "complete text requests; BFCL offline AST, no native tool execution",
                "family": family,
                "scorer": scorer,
                "construction_rng_seed": 11,
            }
        ),
    }


class RealPipeline:
    """One loaded session, one existing physical ledger, one stage controller.

    Every complete-vector request is recorded before generation. Recovery
    reuses only its exact settled observations, never repeats a failed/pending
    model request. Candidate incremental calls use the existing driver's prior
    reservation rather than a second physical reservation.
    """

    def __init__(self, controller, model):
        self.controller, self.model = controller, model
        spec = controller.spec
        ExecutionIdentity(**spec["execution"])
        expected = execution_bindings(controller.store, spec["family"], controller.runtimes)
        if (
            model.model != spec["model"]
            or model.data_root != controller.store.resolve(strict=True)
            or model.profile is None
            or any(spec["execution"].get(k) != v for k, v in model.profile.items())
            or any(spec["execution"].get(k) != v for k, v in expected.items())
        ):
            raise ScoringError("actual model/data/scorer differs from original run freeze")
        controller._no_pending()

    def _vector(self, purpose, identifier, replicates):
        controller = self.controller
        controller._no_pending()
        if purpose == "reference":
            candidate, stage = controller.spec["seed_prompt"], "search"
        elif purpose == "selection":
            ended = controller._one("search_ended")
            controller._absent("selection_ended")
            if identifier not in ended["survivors"]:
                raise ScoringError("selection generation requires a frozen search survivor")
            candidate, stage = controller._candidate(identifier)["candidate"], "selection"
        else:
            raise ScoringError("only complete reference/selection generation is supported")
        if identifier != candidate["id"]:
            raise ScoringError("request prompt differs from frozen stage prompt")
        units = controller.spec[stage + "_ids"]
        if (
            not isinstance(replicates, dict)
            or set(replicates) != set(units)
            or any(not isinstance(value, str) or not value for value in replicates.values())
        ):
            raise ScoringError("complete predeclared random-unit mapping required")
        inputs = rows(controller.store, stage + "/inputs", controller.spec["family"])
        requests = [
            copied(
                {
                    "candidate": candidate,
                    "execution": controller.spec["execution"],
                    "configuration": controller.spec["configuration"],
                    "family": controller.spec["family"],
                    "model": controller.spec["model"],
                    "unit": inputs[unit],
                    "replicate": replicates[unit],
                }
            )
            for unit in units
        ]
        # Validate every complete input before starting the first model call.
        for request in requests:
            task_wire(request)
        frozen = {"purpose": purpose, "identifier": identifier, "requests": requests}
        previous = [
            e
            for e in controller.events
            if e["kind"] == "vector_frozen"
            and e["purpose"] == purpose
            and e["identifier"] == identifier
        ]
        if previous:
            if len(previous) != 1 or any(previous[0][k] != v for k, v in frozen.items()):
                raise ScoringError("complete-vector freeze changed on recovery")
        else:
            controller._record("vector_frozen", **frozen)
        responses = []
        for request in requests:
            unit = request["unit"]["id"]
            original = [
                e
                for e in controller.events
                if e["kind"] == "generation_request"
                and e.get("purpose") == purpose
                and e["identifier"] == identifier
                and e["unit"] == unit
            ]
            if original:
                if len(original) != 1 or original[0]["request"] != request:
                    raise ScoringError("original generation request differs")
                terminals = [
                    e
                    for e in controller.events
                    if e.get("call_id") == original[0]["call_id"]
                    and e["kind"] in {"generation_result", "generation_failed"}
                ]
                if len(terminals) != 1 or terminals[0]["kind"] != "generation_result":
                    raise ScoringError("failed/unresolved original model request; do not replay")
                response = copied(terminals[0]["response"])
            else:
                call_id = f"{controller.spec['run_id']}:{purpose}:{len(controller.events)}"
                controller._record(
                    "generation_request",
                    purpose=purpose,
                    identifier=identifier,
                    unit=unit,
                    call_id=call_id,
                    request=request,
                )
                response = None
                try:
                    response = copied(self.model.metered_task(call_id, request, purpose=purpose))
                    validate(
                        controller._message("score", candidate, responses=[response]),
                        stage + "_scorer",
                        stage,
                    )
                    if response["id"] != unit or response["replicate"] != replicates[unit]:
                        raise ScoringError("actual response differs from frozen random unit")
                except BaseException as exc:
                    physical = getattr(exc, "physical_response", None)
                    controller._record(
                        "generation_failed",
                        call_id=call_id,
                        error=type(exc).__name__,
                        response=response if response is not None else physical,
                    )
                    raise
                controller._record("generation_result", call_id=call_id, response=response)
            responses.append(response)
        return responses

    def reference(self, replicates):
        responses = self._vector("reference", self.controller.spec["seed_prompt"]["id"], replicates)
        completed = [e for e in self.controller.events if e["kind"] == "reference_complete"]
        if completed:
            return self.controller._one("reference_complete")
        return self.controller.reference(responses)

    def selection(self, identifier, replicates):
        return self.controller.selection(
            identifier, self._vector("selection", identifier, replicates)
        )

    def incremental(self):
        """Return the existing driver and its unique-reservation real callback.

        Caller must establish execution_scope and freeze the actual audit.
        This bridge neither supplies ONLINE_PINNED nor changes native selection.
        """
        return IncrementalPipeline(
            self.controller,
            self.model.ledger,
            resource_stage=STAGE,
            input_cap=SETTINGS["context_limit"],
            output_cap=TASK_CAPS[self.controller.spec["family"]],
        ), self.model.task
