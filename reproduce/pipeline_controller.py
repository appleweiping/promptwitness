"""Single-writer, durable stage routing for actual research role applications.

Literal run/candidate/profile/predictor freezes precede application execution.
Both full and incremental routes supply actual observed vectors; native
optimizer engine control flow is separate, not an implied certificate.
No final stage is opened here; scientific confirmation admission is separate.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from promptwitness.incremental.features import ParentTrace
from promptwitness.incremental.identity import ExecutionIdentity
from promptwitness.incremental.predictor import TransitionPredictor
from promptwitness.parser import parse_prompt
from reproduce.bfcl_stage import staged_inventory as bfcl_inventory
from reproduce.prepare_bfcl_fit import MODEL_REVISIONS
from reproduce.process_access import launch_role
from reproduce.role_pipeline import FAMILIES, FORMAT, fields, rows, validate
from reproduce.strict_scoring import ScoringError
from reproduce.text_scorer_stage import staged_inventory as text_inventory


def copied(value):
    return json.loads(json.dumps(value, allow_nan=False))


class PipelineController:
    """Trusted controller; roles see only their explicitly constructed messages.

    Records are private. An unresolved request on resume is not silently replayed
    or zero-scored. Known errors are retained and an explicit subsequent call is
    a new attempt. This is stage provenance, not a malicious-operator defense.
    """

    def __init__(self, store: Path, directory: Path, spec: dict, runtimes: dict[str, Path]):
        fields(
            spec,
            {
                "run_id",
                "family",
                "model",
                "execution",
                "configuration",
                "seed_prompt",
                "search_ids",
                "selection_ids",
            },
        )
        if (
            spec["family"] not in FAMILIES
            or not isinstance(spec["run_id"], str)
            or not spec["run_id"]
            or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in spec["run_id"]
            )
        ):
            raise ScoringError("supported task and path-safe run ID required")
        parse_prompt(spec["seed_prompt"])
        identity = ExecutionIdentity(**spec["execution"])
        if (
            spec["model"] not in MODEL_REVISIONS
            or identity.model_revision != MODEL_REVISIONS[spec["model"]]
        ):
            raise ScoringError("model differs from frozen task revision")
        if not isinstance(spec["configuration"], dict) or not spec["configuration"]:
            raise ScoringError("literal frozen run configuration required")
        self.store, self.directory, self.runtimes = store, directory, runtimes
        self.spec = copied(spec)
        self.events = []
        self._check_populations()
        if (store / "search/reference" / (spec["run_id"] + ".json")).exists():
            raise ScoringError("run reference already exists; resume original controller")
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "workers").mkdir()
        self._record("run_frozen", spec=self.spec)

    @classmethod
    def resume(cls, store: Path, directory: Path, runtimes: dict[str, Path]):
        result = cls.__new__(cls)
        result.store, result.directory, result.runtimes = store, directory, runtimes
        with (directory / "events.jsonl").open(encoding="utf-8") as stream:
            result.events = [json.loads(line) for line in stream]
        if not result.events or result.events[0]["kind"] != "run_frozen":
            raise ScoringError("missing original run freeze")
        result.spec = copied(result.events[0]["spec"])
        result._check_populations()
        result._no_pending()
        return result

    def _check_populations(self):
        for stage in ("search", "selection"):
            units = self.spec[stage + "_ids"]
            actual = rows(self.store, stage + "/inputs", self.spec["family"])
            if (
                not isinstance(units, list)
                or not units
                or len(set(units)) != len(units)
                or set(units) != set(actual)
            ):
                raise ScoringError("frozen population differs from actual inputs")
        if set(self.spec["search_ids"]) & set(self.spec["selection_ids"]):
            raise ScoringError("search/selection IDs overlap")

    def _record(self, kind, **payload):
        event = copied({"kind": kind, **payload})
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(event)
        return copied(event)

    def _no_pending(self):
        if self.spec != self.events[0]["spec"]:
            raise ScoringError("literal frozen run configuration changed")
        requests = {e["attempt"] for e in self.events if e["kind"] == "request"}
        terminal = {e["attempt"] for e in self.events if e["kind"] in {"result", "failed"}}
        if requests != terminal:
            raise ScoringError("unresolved worker attempt; inspect original logs, do not replay")
        generations = {e["call_id"] for e in self.events if e["kind"] == "generation_request"}
        settled = {
            e["call_id"]
            for e in self.events
            if e["kind"] in {"generation_result", "generation_failed"}
        }
        if generations != settled:
            raise ScoringError("unresolved generation attempt; inspect cost ledger, do not replay")

    def _one(self, kind):
        self._no_pending()
        found = [e for e in self.events if e["kind"] == kind]
        if len(found) != 1:
            raise ScoringError("missing or duplicate actual stage receipt: " + kind)
        return found[0]

    def _absent(self, kind):
        self._no_pending()
        if any(e["kind"] == kind for e in self.events):
            raise ScoringError("stage already ended: " + kind)

    def _message(self, kind, candidate, **payload):
        return copied(
            {
                "format": FORMAT,
                "kind": kind,
                "family": self.spec["family"],
                "model": self.spec["model"],
                "execution": self.spec["execution"],
                "candidate": candidate,
                **payload,
            }
        )

    def _dispatch(self, role, stage, message, tag):
        self._no_pending()
        validate(message, role, stage)
        if role != "predictor":
            if self.spec["family"] == "bfcl":
                bfcl_inventory(self.runtimes["bfcl"])
            else:
                text_inventory(self.runtimes["text"])
        attempt = len(self.events)
        scratch = self.directory / "workers" / str(attempt)
        scratch.mkdir(exist_ok=False)
        self._record("request", attempt=attempt, tag=tag, role=role, stage=stage, message=message)
        start = time.monotonic()
        try:
            result = launch_role(
                role,
                stage,
                self.store,
                scratch,
                Path(__file__).with_name("role_pipeline.py"),
                [
                    str(self.store),
                    str(self.runtimes["bfcl"]),
                    str(self.runtimes["text"]),
                    str(scratch),
                ],
                message=message,
                timeout=900,
            )
            for name, content in (("stdout", result.stdout), ("stderr", result.stderr)):
                with (scratch / name).open("x", encoding="utf-8") as stream:
                    stream.write(content)
            if result.returncode:
                raise ScoringError(f"worker exit {result.returncode}; see private stderr")
            report = json.loads(result.stdout)
            for name in ("kind", "family", "model", "execution", "candidate"):
                if report[name] != message[name]:
                    raise ScoringError("worker result disagrees with frozen request")
            if (
                report["format"] != "promptwitness.role-result/v1"
                or report["role"] != role
                or report["stage"] != stage
                or report["worker_pid"] == os.getpid()
                or type(report["landlock_abi"]) is not int
                or report["landlock_abi"] < 1
            ):
                raise ScoringError("missing independent restricted worker receipt")
            if role == "predictor":
                if set(report["predictions"]) != set(self.spec["search_ids"]):
                    raise ScoringError("incomplete predictor population")
            else:
                expected = {r["id"]: r for r in message["responses"]}
                observations = report["observations"]
                if set(observations) != set(expected):
                    raise ScoringError("missing actual score entries")
                for unit, obs in observations.items():
                    trace = ParentTrace(**obs["trace"])
                    if (
                        trace.correct != obs["score"]
                        or type(obs["score"]) is not int
                        or obs["score"] not in (0, 1)
                        or obs["replicate"] != expected[unit]["replicate"]
                        or any(
                            getattr(trace, key) != expected[unit][key]
                            for key in ("input_tokens", "output_tokens", "allocated_seconds")
                        )
                    ):
                        raise ScoringError("invalid actual score or execution trace")
        except Exception as exc:
            if isinstance(exc, subprocess.TimeoutExpired):
                for name, content in (("stdout", exc.stdout), ("stderr", exc.stderr)):
                    with (scratch / name).open("xb") as stream:
                        stream.write(
                            content.encode("utf-8") if isinstance(content, str) else content or b""
                        )
            self._record(
                "failed",
                attempt=attempt,
                error=type(exc).__name__,
                CPU_wall_seconds=time.monotonic() - start,
            )
            raise
        self._record(
            "result",
            attempt=attempt,
            tag=tag,
            report=report,
            CPU_wall_seconds=time.monotonic() - start,
        )
        return report

    def _score(self, candidate, responses, stage, tag):
        # Full actual vector is the working native-selector route. Early
        # rejection must later extend this path with the existing AuditPlan;
        # no partial vector is accepted or filled with predictions here.
        expected = self.spec[stage + "_ids"]
        if set(r["id"] for r in responses) != set(expected) or len(responses) != len(expected):
            raise ScoringError("complete actual response vector required")
        return self._dispatch(
            stage + "_scorer", stage, self._message("score", candidate, responses=responses), tag
        )

    def reference(self, responses):
        self._absent("reference_complete")
        completed = [e for e in self.events if e["kind"] == "result" and e["tag"] == "reference"]
        if completed:
            if len(completed) != 1:
                raise ScoringError("ambiguous completed reference")
            previous = next(
                e
                for e in self.events
                if e["kind"] == "request" and e["attempt"] == completed[0]["attempt"]
            )
            if previous["message"] != self._message(
                "score", self.spec["seed_prompt"], responses=responses
            ):
                raise ScoringError("completed reference response provenance changed")
            report = completed[0]["report"]
        else:
            report = self._score(self.spec["seed_prompt"], responses, "search", "reference")
        reference = {unit: row["score"] for unit, row in report["observations"].items()}
        # Publish only a fully observed fixed reference. Read-only role grants
        # never allow a worker to write this leaf. No scientific freeze claim.
        path = self.store / "search/reference" / (self.spec["run_id"] + ".json")
        publication = {"spec": self.spec, "reference": reference}
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != publication:
                raise ScoringError("original reference publication differs; do not overwrite")
        else:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(publication, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
        return self._record(
            "reference_complete",
            reference=reference,
            observations=report["observations"],
            path=str(path),
        )

    def freeze_candidate(self, parent, candidate, predictor: dict, *, structured: bool):
        self._one("reference_complete")
        self._absent("search_ended")
        parent_id, candidate_id = parse_prompt(parent).prompt_id, parse_prompt(candidate).prompt_id
        TransitionPredictor.from_dict(predictor)
        if (
            parent_id == candidate_id
            or candidate_id == parse_prompt(self.spec["seed_prompt"]).prompt_id
        ):
            raise ScoringError("distinct new candidate identity required")
        if parent != self.spec["seed_prompt"]:
            previous = self._candidate(parent_id)
            if parent != previous["candidate"] or not self._candidate_scores(parent_id):
                raise ScoringError("parent must be an already queried prompt")
        if type(structured) is not bool:
            raise ScoringError("explicit feature mode required")
        if any(
            e["kind"] == "candidate_frozen" and e["candidate"]["id"] == candidate_id
            for e in self.events
        ):
            raise ScoringError("candidate identity already frozen")
        return self._record(
            "candidate_frozen",
            parent=parent,
            candidate=candidate,
            predictor=predictor,
            structured=structured,
            configuration=self.spec["configuration"],
            execution=self.spec["execution"],
            reference=self._one("reference_complete")["reference"],
        )

    def _candidate(self, identifier):
        original = self.events[0]["spec"]
        if identifier == original["seed_prompt"]["id"]:
            # Native optimizers may retain the original incumbent. Its freeze
            # is the original run, not an invented candidate_frozen event.
            return {
                "kind": "run_frozen",
                "candidate": original["seed_prompt"],
                "configuration": original["configuration"],
                "execution": original["execution"],
            }
        matches = [
            e
            for e in self.events
            if e["kind"] == "candidate_frozen" and e["candidate"]["id"] == identifier
        ]
        if len(matches) != 1:
            raise ScoringError("candidate not frozen")
        return matches[0]

    def _candidate_scores(self, identifier, stage="search"):
        if stage == "search" and identifier == self.events[0]["spec"]["seed_prompt"]["id"]:
            return [self._one("reference_complete")["observations"]]
        vectors = [
            e["report"]["observations"]
            for e in self.events
            if e["kind"] == "result" and e["tag"] == stage + ":" + identifier
        ]
        merged = {}
        for vector in vectors:
            if set(merged) & set(vector):
                raise ScoringError("duplicate actual candidate observations")
            merged.update(vector)
        return [merged] if vectors else []

    def _search_requested(self, identifier):
        return any(
            (e["kind"] == "request" and e["tag"] == "search:" + identifier)
            or (e["kind"] == "generation_request" and e["identifier"] == identifier)
            for e in self.events
        )

    def predict(self, identifier):
        self._absent("search_ended")
        frozen = self._candidate(identifier)
        if self._candidate_scores(identifier) or self._search_requested(identifier):
            raise ScoringError("prediction must precede current candidate outcomes")
        parent = frozen["parent"]
        if parent == self.spec["seed_prompt"]:
            observations = self._one("reference_complete")["observations"]
        else:
            observed = self._candidate_scores(parent["id"])
            if len(observed) != 1:
                raise ScoringError("missing or ambiguous queried parent")
            observations = observed[0]
        traces = {unit: row["trace"] for unit, row in observations.items()}
        return self._dispatch(
            "predictor",
            "search",
            self._message(
                "predict",
                frozen["candidate"],
                parent=parent,
                predictor=frozen["predictor"],
                structured=frozen["structured"],
                reference=frozen["reference"],
                parent_observations=traces,
            ),
            "predict:" + identifier,
        )

    def score_candidate(self, identifier, responses):
        self._absent("search_ended")
        candidate = self._candidate(identifier)["candidate"]
        if self._candidate_scores(identifier):
            raise ScoringError("candidate already fully scored")
        return self._score(candidate, responses, "search", "search:" + identifier)

    def score_unit(self, identifier, response):
        """Score one actual response under a previously frozen audit plan.

        The incremental driver controls designated prefixes and episode charges.
        A partial vector cannot enter end_search until every unit is observed.
        """
        self._absent("search_ended")
        candidate = self._candidate(identifier)["candidate"]
        plans = [
            e for e in self.events if e["kind"] == "audit_frozen" and e["identifier"] == identifier
        ]
        if len(plans) != 1 or response["id"] not in plans[0]["replicates"]:
            raise ScoringError("unit requires original frozen audit population")
        if response["replicate"] != plans[0]["replicates"][response["id"]]:
            raise ScoringError("actual random replicate differs from audit freeze")
        observed = self._candidate_scores(identifier)
        if observed and response["id"] in observed[0]:
            raise ScoringError("actual unit already scored; do not replay")
        return self._dispatch(
            "search_scorer",
            "search",
            self._message("score", candidate, responses=[response]),
            "search:" + identifier,
        )["observations"][response["id"]]

    def end_search(self, survivors: list[str]):
        self._no_pending()
        self._one("reference_complete")
        self._absent("search_ended")
        if not survivors or len(set(survivors)) != len(survivors):
            raise ScoringError("explicit unique native survivors required")
        for candidate in survivors:
            self._candidate(candidate)
            scores = self._candidate_scores(candidate)
            if len(scores) != 1 or set(scores[0]) != set(self.spec["search_ids"]):
                raise ScoringError("native survivor lacks actual full score vector")
        return self._record("search_ended", survivors=survivors)

    def selection(self, identifier, responses):
        ended = self._one("search_ended")
        self._absent("selection_ended")
        if identifier not in ended["survivors"] or self._candidate_scores(identifier, "selection"):
            raise ScoringError("selection requires a new frozen search survivor")
        return self._score(
            self._candidate(identifier)["candidate"],
            responses,
            "selection",
            "selection:" + identifier,
        )

    def end_selection(self, selected: str):
        self._no_pending()
        ended = self._one("search_ended")
        self._absent("selection_ended")
        if selected not in ended["survivors"]:
            raise ScoringError("chosen prompt not a frozen survivor")
        for identifier in ended["survivors"]:
            scores = self._candidate_scores(identifier, "selection")
            if len(scores) != 1 or set(scores[0]) != set(self.spec["selection_ids"]):
                raise ScoringError("selection lacks actual complete vectors")
        # Caller retains the native selection rule. This records its actual
        # choice, never substitutes this controller's own maximization policy.
        return self._record(
            "selection_ended",
            selected=selected,
            frozen_prompt=self._candidate(selected)["candidate"],
            execution=self.spec["execution"],
            configuration=self.spec["configuration"],
            final_admission="NOT_GRANTED_BY_THIS_CONTROLLER",
        )
