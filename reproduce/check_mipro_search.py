"""CPU-only native MIPRO search qualification on authored data, not research results."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

# Fixed DSPy installs a lazy NumPy proxy. Loading it before NumPy1.26 can cause
# a partial numpy.linalg cycle (observed in the native CPU import preflight).
import numpy

# isort: split
import dspy
import optuna
from dspy.evaluate import Evaluate
from dspy.teleprompt import mipro_optimizer_v2 as native
from dspy.teleprompt import utils

from promptwitness.incremental.budget import ResourceLedger, ResourceLimit
from reproduce import pipeline_controller as control
from reproduce import role_pipeline as roles
from reproduce.check_mipro_native import FixtureSingleResponseAdapter, RecordingFixtureLM
from reproduce.incremental_pipeline import IncrementalPipeline
from reproduce.mipro_search import MIPROGateEvaluator, mipro_prompt, strict_mipro_search
from reproduce.prepare_bfcl_fit import MODEL_REVISIONS, write_jsonl
from reproduce.process_access import LEAVES
from reproduce.strict_scoring import ScoringError


class SearchFixtureLM(RecordingFixtureLM):
    """Scripted proposer variety and outputs; every invocation remains a fixture."""

    def __call__(self, prompt=None, messages=None, **kwargs):
        role = "proposer" if self.expected_fields else "task"
        self.calls.append({"role": role, "messages": messages, "kwargs": kwargs})
        if self.expected_fields:
            number = sum(call["role"] == "proposer" for call in self.calls)
            instruction = "authored reject" if number % 2 == 0 else "authored retain"
            return [json.dumps({name: instruction for name in self.expected_fields})]
        instruction, unit = messages[0]["content"], messages[-1]["content"]
        baseline_error = (
            instruction == "authored seed" and unit.startswith("s") and int(unit[1:]) % 4 == 0
        )
        return ["wrong" if "reject" in instruction or baseline_error else "authored answer"]


def response(unit, output):
    return {
        "id": unit,
        "output": output,
        "status": "completed",
        "replicate": "sample-0",
        "input_tokens": 12,
        "output_tokens": 3,
        "allocated_seconds": None,
    }


def authored_driver(root, student, lm, stack):
    """Real controller/driver, explicitly authored in-process scoring transport.

    Synthetic ABI/PID fields satisfy the application protocol, not process proof.
    This fresh fixture ledger must never be used as real resource history.
    """
    store = root / "store"
    for leaf in LEAVES:
        (store / leaf).mkdir(parents=True)
    units = [f"s{i}" for i in range(64)]
    for stage, ids in (("search", units), ("selection", ["v"])):
        write_jsonl(
            store / f"{stage}/inputs/hotpotqa.jsonl",
            [{"id": u, "messages": [{"role": "user", "content": u}]} for u in ids],
        )
        write_jsonl(
            store / f"{stage}/gold/hotpotqa.jsonl",
            [{"id": u, "answer": "authored answer"} for u in ids],
        )
    spec = {
        "run_id": "authored-native-mipro",
        "family": "hotpotqa",
        "model": next(iter(MODEL_REVISIONS)),
        "execution": {
            "model_revision": next(iter(MODEL_REVISIONS.values())),
            "tokenizer_revision": "authored",
            "backend_version": "authored",
            **{
                k: "a" * 64
                for k in (
                    "backend_config_digest",
                    "template_digest",
                    "decoding_digest",
                    "scorer_digest",
                    "data_digest",
                    "tool_environment_digest",
                )
            },
        },
        "configuration": {"evaluator": "AUTHORED_IN_PROCESS", "max_evaluation_episodes": 2048},
        "seed_prompt": mipro_prompt(student, "seed"),
        "search_ids": units,
        "selection_ids": ["v"],
    }
    stack.enter_context(patch.object(control, "text_inventory", lambda path: {}))
    stack.enter_context(
        patch.object(roles, "load_staged", lambda path: ({}, None, lambda a, b: a == b))
    )

    def launch(role, stage, data, scratch, entry, arguments, *, message, timeout):
        with patch.dict(
            os.environ, PW_ACCESS_ROLE=role, PW_ACCESS_STAGE=stage, PW_LANDLOCK_ABI="1"
        ):
            result = roles.application(data, {"bfcl": root, "text": root}, scratch, message)
        result["worker_pid"] = os.getpid() + 1  # AUTHORED transport, never actual IPC proof
        return subprocess.CompletedProcess([], 0, json.dumps(result), "")

    stack.enter_context(patch.object(control, "launch_role", launch))
    controller = control.PipelineController(store, root / "run", spec, {"bfcl": root, "text": root})
    controller.reference(
        [
            response(
                u,
                lm(
                    messages=[
                        {"role": "system", "content": student.signature.instructions},
                        {"role": "user", "content": u},
                    ]
                )[0],
            )
            for u in units
        ]
    )
    limit = ResourceLimit(800000, 2000000000, 200000000, 1000)
    ledger = ResourceLedger(
        root / "AUTHORED-resources.sqlite",
        historical_usage={"calls": 0, "input_tokens": 0, "output_tokens": 0, "gpu_hours": 0},
        historical_digest="a" * 64,
        global_limit=limit,
        stage_limits={"authored": limit},
        gpu_uuid="GPU-authored-no-allocation",
    )
    stack.callback(ledger.close)
    driver = IncrementalPipeline(
        controller, ledger, resource_stage="authored", input_cap=32768, output_cap=64
    )
    stack.callback(driver.close)
    documents = [spec["seed_prompt"]]

    def resolve(program):
        for document in documents:
            if mipro_prompt(program, document["id"]) == document:
                return document["id"]
        identifier = f"c{len(documents)}"
        document = mipro_prompt(program, identifier)
        controller.freeze_candidate(
            spec["seed_prompt"],
            document,
            {"format": "promptwitness.delta.predictor/v1.1", "status": "UNFITTED"},
            structured=True,
        )
        driver.freeze(
            identifier,
            replicates={u: "sample-0" for u in units},
            execution_scope="FROZEN_TABLE",
            fixture_seed=7,
        )
        documents.append(document)
        return identifier

    def execute(request):
        messages = [
            {"role": m["role"], "content": m["content"]} for m in request["candidate"]["messages"]
        ]
        messages.extend(request["unit"]["messages"])
        return response(request["unit"]["id"], lm(messages=messages)[0])

    return MIPROGateEvaluator(driver, execute, resolve), controller, ledger


def compile_fixture(lm, student, evaluator, studies, *, minibatch):
    original_create = optuna.create_study

    def capture(*args, **kwargs):
        study = original_create(*args, **kwargs)
        studies.append(study)
        return study

    optimizer = dspy.MIPROv2(
        metric=lambda example, prediction, trace=None: float(
            prediction.task_output == example.task_output
        ),
        prompt_model=lm,
        task_model=lm,
        auto=None,
        num_candidates=4,
        num_threads=1,
        max_errors=1,
        seed=11,
        init_temperature=0,
        verbose=False,
    )
    training = [
        dspy.Example(task_input=f"training-{i}", task_output="authored answer").with_inputs(
            "task_input"
        )
        for i in range(12)
    ]
    search = [
        dspy.Example(
            unit_id=f"s{i}", task_input=f"s{i}", task_output="authored answer"
        ).with_inputs("task_input")
        for i in range(64)
    ]
    with (
        patch.object(optuna, "create_study", capture),
        dspy.context(lm=lm, adapter=FixtureSingleResponseAdapter()),
        strict_mipro_search(evaluator),
    ):
        return optimizer.compile(
            student,
            trainset=training,
            valset=search,
            num_trials=12,
            minibatch=minibatch,
            minibatch_size=8,
            minibatch_full_eval_steps=3,
            program_aware_proposer=False,
            data_aware_proposer=False,
            tip_aware_proposer=False,
            fewshot_aware_proposer=True,
        )


def check(source, output):
    # Existing scientific pins: compare original installed files before runtime adaptation.
    for module, relative in (
        (native, "teleprompt/mipro_optimizer_v2.py"),
        (utils, "teleprompt/utils.py"),
        (inspect.getmodule(Evaluate), "evaluate/evaluate.py"),
    ):
        if Path(module.__file__).read_bytes() != (source / "dspy" / relative).read_bytes():
            raise ScoringError("installed DSPy source differs from the fixed checkout")
    original = native.Evaluate, native.eval_candidate_program

    def prune(*args, **kwargs):
        raise optuna.TrialPruned("authored upstream failure witness")

    swallowed = utils.eval_candidate_program(1, [{}], None, prune)
    if swallowed.score != 0 or swallowed.results != []:
        raise ScoringError("original helper failure-to-zero premise changed")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for minibatch in (False, True):
        lm = SearchFixtureLM()
        student = dspy.Predict(
            dspy.Signature("task_input -> task_output", instructions="authored seed")
        )
        studies = []
        with ExitStack() as stack:
            root = output / ("minibatch" if minibatch else "full")
            root.mkdir()
            evaluator, controller, ledger = authored_driver(root, student, lm, stack)
            evaluations = []

            def observed(
                program,
                *,
                devset,
                callback_metadata,
                studies=studies,
                evaluations=evaluations,
                evaluator=evaluator,
            ):
                trial = studies[-1].trials[-1] if studies else None
                number = (
                    trial.number if trial is not None and trial.state.name == "RUNNING" else None
                )
                evaluations.append({"trial": number, "metric_key": callback_metadata["metric_key"]})
                return evaluator(program, devset=devset, callback_metadata=callback_metadata)

            compiled = compile_fixture(lm, student, observed, studies, minibatch=minibatch)
            trials = studies[0].trials
            states = Counter(t.state.name for t in trials)
            if not states["PRUNED"] or not states["COMPLETE"] or states["FAIL"]:
                raise ScoringError(
                    "original Optuna search did not exercise both survivors and prunes"
                )
            if any(t.value is not None for t in trials if t.state.name == "PRUNED"):
                raise ScoringError("rejection became a completed zero")
            completed = [
                e for e in controller.events if e["kind"] == "native_mipro_evaluation_complete"
            ]
            if (
                not completed
                or compiled is None
                or compiled.signature.instructions != "authored retain"
                or compiled.score != 100
            ):
                raise ScoringError(
                    "native selector failed to advance to the authored complete winner"
                )
            if completed[0]["identifier"] != "seed" or completed[0]["native_score"] != 75:
                raise ScoringError("original lower full reference missing")
            nonseed = [e for e in completed if e["identifier"] != "seed"]
            if not nonseed or not any(e["native_score"] == 100 for e in nonseed):
                raise ScoringError("complete non-seed survivor never reached the selector")
            for event in completed:
                if set(controller._candidate_scores(event["identifier"])[0]) != set(
                    controller.spec["search_ids"]
                ):
                    raise ScoringError("partial vector reached original selector")
            scheduled_pruned = [
                t.number for t in trials if t.state.name == "PRUNED" and (t.number + 1) % 4 == 0
            ]
            if minibatch:
                if not any(
                    e["trial"] is not None and e["metric_key"] == "eval_full" for e in evaluations
                ):
                    raise ScoringError("post-baseline periodic full-evaluation path not exercised")
                if not scheduled_pruned or any(
                    e["trial"] in scheduled_pruned and e["metric_key"] == "eval_full"
                    for e in evaluations
                ):
                    raise ScoringError("prune-at-periodic-boundary behavior not witnessed")
            results.append(
                {
                    "minibatch": minibatch,
                    "trial_states": dict(states),
                    "pruned_trial_values": [t.value for t in trials if t.state.name == "PRUNED"],
                    "complete_native_evaluations": len(completed),
                    "authored_candidate_ledger_calls": ledger.usage()["calls"],
                    "fixture_LM_calls_all_roles": dict(Counter(c["role"] for c in lm.calls)),
                    "scientific_cost_measurement": False,
                    "native_winner_instructions": compiled.signature.instructions,
                    "native_winner_score": compiled.score,
                    "baseline_score": completed[0]["native_score"],
                    "evaluation_cadence": evaluations,
                    "scheduled_pruned_trials": scheduled_pruned if minibatch else [],
                    "pruning_skips_original_periodic_full": minibatch,
                }
            )
    if (native.Evaluate, native.eval_candidate_program) != original:
        raise ScoringError("native runtime symbols not restored")
    report = {
        "format": "promptwitness.delta.mipro-search-qualification/v1",
        "status": "DIAGNOSTIC_GATE_BOUNDARY_OK_CADENCE_UNQUALIFIED",
        "dspy_revision": "da1736e21ffda8cc4b86379d4748b011764d507c",
        "numpy_version": numpy.__version__,
        "selected_installed_source_byte_equal": True,
        "original_helper_swallowed_prune_as_zero": True,
        "source_checkout_modified": False,
        "qualifications": results,
        "real_model_calls": 0,
        "new_gpu_allocations": 0,
        "scientific_admission": False,
        "strict_native_control_compile_qualified": False,
        "limitations": [
            "Authored 64-unit data, scripted LM/proposer, fixture seeds and 12 trials only.",
            "In-process scoring harness, not actual Linux process isolation "
            "or official scorer proof.",
            "Candidate ledger excludes fixture reference/bootstrap/proposal calls; "
            "all-role LM counts separate.",
            "No real backend/full optimizer cost forecast, ONLINE_PINNED, "
            "real M1, Pilot or GEPA qualification.",
            "Strict exception/max_errors/failure_score boundary is an explicit "
            "shared control adaptation.",
            "TrialPruned exits an original objective before its scheduled full-eval block; "
            "this altered cadence is not scientific native-equivalence qualification.",
        ],
    }
    with (output / "qualification.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="existing fixed DSPy checkout")
    parser.add_argument("output", type=Path, help="new exclusive private fixture directory")
    args = parser.parse_args()
    check(args.source, args.output)
