"""Mechanical pinned-DSPy compile check; synthetic calls are NOT research results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import dspy
from dspy.adapters.base import Adapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.base_lm import BaseLM


class RecordingFixtureLM(BaseLM):
    """No network, GPU, model generation or response-quality measurement."""

    def __init__(self) -> None:
        super().__init__("synthetic-accounting-fixture", temperature=0, max_tokens=64, cache=False)
        self.calls: list[dict] = []
        self.expected_fields: list[str] = []

    def __call__(self, prompt=None, messages=None, **kwargs):
        role = "proposer" if self.expected_fields else "task"
        self.calls.append({"role": role, "messages": messages, "kwargs": kwargs})
        if self.expected_fields:
            return [
                json.dumps(
                    {name: "mechanical fixture instruction" for name in self.expected_fields}
                )
            ]
        return ["synthetic-answer"]


class FixtureSingleResponseAdapter(Adapter):
    """Exercise a shared single-output task renderer without patching MIPRO."""

    def format(self, signature, demos, inputs):
        if list(signature.input_fields) != ["task_input"] or list(signature.output_fields) != [
            "task_output"
        ]:
            raise ValueError("task formatter only supports the explicitly declared task interface")
        messages = [{"role": "system", "content": signature.instructions}]
        for demo in demos:
            messages.extend(
                [
                    {"role": "user", "content": demo["task_input"]},
                    {"role": "assistant", "content": demo["task_output"]},
                ]
            )
        messages.append({"role": "user", "content": inputs["task_input"]})
        return messages

    def parse(self, signature, completion):
        return {"task_output": completion}

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        if list(signature.input_fields) == ["task_input"] and list(signature.output_fields) == [
            "task_output"
        ]:
            lm.expected_fields = []
            result = lm(messages=self.format(signature, demos, inputs), **lm_kwargs)
            return [{"task_output": value} for value in result]
        # Synthetic response-schema knowledge is confined to this fixture LM.
        # The research provider cannot be supplied with these generated outputs.
        lm.expected_fields = list(signature.output_fields)
        try:
            return JSONAdapter(use_native_function_calling=False)(
                lm, lm_kwargs, signature, demos, inputs
            )
        finally:
            lm.expected_fields = []


def check() -> dict:
    lm = RecordingFixtureLM()
    adapter = FixtureSingleResponseAdapter()
    signature = dspy.Signature("task_input -> task_output", instructions="fixed task instruction")
    student = dspy.Predict(signature)
    training = [
        dspy.Example(
            task_input=f"synthetic training {index}", task_output="synthetic-answer"
        ).with_inputs("task_input")
        for index in range(12)
    ]
    validation = [
        dspy.Example(
            task_input=f"synthetic search {index}", task_output="synthetic-answer"
        ).with_inputs("task_input")
        for index in range(6)
    ]
    optimizer = dspy.MIPROv2(
        metric=lambda example, prediction, trace=None: (
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
    with dspy.context(lm=lm, adapter=adapter):
        compiled = optimizer.compile(
            student,
            trainset=training,
            valset=validation,
            num_trials=2,
            minibatch=False,
            program_aware_proposer=False,
            data_aware_proposer=False,
            tip_aware_proposer=False,
            fewshot_aware_proposer=True,
        )
    tasks = [call for call in lm.calls if call["role"] == "task"]
    proposals = [call for call in lm.calls if call["role"] == "proposer"]
    bootstrap = [
        call for call in tasks if call["messages"][-1]["content"].startswith("synthetic training")
    ]
    search = [
        call for call in tasks if call["messages"][-1]["content"].startswith("synthetic search")
    ]
    expected_seed = [
        adapter.format(signature, [], {"task_input": example.task_input}) for example in validation
    ]
    if len(search) < 6 or [call["messages"] for call in search[:6]] != expected_seed:
        raise ValueError(
            "native default evaluation does not match the canonical demo-free seed requests"
        )
    if compiled is None or not tasks or not proposals or not bootstrap or student.demos:
        raise ValueError("native compile/fixture isolation failed")
    maximum_demos_seen = max(
        sum(message["role"] == "assistant" for message in call["messages"]) for call in tasks
    )
    if maximum_demos_seen > 4:
        raise ValueError("native default total demonstration bound exceeded")
    return {
        "format": "promptwitness.delta.native-mipro-mechanical-check/v1",
        "status": "PASSED_MECHANICAL_ONLY",
        "dspy_source_commit": "da1736e21ffda8cc4b86379d4748b011764d507c",
        "dspy_version": dspy.__version__,
        "optimizer_source_modified": False,
        "task_renderer": (
            "Explicit single task input/output, same canonical renderer planned for GEPA; "
            "different from DSPy default ChatAdapter"
        ),
        "synthetic_task_calls": len(tasks),
        "synthetic_proposer_calls": len(proposals),
        "synthetic_training_bootstrap_calls": len(bootstrap),
        "native_default_seed_vector_genuine_fixture_completions": len(expected_seed),
        "native_default_seed_request_identity_matches_canonical_renderer": True,
        "default_demo_limits_retained": {"bootstrapped": 4, "labeled": 4},
        "maximum_total_demonstrations_seen": maximum_demos_seen,
        "default_total_demonstration_limit": 4,
        "source_total_demo_limit_basis": (
            "bootstrap.py:_train subtracts augmented demos from the labeled limit, "
            "then concatenates; defaults 4/4 do not mean eight demos."
        ),
        "original_student_remains_demo_free": True,
        "real_model_calls": 0,
        "real_task_scores_or_optimizer_benefits_measured": False,
        "limitations": [
            "Uses synthetic completions and a six-unit search pool; "
            "not the 2048-episode campaign or real model adapter.",
            "Canonical message identity alone does not validate the full production cache key "
            "or nondeterministic execution assumptions.",
            "Fixture proposer output fields are scripted for parsing. "
            "Real proposer failures must remain charged, not replaced by scripts.",
            "No MIPRO selector code was replaced; "
            "the adapter setting is an explicit task-interface configuration deviation.",
        ],
        "supersedes": (
            "mipro-native-mechanical-v1.json: two candidate sets exercised only "
            "zero-shot/labeled special sets; this check includes the actual bootstrap path."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = check()
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
