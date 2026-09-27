"""CPU official-scorer checks on authored fixtures and prior fit-only responses.

No model inference and no final examples. This is an integration qualification,
not a Pilot, baseline reproduction or demonstration of the proposed method.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import random
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from reproduce.strict_scoring import (
    ScoringError,
    UnsupportedScoring,
    score_hotpotqa,
    score_ifbench,
    score_iftrain,
)

SOURCE_REVISIONS = {
    "IFBench": "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d",
    "open-instruct-verifiers": "99b1ee970490a2d0d5664663eb0b1acc410b944c",
}
NLTK_RESOURCES = ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger_eng")


@contextmanager
def construction_rng(seed=11):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def prepare_nltk(source):
    """Explicit build step; official IFBench import would otherwise auto-download."""
    import nltk

    target = source / "nltk-data"
    target.mkdir(exist_ok=True)
    for resource in NLTK_RESOURCES:
        if not nltk.download(resource, download_dir=str(target), quiet=True, raise_on_error=True):
            raise ScoringError("NLTK resource preparation failed")


def verify_native_sources(source):
    # Check only scorer source paths, never bundled final data or upstream scores.
    for directory, revision in SOURCE_REVISIONS.items():
        actual = subprocess.check_output(
            ["git", "-C", str(source / directory), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != revision:
            raise ScoringError("unexpected native source revision")
        paths = (
            [
                "evaluation_lib.py",
                "ifbench/instructions.py",
                "ifbench/classic_instructions.py",
                "ifbench/instructions_registry.py",
                "ifbench/instructions_util.py",
            ]
            if directory == "IFBench"
            else [
                "open_instruct/IFEvalG/instructions.py",
                "open_instruct/IFEvalG/instructions_registry.py",
                "open_instruct/IFEvalG/instructions_util.py",
            ]
        )
        subprocess.run(
            ["git", "-C", str(source / directory), "diff", "--exit-code", "HEAD", "--", *paths],
            check=True,
            capture_output=True,
        )


def load_native(source):
    import nltk
    from langdetect import DetectorFactory

    verify_native_sources(source)
    nltk.data.path.insert(0, str(source / "nltk-data"))
    # Fail before importing the native module's automatic downloader.
    for resource in (
        "tokenizers/punkt",
        "tokenizers/punkt_tab",
        "corpora/stopwords",
        "taggers/averaged_perceptron_tagger_eng",
    ):
        nltk.data.find(resource, paths=[str(source / "nltk-data")])
    DetectorFactory.seed = 11
    sys.path.insert(0, str(source / "open-instruct-verifiers"))
    from open_instruct.IFEvalG.instructions_registry import INSTRUCTION_DICT

    sys.path.insert(0, str(source / "IFBench"))
    from ifbench.instructions_registry import INSTRUCTION_DICT as bench_registry

    def load_file(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ScoringError("official scorer module cannot be loaded")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    bench = load_file("promptwitness_native_ifbench", source / "IFBench/evaluation_lib.py")
    hotpot = load_file(
        "promptwitness_native_hotpot", source / "hotpot-scorer/hotpot_evaluate_v1.py"
    )

    def strict(prompt, ids, kwargs, response):
        if any(identifier not in bench_registry for identifier in ids):
            raise UnsupportedScoring("unknown IFBench constraint")
        inp = bench.InputExample(key=0, instruction_id_list=ids, prompt=prompt, kwargs=kwargs)
        result = bench.test_instruction_following_strict(inp, {prompt: response})
        if (
            result.response != response
            or result.prompt != prompt
            or result.instruction_id_list != ids
            or type(result.follow_all_instructions) is not bool
            or result.follow_all_instructions != all(result.follow_instruction_list)
        ):
            raise ScoringError("native IFBench output does not match the requested response")
        return result.follow_instruction_list

    return INSTRUCTION_DICT, strict, hotpot.exact_match_score


def mechanical_checks(train_registry, bench_strict, hotpot_em):
    results = []
    for text, expected in (("The Red Bridge.", 1), ("Red", 0), ("", 0)):
        actual = score_hotpotqa(text, "completed", "red bridge", hotpot_em).value
        results.append(
            {"case": f"hotpot_em_{len(results)}", "expected": expected, "actual": actual}
        )
    for scorer in ("iftrain", "ifbench"):
        ids = ["keywords:existence", "punctuation:no_comma"]
        arguments = [{"keywords": ["apricot", "cedar"]}, {}]
        for text, expected in (
            ("apricot cedar", 1),
            ("apricot", 0),
            ("apricot, cedar", 0),
            ("", 0),
        ):
            with construction_rng():
                score = (
                    score_iftrain(text, "completed", ids, arguments, train_registry)
                    if scorer == "iftrain"
                    else score_ifbench(
                        text, "completed", "Fixture only", ids, arguments, bench_strict
                    )
                )
            results.append(
                {"case": f"{scorer}_{len(results)}", "expected": expected, "actual": score.value}
            )
    # Exercise an OOD checker with NLTK and the native strict loop, without final data.
    for text, expected in (("apricot cedar", 1), ("apricot", 0)):
        score = score_ifbench(
            text,
            "completed",
            "Write two words.",
            ["count:word_count_range"],
            [{"min_words": 2, "max_words": 2}],
            bench_strict,
        )
        results.append(
            {"case": f"ifbench_ood_{len(results)}", "expected": expected, "actual": score.value}
        )
    if any(row["actual"] != row["expected"] for row in results):
        raise ScoringError("native mechanical fixture mismatch")
    return results


def audit_prior_fit(source, pools, train_registry, hotpot_em):
    import pyarrow.parquet as parquet

    # Membership is fixed from metadata before scoring, never chosen from correctness.
    selected = {
        family: {row["id"] for row in pools[family]["pools"]["fit"]}
        for family in ("hotpotqa", "instruction_following")
    }
    responses = {family: {} for family in selected}
    for line in (source / "infra1/qwen-infra1.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        family = row["family"]
        if family not in selected or row["role"] != "task" or row.get("profile", "base") != "base":
            continue
        prefix = family + ":"
        if not row["id"].startswith(prefix):
            continue
        unit = row["id"][len(prefix) :]
        if unit not in selected[family]:
            continue
        if (
            row["model"] != "Qwen/Qwen3.5-9B"
            or row["revision"] != "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
            or row["scoring_executed"] is not False
            or unit in responses[family]
        ):
            raise ScoringError("changed or duplicated archived fit response")
        responses[family][unit] = row
    output = {}
    for family, by_id in responses.items():
        if not by_id:
            raise ScoringError("no archived fit responses for official integration check")
        ids = sorted(by_id)
        if family == "hotpotqa":
            records = [
                row
                for name in ("hotpot-train-0.parquet", "hotpot-train-1.parquet")
                for row in parquet.read_table(
                    source / name, columns=["id", "answer"], filters=[("id", "in", ids)]
                ).to_pylist()
            ]
            annotations = {row["id"]: row["answer"] for row in records}
        else:
            records = parquet.read_table(
                source / "if-train.parquet",
                columns=["key", "ground_truth"],
                filters=[("key", "in", ids)],
            ).to_pylist()
            annotations = {row["key"]: ast.literal_eval(row["ground_truth"])[0] for row in records}
        if set(annotations) != set(ids):
            raise ScoringError("missing fit annotation")
        checks = []
        for unit in ids:
            row = by_id[unit]
            annotation = annotations[unit]
            with construction_rng():
                score = (
                    score_hotpotqa(row["output"], row["status"], annotation, hotpot_em)
                    if family == "hotpotqa"
                    else score_iftrain(
                        row["output"],
                        row["status"],
                        annotation["instruction_id"],
                        annotation["kwargs"],
                        train_registry,
                    )
                )
            checks.append({"unit": unit, "score": score.value, "profile": score.profile})
        output[family] = {
            "units": len(checks),
            "correct": sum(row["score"] for row in checks),
            "checks": checks,
        }
    return output


def check(source, pools_path):
    train_registry, bench_strict, hotpot_em = load_native(source)
    fixtures = mechanical_checks(train_registry, bench_strict, hotpot_em)
    pools = json.loads(pools_path.read_text(encoding="utf-8"))
    fit = audit_prior_fit(source, pools, train_registry, hotpot_em)
    return {
        "format": "promptwitness.native-strict-scorer-check/v1",
        "scope": "authored_mechanical_fixtures_and_preexisting_fit_only_cost_responses",
        "native_sources": {
            **SOURCE_REVISIONS,
            "HotpotQA": "3635853403a8735609ee997664e1528f4480762a",
        },
        "mechanical_checks": fixtures,
        "fit_response_audit": fit,
        "new_model_calls": 0,
        "new_allocated_GPU_hours": 0,
        "final_examples_or_scores_accessed": False,
        "method_effect_or_pilot_measured": False,
        "BFCL_official_runtime": "NOT_QUALIFIED_vendor_SDK_import_block_remains",
        "all_scorers_gate": "PARTIAL_NOT_PASSED",
        "process_split_access_gate": "NOT_PASSED_BY_THIS_CPU_CHECK",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("pools", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prepare-nltk", action="store_true")
    args = parser.parse_args()
    if args.prepare_nltk:
        prepare_nltk(args.source)
    result = check(args.source, args.pools)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "mechanical_checks": len(result["mechanical_checks"]),
                "fit_units": {
                    key: value["units"] for key, value in result["fit_response_audit"].items()
                },
                "all_scorers_gate": result["all_scorers_gate"],
                "new_model_calls": 0,
            }
        )
    )
