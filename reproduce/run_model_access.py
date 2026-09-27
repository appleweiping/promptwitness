"""Real fixed fit-input execution under the new inference filesystem boundary.

Only the three originally preselected input-only fit rows per model, no proposer
execution, gold-store/scoring/final access or scientific admission. The original
ten-row source also contains unused, previously disclosed fit-gold examples;
the trusted controller parses that file, but sends only input-only fit rows.
Full actual costs
continue the previous closed ledger. No automatic retry or response reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from promptwitness.incremental.sampling import digest
from reproduce.online_resources import closed_history, continue_ledger
from reproduce.persistent_model import PersistentModel
from reproduce.prepare_bfcl_fit import MODEL_REVISIONS
from reproduce.torch_runtime import TASK_CAPS, task_wire


def training_records(raw: bytes, sha: str, model: str) -> list[dict]:
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("original predeclared input source changed")
    all_rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if len(all_rows) != 10 or Counter(row["model"] for row in all_rows) != Counter(
        {m: 5 for m in MODEL_REVISIONS}
    ):
        raise ValueError("original prepared two-model source required")
    rows = [row for row in all_rows if row["model"] == model and row["purpose"] == "fit"]
    if len({row["id"] for row in rows}) != 3 or Counter(row["family"] for row in rows) != {
        family: 1 for family in TASK_CAPS
    }:
        raise ValueError("one predeclared fit input per original family required")
    return rows


def run(args):
    records = training_records(args.requests.read_bytes(), args.request_sha256, args.model)
    history = closed_history(args.history, args.history_sha256)
    account = continue_ledger(args.ledger, history)
    before = account.usage()
    observations = []
    try:
        with PersistentModel(
            args.output,
            args.snapshot,
            args.model,
            account,
            args.run_id,
            model_python=args.model_python,
            data_root=args.data_root,
        ) as process:
            execution = {
                **process.profile,
                "data_digest": args.request_sha256,
                "scorer_digest": digest("NOT_RUN_inference_boundary_only"),
                "tool_environment_digest": digest("BFCL_text_JSON_no_native_tool_execution"),
            }
            for row in records:
                request = {
                    "candidate": row["candidate"],
                    "unit": row["unit"],
                    "family": row["family"],
                    "model": args.model,
                    "execution": execution,
                    "configuration": {"purpose": "INFERENCE_ACCESS_NOT_PILOT"},
                    "replicate": args.run_id + ":" + row["id"],
                }
                response = process.metered(
                    request["replicate"], execution, task_wire(request), purpose="fit"
                )
                observations.append(
                    {
                        "family": row["family"],
                        "input_tokens": response["input_tokens"],
                        "output_tokens": response["output_tokens"],
                    }
                )
                print(
                    json.dumps({"event": "restricted_real_fit_wire", **observations[-1]}),
                    flush=True,
                )
        after = account.usage()
        report = {
            "format": "promptwitness.inference-access-execution/v1",
            "model": args.model,
            "revision": MODEL_REVISIONS[args.model],
            "requests": len(observations),
            "families": observations,
            "identity": execution,
            "access": process.access,
            "allocated_GPU_hours": after["gpu_hours"] - before["gpu_hours"],
            "before": before,
            "after": after,
            "scoring_executed": False,
            "online_certificate_validated": False,
            "native_optimizer_or_full_proposer_workflow": False,
            "real_M1_fit": "UNFITTED",
            "Pilot": "NOT_RUN",
            "method_effects": "NOT_MEASURED",
            "boundary_scope": (
                "filesystem read/write only, not ABI1 network/stat/truncate or malicious operator"
            ),
        }
        with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        return report
    finally:
        account.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "requests",
        "history",
        "ledger",
        "output",
        "snapshot",
        "model-python",
        "data-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("request-sha256", "history-sha256", "model", "run-id"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(run(parser.parse_args()), allow_nan=False))
