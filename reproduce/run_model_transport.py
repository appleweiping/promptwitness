"""Real persistent-model wire/cost admission, not a scientific Pilot or effect.

Open only the fixed prepared messages and verified closed history. All responses
and physical ledgers stay private. No final bodies or gold stores are opened;
the original proposer cost fixtures do contain already-disclosed fit examples.
No scoring is executed. Reuse one continued DB, never reprice from zero.
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
from reproduce.torch_runtime import task_wire


def run(args):
    raw = args.requests.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.request_sha256:
        raise ValueError("fixed original complete wire source changed")
    all_rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if len(all_rows) != 10 or Counter(row["model"] for row in all_rows) != Counter(
        {m: 5 for m in MODEL_REVISIONS}
    ):
        raise ValueError("fixed two-model ten-request training-side qualification required")
    records = [row for row in all_rows if row["model"] == args.model]
    if len({row["id"] for row in records}) != 5 or Counter(row["purpose"] for row in records) != {
        "fit": 3,
        "proposer": 2,
    }:
        raise ValueError("complete predeclared task/proposer transport required")
    history = closed_history(args.history, args.history_sha256)
    account = continue_ledger(args.ledger, history)
    before = account.usage()
    responses = []
    try:
        with PersistentModel(
            args.output,
            args.snapshot,
            args.model,
            account,
            args.run_id,
            model_python=args.model_python,
        ) as process:
            execution = {
                **process.profile,
                "data_digest": args.request_sha256,
                "scorer_digest": digest("NOT_RUN_transport_admission_only"),
                "tool_environment_digest": digest("BFCL_text_JSON_no_native_tool_execution"),
            }
            with (args.output / "freeze.json").open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "run_id": args.run_id,
                        "model": args.model,
                        "model_python": str(args.model_python),
                        "execution": execution,
                        "history": history,
                        "records": records,
                        "science_or_online_admission": False,
                    },
                    stream,
                    allow_nan=False,
                )
                stream.flush()
                import os

                os.fsync(stream.fileno())
            for row in records:
                if row["purpose"] == "fit":
                    wire = task_wire(
                        {
                            "candidate": row["candidate"],
                            "unit": row["unit"],
                            "family": row["family"],
                            "model": args.model,
                            "execution": execution,
                            "configuration": {"purpose": "TRANSPORT_ONLY_NOT_PILOT"},
                            "replicate": args.run_id + ":" + row["id"],
                        }
                    )
                else:
                    original = row["original"]
                    wire = {
                        k: original[k]
                        for k in ("id", "role", "family", "messages", "max_new_tokens")
                    }
                    wire["replicate"] = args.run_id + ":" + row["id"]
                value = process.metered(
                    args.run_id + ":" + row["id"], execution, wire, purpose=row["purpose"]
                )
                responses.append(value)
                print(
                    json.dumps(
                        {
                            "event": "completed_original_wire",
                            "completed": len(responses),
                            "purpose": row["purpose"],
                            "family": wire["family"],
                            "input_tokens": value["input_tokens"],
                            "output_tokens": value["output_tokens"],
                        }
                    ),
                    flush=True,
                )
        after = account.usage()
        summary = {
            "format": "promptwitness.model-transport-admission/v1",
            "model": args.model,
            "revision": MODEL_REVISIONS[args.model],
            "requests": len(responses),
            "input_tokens": sum(r["input_tokens"] for r in responses),
            "output_tokens": sum(r["output_tokens"] for r in responses),
            "allocated_GPU_hours": after["gpu_hours"] - before["gpu_hours"],
            "before": before,
            "after": after,
            "identity": execution,
            "scoring_executed": False,
            "online_certificate_validated": False,
            "inference_process_data_sandbox": False,
            "native_optimizer_or_full_proposer_workflow": False,
            "real_M1_fit": "UNFITTED",
            "Pilot": "NOT_RUN",
            "method_effects": "NOT_MEASURED",
        }
        with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
        return summary
    finally:
        account.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("requests", "history", "ledger", "output", "snapshot", "model-python"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("request-sha256", "history-sha256", "model", "run-id"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(run(parser.parse_args()), allow_nan=False))
