"""Owned persistent-model throughput probe. Cost evidence, NOT a method pilot.

The supervisor charges one reservation through child exit, including idle/cold
time. Each attempted row is reserved before generate. No response cache, cap or
precision change; no online statistical certificate from matching output text.
Raw responses and SQLite stay in the external controlled task directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from promptwitness.incremental.budget import ResourceLedger, ResourceLimit
from promptwitness.incremental.sampling import digest

GPU = "GPU-18ce5833-27a4-ac4b-f952-49cc52045714"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL = "Qwen/Qwen3.5-9B"
HISTORY_SHA = "e3807cc3c9713152193c5c80f0acdad8161cccf8da2cd8464c73e0bb31c04d04"


def ledger(args: argparse.Namespace) -> ResourceLedger:
    raw = args.history.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HISTORY_SHA:
        raise ValueError("verified immutable v1 summary required")
    value = json.loads(raw)["consumed_all_g0_phases"]
    return ResourceLedger(
        args.ledger,
        historical_usage={
            "calls": value["requests"],
            "input_tokens": value["input_tokens"],
            "output_tokens": value["output_tokens"],
            "gpu_hours": value["allocated_gpu_hours"],
        },
        historical_digest=HISTORY_SHA,
        global_limit=ResourceLimit(800000, 2000000000, 200000000, 1000),
        stage_limits={
            "unblocking_v1_1": ResourceLimit(400, 20000000, 2000000, 8),
            "pilot": ResourceLimit(12000, 200000000, 20000000, 30),
        },
        gpu_uuid=GPU,
    )


def run_child(args: argparse.Namespace) -> None:
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != GPU
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or args.snapshot.name != REVISION
    ):
        raise ValueError("only the assigned device and frozen snapshot/runtime are permitted")
    raw = args.requests.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.request_sha256:
        raise ValueError("frozen benchmark request hash changed")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if len(rows) != 32 or len({r["id"] for r in rows}) != 32:
        raise ValueError("fixed 32-unit workload required")
    caps = {"bfcl": 1024, "hotpotqa": 64, "instruction_following": 4096}
    if any(r["role"] != "task" or r["max_new_tokens"] != caps[r["family"]] for r in rows):
        raise ValueError("frozen task interfaces/caps changed")
    account = ledger(args)
    started = time.monotonic()
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.manual_seed(11)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    tokenizer = AutoTokenizer.from_pretrained(
        args.snapshot, local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForImageTextToText.from_pretrained(
        args.snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    )
    model.eval()
    torch.cuda.synchronize()
    cold = time.monotonic() - started
    rendered = {
        row["id"]: tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for row in rows
    }
    length = {key: len(tokenizer.encode(value)) for key, value in rendered.items()}
    if max(length.values()) > 32768:
        raise ValueError("context envelope exceeded: truncation is forbidden")
    identity = {
        "model": MODEL,
        "revision": REVISION,
        "torch": str(torch.__version__),
        "transformers": __import__("transformers").__version__,
        "template_sha256": digest(tokenizer.chat_template),
        "gpu_uuid": GPU,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "seed": 11,
        "do_sample": False,
        "enable_thinking": False,
        "compile": False,
        "deterministic_algorithms": True,
        "tf32": False,
        "cpu_threads": 4,
        "request_source_sha256": args.request_sha256,
        "scorer": "NOT_RUN_cost_only",
        "cache": "no_response_cache",
        "backend_config": "Torch_generate_left_padded_length_scheduled",
    }
    print(
        json.dumps({"event": "loaded", "cold_start_seconds": cold, "identity": identity}),
        flush=True,
    )
    configs = []
    with args.responses.open("x", encoding="utf-8") as output:
        for config_index in range(4):
            batch_size = (
                (1, 4, 8)[config_index]
                if config_index < 3
                else min(configs, key=lambda result: result["elapsed_seconds"])["batch_size"]
            )
            config_id = f"config-{config_index}-batch-{batch_size}"
            torch.cuda.reset_peak_memory_stats()
            tick_config = time.monotonic()
            inputs_total = outputs_total = cap_hits = completed = 0
            profiles = sorted({(r["profile"], r["max_new_tokens"]) for r in rows})
            for profile, cap in profiles:
                group = sorted(
                    [r for r in rows if (r["profile"], r["max_new_tokens"]) == (profile, cap)],
                    key=lambda r: (length[r["id"]], r["id"]),
                )
                for offset in range(0, len(group), batch_size):
                    batch = group[offset : offset + batch_size]
                    if account.usage("unblocking_v1_1")["gpu_hours"] >= 7.9:
                        raise RuntimeError("stage allocation guard reached")
                    inputs = tokenizer(
                        [rendered[r["id"]] for r in batch],
                        padding=True,
                        truncation=False,
                        return_tensors="pt",
                    ).to("cuda:0")
                    if inputs["input_ids"].shape[1] > 32768:
                        raise ValueError("padded context exceeded; no truncation")
                    attempts = []
                    for i, row in enumerate(batch):
                        call_id = config_id + ":" + row["id"]
                        count = int(inputs["attention_mask"][i].sum())
                        account.reserve_call(
                            call_id,
                            "unblocking_v1_1",
                            digest(
                                {
                                    "execution": identity,
                                    "batch_size": batch_size,
                                    "request": row,
                                    "independent_replicate": config_index,
                                }
                            ),
                            input_cap=count,
                            output_cap=cap,
                        )
                        attempts.append((call_id, count))
                    tick = time.monotonic()
                    try:
                        with torch.inference_mode():
                            generated = model.generate(
                                **inputs,
                                do_sample=False,
                                max_new_tokens=cap,
                                pad_token_id=tokenizer.pad_token_id,
                                use_cache=True,
                                disable_compile=True,
                            )
                        torch.cuda.synchronize()
                    except Exception:
                        for call_id, _ in attempts:
                            account.settle_call(
                                call_id, succeeded=False, input_tokens=None, output_tokens=None
                            )
                        raise
                    elapsed = time.monotonic() - tick
                    terminators = model.generation_config.eos_token_id
                    stops = (
                        {terminators} if isinstance(terminators, int) else set(terminators or [])
                    )
                    for index, (row, (call_id, input_tokens)) in enumerate(
                        zip(batch, attempts, strict=True)
                    ):
                        tokens = generated[index, inputs["input_ids"].shape[1] :].tolist()
                        stop = next(
                            (p + 1 for p, token in enumerate(tokens) if token in stops), len(tokens)
                        )
                        tokens = tokens[:stop]
                        account.settle_call(
                            call_id,
                            succeeded=True,
                            input_tokens=input_tokens,
                            output_tokens=len(tokens),
                        )
                        record = {
                            "id": row["id"],
                            "call_id": call_id,
                            "config": config_id,
                            "batch_size": len(batch),
                            "family": row["family"],
                            "profile": profile,
                            "input_tokens": input_tokens,
                            "output_tokens": len(tokens),
                            "cap_hit": len(tokens) == cap,
                            "output": tokenizer.decode(tokens, skip_special_tokens=True),
                            "batch_elapsed_seconds": elapsed,
                            "scoring_executed": False,
                        }
                        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        inputs_total += input_tokens
                        outputs_total += len(tokens)
                        cap_hits += int(len(tokens) == cap)
                        completed += 1
                    output.flush()
                    print(
                        json.dumps(
                            {
                                "event": "batch",
                                "config": config_id,
                                "completed": completed,
                                "seconds": elapsed,
                                "attempts": account.usage("unblocking_v1_1")["calls"],
                            }
                        ),
                        flush=True,
                    )
            elapsed_config = time.monotonic() - tick_config
            result = {
                "config": config_id,
                "batch_size": batch_size,
                "requests": completed,
                "elapsed_seconds": elapsed_config,
                "requests_per_second": completed / elapsed_config,
                "input_tokens": inputs_total,
                "output_tokens": outputs_total,
                "input_tokens_per_second": inputs_total / elapsed_config,
                "output_tokens_per_second": outputs_total / elapsed_config,
                "cap_hits": cap_hits,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "queue_policy": "offline fixed profile then input length; no outcome scheduling",
                "prefix_hits": "NOT_SUPPORTED_measured_tokenizer_cache_only",
                "repeat_of_batch_size": batch_size if config_index == 3 else None,
            }
            configs.append(result)
            print(json.dumps({"event": "config_complete", **result}), flush=True)
    with args.summary.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "identity": identity,
                "cold_start_seconds": cold,
                "configs": configs,
                "online_formal_certificate_validated": False,
                "scientific_effects_measured": False,
            },
            stream,
            indent=2,
        )
    account.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "history", "requests", "ledger", "responses", "summary"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        run_child(args)
        return
    account = ledger(args)
    allocation = "execution-benchmark-v1.1"
    account.begin_gpu(allocation, "unblocking_v1_1", GPU, start=time.time())
    process = None

    def stop(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        # Fixed executable/self argv, shell=False. Ownership is this exact child.
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"]
        )
        try:
            code = process.wait(timeout=7.9 * 3600)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    finally:
        account.end_gpu(allocation, end=time.time())
        print(
            json.dumps(
                {
                    "event": "reservation_released",
                    "total": account.usage(),
                    "stage": account.usage("unblocking_v1_1"),
                }
            ),
            flush=True,
        )
        account.close()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
