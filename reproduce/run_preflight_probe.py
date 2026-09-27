"""Measure a bounded G0 cost envelope; immutable v2 is retained in archive/."""

from __future__ import annotations

import argparse
import atexit
import faulthandler
import hashlib
import json
import sqlite3
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--probe-id")
    parser.add_argument("--attempt-ceiling", type=int, default=200)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--diagnostic-stack-interval", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--proposers-batch-one", action="store_true")
    parser.add_argument("--stress-batch-size", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 8:
        raise ValueError("batch size must be in [1,8]")
    if args.attempt_ceiling not in (200, 400, 600, 700):
        raise ValueError("unregistered G0 attempt ceiling")
    if args.attempt_ceiling == 700 and args.probe_id != "cost-envelope-v1":
        raise ValueError("the 700 ceiling is reserved for the frozen 300-request cost envelope")
    if args.max_input_tokens not in (16384, 32768):
        raise ValueError("unsupported context envelope")
    if not 1 <= args.stress_batch_size <= args.batch_size:
        raise ValueError("invalid stress batch size")
    if args.cpu_threads is not None and not 1 <= args.cpu_threads <= 8:
        raise ValueError("explicit CPU threads must be in [1,8]")
    if args.probe_id is None and args.attempt_ceiling != 200:
        raise ValueError("additional attempts require a named, separately frozen repair phase")
    if args.diagnostic_stack_interval not in (0, 60):
        raise ValueError("diagnostic stack interval must be 0 or 60 seconds")
    ledger_model = args.model if args.probe_id is None else f"{args.model}@{args.probe_id}"
    rows = [json.loads(line) for line in args.requests.read_text().splitlines()]
    if len(rows) != 100 or len({row["id"] for row in rows}) != 100:
        raise ValueError("probe requires exactly 100 distinct training-side requests per model")
    if any(row["max_new_tokens"] not in (64, 256, 512, 1024, 2048, 4096) for row in rows):
        raise ValueError("unsupported output cap; no silent skipping of requests")
    if any(row["role"] not in ("task", "proposer") for row in rows):
        raise ValueError("unsupported probe role")
    requests_sha = hashlib.sha256(args.requests.read_bytes()).hexdigest()
    ledger = sqlite3.connect(args.ledger)
    ledger.execute(
        "CREATE TABLE IF NOT EXISTS runs (model TEXT PRIMARY KEY, revision TEXT, "
        "requests_sha TEXT, started REAL, elapsed REAL, cold_start REAL)"
    )
    ledger.execute(
        "CREATE TABLE IF NOT EXISTS attempts (model TEXT, id TEXT, role TEXT, "
        "status TEXT, PRIMARY KEY(model,id))"
    )
    ledger.commit()
    if ledger.execute("SELECT 1 FROM runs WHERE model=?", (ledger_model,)).fetchone():
        raise ValueError("this model probe already started; no silent replay or duplicate charging")
    started = time.monotonic()
    ledger.execute(
        "INSERT INTO runs VALUES (?,?,?,?,NULL,NULL)",
        (ledger_model, args.revision, requests_sha, time.time()),
    )
    ledger.commit()

    def record_elapsed() -> None:
        elapsed = time.monotonic() - started
        ledger.execute("UPDATE runs SET elapsed=? WHERE model=?", (elapsed, ledger_model))
        ledger.commit()

    atexit.register(record_elapsed)
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(args.cpu_threads)
    if args.diagnostic_stack_interval:
        faulthandler.dump_traceback_later(args.diagnostic_stack_interval, repeat=True)
        atexit.register(faulthandler.cancel_dump_traceback_later)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        args.snapshot, local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_type = (
        AutoModelForImageTextToText if args.model.startswith("Qwen/") else AutoModelForCausalLM
    )
    model = model_type.from_pretrained(
        args.snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    )
    model.eval()
    torch.cuda.synchronize()
    cold_start = time.monotonic() - started
    ledger.execute("UPDATE runs SET cold_start=? WHERE model=?", (cold_start, ledger_model))
    ledger.commit()
    with args.output.open("x", encoding="utf-8") as output:
        profiles = sorted(
            {(row["role"], row["max_new_tokens"], row.get("profile", "base")) for row in rows}
        )
        for role, limit, profile in profiles:
            group = [
                row
                for row in rows
                if (row["role"], row["max_new_tokens"], row.get("profile", "base"))
                == (role, limit, profile)
            ]
            batch_size = 1 if role == "proposer" and args.proposers_batch_one else args.batch_size
            if profile == "eight_demo_stress_proxy":
                batch_size = min(batch_size, args.stress_batch_size)
            for offset in range(0, len(group), batch_size):
                batch = group[offset : offset + batch_size]
                if time.monotonic() - started > 3600:
                    raise RuntimeError("preflight one-hour allocated GPU ceiling reached")
                prompts = [
                    tokenizer.apply_chat_template(
                        row["messages"],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for row in batch
                ]
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(
                    "cuda:0"
                )
                if inputs["input_ids"].shape[1] > args.max_input_tokens:
                    raise ValueError(
                        "input exceeds the frozen preflight context bound; truncation forbidden"
                    )
                ledger.execute("BEGIN IMMEDIATE")
                if (
                    ledger.execute("SELECT count(*) FROM attempts").fetchone()[0] + len(batch)
                    > args.attempt_ceiling
                ):
                    ledger.rollback()
                    raise RuntimeError("preflight aggregate attempt hard limit reached")
                for row in batch:
                    ledger.execute(
                        "INSERT INTO attempts VALUES (?,?,?,?)",
                        (ledger_model, row["id"], row["role"], "reserved"),
                    )
                ledger.commit()
                tick = time.monotonic()
                try:
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            do_sample=False,
                            max_new_tokens=limit,
                            pad_token_id=tokenizer.pad_token_id,
                            use_cache=True,
                            disable_compile=args.disable_compile,
                        )
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - tick
                    for index, row in enumerate(batch):
                        tokens = generated[index, inputs["input_ids"].shape[1] :].tolist()
                        terminators = model.generation_config.eos_token_id
                        stops = (
                            {terminators}
                            if isinstance(terminators, int)
                            else set(terminators or [])
                        )
                        first_stop = next(
                            (
                                position + 1
                                for position, token in enumerate(tokens)
                                if token in stops
                            ),
                            len(tokens),
                        )
                        tokens = tokens[:first_stop]
                        record = {
                            "id": row["id"],
                            "model": args.model,
                            "ledger_model": ledger_model,
                            "probe_id": args.probe_id,
                            "revision": args.revision,
                            "requests_sha256": requests_sha,
                            "family": row["family"],
                            "role": row["role"],
                            "profile": row.get("profile", "base"),
                            "status": "completed",
                            "input_tokens": int(inputs["attention_mask"][index].sum()),
                            "output_tokens": len(tokens),
                            "max_new_tokens": limit,
                            "batch_size": len(batch),
                            "batch_elapsed_seconds": elapsed,
                            "allocated_seconds_share": elapsed / len(batch),
                            "output": tokenizer.decode(tokens, skip_special_tokens=True),
                            "decoding": {"do_sample": False, "enable_thinking": False},
                            "scoring_executed": False,
                            "torch_version": torch.__version__,
                            "cpu_threads": torch.get_num_threads(),
                            "disable_compile": args.disable_compile,
                            "max_input_tokens": args.max_input_tokens,
                            "proposers_batch_one": args.proposers_batch_one,
                            "script_sha256": hashlib.sha256(
                                Path(__file__).read_bytes()
                            ).hexdigest(),
                        }
                        output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                        ledger.execute(
                            "UPDATE attempts SET status='completed' WHERE model=? AND id=?",
                            (ledger_model, row["id"]),
                        )
                    output.flush()
                    ledger.commit()
                    print(
                        json.dumps(
                            {
                                "model": args.model,
                                "completed_batch": len(batch),
                                "batch_elapsed_seconds": elapsed,
                                "charged_attempts": ledger.execute(
                                    "SELECT count(*) FROM attempts"
                                ).fetchone()[0],
                            }
                        ),
                        flush=True,
                    )
                except Exception as error:
                    elapsed = time.monotonic() - tick
                    for row in batch:
                        output.write(
                            json.dumps(
                                {
                                    "id": row["id"],
                                    "model": args.model,
                                    "ledger_model": ledger_model,
                                    "probe_id": args.probe_id,
                                    "family": row["family"],
                                    "role": row["role"],
                                    "status": "failed",
                                    "error_type": type(error).__name__,
                                    "reserved_output_tokens": limit,
                                    "allocated_seconds_share": elapsed / len(batch),
                                }
                            )
                            + "\n"
                        )
                        ledger.execute(
                            "UPDATE attempts SET status='failed' WHERE model=? AND id=?",
                            (ledger_model, row["id"]),
                        )
                    output.flush()
                    ledger.commit()
                    raise
    elapsed = time.monotonic() - started
    ledger.execute("UPDATE runs SET elapsed=? WHERE model=?", (elapsed, ledger_model))
    ledger.commit()
    print(
        json.dumps(
            {
                "model": args.model,
                "ledger_model": ledger_model,
                "probe_id": args.probe_id,
                "elapsed_seconds": elapsed,
                "cold_start_seconds": cold_start,
                "allocated_gpu_hours": elapsed / 3600,
                "model_probe_completed": True,
                "peak_allocated_memory_mib": torch.cuda.max_memory_allocated() / (1024**2),
                "peak_reserved_memory_mib": torch.cuda.max_memory_reserved() / (1024**2),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
