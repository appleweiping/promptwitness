"""Persistent fixed single-unit Torch execution, with literal wire receipts.

Reuses the measured backend settings, not its cost-only batch scheduler or old
stage quotas. Native task inputs are complete and never truncated. Matching
outputs or do_sample=False do NOT establish ONLINE_PINNED. Runtime identity
must be frozen into the controller before its reference or candidate calls.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from promptwitness.incremental.identity import ExecutionIdentity
from promptwitness.incremental.sampling import digest
from promptwitness.parser import parse_prompt
from promptwitness.variables import inspect_variables, render_template
from reproduce.online_resources import GPU
from reproduce.prepare_bfcl_fit import MODEL_REVISIONS
from reproduce.prepare_native_context import task_messages
from reproduce.prepare_preflight_requests import SEEDS

TASK_CAPS = {"bfcl": 1024, "hotpotqa": 64, "instruction_following": 4096}
ROLE_CAPS = {"proposer": 2048, "reflection": 2048}
DECODE = {
    "do_sample": False,
    "enable_thinking": False,
    "seed": 11,
    "use_cache": True,
    "disable_compile": True,
}
SETTINGS = {
    "batch_size": 1,
    "dtype": "bfloat16",
    "attention": "sdpa",
    "padding_side": "left",
    "context_limit": 32768,
    "cpu_threads": 4,
    "deterministic_algorithms": True,
    "tf32": False,
    "gpu_uuid": GPU,
    "task_caps": TASK_CAPS,
    "role_caps": ROLE_CAPS,
}


def task_wire(request: dict) -> dict:
    """Render instruction/demonstration prefix or an explicit task_input template.

    ToolSpec/multimodal/native tool APIs are not this fixed text task interface;
    reject them, rather than discard structure and pretend it was executed.
    All original BFCL questions/functions, Hotpot contexts and IFTrain messages
    are retained. Prompt templates bind only the actual input, never a label.
    """
    if set(request) != {
        "candidate",
        "execution",
        "configuration",
        "family",
        "model",
        "unit",
        "replicate",
    }:
        raise ValueError("literal input-only controller request required")
    document = parse_prompt(request["candidate"])
    if (
        document.tools
        or not document.messages
        or any(
            m.content_parts or m.name is not None or m.role not in {"system", "user", "assistant"}
            for m in document.messages
        )
    ):
        raise ValueError("unsupported prompt interface; do not flatten tools or multimodal blocks")
    family, unit = request["family"], request["unit"]
    expected = {
        "bfcl": {"id", "category", "question", "function"},
        "hotpotqa": {"id", "family", "messages"},
        "instruction_following": {"id", "family", "messages"},
    }
    if family not in expected or set(unit) != expected[family]:
        raise ValueError("unknown or annotation-bearing task input fields")
    if family == "bfcl":
        original = task_messages(family, unit)
    else:
        # Actual prepare_text_fit.convert / prepare_training_store schema:
        # the first message is the original seed, not an immutable task input.
        # Replace ONLY this verified seed. Preserve every original constraint
        # and complete context message after it, including any input system text.
        if (
            unit["family"] != family
            or not isinstance(unit["messages"], list)
            or len(unit["messages"]) < 2
            or unit["messages"][0] != {"role": "system", "content": SEEDS[family]}
        ):
            raise ValueError("prepared original seed/family interface changed")
        original = json.loads(json.dumps(unit["messages"][1:]))
    variables = set().union(*(inspect_variables(m.content).names for m in document.messages))
    if variables - {"task_input"}:
        raise ValueError("unbound prompt variable; never invent input or drop fields")
    if variables and (len(original) != 1 or original[0]["role"] != "user"):
        raise ValueError("task_input template requires one complete original user message")
    values = {"task_input": original[0]["content"]} if variables else {}
    messages = [
        {"role": m.role, "content": render_template(m.content, values)} for m in document.messages
    ]
    if not variables:
        messages.extend(original)
    if messages[-1]["role"] != "user":
        raise ValueError("frozen task request must end in a user message")
    return {
        "id": unit["id"],
        "replicate": request["replicate"],
        "role": "task",
        "family": family,
        "messages": messages,
        "max_new_tokens": TASK_CAPS[family],
    }


class TorchRuntime:
    """One model load, single-unit generation, no response cache or outcome batching.

    This backend consumes only messages. The persistent launcher restricts its
    process through model_access before importing this application module.
    Direct invocation of this module does not establish that access boundary.
    Whole GPU allocation is owned by the parent through this process's exit.
    """

    def __init__(self, snapshot: Path, model: str):
        if (
            model not in MODEL_REVISIONS
            or snapshot.name != MODEL_REVISIONS[model]
            or os.environ.get("CUDA_VISIBLE_DEVICES") != GPU
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("fixed model snapshot/device/workspace required")
        tick = time.monotonic()
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        if (
            str(torch.__version__) != "2.7.1+cu118"
            or __import__("transformers").__version__ != "5.3.0"
        ):
            raise ValueError("frozen measured Torch/Transformers environment required")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        torch.manual_seed(DECODE["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        cls = AutoModelForImageTextToText if model.startswith("Qwen/") else AutoModelForCausalLM
        loaded = cls.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
        )
        loaded.eval()
        torch.cuda.synchronize()
        self.torch, self.tokenizer, self.model, self.name = torch, tokenizer, loaded, model
        self.profile = {
            "model_revision": MODEL_REVISIONS[model],
            "tokenizer_revision": MODEL_REVISIONS[model],
            "backend_version": "Torch2.7.1+cu118/Transformers5.3.0",
            "backend_config_digest": digest(SETTINGS),
            "template_digest": digest(tokenizer.chat_template),
            "decoding_digest": digest(
                {
                    "overrides": DECODE,
                    "generation_config": loaded.generation_config.to_dict(),
                    "pad_token_id": tokenizer.pad_token_id,
                }
            ),
        }
        self.cold_start_seconds = time.monotonic() - tick

    def generate(self, execution: dict, wire: dict) -> dict:
        identity = ExecutionIdentity(**execution)
        if any(asdict(identity)[key] != value for key, value in self.profile.items()):
            raise ValueError("observed runtime differs from literal frozen execution")
        if (
            digest(self.tokenizer.chat_template) != self.profile["template_digest"]
            or digest(
                {
                    "overrides": DECODE,
                    "generation_config": self.model.generation_config.to_dict(),
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
            )
            != self.profile["decoding_digest"]
        ):
            raise ValueError("loaded template/decoding changed since runtime freeze")
        if set(wire) != {"id", "replicate", "role", "family", "messages", "max_new_tokens"}:
            raise ValueError("complete literal wire fields required")
        if wire["family"] not in TASK_CAPS or wire["role"] not in {"task", *ROLE_CAPS}:
            raise ValueError("unknown model role/family")
        cap = TASK_CAPS[wire["family"]] if wire["role"] == "task" else ROLE_CAPS[wire["role"]]
        if wire["max_new_tokens"] != cap or not wire["id"] or not wire["replicate"]:
            raise ValueError("frozen output cap and actual random unit required")
        if (
            not isinstance(wire["messages"], list)
            or not wire["messages"]
            or any(
                set(m) != {"role", "content"}
                or not isinstance(m["content"], str)
                or m["role"] not in {"system", "user", "assistant"}
                for m in wire["messages"]
            )
        ):
            raise ValueError("original text-only complete messages required")
        prompt = self.tokenizer.apply_chat_template(
            wire["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer([prompt], padding=True, truncation=False, return_tensors="pt").to(
            "cuda:0"
        )
        count = int(inputs["attention_mask"][0].sum())
        if inputs["input_ids"].shape[1] > SETTINGS["context_limit"]:
            raise ValueError("context envelope exceeded; truncation forbidden")
        tick = time.monotonic()
        # Batch shape/order is fixed. A recorded replicate is not a claim of
        # stochastic independence or universal numerical determinism.
        self.torch.manual_seed(DECODE["seed"])
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=cap,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                disable_compile=True,
            )
        self.torch.cuda.synchronize()
        tokens = generated[0, inputs["input_ids"].shape[1] :].tolist()
        eos = self.model.generation_config.eos_token_id
        stops = {eos} if isinstance(eos, int) else set(eos or [])
        end = next((i + 1 for i, token in enumerate(tokens) if token in stops), len(tokens))
        tokens = tokens[:end]
        return {
            "id": wire["id"],
            "replicate": wire["replicate"],
            "status": "completed",
            "input_tokens": count,
            "output_tokens": len(tokens),
            "output": self.tokenizer.decode(tokens, skip_special_tokens=True),
            # The union of cold/idle/exit intervals is separate in ResourceLedger.
            "allocated_seconds": time.monotonic() - tick,
        }


def worker(snapshot: Path, model: str, *, access=None):
    runtime = TorchRuntime(snapshot, model)
    print(
        json.dumps(
            {
                "kind": "loaded",
                "profile": runtime.profile,
                "cold_start_seconds": runtime.cold_start_seconds,
                "access": access,
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        message = json.loads(line)
        if set(message) != {"execution", "wire"}:
            raise ValueError("unknown persistent request fields")
        result = runtime.generate(message["execution"], message["wire"])
        print(json.dumps({"kind": "response", "response": result}, allow_nan=False), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m reproduce.torch_runtime SNAPSHOT MODEL")
    worker(Path(sys.argv[1]), sys.argv[2])
