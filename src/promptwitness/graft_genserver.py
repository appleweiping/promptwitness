"""A vLLM generation service for search loops that keep the HF model only for scoring.

Search spends most of its time generating (incumbent reasoning, label-free proposals,
self-samples, fresh verification). ``serve`` runs one vLLM engine in its own process and
answers token-id requests over a local authenticated socket; ``graft_runtime`` routes
generation to it after ``graft_runtime.use_remote_generation(address)``. Requests carry
token ids in and out, so rendering, stop tokens and parsing stay those of the caller.
Greedy requests use temperature 0; sampled requests carry one seed per sequence.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from multiprocessing.connection import Client, Listener
from typing import Any

AUTHKEY = b"graft-generation"


def serve(model_path: str, host: str, port: int, gpu_memory_utilization: float, max_model_len: int,
          prefix_caching: bool = False, max_num_seqs: int | None = None) -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    import time

    for attempt in range(5):
        try:
            # Cached prefixes and preempted-then-recomputed sequences change numerics, so
            # repeated reads differ; for exact repeats keep prefix caching off and cap the
            # number of concurrent sequences so that the KV cache never forces preemption.
            # The cached configuration exists only to measure its same-prompt flip rate.
            extra = {"max_num_seqs": max_num_seqs} if max_num_seqs else {}
            llm = LLM(model=model_path, tokenizer=model_path, dtype="bfloat16", seed=0,
                      gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                      enable_prefix_caching=prefix_caching, **extra)
            break
        except (AssertionError, RuntimeError) as error:
            # vLLM's start-up memory profiling fails if a process sharing the GPU frees
            # memory meanwhile (e.g. a search process between phases); retry.
            if attempt == 4:
                raise
            print(f"engine start failed ({error!r}); retrying", flush=True)
            time.sleep(20)
    print(f"READY {host}:{port}", flush=True)
    with Listener((host, port), authkey=AUTHKEY) as listener:
        while True:
            with listener.accept() as conn:
                while True:
                    try:
                        request = conn.recv()
                    except EOFError:
                        break
                    if request.get("op") == "shutdown":
                        conn.send("bye")
                        return
                    seeds = request.get("seeds") or [None] * len(request["prompts"])
                    params = [SamplingParams(temperature=request["temperature"], top_p=request["top_p"],
                                             max_tokens=request["max_new_tokens"], seed=seed,
                                             stop_token_ids=request["stop"], skip_special_tokens=True)
                              for seed in seeds]
                    outputs = llm.generate([TokensPrompt(prompt_token_ids=list(p)) for p in request["prompts"]],
                                           params, use_tqdm=False)
                    conn.send([(list(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in outputs])


class GenClient:
    def __init__(self, address: str) -> None:
        host, port = address.rsplit(":", 1)
        self.conn = Client((host, int(port)), authkey=AUTHKEY)

    def generate(self, prompts: Sequence[Sequence[int]], *, max_new_tokens: int, stop: Sequence[int],
                 temperature: float = 0.0, top_p: float = 1.0,
                 seeds: Sequence[int] | None = None) -> list[tuple[list[int], bool]]:
        """Token ids without trailing stop tokens, and whether generation stopped (not truncated)."""
        self.conn.send({"prompts": [list(p) for p in prompts], "max_new_tokens": max_new_tokens,
                        "stop": list(stop), "temperature": temperature, "top_p": top_p,
                        "seeds": list(seeds) if seeds is not None else None})
        out: list[tuple[list[int], bool]] = []
        stops = set(stop)
        for tokens, reason in self.conn.recv():
            ended = reason == "stop" or bool(tokens and tokens[-1] in stops)
            while tokens and tokens[-1] in stops:
                tokens.pop()
            out.append((tokens, ended))
        return out

    def shutdown(self) -> Any:
        self.conn.send({"op": "shutdown"})
        return self.conn.recv()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.42)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--prefix-caching", action="store_true",
                        help="enable vLLM prefix caching (nondeterministic re-reads; for the null measurement only)")
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="cap on concurrent sequences (no preemption)")
    parser.add_argument("--inproc", action="store_true",
                        help="run the vLLM engine core in this process, so every request of a call is "
                             "enqueued before the first step (batch composition independent of timing)")
    args = parser.parse_args()
    if args.inproc:
        import os

        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    serve(args.model_path, args.host, args.port, args.gpu_memory_utilization, args.max_model_len,
          args.prefix_caching, args.max_num_seqs)


if __name__ == "__main__":
    main()
