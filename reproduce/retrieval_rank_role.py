"""Persistent input-only CIR search ranker behind the existing Landlock role.

The worker owns its CLIP model and full gallery index for the session. Only
query IDs and complete rankings cross the pipe. The caller must supply a
qualified input-only image/weight source and account for whole-process cost.
"""

from __future__ import annotations

import json
import math
import os
import select
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from reproduce.process_access import AccessBoundaryError, landlock_abi, start_role
from reproduce.retrieval_direct_clip_ranker import DirectClipRanker
from reproduce.retrieval_work_ledger import RetrievalWorkLedger

FORMAT = "promptwitness.retrieval-rank-session/v1"
ENTRYPOINT = Path(__file__)


def _input_file(directory: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("nonempty relative input-side file path required")
    path = (directory / relative).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(directory.resolve(strict=True)):
        raise ValueError("ranker files must remain inside the input-only leaf")
    return path


def serve(
    store: Path,
    ranker_factory: Callable[..., Any] = DirectClipRanker,
) -> None:
    """Run after process_access.worker has restricted filesystem access."""
    if (
        os.environ.get("PW_ACCESS_ROLE") != "retrieval_search_ranker"
        or os.environ.get("PW_ACCESS_STAGE") != "search"
    ):
        raise AccessBoundaryError("retrieval ranker needs its search-only role")
    initial = json.loads(sys.stdin.readline())
    if (
        not isinstance(initial, dict)
        or set(initial)
        != {"format", "dataset", "image_paths", "checkpoint", "image_weight", "attempt_prefix"}
        or initial["format"] != FORMAT
    ):
        raise ValueError("invalid retrieval ranker initialization")
    if initial["dataset"] not in {"cirr", "fashioniq"}:
        raise ValueError("only CIRR/FashionIQ ranker inputs supported")
    if not isinstance(initial["image_paths"], dict) or any(
        not isinstance(item, str) or not item for item in initial["image_paths"]
    ):
        raise ValueError("image paths must be keyed by image ID")
    if (
        not isinstance(initial["image_weight"], float)
        or not math.isfinite(initial["image_weight"])
        or not 0 <= initial["image_weight"] <= 1
    ):
        raise ValueError("explicit finite fusion weight required")
    input_dir = store / "search/inputs"
    paths = {
        image_id: _input_file(input_dir, relative)
        for image_id, relative in initial["image_paths"].items()
    }
    checkpoint = _input_file(input_dir, initial["checkpoint"])
    ledger = RetrievalWorkLedger(Path.cwd() / "ranker-work.sqlite")
    try:
        ranker = ranker_factory(
            input_dir=input_dir,
            dataset=initial["dataset"],
            image_paths=paths,
            checkpoint=checkpoint,
            image_weight=initial["image_weight"],
            ledger=ledger,
            attempt_prefix=initial["attempt_prefix"],
        )
        print(
            json.dumps(
                {
                    "format": FORMAT,
                    "op": "ready",
                    "dataset": initial["dataset"],
                    "role": os.environ["PW_ACCESS_ROLE"],
                    "pid": os.getpid(),
                    "landlock_abi": int(os.environ["PW_LANDLOCK_ABI"]),
                    "query_count": len(ranker.queries),
                    "gallery_sizes": {key: len(value) for key, value in ranker.galleries.items()},
                },
                allow_nan=False,
            ),
            flush=True,
        )
        for line in sys.stdin:
            request = json.loads(line)
            if not isinstance(request, dict) or set(request) != {"op", "query_id"}:
                raise ValueError("invalid rank request")
            if request["op"] != "rank" or not isinstance(request["query_id"], str):
                raise ValueError("only query-ID rank requests are supported")
            ranking = ranker.rank_one(request["query_id"])
            print(
                json.dumps(
                    {
                        "format": FORMAT,
                        "op": "ranking",
                        "query_id": request["query_id"],
                        "ranking": ranking,
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
    finally:
        ledger.close()


class RestrictedRankerSession:
    """One live search ranker; its private operation ledger is separate from the bridge."""

    def __init__(
        self,
        *,
        store: Path,
        scratch: Path,
        dataset: str,
        image_paths: Mapping[str, str],
        checkpoint: str,
        image_weight: float,
        attempt_prefix: str,
        timeout: float = 900.0,
    ) -> None:
        if not scratch.is_dir() or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("existing ranker scratch and positive finite timeout required")
        landlock_abi()
        self.timeout = timeout
        self.process: subprocess.Popen[str] | None = None
        self._closed = False
        self._read_buffer = bytearray()
        self.stderr_path = scratch / "ranker-stderr.log"
        with self.stderr_path.open("xb") as errors:
            self.process = start_role(
                "retrieval_search_ranker",
                "search",
                store,
                scratch,
                ENTRYPOINT,
                [str(store.resolve(strict=True))],
                stderr=errors,
            )
        try:
            if self.process is None or self.process.stdin is None or self.process.stdout is None:
                raise AccessBoundaryError("ranker did not establish protocol pipes")
            set_blocking = getattr(os, "set_blocking", None)
            if set_blocking is None:
                raise AccessBoundaryError("nonblocking role pipes are unavailable")
            set_blocking(self.process.stdin.fileno(), False)
            set_blocking(self.process.stdout.fileno(), False)
            deadline = time.monotonic() + self.timeout
            self._send(
                {
                    "format": FORMAT,
                    "dataset": dataset,
                    "image_paths": dict(image_paths),
                    "checkpoint": checkpoint,
                    "image_weight": image_weight,
                    "attempt_prefix": attempt_prefix,
                },
                deadline,
            )
            ready = self._receive(deadline)
            if (
                not isinstance(ready, dict)
                or ready.get("format") != FORMAT
                or ready.get("op") != "ready"
                or ready.get("dataset") != dataset
                or ready.get("role") != "retrieval_search_ranker"
                or type(ready.get("landlock_abi")) is not int
                or ready["landlock_abi"] < 1
                or type(ready.get("pid")) is not int
                or ready["pid"] <= 0
                or ready["pid"] == os.getpid()
            ):
                raise AccessBoundaryError("ranker did not establish its restricted role")
            self.ready = ready
        except BaseException:
            self.close(abort=True)
            raise

    def _send(self, message: dict[str, Any], deadline: float) -> None:
        process = self.process
        if self._closed or process is None or process.stdin is None or process.poll() is not None:
            raise ValueError("retrieval ranker process is not live")
        payload = (json.dumps(message, allow_nan=False) + "\n").encode("utf-8")
        descriptor = process.stdin.fileno()
        offset = 0
        while offset < len(payload):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("restricted retrieval ranker input timed out")
            _, writable, _ = select.select([], [descriptor], [], remaining)
            if not writable:
                raise TimeoutError("restricted retrieval ranker input timed out")
            try:
                offset += os.write(descriptor, payload[offset:])
            except BlockingIOError:
                continue

    def _receive(self, deadline: float) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdout is None:
            raise ValueError("retrieval ranker output pipe is missing")
        descriptor = process.stdout.fileno()
        while b"\n" not in self._read_buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("restricted retrieval ranker did not answer")
            readable, _, _ = select.select([descriptor], [], [], remaining)
            if not readable:
                raise TimeoutError("restricted retrieval ranker did not answer")
            try:
                chunk = os.read(descriptor, 65536)
            except BlockingIOError:
                continue
            if not chunk:
                raise ValueError(f"restricted retrieval ranker exited; inspect {self.stderr_path}")
            self._read_buffer.extend(chunk)
        line, _, rest = self._read_buffer.partition(b"\n")
        self._read_buffer = bytearray(rest)
        result = json.loads(line)
        if not isinstance(result, dict):
            raise ValueError("restricted ranker returned a non-object frame")
        return result

    def rank_one(self, query_id: str) -> tuple[str, ...]:
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("nonempty query ID required")
        try:
            deadline = time.monotonic() + self.timeout
            self._send({"op": "rank", "query_id": query_id}, deadline)
            result = self._receive(deadline)
            ranking = result.get("ranking")
            if (
                result.get("format") != FORMAT
                or result.get("op") != "ranking"
                or result.get("query_id") != query_id
                or not isinstance(ranking, list)
                or not ranking
                or any(not isinstance(item, str) or not item for item in ranking)
                or len(set(ranking)) != len(ranking)
            ):
                raise ValueError("restricted ranker returned invalid full-rank frame")
            return tuple(ranking)
        except BaseException:
            self.close(abort=True)
            raise

    def close(self, *, abort: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        process = self.process
        if process is None:
            return
        if process.stdin is not None:
            with suppress(BrokenPipeError):
                process.stdin.close()
        if abort and process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdout is not None:
            process.stdout.close()
        if not abort and process.returncode != 0:
            raise ValueError(f"restricted ranker exited unsuccessfully; inspect {self.stderr_path}")

    def __enter__(self) -> RestrictedRankerSession:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close(abort=bool(_exc[0]))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: retrieval_rank_role.py STORE")
    serve(Path(sys.argv[1]))
