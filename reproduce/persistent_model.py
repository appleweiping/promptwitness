"""Owned model process through actual exit; complete physical role accounting.

One cooperating trusted controller owns one session and its original ledger.
No timer/daemon, automatic retry, cache, fake partial vector or scientific
admission. The child receives no direct benchmark-data filesystem grant.
"""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

from promptwitness.incremental.features import ParentTrace
from promptwitness.incremental.identity import ExecutionIdentity
from reproduce.online_resources import CEILING, GPU, STAGE
from reproduce.role_pipeline import RESPONSE_FIELDS, fields
from reproduce.strict_scoring import completed_text
from reproduce.torch_runtime import SETTINGS, TASK_CAPS, task_wire


class PhysicalCallFailure(RuntimeError):
    """Retain any actually received response/cost even if validation fails."""

    def __init__(self, message, response):
        super().__init__(message)
        self.physical_response = response


def model_environment(model_python: Path, scratch: Path) -> dict:
    work = scratch.resolve()
    return {
        "PATH": os.pathsep.join((str(model_python.parent), "/usr/bin", "/bin")),
        "LANG": "C.UTF-8",
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": str(work),
        "TMPDIR": str(work),
        "CUDA_CACHE_PATH": str(work / "cuda-cache"),
        "HF_HOME": str(work / "hf-cache"),
        "CUDA_VISIBLE_DEVICES": GPU,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


class PersistentModel:
    """Reuse the existing backend, with a non-batched request/response stream.

    The supplied ledger MUST be the continued actual physical ledger. Callers
    cannot infer allocated GPU time from per-response kernel/CPU seconds.
    Module entrypoints are fixed, shell=False, credentials are not inherited.
    """

    def __init__(
        self,
        directory: Path,
        snapshot: Path,
        model: str,
        ledger,
        allocation_id: str,
        *,
        model_python: Path,
        data_root: Path,
    ):
        if sys.platform != "linux":
            raise ValueError("registered Linux native inference environment required")
        if not model_python.is_absolute() or not model_python.is_file():
            raise ValueError("registered inference interpreter must be an existing absolute file")
        if not snapshot.is_absolute():
            raise ValueError("registered model snapshot must be an absolute path")
        self.directory, self.snapshot, self.model = directory, snapshot, model
        self.model_python = model_python
        self.data_root = data_root.resolve(strict=True)
        self.ledger, self.allocation_id = ledger, allocation_id
        self.process = self.stderr = None
        self.started = False
        self.failed = False
        self.profile = None
        self.old_sigterm = None
        directory.mkdir(parents=True, exist_ok=False)

    def _record(self, event, **payload):
        with (self.directory / "wire.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": event, **payload}, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _read(self):
        with selectors.DefaultSelector() as ready:
            ready.register(self.process.stdout, selectors.EVENT_READ)
            while not ready.select(timeout=1):
                if self.ledger.usage()["gpu_hours"] >= CEILING.gpu_hours:
                    raise RuntimeError("actual total allocated GPU ceiling reached")
            line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                "original model process ended; inspect private stderr, do not replay"
            )
        return json.loads(line)

    def __enter__(self):
        if (
            self.ledger.connection.execute(
                "SELECT 1 FROM model_calls WHERE status='reserved'"
            ).fetchone()
            or self.ledger.connection.execute(
                "SELECT 1 FROM gpu_allocations WHERE end IS NULL"
            ).fetchone()
        ):
            raise ValueError("unresolved original physical accounting; inspect, do not restart")
        self.ledger.begin_gpu(self.allocation_id, STAGE, GPU, start=time.time())
        self.started = True
        try:
            self.old_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self._stop)
            self.stderr = (self.directory / "stderr").open("x", encoding="utf-8")
            root = Path(__file__).resolve().parents[1]
            scratch = self.directory / "scratch"
            scratch.mkdir()
            environment = model_environment(self.model_python, scratch)
            self.process = subprocess.Popen(
                [
                    str(self.model_python),
                    "-u",
                    "-I",
                    str(root / "reproduce/model_access.py"),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr,
                text=True,
                encoding="utf-8",
                bufsize=1,
                close_fds=True,
                env=environment,
                cwd=scratch,
            )
            launch = {
                "snapshot": str(self.snapshot),
                "model": self.model,
                "data_root": str(self.data_root),
                "scratch": str(scratch.resolve()),
            }
            self.process.stdin.write(json.dumps(launch) + "\n")
            self.process.stdin.flush()
            hello = self._read()
            if (
                set(hello) != {"kind", "profile", "cold_start_seconds", "access"}
                or hello["kind"] != "loaded"
                or not isinstance(hello["access"], dict)
                or hello["access"].get("role") != "inference_message_only"
                or type(hello["access"].get("landlock_abi")) is not int
                or hello["access"]["landlock_abi"] < 1
                or hello["access"].get("benchmark_data_read_grants") != 0
                or hello["access"].get("restriction_before_application_imports") is not True
                or hello["access"].get("worker_pid") != self.process.pid
            ):
                raise ValueError("missing original runtime load receipt")
            self.profile = hello["profile"]
            self._record("loaded", **hello)
            self.access = hello["access"]
            return self
        except BaseException:
            self.close(abort=True)
            raise

    @staticmethod
    def _stop(_signum, _frame):
        raise KeyboardInterrupt("owned model supervisor stopped")

    def close(self, *, abort=False):
        """Abort errors immediately; successful EOF exit may wait gracefully.

        Charge through terminal wait, even when forced cleanup was necessary.
        """
        if not self.started:
            return None
        code = None
        try:
            if self.process is not None:
                if abort and self.process.poll() is None:
                    self.process.terminate()
                if self.process.stdin is not None:
                    with suppress(BrokenPipeError):
                        self.process.stdin.close()
                try:
                    code = self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if abort:
                        self.process.kill()
                        code = self.process.wait()
                    else:
                        self.process.terminate()
                        try:
                            code = self.process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            self.process.kill()
                            code = self.process.wait()
                if self.process.stdout is not None:
                    self.process.stdout.close()
                self._record("original_process_exit", returncode=code)
        finally:
            terminal = self.process is None or self.process.poll() is not None
            if self.started and terminal:
                self.ledger.end_gpu(self.allocation_id, end=time.time())
                self.started = False
                self._record("allocation_closed", actual_usage=self.ledger.usage())
            elif self.started:
                self._record("allocation_unresolved", reason="original child not proven terminal")
            if self.stderr is not None:
                self.stderr.close()
            if self.old_sigterm is not None:
                signal.signal(signal.SIGTERM, self.old_sigterm)
        return code

    def __exit__(self, exc_type, *_exception):
        code = self.close(abort=exc_type is not None or self.failed)
        if exc_type is None and code not in (None, 0):
            raise RuntimeError("original model process exited unsuccessfully; see private stderr")

    def _execute(self, execution, wire):
        if self.failed:
            raise RuntimeError("failed session cannot execute again; inspect original receipt")
        identity = ExecutionIdentity(**execution)
        if self.profile is None or any(execution[k] != v for k, v in self.profile.items()):
            raise ValueError("controller must freeze observed actual runtime identity first")
        key = identity.request_key(wire, unit_id=wire["id"], replicate_id=wire["replicate"])
        self._record("wire_request", key=key, execution=execution, wire=wire)
        response = None
        try:
            self.process.stdin.write(
                json.dumps({"execution": execution, "wire": wire}, allow_nan=False) + "\n"
            )
            self.process.stdin.flush()
            result = self._read()
            self._record("wire_received", key=key, result=result)
            response = result.get("response") if isinstance(result, dict) else None
            if set(result) != {"kind", "response"} or result["kind"] != "response":
                raise ValueError("invalid model response receipt")
            fields(response, RESPONSE_FIELDS)
            completed_text(response["output"], response["status"])
            ParentTrace(
                **{k: response[k] for k in ("input_tokens", "output_tokens", "allocated_seconds")}
            )
            if (
                response["id"] != wire["id"]
                or response["replicate"] != wire["replicate"]
                or response["input_tokens"] is None
                or response["output_tokens"] is None
            ):
                raise ValueError("missing actual random unit or measured token cost")
            self._record("wire_result", key=key, response=response)
            return response
        except BaseException as exc:
            self._record("wire_failed", key=key, error=type(exc).__name__)
            # The evaluator deliberately converts executor errors to INCONCLUSIVE.
            # Stop here, before that supported caller can swallow our exception.
            self.failed = True
            try:
                self.close(abort=True)
            except BaseException as cleanup:
                self._record("abort_cleanup_failed", key=key, error=type(cleanup).__name__)
            if isinstance(exc, Exception):
                raise PhysicalCallFailure(type(exc).__name__, response) from exc
            raise

    def task(self, request):
        """Executor callback for IncrementalPipeline, whose ledger reserves first.

        Supply this same ledger/stage, 32768 input cap and the frozen family cap
        to that existing driver. Do NOT also reserve a second physical call here.
        Reference/full-vector/selection callers must use metered_task instead.
        """
        if self.failed:
            raise RuntimeError("failed session cannot execute again; inspect original receipt")
        if request["model"] != self.model:
            raise ValueError("task model differs from loaded process")
        key = ExecutionIdentity(**request["execution"]).request_key(
            request, unit_id=request["unit"]["id"], replicate_id=request["replicate"]
        )
        reservations = self.ledger.connection.execute(
            "SELECT input_tokens,output_tokens FROM model_calls "
            "WHERE request_digest=? AND stage=? AND status='reserved'",
            (key, STAGE),
        ).fetchall()
        if reservations != [(SETTINGS["context_limit"], TASK_CAPS[request["family"]])]:
            raise ValueError("incremental execution requires its unique prior physical reservation")
        return self._execute(request["execution"], task_wire(request))

    def metered(self, call_id, execution, wire, *, purpose):
        """Account real task/proposer/reflection requests, including failures.

        purpose records reference/search/selection/proposal/reflection provenance;
        it does not grant stage/final data access or authorize caller messages.
        """
        if self.failed:
            raise RuntimeError("failed session cannot execute again; inspect original receipt")
        if purpose not in {"fit", "reference", "search", "selection", "proposer", "reflection"}:
            raise ValueError("unregistered role purpose; final admission is separate")
        expected_role = purpose if purpose in {"proposer", "reflection"} else "task"
        if wire["role"] != expected_role:
            raise ValueError("model role differs from recorded call purpose")
        identity = ExecutionIdentity(**execution)
        self.ledger.reserve_call(
            call_id,
            STAGE,
            identity.request_key(
                {"purpose": purpose, "wire": wire},
                unit_id=wire["id"],
                replicate_id=wire["replicate"],
            ),
            input_cap=SETTINGS["context_limit"],
            output_cap=wire["max_new_tokens"],
        )
        self._record("call_reserved", call_id=call_id, purpose=purpose)
        response = None
        try:
            response = self._execute(execution, wire)
        except BaseException as exc:
            actual = getattr(exc, "physical_response", None)
            known = isinstance(actual, dict) and all(
                type(actual.get(k)) is int and actual[k] >= 0
                for k in ("input_tokens", "output_tokens")
            )
            self.ledger.settle_call(
                call_id,
                succeeded=False,
                input_tokens=actual["input_tokens"] if known else None,
                output_tokens=actual["output_tokens"] if known else None,
            )
            raise
        self.ledger.settle_call(
            call_id,
            succeeded=True,
            input_tokens=response["input_tokens"],
            output_tokens=response["output_tokens"],
        )
        return response

    def metered_task(self, call_id, request, *, purpose):
        if request["model"] != self.model:
            raise ValueError("task model differs from loaded process")
        if request["family"] not in TASK_CAPS:
            raise ValueError("unknown task family")
        return self.metered(call_id, request["execution"], task_wire(request), purpose=purpose)
