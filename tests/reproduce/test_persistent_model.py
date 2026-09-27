"""Authored model-process stand-ins; no GPU or real physical history."""

import io
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from promptwitness.incremental.budget import ResourceLedger
from reproduce.online_resources import CEILING, GPU, STAGE
from reproduce.persistent_model import PersistentModel, PhysicalCallFailure
from reproduce.prepare_bfcl_fit import MODEL_REVISIONS


def identity():
    return {
        "model_revision": next(iter(MODEL_REVISIONS.values())),
        "tokenizer_revision": "AUTHORED",
        "backend_version": "AUTHORED_NO_INFERENCE",
        **{
            k: "a" * 64
            for k in (
                "backend_config_digest",
                "template_digest",
                "decoding_digest",
                "scorer_digest",
                "data_digest",
                "tool_environment_digest",
            )
        },
    }


def wire():
    return {
        "id": "u",
        "replicate": "r",
        "role": "task",
        "family": "hotpotqa",
        "messages": [{"role": "user", "content": "AUTHORED"}],
        "max_new_tokens": 64,
    }


def response():
    return {
        "id": "u",
        "replicate": "r",
        "status": "completed",
        "input_tokens": 12,
        "output_tokens": 3,
        "allocated_seconds": 0.25,
        "output": "AUTHORED",
    }


def client(tmp_path):
    ledger = ResourceLedger(
        tmp_path / "AUTHORED.sqlite",
        historical_usage={"calls": 0, "input_tokens": 0, "output_tokens": 0, "gpu_hours": 0},
        historical_digest="a" * 64,
        global_limit=CEILING,
        stage_limits={STAGE: CEILING},
        gpu_uuid=GPU,
    )
    value = PersistentModel.__new__(PersistentModel)
    value.directory = tmp_path / "AUTHORED_process"
    value.directory.mkdir()
    value.ledger, value.allocation_id = ledger, "AUTHORED"
    value.profile = {"backend_version": identity()["backend_version"]}
    value.model = next(iter(MODEL_REVISIONS))
    value.snapshot = tmp_path / next(iter(MODEL_REVISIONS.values()))
    value.process = value.stderr = None
    value.started = False
    value.failed = False
    value.old_sigterm = None
    return value


@pytest.mark.parametrize(
    "purpose", ["fit", "reference", "search", "selection", "proposer", "reflection"]
)
def test_all_model_roles_charged_once(tmp_path, monkeypatch, purpose):
    c = client(tmp_path)
    monkeypatch.setattr(c, "_execute", lambda *_: response())
    w = wire()
    if purpose in {"proposer", "reflection"}:
        w.update(role=purpose, max_new_tokens=2048)
    assert c.metered("one", identity(), w, purpose=purpose) == response()
    assert c.ledger.usage()["calls"] == 1
    assert c.ledger.usage()["input_tokens"] == 12
    assert c.ledger.usage()["output_tokens"] == 3
    with pytest.raises(sqlite3.IntegrityError):
        c.metered("one", identity(), w, purpose=purpose)
    assert c.ledger.usage()["calls"] == 1
    c.ledger.close()


@pytest.mark.parametrize("known", [True, False])
def test_failed_response_costs_not_forgotten(tmp_path, monkeypatch, known):
    c = client(tmp_path)

    def fail(*_):
        raise PhysicalCallFailure("authored wrong response identity", response() if known else None)

    monkeypatch.setattr(c, "_execute", fail)
    with pytest.raises(PhysicalCallFailure):
        c.metered("failed", identity(), wire(), purpose="reference")
    usage = c.ledger.usage()
    assert usage["calls"] == 1
    assert usage["input_tokens"] == (12 if known else 32768)
    assert usage["output_tokens"] == (3 if known else 64)
    assert usage["calls_with_reserved_or_unknown_token_cost"] == int(not known)
    c.ledger.close()


def test_no_final_or_role_mismatch_grant(tmp_path):
    c = client(tmp_path)
    for purpose in ("final", "proposer"):
        with pytest.raises(ValueError):
            c.metered("forbidden", identity(), wire(), purpose=purpose)
    assert c.ledger.usage()["calls"] == 0
    c.ledger.close()


def test_actual_received_tokens_survive_wrong_unit(tmp_path, monkeypatch):
    c = client(tmp_path)
    c.process = type("Fake", (), {"stdin": io.StringIO()})()
    r = response()
    r["id"] = "wrong"
    monkeypatch.setattr(c, "_read", lambda: {"kind": "response", "response": r})
    with pytest.raises(PhysicalCallFailure) as failure:
        c._execute(identity(), wire())
    assert failure.value.physical_response == r
    events = [json.loads(line) for line in (c.directory / "wire.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["wire_request", "wire_received", "wire_failed"]
    assert events[1]["result"]["response"] == r
    c.ledger.close()


class FakeChild:
    def __init__(self, timeouts=0):
        self.stdin, self.stdout = io.StringIO(), io.StringIO()
        self.timeouts, self.returncode, self.operations = timeouts, None, []
        self.pid = 1123

    def wait(self, timeout=None):
        self.operations.append("wait")
        if self.timeouts:
            self.timeouts -= 1
            raise subprocess.TimeoutExpired("AUTHORED", timeout)
        self.returncode = 0
        return 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.operations.append("terminate")

    def kill(self):
        self.operations.append("kill")


@pytest.mark.parametrize(
    "timeouts,operations",
    [
        (0, ["wait"]),
        (1, ["wait", "terminate", "wait"]),
        (2, ["wait", "terminate", "wait", "kill", "wait"]),
    ],
)
def test_whole_owned_allocation_closed_only_after_wait(tmp_path, monkeypatch, timeouts, operations):
    c = client(tmp_path)
    c.ledger.begin_gpu("AUTHORED", STAGE, GPU, start=100)
    c.started = True
    c.process = FakeChild(timeouts)
    monkeypatch.setattr("reproduce.persistent_model.time.time", lambda: 3700)
    assert c.close() == 0
    assert c.process.operations == operations
    assert c.ledger.usage()["gpu_hours"] == 1
    assert c.close() is None
    c.ledger.close()


def test_unproven_child_exit_keeps_interval_open(tmp_path, monkeypatch):
    c = client(tmp_path)
    c.ledger.begin_gpu("AUTHORED", STAGE, GPU, start=100)
    c.started = True
    c.process = FakeChild()

    def wait(**_):
        raise OSError("AUTHORED observation failure")

    c.process.wait = wait
    monkeypatch.setattr("reproduce.persistent_model.time.time", lambda: 3700)
    with pytest.raises(OSError):
        c.close()
    assert c.ledger.connection.execute("SELECT end FROM gpu_allocations").fetchone()[0] is None
    assert c.ledger.usage()["gpu_hours"] == 1
    c.ledger.close()


@pytest.mark.parametrize(
    "timeouts,operations",
    [(0, ["terminate", "wait"]), (1, ["terminate", "wait", "kill", "wait"])],
)
def test_exception_exit_terminates_before_wait(tmp_path, monkeypatch, timeouts, operations):
    c = client(tmp_path)
    c.ledger.begin_gpu("AUTHORED", STAGE, GPU, start=100)
    c.started = True
    c.process = FakeChild(timeouts)
    monkeypatch.setattr("reproduce.persistent_model.time.time", lambda: 3700)
    c.__exit__(RuntimeError, RuntimeError("AUTHORED ceiling interruption"), None)
    assert c.process.operations == operations
    assert c.ledger.usage()["gpu_hours"] == 1
    assert not c.started
    c.ledger.close()


def test_exception_wait_failure_keeps_original_allocation_open(tmp_path, monkeypatch):
    c = client(tmp_path)
    c.ledger.begin_gpu("AUTHORED", STAGE, GPU, start=100)
    c.started = True
    c.process = FakeChild()

    def wait(**_):
        raise OSError("AUTHORED observation failure")

    c.process.wait = wait
    monkeypatch.setattr("reproduce.persistent_model.time.time", lambda: 3700)
    with pytest.raises(OSError):
        c.__exit__(RuntimeError, RuntimeError("AUTHORED interruption"), None)
    assert c.process.operations == ["terminate"]
    assert c.ledger.connection.execute("SELECT end FROM gpu_allocations").fetchone()[0] is None
    c.ledger.close()


@pytest.mark.parametrize("load_failure", [False, True])
def test_explicit_model_interpreter_and_load_abort(tmp_path, monkeypatch, load_failure):
    account = client(tmp_path).ledger
    monkeypatch.setattr("reproduce.persistent_model.sys.platform", "linux")
    monkeypatch.setenv("AUTHORED_CREDENTIAL", "must not be inherited")
    interpreter = Path(sys.executable)
    c = PersistentModel(
        tmp_path / "AUTHORED_NEW_SESSION",
        tmp_path / "AUTHORED_snapshot",
        next(iter(MODEL_REVISIONS)),
        account,
        "AUTHORED_NEW_ALLOCATION",
        model_python=interpreter,
        data_root=tmp_path,
    )
    child = FakeChild()
    started = []

    def popen(command, **kwargs):
        started.append((command, kwargs))
        return child

    def loaded():
        if load_failure:
            raise RuntimeError("AUTHORED failed load")
        return {
            "kind": "loaded",
            "profile": {"authored": True},
            "cold_start_seconds": 1.0,
            "access": {
                "role": "inference_message_only",
                "landlock_abi": 1,
                "benchmark_data_read_grants": 0,
                "restriction_before_application_imports": True,
                "worker_pid": child.pid,
            },
        }

    monkeypatch.setattr("reproduce.persistent_model.subprocess.Popen", popen)
    monkeypatch.setattr(c, "_read", loaded)
    if load_failure:
        with pytest.raises(RuntimeError, match="failed load"):
            c.__enter__()
        assert child.operations == ["terminate", "wait"]
    else:
        with c:
            assert c.profile == {"authored": True}
        assert child.operations == ["wait"]
    command, options = started[0]
    assert command[:3] == [str(interpreter), "-u", "-I"]
    assert Path(command[3]).name == "model_access.py"
    assert "AUTHORED_CREDENTIAL" not in options["env"]
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == GPU
    assert options["env"]["HF_HUB_OFFLINE"] == "1"
    assert "PYTHONPATH" not in options["env"]
    assert options["env"]["HOME"] == str((c.directory / "scratch").resolve())
    assert options["close_fds"] is True
    assert account.connection.execute("SELECT end FROM gpu_allocations").fetchone()[0] is not None
    account.close()


def test_unregistered_interpreter_rejected_before_allocation(tmp_path, monkeypatch):
    account = client(tmp_path).ledger
    monkeypatch.setattr("reproduce.persistent_model.sys.platform", "linux")
    with pytest.raises(ValueError, match="interpreter"):
        PersistentModel(
            tmp_path / "AUTHORED_rejected",
            tmp_path,
            next(iter(MODEL_REVISIONS)),
            account,
            "AUTHORED",
            model_python=tmp_path / "missing_python",
            data_root=tmp_path,
        )
    assert account.connection.execute("SELECT COUNT(*) FROM gpu_allocations").fetchone()[0] == 0
    assert not (tmp_path / "AUTHORED_rejected").exists()
    account.close()


def test_relative_snapshot_rejected_before_child_or_allocation(tmp_path, monkeypatch):
    account = client(tmp_path).ledger
    monkeypatch.setattr("reproduce.persistent_model.sys.platform", "linux")
    with pytest.raises(ValueError, match=r"snapshot.*absolute"):
        PersistentModel(
            tmp_path / "AUTHORED_no_launch",
            Path("relative_snapshot"),
            next(iter(MODEL_REVISIONS)),
            account,
            "AUTHORED",
            model_python=Path(sys.executable),
            data_root=tmp_path,
        )
    assert account.connection.execute("SELECT COUNT(*) FROM gpu_allocations").fetchone()[0] == 0
    assert not (tmp_path / "AUTHORED_no_launch").exists()
    account.close()


def test_caught_physical_failure_already_stops_child_and_forbids_reuse(tmp_path, monkeypatch):
    c = client(tmp_path)
    c.ledger.begin_gpu("AUTHORED", STAGE, GPU, start=100)
    c.started = True
    c.process = FakeChild()
    monkeypatch.setattr("reproduce.persistent_model.time.time", lambda: 3700)

    def budget_interruption():
        raise RuntimeError("AUTHORED actual GPU ceiling reached")

    monkeypatch.setattr(c, "_read", budget_interruption)
    # The actual IncrementalPipeline/evaluate_candidate can swallow this error
    # and return INCONCLUSIVE. Transport must stop before handing it upstream.
    with pytest.raises(PhysicalCallFailure):
        c.metered("interrupted", identity(), wire(), purpose="search")
    assert c.process.operations == ["terminate", "wait"]
    assert not c.started
    assert c.failed
    assert c.ledger.usage()["calls"] == 1
    assert c.ledger.usage()["calls_with_reserved_or_unknown_token_cost"] == 1
    with pytest.raises(RuntimeError, match="failed session"):
        c.metered("must_not_reserve", identity(), wire(), purpose="search")
    assert c.ledger.usage()["calls"] == 1
    c.__exit__(None, None, None)
    assert c.process.operations == ["terminate", "wait"]
    c.ledger.close()
