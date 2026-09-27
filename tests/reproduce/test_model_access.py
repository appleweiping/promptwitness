"""Authored structural checks; Linux actual enforcement is a separate witness."""

import copy
import io
import json
import stat
import sys
from pathlib import Path

import pytest

from reproduce import model_access as boundary


@pytest.mark.parametrize("missing", [None, "/proc/cpuinfo", "/proc/sys/vm/mmap_min_addr"])
def test_only_exact_cpu_cuda_metadata_files_are_readable(monkeypatch, missing):
    class NativePath:
        def __init__(self, name):
            self.name = name

        def stat(self):
            return type("Metadata", (), {"st_mode": stat.S_IFCHR})()

        def is_file(self):
            return self.name != missing

    monkeypatch.setattr(boundary, "Path", NativePath)
    if missing is not None:
        with pytest.raises(boundary.AccessBoundaryError, match="metadata files unavailable"):
            boundary.native_files()
        return
    read, write = boundary.native_files()
    assert {path.name for path in read} == {
        "/dev/urandom",
        "/proc/cpuinfo",
        "/proc/sys/vm/mmap_min_addr",
    }
    assert {path.name for path in write} == {"/dev/null", *boundary.DEVICES}
    assert "/proc" not in {path.name for path in (*read, *write)}


def layout(tmp_path, monkeypatch):
    data, runtime, snapshot, scratch = (
        tmp_path / name for name in ("store", "runtime", "snapshots/revision", "scratch")
    )
    for path in (runtime, snapshot, scratch):
        path.mkdir(parents=True)
    for leaf in boundary.LEAVES:
        (data / leaf).mkdir(parents=True)
    monkeypatch.setattr(boundary, "runtime_roots", lambda: (runtime,))
    monkeypatch.setattr(boundary, "native_files", lambda: ([], []))
    (runtime / "own-task").mkdir()
    monkeypatch.setattr(boundary, "thread_metadata_root", lambda: runtime / "own-task")
    return data, runtime, snapshot, scratch


def test_no_benchmark_or_cache_ancestor_granted(tmp_path, monkeypatch):
    data, runtime, snapshot, scratch = layout(tmp_path, monkeypatch)
    policy = boundary.model_policy(snapshot, data, scratch)
    assert policy["benchmark_data_read"] == []
    assert policy["runtime_read_execute"] == [str(runtime)]
    assert str(snapshot) in policy["data_read"]
    assert str(snapshot.parent.parent) not in policy["data_read"]
    assert str(data) not in policy["data_read"]
    assert policy["scratch_read_write"] == str(scratch)
    assert policy["write_file_trees"] == [str(runtime / "own-task")]


@pytest.mark.parametrize("which", ["runtime", "scratch", "snapshot", "thread_metadata"])
def test_actual_grant_cannot_cover_store(tmp_path, monkeypatch, which):
    data, _runtime, snapshot, scratch = layout(tmp_path, monkeypatch)
    if which == "runtime":
        monkeypatch.setattr(boundary, "runtime_roots", lambda: (tmp_path,))
    elif which == "scratch":
        scratch = data / "fit/records"
    elif which == "snapshot":
        snapshot = data / "fit/gold"
    else:
        monkeypatch.setattr(boundary, "thread_metadata_root", lambda: data / "fit/gold")
    with pytest.raises(boundary.AccessBoundaryError, match="overlap"):
        boundary.model_policy(snapshot, data, scratch)


def test_exact_symlink_blob_not_whole_cache(tmp_path, monkeypatch):
    data, _, snapshot, scratch = layout(tmp_path, monkeypatch)
    blob = snapshot.parent.parent / "blobs" / "original-weight-file"
    blob.parent.mkdir()
    blob.write_text("AUTHORED_ONLY")
    try:
        (snapshot / "model.safetensors").symlink_to(blob)
    except OSError:
        pytest.skip("OS does not permit authored symlink construction")
    policy = boundary.model_policy(snapshot, data, scratch)
    assert policy["read_files"] == [str(blob)]
    assert str(blob.parent) not in policy["data_read"]
    assert str(blob.parent) not in policy["runtime_read_execute"]


def test_symlink_not_arbitrary_private_record(tmp_path, monkeypatch):
    data, _, snapshot, scratch = layout(tmp_path, monkeypatch)
    forbidden = data / "fit/gold" / "authored.json"
    forbidden.write_text("AUTHORED_ONLY")
    try:
        (snapshot / "model.safetensors").symlink_to(forbidden)
    except OSError:
        pytest.skip("OS does not permit authored symlink construction")
    with pytest.raises(boundary.AccessBoundaryError, match="original model blob"):
        boundary.model_policy(snapshot, data, scratch)


@pytest.mark.parametrize("failure", ["fields", "threads", "already_imported", "kernel"])
def test_restriction_failure_does_not_start_app(tmp_path, monkeypatch, failure):
    message = {"snapshot": "a", "model": "b", "data_root": "c", "scratch": str(tmp_path)}
    policy = {
        "read_files": [],
        "read_write_files": [],
        "write_file_trees": [],
        "role": "inference_message_only",
    }
    monkeypatch.setattr(boundary, "model_policy", lambda *_: policy)
    monkeypatch.setattr(boundary.os, "listdir", lambda _: ["one"])
    monkeypatch.delitem(sys.modules, "promptwitness")
    if failure == "fields":
        message["gold"] = "forbidden"
    elif failure == "threads":
        monkeypatch.setattr(boundary.os, "listdir", lambda _: ["one", "two"])
    elif failure == "already_imported":
        monkeypatch.setitem(sys.modules, "torch", object())

    def kernel(*_, **__):
        raise boundary.AccessBoundaryError("AUTHORED unavailable kernel")

    monkeypatch.setattr(boundary, "enforce_policy", kernel)
    with pytest.raises(boundary.AccessBoundaryError):
        boundary.restrict(message)


def test_worker_restricts_before_binding_and_reading_app_messages(tmp_path, monkeypatch):
    launch = {"snapshot": "a", "model": "b", "data_root": "c", "scratch": str(tmp_path)}
    application = {"execution": "AUTHORED", "wire": "AUTHORED"}
    pipe = io.StringIO(json.dumps(launch) + "\n" + json.dumps(application))
    monkeypatch.setattr(boundary.sys, "stdin", pipe)
    events = []
    access = {"AUTHORED_RECEIPT": True}

    def restrict(message):
        assert message == launch
        assert pipe.tell() == len(json.dumps(launch)) + 1
        events.append("restricted")
        return copy.deepcopy(access)

    def bind():
        assert events == ["restricted"]
        events.append("bind_application")

    def worker(snapshot, model, *, access):
        assert events == ["restricted", "bind_application"]
        assert json.loads(pipe.read()) == application
        assert access == {"AUTHORED_RECEIPT": True}
        events.append("model")

    monkeypatch.setattr(boundary, "restrict", restrict)
    monkeypatch.setattr(boundary, "bind_source", bind)
    monkeypatch.setattr("reproduce.torch_runtime.worker", worker)
    boundary.worker()
    assert events == ["restricted", "bind_application", "model"]


def test_isolated_script_can_load_helper_without_package(tmp_path):
    import subprocess

    script = Path(boundary.__file__)
    result = subprocess.run(
        [sys.executable, "-I", str(script), "--probe"],
        input=json.dumps({"bad_policy": True}) + "\n",
        text=True,
        capture_output=True,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode != 0
    assert "literal model launch fields" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
