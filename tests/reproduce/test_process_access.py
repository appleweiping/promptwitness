"""Role policy checks and optional real Linux sentinel qualification."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from reproduce import process_access as access
from reproduce.check_process_access import EXPECTED, check, private_env_witness


def layout(tmp_path):
    data, code, scratch = (tmp_path / name for name in ("store", "code", "scratch"))
    for leaf in access.LEAVES:
        (data / leaf).mkdir(parents=True)
    code.mkdir()
    scratch.mkdir()
    return data, code, scratch


@pytest.mark.parametrize(
    "role,stage",
    [
        ("fit_learner", "fit"),
        ("optimizer", "search"),
        ("predictor", "search"),
        ("search_scorer", "search"),
        ("selection_scorer", "selection"),
        ("final_scorer", "final"),
        ("retrieval_fit_learner", "fit"),
        ("retrieval_fit_scorer", "fit"),
        ("retrieval_optimizer", "search"),
        ("retrieval_predictor", "search"),
        ("retrieval_search_ranker", "search"),
        ("retrieval_selection_ranker", "selection"),
        ("retrieval_search_scorer", "search"),
        ("retrieval_selection_scorer", "selection"),
        ("retrieval_final_scorer", "final"),
    ],
)
def test_grants_match_independent_matrix(tmp_path, role, stage):
    data, code, scratch = layout(tmp_path)
    policy = access.build_policy(role, stage, data, [code], scratch)
    assert {Path(path).relative_to(data).as_posix() for path in policy["data_read"]} == EXPECTED[
        role
    ]
    assert policy["runtime_read_execute"] == [str(code)]


def test_retrieval_non_scorers_cannot_read_any_gold(tmp_path):
    data, code, scratch = layout(tmp_path)
    for role, stage in (
        ("retrieval_fit_learner", "fit"),
        ("retrieval_optimizer", "search"),
        ("retrieval_predictor", "search"),
        ("retrieval_search_ranker", "search"),
        ("retrieval_selection_ranker", "selection"),
    ):
        policy = access.build_policy(role, stage, data, [code], scratch)
        assert all("gold" not in Path(path).parts for path in policy["data_read"])
    assert "fit/gold" in access.role_leaves("retrieval_fit_scorer", "fit")


@pytest.mark.parametrize(
    "role,stage",
    [
        ("optimizer", "final"),
        ("predictor", "selection"),
        ("final_scorer", "search"),
        ("selection_scorer", "fit"),
        ("unknown", "search"),
        ("optimizer", "unknown"),
        ("retrieval_selection_ranker", "search"),
        ("retrieval_search_ranker", "final"),
    ],
)
def test_invalid_role_stage_denied(role, stage):
    with pytest.raises(access.AccessBoundaryError, match="authorized"):
        access.role_leaves(role, stage)


@pytest.mark.parametrize(
    "kind", ["runtime_ancestor", "runtime_inside", "scratch_inside", "scratch_runtime"]
)
def test_data_or_scratch_cannot_be_inside_runtime(tmp_path, kind):
    data, code, scratch = layout(tmp_path)
    runtime = [tmp_path] if kind == "runtime_ancestor" else [code]
    if kind == "runtime_inside":
        runtime = [data / "fit/inputs"]
    if kind == "scratch_inside":
        scratch = data / "final/gold"
    if kind == "scratch_runtime":
        scratch = code
    with pytest.raises(access.AccessBoundaryError, match="separate"):
        access.build_policy("predictor", "search", data, runtime, scratch)


def test_missing_leaf_does_not_silently_widen_grant(tmp_path):
    data, code, scratch = layout(tmp_path)
    (data / "final/gold").rmdir()
    with pytest.raises(FileNotFoundError):
        access.build_policy("optimizer", "search", data, [code], scratch)


def test_no_runtime_roots_denied(tmp_path):
    data, _, scratch = layout(tmp_path)
    with pytest.raises(access.AccessBoundaryError, match="separate"):
        access.build_policy("optimizer", "search", data, [], scratch)


def test_file_not_directory_denied(tmp_path):
    file = tmp_path / "file"
    file.write_text("fixture", encoding="utf-8")
    with pytest.raises(access.AccessBoundaryError, match="directories"):
        access._directory(file)


def test_launch_has_clean_environment_closed_descriptors_and_no_shell(tmp_path, monkeypatch):
    data, code, scratch = layout(tmp_path)
    entry = code / "application.py"
    entry.write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(access, "runtime_roots", lambda: (code,))
    monkeypatch.setenv("PW_SENTINEL_PRIVATE", "not-for-worker")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 3, "", "application failure")

    monkeypatch.setattr(access.subprocess, "run", run)
    result = access.launch_role("optimizer", "search", data, scratch, entry, ["plain"])
    command, kwargs = calls[0]
    assert command[1] == "-I"
    assert "shell" not in kwargs and kwargs["close_fds"] is True
    assert "PW_SENTINEL_PRIVATE" not in kwargs["env"]
    assert kwargs["env"]["HOME"] == str(scratch.resolve())
    assert json.loads(kwargs["input"])["arguments"] == ["plain"]
    assert result.returncode == 3  # not replaced by a score


def test_persistent_launch_sends_policy_before_application_input(tmp_path, monkeypatch):
    data, code, scratch = layout(tmp_path)
    entry = code / "application.py"
    entry.write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(access, "runtime_roots", lambda: (code,))
    monkeypatch.setattr(access, "landlock_abi", lambda: 1)
    monkeypatch.setenv("PW_SENTINEL_PRIVATE", "not-for-worker")
    calls = []

    class FakeProcess:
        stdin = io.StringIO()

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(access.subprocess, "Popen", popen)
    process = access.start_role(
        "retrieval_search_ranker", "search", data, scratch, entry, stderr=io.StringIO()
    )
    command, kwargs = calls[0]
    assert command[1] == "-I" and kwargs["close_fds"] is True
    assert "PW_SENTINEL_PRIVATE" not in kwargs["env"]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert json.loads(process.stdin.getvalue())["data_read"] == [str(data / "search/inputs")]


def test_entrypoint_outside_runtime_denied(tmp_path, monkeypatch):
    data, code, scratch = layout(tmp_path)
    entry = tmp_path / "outside.py"
    entry.write_text("pass", encoding="utf-8")
    monkeypatch.setattr(access, "runtime_roots", lambda: (code,))
    with pytest.raises(access.AccessBoundaryError, match="entrypoint"):
        access.launch_role("optimizer", "search", data, scratch, entry)


def test_unsupported_host_is_not_unrestricted_fallback(monkeypatch):
    monkeypatch.setattr(access.sys, "platform", "win32")
    with pytest.raises(access.AccessBoundaryError, match="Linux"):
        access.landlock_abi()


@pytest.mark.parametrize("previous", [None, "original-sentinel"])
def test_environment_probe_seeds_and_restores_parent(monkeypatch, previous):
    if previous is None:
        monkeypatch.delenv("PW_SENTINEL_PRIVATE", raising=False)
    else:
        monkeypatch.setenv("PW_SENTINEL_PRIVATE", previous)
    with pytest.raises(ValueError, match="fixture"), private_env_witness():
        assert access.os.environ["PW_SENTINEL_PRIVATE"] == "AUTHORED_NOT_A_CREDENTIAL"
        raise ValueError("fixture failure")
    assert access.os.environ.get("PW_SENTINEL_PRIVATE") == previous


def test_real_linux_process_sentinels(tmp_path, monkeypatch):
    if sys.platform != "linux":
        pytest.skip("actual Landlock witness requires Linux")
    try:
        access.landlock_abi()
    except access.AccessBoundaryError as exc:
        pytest.skip(str(exc))
    monkeypatch.setenv("PW_SENTINEL_PRIVATE", "not-for-worker")
    result = check(tmp_path / "attempt", tmp_path / "witness.json")
    assert result["read_checks"] == result["write_checks"] == 165
    assert result["scientific_admission"] == "PARTIAL_NOT_PASSED"
    assert len({row["pid"] for row in result["workers"]}) == 15
    assert all(row["helper_import_after_restriction"] for row in result["workers"])


@pytest.mark.parametrize(
    "role,stage,random_allowed,cpuinfo_allowed,gold_allowed",
    [
        ("retrieval_search_ranker", "search", True, True, False),
        ("retrieval_selection_ranker", "selection", True, True, False),
        ("retrieval_search_scorer", "search", False, False, True),
        ("retrieval_selection_scorer", "selection", False, False, True),
    ],
)
def test_real_linux_ranker_device_grant_preserves_role_boundary(
    tmp_path, monkeypatch, role, stage, random_allowed, cpuinfo_allowed, gold_allowed
):
    if sys.platform != "linux":
        pytest.skip("actual Landlock device witness requires Linux")
    try:
        access.landlock_abi()
    except access.AccessBoundaryError as exc:
        pytest.skip(str(exc))
    data, code, scratch = layout(tmp_path)
    (data / stage / "gold/secret.txt").write_text("authored-gold", encoding="utf-8")
    entry = code / "device_probe.py"
    entry.write_text(
        "from pathlib import Path\n"
        "def readable(path):\n"
        "    try:\n"
        "        with Path(path).open('rb') as stream: stream.read(1)\n"
        "    except PermissionError: return False\n"
        "    return True\n"
        "print(readable('/dev/urandom'), readable('/proc/cpuinfo'), "
        + f"readable({str(data / stage / 'gold/secret.txt')!r}), sep=',')\n",
        encoding="utf-8",
    )
    original_roots = access.runtime_roots()
    monkeypatch.setattr(access, "runtime_roots", lambda: (code, *original_roots))
    outcome = access.launch_role(role, stage, data, scratch, entry)
    assert outcome.returncode == 0, outcome.stderr
    assert outcome.stdout.strip() == f"{random_allowed},{cpuinfo_allowed},{gold_allowed}"
