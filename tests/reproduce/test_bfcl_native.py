"""Binding mechanics without importing optional SDKs or benchmark data."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reproduce import bfcl_native
from reproduce.check_bfcl_scorer import authored_fixtures
from reproduce.strict_scoring import BFCL_CATEGORIES, ScoringError, UnsupportedScoring


def fake_native(monkeypatch, tmp_path, *, config=None):
    package = tmp_path / "source" / bfcl_native.PACKAGE_RELATIVE
    calls = []

    def native(*args):
        calls.append(args)
        return {"valid": True}

    model = bfcl_native.TASK_MODELS[0]
    mapping = {} if config is None else {model: config}
    modules = {
        "bfcl_eval.eval_checker.ast_eval.ast_checker": SimpleNamespace(
            __file__=str(package / "eval_checker/ast_eval/ast_checker.py"), ast_checker=native
        ),
        "bfcl_eval.constants.model_config": SimpleNamespace(
            __file__=str(package / "constants/model_config.py"),
            MODEL_CONFIG_MAPPING=mapping,
            ModelConfig=SimpleNamespace,
        ),
        "bfcl_eval.constants.enums": SimpleNamespace(
            __file__=str(package / "constants/enums.py"),
            Language=SimpleNamespace(PYTHON="native-Python-enum"),
        ),
    }
    monkeypatch.setattr(bfcl_native, "verify_bfcl_source", lambda _: package)
    monkeypatch.setattr(bfcl_native.importlib, "import_module", modules.__getitem__)
    monkeypatch.delitem(bfcl_native.sys.modules, "bfcl_eval.constants.eval_config", raising=False)
    # Keep process-global import/environment changes local to this test.
    monkeypatch.setattr(bfcl_native.sys, "path", list(bfcl_native.sys.path))
    for name in ("BFCL_PROJECT_ROOT", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "test-original-value")
    return package, calls, mapping, modules


def test_binding_dispatches_native_exact_arguments(monkeypatch, tmp_path):
    package, calls, mapping, _ = fake_native(monkeypatch, tmp_path)
    model = bfcl_native.TASK_MODELS[0]
    checker, metadata = bfcl_native.load_bfcl_native(
        package.parents[1], tmp_path / "scratch", model
    )
    functions, decoded, answers = [{"name": "a.b"}], [{"a.b": {}}], [{"a.b": {}}]
    assert checker(functions, decoded, answers, "parallel") == {"valid": True}
    assert calls == [(functions, decoded, answers, "native-Python-enum", "parallel", model)]
    assert mapping[model].model_handler == "SCORER_ONLY_NOT_INFERENCE"
    assert mapping[model].is_fc_model is False
    assert mapping[model].underscore_to_dot is False
    assert metadata["scorer_only_registry_entry_added"] is True
    assert bfcl_native.os.environ["BFCL_PROJECT_ROOT"] == str((tmp_path / "scratch").resolve())


def test_conflicting_native_metadata_is_not_overwritten(monkeypatch, tmp_path):
    original = SimpleNamespace(underscore_to_dot=True)
    package, _, mapping, _ = fake_native(monkeypatch, tmp_path, config=original)
    with pytest.raises(ScoringError, match="conflicts"):
        bfcl_native.load_bfcl_native(
            package.parents[1], tmp_path / "scratch", bfcl_native.TASK_MODELS[0]
        )
    assert mapping[bfcl_native.TASK_MODELS[0]] is original


def test_matching_native_metadata_is_preserved_and_checked_per_call(monkeypatch, tmp_path):
    original = SimpleNamespace(underscore_to_dot=False)
    package, calls, _, _ = fake_native(monkeypatch, tmp_path, config=original)
    checker, metadata = bfcl_native.load_bfcl_native(
        package.parents[1], tmp_path / "scratch", bfcl_native.TASK_MODELS[0]
    )
    assert metadata["scorer_only_registry_entry_added"] is False
    original.underscore_to_dot = True
    with pytest.raises(ScoringError, match="conflicts"):
        checker([], [], [], "simple_python")
    assert not calls


@pytest.mark.parametrize("category", ["irrelevance", "simple_java", "multi_turn_base"])
def test_binding_rejects_outside_AST_categories(monkeypatch, tmp_path, category):
    package, calls, _, _ = fake_native(monkeypatch, tmp_path)
    checker, _ = bfcl_native.load_bfcl_native(
        package.parents[1], tmp_path / "scratch", bfcl_native.TASK_MODELS[0]
    )
    with pytest.raises(UnsupportedScoring):
        checker([], [], [], category)
    assert not calls


def test_unknown_model_rejected_before_import(monkeypatch, tmp_path):
    monkeypatch.setattr(bfcl_native, "verify_bfcl_source", lambda _: pytest.fail("must not import"))
    with pytest.raises(UnsupportedScoring):
        bfcl_native.load_bfcl_native(tmp_path / "source", tmp_path / "scratch", "other/model")


def test_scratch_cannot_mutate_source_checkout(monkeypatch, tmp_path):
    package, calls, _, _ = fake_native(monkeypatch, tmp_path)
    with pytest.raises(ScoringError, match="outside"):
        bfcl_native.load_bfcl_native(
            package.parents[1], package / "scratch", bfcl_native.TASK_MODELS[0]
        )
    assert not calls


def test_cached_different_scratch_rejected(monkeypatch, tmp_path):
    package, _, _, _ = fake_native(monkeypatch, tmp_path)
    monkeypatch.setitem(
        bfcl_native.sys.modules,
        "bfcl_eval.constants.eval_config",
        SimpleNamespace(PROJECT_ROOT=tmp_path / "old-scratch"),
    )
    with pytest.raises(ScoringError, match="different scratch"):
        bfcl_native.load_bfcl_native(
            package.parents[1], tmp_path / "scratch", bfcl_native.TASK_MODELS[0]
        )


def test_shadowed_native_module_rejected(monkeypatch, tmp_path):
    package, _, _, modules = fake_native(monkeypatch, tmp_path)
    modules["bfcl_eval.constants.enums"].__file__ = str(tmp_path / "other/enums.py")
    with pytest.raises(ScoringError, match="pinned checkout"):
        bfcl_native.load_bfcl_native(
            package.parents[1], tmp_path / "scratch", bfcl_native.TASK_MODELS[0]
        )


def test_source_verification_only_requests_code_paths(monkeypatch, tmp_path):
    source = tmp_path / "source"
    native_file = source / bfcl_native.PACKAGE_RELATIVE / "eval_checker/ast_eval/ast_checker.py"
    native_file.parent.mkdir(parents=True)
    native_file.touch()
    calls = []
    monkeypatch.setattr(
        bfcl_native.subprocess, "check_output", lambda *a, **k: bfcl_native.BFCL_REVISION
    )
    monkeypatch.setattr(bfcl_native.subprocess, "run", lambda args, **kwargs: calls.append(args))
    assert bfcl_native.verify_bfcl_source(source) == source / bfcl_native.PACKAGE_RELATIVE
    requested = calls[0][calls[0].index("--") + 1 :]
    assert requested == list(bfcl_native.SOURCE_PATHS)
    assert all("/data" not in path and "/eval/" not in path for path in requested)
    assert all(path.endswith((".py", ".toml")) for path in requested)


def test_wrong_revision_is_error(monkeypatch, tmp_path):
    source = tmp_path / "source"
    native_file = source / bfcl_native.PACKAGE_RELATIVE / "eval_checker/ast_eval/ast_checker.py"
    native_file.parent.mkdir(parents=True)
    native_file.touch()
    monkeypatch.setattr(
        bfcl_native.subprocess, "check_output", lambda *a, **k: "different-revision"
    )
    with pytest.raises(ScoringError, match="revision"):
        bfcl_native.verify_bfcl_source(source)


def test_missing_source_is_error(tmp_path):
    with pytest.raises(ScoringError, match="missing"):
        bfcl_native.verify_bfcl_source(tmp_path)


def test_authored_fixtures_cover_scope_without_benchmark_units():
    fixtures = authored_fixtures()
    assert len(fixtures) == 40
    assert len({row["case"] for row in fixtures}) == len(fixtures)
    assert {row["category"] for row in fixtures if row["error"] is None} == BFCL_CATEGORIES
    assert {row["error"] for row in fixtures if row["error"]} == {
        "ScoringError",
        "UnsupportedScoring",
    }
    assert all("id" not in row for row in fixtures)
    assert any(row["case"] == "native_string_standardization" for row in fixtures)
