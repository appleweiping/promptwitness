"""Local checker mechanics only; actual Torch/published kernels need SSH receipt."""

import math
from pathlib import Path

import pytest

from reproduce.check_retrieval_native_metrics import check, compare_metrics, load_metric_kernels

FUNCTION_SOURCE = """
raise RuntimeError("top-level must not execute")
def cirr_compute_val_metrics(value: UnknownType) -> MissingAnnotation:
    return {"cirr": value}
def fiq_compute_val_metrics(value):
    return {"fiq": value}
def unrelated():
    raise RuntimeError("unrelated function must not execute")
"""


def test_only_selected_original_functions_no_top_level_or_annotation_import():
    namespace = {"sentinel": 7}
    kernels = load_metric_kernels(FUNCTION_SOURCE, namespace)
    assert kernels["cirr_compute_val_metrics"](3) == {"cirr": 3}
    assert kernels["fiq_compute_val_metrics"](4) == {"fiq": 4}
    assert namespace == {"sentinel": 7}


@pytest.mark.parametrize(
    "source",
    (
        "",
        "def cirr_compute_val_metrics(): pass",
        FUNCTION_SOURCE + "\ndef fiq_compute_val_metrics(): pass",
    ),
)
def test_missing_or_duplicate_expected_function_fails(source):
    with pytest.raises(ValueError, match="exactly the two"):
        load_metric_kernels(source, {})


def test_original_body_and_decorator_execute_with_supplied_bindings():
    calls = []

    def decorator(fn):
        calls.append(fn.__name__)
        return fn

    source = """
@decorator
def cirr_compute_val_metrics():
    return helper()
def fiq_compute_val_metrics():
    return helper()
"""
    kernels = load_metric_kernels(source, {"decorator": decorator, "helper": lambda: 19})
    assert calls == ["cirr_compute_val_metrics"]
    assert kernels["cirr_compute_val_metrics"]() == kernels["fiq_compute_val_metrics"]() == 19


def test_native_float32_percentage_rounding_tolerance():
    assert compare_metrics({"recall": 20.000000298023224}, {"recall": 20.0}) == 1


@pytest.mark.parametrize("value", (True, "20", math.nan, math.inf, 20.001, None))
def test_invalid_or_different_native_value_fails(value):
    with pytest.raises(ValueError, match="native percentage differs"):
        compare_metrics({"recall": value}, {"recall": 20.0})


def test_exact_metric_coverage_required():
    with pytest.raises(ValueError, match="keys differ"):
        compare_metrics({"recall": 20.0, "extra": 0.0}, {"recall": 20.0})


def test_cpu_visibility_guard_precedes_optional_import_and_output(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="CPU-only"):
        check(tmp_path / "never-opened.py", output)
    assert not output.exists()
