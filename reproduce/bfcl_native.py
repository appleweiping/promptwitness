"""Bind the pinned, unchanged BFCL Python AST matcher without SDK inference.

This is research tooling, not part of the dependency-free public package. The
model registry extension is explicit scorer-only metadata for the charter's
JSON interface; it is not upstream generation support or a replacement matcher.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from reproduce.strict_scoring import BFCL_CATEGORIES, ScoringError, UnsupportedScoring

BFCL_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
TASK_MODELS = (
    "Qwen/Qwen3.5-9B",
    "allenai/Olmo-3-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
PACKAGE_RELATIVE = Path("berkeley-function-call-leaderboard/bfcl_eval")
# Explicit source paths only: never diff bundled data, final prompts or scores.
SOURCE_PATHS = (
    "berkeley-function-call-leaderboard/pyproject.toml",
    f"{PACKAGE_RELATIVE.as_posix()}/__init__.py",
    f"{PACKAGE_RELATIVE.as_posix()}/utils.py",
    f":(glob){PACKAGE_RELATIVE.as_posix()}/constants/*.py",
    f":(glob){PACKAGE_RELATIVE.as_posix()}/model_handler/**/*.py",
    f":(glob){PACKAGE_RELATIVE.as_posix()}/eval_checker/ast_eval/**/*.py",
    f":(glob){PACKAGE_RELATIVE.as_posix()}/eval_checker/multi_turn_eval/*.py",
)
BFCLChecker = Callable[
    [list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str], Mapping[str, Any]
]


def verify_bfcl_source(source: Path) -> Path:
    """Reject another revision or changed scorer code before importing it."""
    source = source.resolve()
    package = source / PACKAGE_RELATIVE
    if not (package / "eval_checker/ast_eval/ast_checker.py").is_file():
        raise ScoringError("pinned BFCL source is missing")
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != BFCL_REVISION:
        raise ScoringError("unexpected BFCL source revision")
    subprocess.run(
        ["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--", *SOURCE_PATHS],
        check=True,
        capture_output=True,
    )
    return package


def load_bfcl_native(
    source: Path, work_root: Path, model: str
) -> tuple[BFCLChecker, dict[str, Any]]:
    """Import the native matcher and declare exact-name scorer metadata.

    Importing upstream eval_config creates result/score/lock directories. Redirect
    those to task-owned scratch *before* import. No handler is instantiated; no
    API, decoder, dataset loader or generation entry point is invoked.
    """
    if model not in TASK_MODELS:
        raise UnsupportedScoring("model is outside the frozen task model set")
    package = verify_bfcl_source(source)
    work_root = work_root.resolve()
    if work_root.is_relative_to(source.resolve()):
        raise ScoringError("BFCL scratch must be outside the upstream source checkout")
    work_root.mkdir(parents=True, exist_ok=True)
    cached = sys.modules.get("bfcl_eval.constants.eval_config")
    if cached is not None and Path(cached.PROJECT_ROOT).resolve() != work_root:
        raise ScoringError("cached BFCL runtime has a different scratch directory")
    os.environ["BFCL_PROJECT_ROOT"] = str(work_root)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(package.parent))
    native = importlib.import_module("bfcl_eval.eval_checker.ast_eval.ast_checker")
    registry = importlib.import_module("bfcl_eval.constants.model_config")
    enums = importlib.import_module("bfcl_eval.constants.enums")
    for module in (native, registry, enums):
        if module.__file__ is None or not Path(module.__file__).resolve().is_relative_to(package):
            raise ScoringError("BFCL module was not loaded from the pinned checkout")
    added = model not in registry.MODEL_CONFIG_MAPPING
    if added:
        registry.MODEL_CONFIG_MAPPING[model] = registry.ModelConfig(
            model_name=model,
            display_name=f"{model} (PromptWitness scorer-only JSON)",
            url=f"https://huggingface.co/{model}",
            org="scorer-only metadata; not a leaderboard submission",
            license="refer to frozen model license audit; no weights used here",
            model_handler="SCORER_ONLY_NOT_INFERENCE",
            is_fc_model=False,
            underscore_to_dot=False,
        )

    def check(
        functions: list[dict[str, Any]],
        decoded: list[dict[str, Any]],
        answers: list[dict[str, Any]],
        category: str,
    ) -> Mapping[str, Any]:
        if category not in BFCL_CATEGORIES - {"irrelevance"}:
            raise UnsupportedScoring("native AST dispatch requires an offline Python call category")
        if registry.MODEL_CONFIG_MAPPING[model].underscore_to_dot is not False:
            raise ScoringError("native model metadata conflicts with the exact-name JSON interface")
        return cast(
            Mapping[str, Any],
            native.ast_checker(functions, decoded, answers, enums.Language.PYTHON, category, model),
        )

    if registry.MODEL_CONFIG_MAPPING[model].underscore_to_dot is not False:
        raise ScoringError("native model metadata conflicts with the exact-name JSON interface")
    return check, {
        "revision": BFCL_REVISION,
        "model": model,
        "scorer_only_registry_entry_added": added,
        "underscore_to_dot": False,
        "language": "Python",
        "native_module": str(Path(cast(str, native.__file__)).resolve()),
        "no_inference_handler_instantiated": True,
    }
