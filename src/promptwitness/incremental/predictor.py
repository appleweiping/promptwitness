"""Regularized transition heads; synthetic training never proves research value."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from .features import FEATURE_VERSION, FeatureVector
from .sampling import digest


@dataclass(frozen=True, slots=True)
class TrainingRow:
    features: FeatureVector
    old_correct: int
    new_correct: int
    lineage: str
    source_group: str
    split: str
    label_source_digest: str
    kind: str = "real_oracle"
    allocated_seconds: float | None = None

    def __post_init__(self) -> None:
        if any(type(v) is not int or v not in (0, 1) for v in (self.old_correct, self.new_correct)):
            raise ValueError("actual binary paired training labels required")
        if self.split != "fit" or self.kind not in ("real_oracle", "synthetic"):
            raise ValueError("only authorized fit-side rows may train the predictor")
        if not self.lineage or not self.source_group or len(self.label_source_digest) != 64:
            raise ValueError("lineage, source group and source digest required")
        if self.allocated_seconds is not None and (
            isinstance(self.allocated_seconds, bool)
            or not math.isfinite(self.allocated_seconds)
            or self.allocated_seconds < 0
        ):
            raise ValueError("finite nonnegative observed cost required")


@dataclass(frozen=True, slots=True)
class Prediction:
    status: str
    regression_probability: float | None
    improvement_probability: float | None
    estimated_allocated_seconds: float | None
    model_digest: str | None


class TransitionPredictor:
    """Frozen feature vocabulary, independent conditional logistic heads, Ridge cost.

    Checkpoints are plain validated JSON coefficients, never executable pickles.
    The sparse-class fallback is an explicitly smoothed intercept, not a
    zero-risk label for unseen outcomes. Prediction cannot accept audit labels.
    """

    def __init__(self) -> None:
        self._state: dict[str, Any] | None = None

    @property
    def status(self) -> str:
        return "UNFITTED" if self._state is None else str(self._state["status"])

    def fit(
        self,
        rows: tuple[TrainingRow, ...],
        *,
        held_out_lineages: frozenset[str] = frozenset(),
        held_out_sources: frozenset[str] = frozenset(),
    ) -> None:
        if not rows or len(rows) > 100000:
            raise ValueError("bounded nonempty fit records required")
        first = rows[0].features
        if any(
            row.features.names != first.names
            or row.features.version != first.version
            or row.kind != rows[0].kind
            or row.lineage in held_out_lineages
            or row.source_group in held_out_sources
            for row in rows
        ):
            raise ValueError(
                "feature mismatch, mixed training provenance or held-out training leakage"
            )
        try:
            import numpy as np

            linear_models = import_module("sklearn.linear_model")
        except ImportError as error:
            raise ValueError("install promptwitness[research] to fit a predictor") from error
        matrix = np.asarray([row.features.values for row in rows], dtype=float)
        heads: dict[str, Any] = {}
        for name, old in (("regression", 1), ("improvement", 0)):
            selected = [i for i, row in enumerate(rows) if row.old_correct == old]
            if not selected:
                heads[name] = None
                continue
            targets = [int(rows[i].new_correct != old) for i in selected]
            if len(set(targets)) == 1:
                p = (sum(targets) + 1) / (len(targets) + 2)
                heads[name] = {
                    "coefficients": [0.0] * matrix.shape[1],
                    "intercept": math.log(p / (1 - p)),
                    "fit": "laplace_intercept_one_class",
                    "rows": len(selected),
                }
            else:
                model = linear_models.LogisticRegression(
                    C=1.0, solver="liblinear", random_state=11, max_iter=2000
                )
                model.fit(matrix[selected], targets)
                if int(model.n_iter_[0]) >= 2000:
                    raise ValueError("logistic fit did not converge within its frozen limit")
                heads[name] = {
                    "coefficients": model.coef_[0].tolist(),
                    "intercept": float(model.intercept_[0]),
                    "fit": "regularized_logistic",
                    "rows": len(selected),
                }
        cost_rows = [i for i, row in enumerate(rows) if row.allocated_seconds is not None]
        cost = None
        if cost_rows:
            model_cost = linear_models.Ridge(alpha=1.0)
            model_cost.fit(
                matrix[cost_rows],
                [
                    math.log1p(row.allocated_seconds)
                    for row in rows
                    if row.allocated_seconds is not None
                ],
            )
            cost = {
                "coefficients": model_cost.coef_.tolist(),
                "intercept": float(model_cost.intercept_),
            }
        state = {
            "format": "promptwitness.delta.predictor/v1.1",
            "feature_version": FEATURE_VERSION,
            "feature_names": list(first.names),
            "structured": first.structured,
            "status": "FITTED_SYNTHETIC" if rows[0].kind == "synthetic" else "FITTED_REAL_FIT_ONLY",
            "heads": heads,
            "cost": cost,
            "training_digest": digest(
                [
                    {
                        "features": r.features.sha256,
                        "old": r.old_correct,
                        "new": r.new_correct,
                        "lineage": r.lineage,
                        "source": r.source_group,
                        "label_source_digest": r.label_source_digest,
                        "cost": r.allocated_seconds,
                    }
                    for r in rows
                ]
            ),
        }
        self._state = state

    def to_dict(self) -> dict[str, Any]:
        if self._state is None:
            return {"format": "promptwitness.delta.predictor/v1.1", "status": "UNFITTED"}
        import json

        return dict(json.loads(json.dumps(self._state, allow_nan=False)))

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> TransitionPredictor:
        result = cls()
        if state == {"format": "promptwitness.delta.predictor/v1.1", "status": "UNFITTED"}:
            return result
        expected = {
            "format",
            "feature_version",
            "feature_names",
            "structured",
            "status",
            "heads",
            "cost",
            "training_digest",
        }
        if (
            set(state) != expected
            or state["format"] != "promptwitness.delta.predictor/v1.1"
            or state["feature_version"] != FEATURE_VERSION
        ):
            raise ValueError("unknown predictor format or fields")
        if (
            state["status"] not in ("FITTED_SYNTHETIC", "FITTED_REAL_FIT_ONLY")
            or type(state["structured"]) is not bool
        ):
            raise ValueError("unsupported predictor provenance")
        names = state["feature_names"]
        if (
            not isinstance(names, list)
            or not names
            or len(names) != len(set(names))
            or any(not isinstance(n, str) for n in names)
        ):
            raise ValueError("invalid feature names")
        if not isinstance(state["heads"], dict) or set(state["heads"]) != {
            "regression",
            "improvement",
        }:
            raise ValueError("both transition heads must be explicitly represented")
        for head in (*state["heads"].values(), state["cost"]):
            if head is None:
                continue
            if not isinstance(head, dict) or "coefficients" not in head or "intercept" not in head:
                raise ValueError("invalid coefficient head")
            if len(head["coefficients"]) != len(names) or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                for x in (*head["coefficients"], head["intercept"])
            ):
                raise ValueError("non-finite or unaligned model coefficients")
        result._state = state
        # Copy through JSON to remove mutable caller references and non-JSON values.
        result._state = result.to_dict()
        return result

    def predict(self, features: FeatureVector) -> Prediction:
        if self._state is None:
            return Prediction("UNFITTED", None, None, None, None)
        if (
            list(features.names) != self._state["feature_names"]
            or features.structured != self._state["structured"]
        ):
            raise ValueError("prediction feature specification differs from fit")

        def linear(head: dict[str, Any]) -> float:
            return float(
                head["intercept"]
                + sum(a * b for a, b in zip(head["coefficients"], features.values, strict=True))
            )

        probabilities = []
        for name in ("regression", "improvement"):
            head = self._state["heads"][name]
            probabilities.append(
                None if head is None else 1 / (1 + math.exp(-max(-40.0, min(40.0, linear(head)))))
            )
        cost = self._state["cost"]
        estimate = (
            None if cost is None else max(0.0, math.expm1(max(-40.0, min(40.0, linear(cost)))))
        )
        return Prediction(
            self.status, probabilities[0], probabilities[1], estimate, digest(self._state)
        )
