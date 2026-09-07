from __future__ import annotations

import pytest

from promptwitness import evaluate_long_context, make_needle_cases


def test_needle_generation_is_deterministic_and_position_reported() -> None:
    contexts = (("alpha", "beta", "gamma"), ("one", "two"))
    cases = make_needle_cases(
        contexts, needle="Ada works on NLP", query="Who works on NLP?", expected="Ada", seed=7
    )
    assert cases == make_needle_cases(
        contexts, needle="Ada works on NLP", query="Who works on NLP?", expected="Ada", seed=7
    )
    assert all("NEEDLE: Ada works on NLP" in case.prompt for case in cases)
    report = evaluate_long_context(cases, lambda case: "Ada")
    assert report.accuracy == 1.0
    assert report.attempted == 2 and report.failed == 0
    assert set(report.by_position) == {"front", "middle", "back"}
    assert report.to_dict()["by_depth"]["3"] is not None


def test_long_context_failures_are_retained_or_fail_fast() -> None:
    (case,) = make_needle_cases(
        (("first", "last"),), needle="secret", query="What?", expected="yes"
    )
    report = evaluate_long_context(
        (case,), lambda _: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    assert report.accuracy is None and report.failed == 1
    assert "offline" in (report.results[0].error or "")
    with pytest.raises(TypeError, match="boolean"):
        evaluate_long_context((case,), lambda _: "no", strict="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="failed"):
        evaluate_long_context(
            (case,), lambda _: (_ for _ in ()).throw(RuntimeError("offline")), strict=True
        )
