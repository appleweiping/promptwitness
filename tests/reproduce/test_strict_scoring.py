"""Authored mechanical fixtures; not research effects or native-scorer validation."""

import copy

import pytest

from reproduce import check_strict_scorers
from reproduce.strict_scoring import (
    BinaryScore,
    ScoringError,
    UnsupportedScoring,
    score_bfcl,
    score_hotpotqa,
    score_ifbench,
    score_iftrain,
)


class KeywordFixture:
    def __init__(self, identifier):
        self.identifier = identifier

    def build_description(self, **kwargs):
        self.keywords = kwargs["keywords"]

    def check_following(self, text):
        return all(word in text for word in self.keywords)


def unavailable(*args, **kwargs):
    raise RuntimeError("fixture scorer unavailable")


@pytest.mark.parametrize("function", [score_hotpotqa, score_iftrain, score_ifbench, score_bfcl])
@pytest.mark.parametrize(
    "response,status", [(None, "completed"), ("", "failed"), ("answer", "cancelled")]
)
def test_missing_execution_is_never_zero(function, response, status):
    with pytest.raises(ScoringError, match="completed"):
        if function is score_hotpotqa:
            function(response, status, "gold", unavailable)
        elif function is score_iftrain:
            function(response, status, ["k"], [{}], {})
        elif function is score_ifbench:
            function(response, status, "prompt", ["k"], [{}], unavailable)
        else:
            function(response, status, "irrelevance", [{}], None, unavailable)


def test_hotpot_calls_official_metric_without_text_rewriting():
    observed = []

    def native(prediction, gold):
        observed.append((prediction, gold))
        return prediction == gold

    assert score_hotpotqa("answer", "completed", "answer", native).value == 1
    assert score_hotpotqa(" answer ", "completed", "answer", native).value == 0
    assert score_hotpotqa("", "completed", "answer", native).value == 0
    assert observed[1] == (" answer ", "answer")
    with pytest.raises(ScoringError):
        score_hotpotqa("text", "completed", "", native)
    with pytest.raises(ScoringError):
        score_hotpotqa("text", "completed", "gold", unavailable)
    with pytest.raises(ScoringError, match="Boolean"):
        score_hotpotqa("text", "completed", "gold", lambda *_: 0.5)


@pytest.mark.parametrize(
    "ids,args", [([], []), (["k"], []), ([None], [{}]), ("k", [{}]), ([""], [{}]), (["k"], [3])]
)
def test_invalid_annotation_cannot_be_a_vacuous_pass(ids, args):
    with pytest.raises(ScoringError):
        score_iftrain("text", "completed", ids, args, {"k": KeywordFixture})


def test_iftrain_binary_all_constraints_keeps_raw_text_and_arguments():
    args = [{"keywords": ["apricot"], "optional": None}, {"keywords": ["cedar"]}]
    before = copy.deepcopy(args)
    registry = {"k": KeywordFixture}
    assert score_iftrain("apricot cedar", "completed", ["k", "k"], args, registry).value == 1
    partial = score_iftrain("apricot", "completed", ["k", "k"], args, registry)
    assert partial.value == 0 and partial.constraint_flags == (True, False)
    assert score_iftrain("", "completed", ["k", "k"], args, registry).value == 0
    assert (
        score_iftrain("<think>apricot cedar</think>", "completed", ["k", "k"], args, registry).value
        == 1
    )
    assert args == before
    with pytest.raises(UnsupportedScoring):
        score_iftrain("text", "completed", ["unknown"], [None], registry)
    with pytest.raises(ScoringError, match="checker failed"):
        score_iftrain("text", "completed", ["k"], [{}], {"k": unavailable})


def test_ifbench_calls_native_strict_with_original_prompt_and_no_mutation():
    args = [{"keyword": ["apricot"], "unused": None}]
    before = copy.deepcopy(args)

    def native(prompt, ids, kwargs, text):
        assert (prompt, ids, text) == ("original prompt", ["k"], " raw text ")
        assert kwargs == [{"keyword": ["apricot"]}]
        kwargs[0]["keyword"].clear()
        return [False]

    assert (
        score_ifbench(" raw text ", "completed", "original prompt", ["k"], args, native).value == 0
    )
    assert args == before
    with pytest.raises(ScoringError):
        score_ifbench("text", "completed", "", ["k"], [{}], native)
    with pytest.raises(ScoringError):
        score_ifbench("text", "completed", "p", ["k"], [{}], unavailable)
    with pytest.raises(UnsupportedScoring):
        score_ifbench(
            "text",
            "completed",
            "p",
            ["k"],
            [{}],
            lambda *_: (_ for _ in ()).throw(UnsupportedScoring("unknown")),
        )


@pytest.mark.parametrize("flags", [[], [None], [0.5], [True, False]])
def test_ifbench_bad_native_result_is_not_a_score(flags):
    with pytest.raises(ScoringError, match="flags"):
        score_ifbench("text", "completed", "p", ["k"], [{}], lambda *_: flags)


FUNCTIONS = [{"name": "module.f", "parameters": {"type": "dict"}}]
ANSWERS = [{"module.f": {"x": [1]}}]


def test_bfcl_preserves_names_and_delegates_native_matching():
    before = copy.deepcopy((FUNCTIONS, ANSWERS))

    def native(functions, decoded, answers, category):
        assert (functions, answers) == before
        assert decoded == [{"module.f": {"x": 1}}]
        assert category == "parallel_multiple"
        functions.clear()
        answers.clear()
        return {"valid": True}

    result = score_bfcl(
        '[{"name":"module.f","arguments":{"x":1}}]',
        "completed",
        "parallel_multiple",
        FUNCTIONS,
        ANSWERS,
        native,
    )
    assert result.value == 1 and before == (FUNCTIONS, ANSWERS)


@pytest.mark.parametrize(
    "response",
    [
        "",
        "[]",
        "not JSON",
        "{}",
        '[{"function":"f","parameters":{}}]',
        '[{"name":"f","arguments":null}]',
    ],
)
def test_bfcl_wrong_format_and_irrelevance_follow_distinct_semantics(response):
    assert (
        score_bfcl(response, "completed", "simple_python", FUNCTIONS, ANSWERS, unavailable).value
        == 0
    )
    assert score_bfcl(response, "completed", "irrelevance", FUNCTIONS, None, unavailable).value == 1
    assert (
        score_bfcl(
            '[{"name":"f","arguments":{}}]',
            "completed",
            "irrelevance",
            FUNCTIONS,
            None,
            unavailable,
        ).value
        == 0
    )


def test_bfcl_unsupported_missing_gold_and_native_failures_are_not_zero():
    text = '[{"name":"f","arguments":{}}]'
    with pytest.raises(UnsupportedScoring):
        score_bfcl(text, "completed", "multi_turn", FUNCTIONS, ANSWERS, unavailable)
    with pytest.raises(ScoringError):
        score_bfcl(text, "completed", "multiple", [], ANSWERS, unavailable)
    with pytest.raises(ScoringError):
        score_bfcl(text, "completed", "multiple", FUNCTIONS, None, unavailable)
    with pytest.raises(ScoringError, match="checker failed"):
        score_bfcl(text, "completed", "multiple", FUNCTIONS, ANSWERS, unavailable)
    with pytest.raises(ScoringError, match="Boolean"):
        score_bfcl(text, "completed", "multiple", FUNCTIONS, ANSWERS, lambda *_: {"valid": None})
    assert (
        score_bfcl(
            text, "completed", "multiple", FUNCTIONS, ANSWERS, lambda *_: {"valid": False}
        ).value
        == 0
    )


@pytest.mark.parametrize("value", [True, 0.5, -1, 2])
def test_only_binary_observed_values(value):
    with pytest.raises(ScoringError):
        BinaryScore(value, "fixture")


def test_source_audit_never_selects_bundled_data(monkeypatch, tmp_path):
    commands = []

    def revision(command, **kwargs):
        directory = command[2].split("\\")[-1].split("/")[-1]
        return check_strict_scorers.SOURCE_REVISIONS[directory] + "\n"

    def diff(command, **kwargs):
        commands.append(command)

    monkeypatch.setattr(check_strict_scorers.subprocess, "check_output", revision)
    monkeypatch.setattr(check_strict_scorers.subprocess, "run", diff)
    check_strict_scorers.verify_native_sources(tmp_path)
    assert len(commands) == 2
    for command in commands:
        paths = command[command.index("--") + 1 :]
        assert paths and all(path.endswith(".py") for path in paths)
        assert all("data/" not in path and "eval/" not in path for path in paths)


def test_source_revision_mismatch_stops_before_diff(monkeypatch, tmp_path):
    monkeypatch.setattr(check_strict_scorers.subprocess, "check_output", lambda *a, **kw: "wrong")
    monkeypatch.setattr(check_strict_scorers.subprocess, "run", unavailable)
    with pytest.raises(ScoringError, match="revision"):
        check_strict_scorers.verify_native_sources(tmp_path)
