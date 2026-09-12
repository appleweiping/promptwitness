"""Closed six-family schemas; semantic correctness of reference answers is separate."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from .procedure_json_data import (
    ProcedureDataError,
    canonical_bytes,
    integer,
    object_fields,
    sequence,
    sha256,
    string,
)

FAMILIES = (
    "countdown",
    "path_traversal",
    "html_to_tsv",
    "tom_tracking",
    "travel_planning",
    "pseudo_to_code",
)
OUTPUT_BUCKETS = ("0.5k", "2k", "8k")
TRANSIT_METHODS = ("bus", "train", "plane", "ferry")


def relative_path(value: Any) -> str:
    result = string(value)
    parts = result.split("/")
    if (
        "\\" in result
        or ":" in result
        or "\x00" in result
        or any(part in ("", ".", "..") for part in parts)
        or PurePosixPath(result).is_absolute()
    ):
        raise ProcedureDataError("procedure source path must be relative and confined")
    return result


def strings(value: Any, *, minimum: int = 0, nonempty: bool = False) -> None:
    for item in sequence(value, minimum=minimum):
        string(item, nonempty=nonempty)


def edges(value: Any) -> None:
    for edge in sequence(value):
        object_fields(edge, "source method target")
        string(edge["source"])
        string(edge["target"])
        if edge["method"] not in TRANSIT_METHODS:
            raise ProcedureDataError("procedure transit method is unsupported")


def travel_cities(constraints: Any) -> list[dict[str, Any]]:
    durations: dict[str, int] = {}
    fixed: dict[str, tuple[int, int]] = {}
    for item in sequence(constraints, maximum=200, minimum=1):
        if not isinstance(item, Mapping):
            raise ProcedureDataError("procedure constraint must be an object")
        if item.get("type") == "duration":
            object_fields(item, "type city num_days")
            city = string(item["city"])
            if city in durations:
                raise ProcedureDataError("procedure has duplicate duration constraints")
            durations[city] = integer(item["num_days"], minimum=1, maximum=10_000)
        elif item.get("type") == "fixed":
            object_fields(item, "type city start_day end_day")
            city = string(item["city"])
            if city in fixed:
                raise ProcedureDataError("procedure has duplicate fixed constraints")
            start = integer(item["start_day"], minimum=1, maximum=10_000)
            end = integer(item["end_day"], minimum=start, maximum=10_000)
            fixed[city] = (start, end)
        else:
            raise ProcedureDataError("procedure constraint type is unsupported")
    if not set(fixed) <= set(durations):
        raise ProcedureDataError("procedure fixed constraint has no duration")
    return [
        {
            "city": city,
            "duration": duration,
            "fixed_start": fixed[city][0] if city in fixed else None,
            "fixed_end": fixed[city][1] if city in fixed else None,
        }
        for city, duration in durations.items()
    ]


def travel_stays(cities: Any, durations: Any) -> list[dict[str, Any]]:
    names = string(cities).split("**")
    days = string(durations).split("**")
    if len(names) != len(days) or len(names) > 100:
        raise ProcedureDataError("procedure reference stay arrays do not align")
    result = []
    start = 1
    for city, day in zip(names, days, strict=True):
        string(city)
        if not day.isascii() or not day.isdecimal() or len(day) > 5:
            raise ProcedureDataError("procedure reference duration is invalid")
        duration = integer(int(day), minimum=1, maximum=10_000)
        end = start + duration - 1
        result.append({"city": city, "start_day": start, "end_day": end})
        start = end
    return result


def validate_source(source: Any) -> None:
    if isinstance(source, Mapping) and source.get("kind") == "authored":
        object_fields(source, "kind label")
        string(source["label"])
        return
    source = object_fields(
        source,
        "kind dataset file file_sha256 row_index record_id record_sha256 assets prompt_sha256",
    )
    if source["kind"] != "longproc":
        raise ProcedureDataError("procedure source kind is unsupported")
    string(source["dataset"])
    relative_path(source["file"])
    sha256(source["file_sha256"])
    sha256(source["record_sha256"])
    sha256(source["prompt_sha256"])
    integer(source["row_index"])
    string(source["record_id"])
    paths = set()
    for asset in sequence(source["assets"], maximum=1):
        object_fields(asset, "path sha256 size")
        path = relative_path(asset["path"])
        if path in paths:
            raise ProcedureDataError("procedure source assets are duplicated")
        paths.add(path)
        sha256(asset["sha256"])
        integer(asset["size"])


def validate_case(family: str, inputs: Any, reference: Any) -> None:
    if family == "countdown":
        object_fields(inputs, "numbers target min_intermediate max_intermediate")
        for number in sequence(inputs["numbers"], minimum=4, maximum=4):
            integer(number, minimum=1, maximum=2000)
        integer(inputs["target"], minimum=1, maximum=2000)
        if type(inputs["min_intermediate"]) is not int or inputs["min_intermediate"] != 1:
            raise ProcedureDataError("procedure countdown policy must be positive integers")
        if type(inputs["max_intermediate"]) is not int or inputs["max_intermediate"] != 2000:
            raise ProcedureDataError("procedure countdown upper bound must be 2000 inclusive")
        object_fields(
            reference, "solution solution_text demonstration search_steps num_search_tokens"
        )
        strings(reference["solution"], minimum=1, nonempty=True)
        string(reference["solution_text"])
        string(reference["demonstration"])
        number = reference["search_steps"]
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            raise ProcedureDataError("procedure search difficulty must be finite and non-negative")
        integer(reference["num_search_tokens"])
    elif family == "path_traversal":
        object_fields(inputs, "edges source target problem_description context_nl question_nl")
        edges(inputs["edges"])
        sources = [edge["source"] for edge in inputs["edges"]]
        if len(set(sources)) != len(sources):
            raise ProcedureDataError("procedure graph requires one outgoing edge per source")
        for name in ("source", "target", "problem_description", "context_nl", "question_nl"):
            string(inputs[name])
        object_fields(reference, "steps answer_nl")
        edges(reference["steps"])
        string(reference["answer_nl"], nonempty=False)
    elif family == "html_to_tsv":
        object_fields(
            inputs, "html header task_topic task_description filtering_instruction website_id"
        )
        for name in ("html", "task_topic", "task_description", "website_id"):
            string(inputs[name])
        string(inputs["filtering_instruction"], nonempty=False)
        strings(inputs["header"], minimum=1, nonempty=True)
        if any("\t" in item or "\n" in item or "\r" in item for item in inputs["header"]):
            raise ProcedureDataError("procedure TSV header cells must be single fields")
        object_fields(reference, "tsv")
        string(reference["tsv"], nonempty=False)
    elif family == "tom_tracking":
        object_fields(inputs, "story_components story question")
        for value in inputs.values():
            string(value)
        object_fields(reference, "solution answer trace")
        string(reference["solution"])
        strings(reference["answer"])
        strings(reference["trace"])
        expected = [
            line for line in reference["solution"].splitlines() if line.strip().startswith("-")
        ]
        if canonical_bytes(expected) != canonical_bytes(reference["trace"]):
            raise ProcedureDataError("procedure belief trace differs from declared solution")
    elif family == "travel_planning":
        object_fields(
            inputs, "problem original_question num_cities total_days constraints cities flights"
        )
        string(inputs["problem"])
        string(inputs["original_question"])
        integer(inputs["num_cities"], minimum=1, maximum=100)
        integer(inputs["total_days"], minimum=1, maximum=10_000)
        expected_cities = travel_cities(inputs["constraints"])
        if canonical_bytes(expected_cities) != canonical_bytes(inputs["cities"]):
            raise ProcedureDataError("procedure cities differ from declared constraints")
        if len(expected_cities) != inputs["num_cities"]:
            raise ProcedureDataError("procedure city count differs from constraints")
        for flight in sequence(inputs["flights"], maximum=10_000):
            strings(sequence(flight, minimum=2, maximum=2), nonempty=True)
        object_fields(
            reference,
            "ground_truth_cities ground_truth_durations ground_truth_plan solving_procedure "
            "estimated_output_tokens stays",
        )
        string(reference["ground_truth_plan"])
        string(reference["solving_procedure"], nonempty=False)
        integer(reference["estimated_output_tokens"])
        stays = travel_stays(reference["ground_truth_cities"], reference["ground_truth_durations"])
        if canonical_bytes(stays) != canonical_bytes(reference["stays"]):
            raise ProcedureDataError("procedure stays differ from declared reference")
    elif family == "pseudo_to_code":
        object_fields(inputs, "pseudocode_lines")
        strings(inputs["pseudocode_lines"], minimum=1)
        object_fields(reference, "code_lines testcases")
        strings(reference["code_lines"], minimum=1)
        for case in sequence(reference["testcases"]):
            sequence(case, minimum=2, maximum=2)
            strings(case[0])
            strings(case[1])
    else:
        raise ProcedureDataError("procedure family is unsupported")
