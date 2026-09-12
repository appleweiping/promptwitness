"""Directed route and inclusive-day travel constraints, independent of gold order."""

from __future__ import annotations

import re
from collections.abc import Mapping
from itertools import pairwise
from typing import Any

from .procedure_common_score import ScoreWork, prefix_fraction, result, tagged

_DAY = re.compile(r"[1-9][0-9]{0,5}")


def _parse_edge(line: str) -> tuple[str, str, str] | None:
    if not line.startswith("From ") or not line.endswith("."):
        return None
    source, separator, rest = line[5:-1].partition(", take a ")
    method, direction, target = rest.partition(" to ")
    if not separator or not direction or not source or not target:
        return None
    if method not in {"bus", "train", "plane", "ferry"}:
        return None
    return source, method, target


def _edge(edge: Mapping[str, Any]) -> tuple[str, str, str]:
    return edge["source"], edge["method"], edge["target"]


def _path_valid(
    route: list[tuple[str, str, str]], data: Mapping[str, Any], edges: set[tuple[str, str, str]]
) -> tuple[bool, int, str | None]:
    current = data["source"]
    seen = {current}
    for index, (source, method, target) in enumerate(route):
        if current == data["target"]:
            return False, index, "route_continues_after_destination"
        if source != current:
            return False, index, "disconnected_route"
        if (source, method, target) not in edges:
            return False, index, "unknown_directed_edge"
        if target in seen:
            return False, index, "repeated_city"
        seen.add(target)
        current = target
    if current != data["target"]:
        return False, len(route), "destination_not_reached"
    return True, len(route), None


def score_path(
    data: Mapping[str, Any], reference: Mapping[str, Any], prediction: str, work: ScoreWork
) -> dict[str, Any]:
    work.use(len(data["edges"]) + len(reference["steps"]))
    names = [data["source"], data["target"]]
    names.extend(value for edge in data["edges"] for value in (edge["source"], edge["target"]))
    if any("\n" in name or "\r" in name for name in names):
        return result(
            {"output_grammar_supported": 0.0},
            None,
            None,
            unsupported={"route_valid": "multiline_city_names_not_supported"},
        )
    edges = {_edge(edge) for edge in data["edges"]}
    outgoing = {edge[0]: edge for edge in edges}
    # Construct each known edge spelling once. Re-rendering a very long source
    # name for every short invalid output line would multiply admitted work.
    spellings = {
        source: f"From {source}, take a {method} to {target}." for source, method, target in edges
    }
    expected = [_edge(edge) for edge in reference["steps"]]
    gold_valid = _path_valid(expected, data, edges)[0]
    text = tagged(prediction, "Route")
    route: list[tuple[str, str, str]] | None = [] if text is not None else None
    if text:
        lines = work.lines(text)
        work.use(len(lines))
        current = data["source"]
        for line in lines:
            line = line.strip()
            expected_edge = outgoing.get(current)
            parsed_edge: tuple[str, str, str] | None
            if expected_edge is not None and line == spellings[current]:
                parsed_edge = expected_edge
            else:
                parsed_edge = _parse_edge(line)
            if parsed_edge is None:
                route = None
                break
            if route is not None:
                route.append(parsed_edge)
                current = parsed_edge[2]
    valid, connected, reason = (
        _path_valid(route, data, edges) if route is not None else (False, 0, "route_format")
    )
    return result(
        {
            "route_valid": float(valid),
            "format_valid": float(route is not None),
            "reference_route_exact_match": float(route is not None and route == expected),
            "reference_prefix_fraction": prefix_fraction(route, expected)
            if route is not None
            else 0.0,
            "reference_valid": float(gold_valid),
        },
        "route_valid",
        [dict(source=a, method=m, target=b) for a, m, b in route] if route is not None else None,
        diagnostics={"failure": reason, "connected_steps": connected},
    )


def _parse_plan(
    text: str | None, work: ScoreWork, cities: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    if not text:
        return None
    lines = work.lines(text)
    work.use(len(lines))
    if not len(lines) % 2:
        return None
    stays: list[dict[str, Any]] = []
    flights: list[dict[str, Any]] = []
    # Typed names may contain English delimiter words. Match known names as
    # entire fields rather than interpreting parts of a city as syntax.
    arrival_names = {f"Arriving in {city} and visit {city}": city for city in cities}
    flight_actions = []
    for index, line in enumerate(lines):
        line = line.strip()
        if not line.startswith("**Day ") or not line.endswith("."):
            return None
        days, separator, action = line[6:-1].partition(":** ")
        if not separator:
            return None
        if index % 2:
            if _DAY.fullmatch(days) is None or not action.startswith("Fly from "):
                return None
            source, separator, target = action[9:].partition(" to ")
            if not separator or not source or not target:
                return None
            flights.append({"day": int(days), "source": source, "target": target})
            flight_actions.append(action)
        else:
            start, separator, end = days.partition("-")
            if not separator or _DAY.fullmatch(start) is None or _DAY.fullmatch(end) is None:
                return None
            visit, separator, duration_text = action.rpartition(" for ")
            if not separator or not duration_text.endswith(" days"):
                return None
            duration = duration_text[:-5]
            if _DAY.fullmatch(duration) is None:
                return None
            if visit.startswith("Visit "):
                city = visit[6:]
            elif index == 0 and visit in arrival_names:
                city = arrival_names[visit]
            elif index == 0 and visit.startswith("Arriving in "):
                city, separator, repeated = visit[12:].partition(" and visit ")
                if not separator or city != repeated:
                    return None
            else:
                return None
            if not city:
                return None
            stays.append(
                {
                    "city": city,
                    "start_day": int(start),
                    "end_day": int(end),
                    "duration": int(duration),
                }
            )
    for flight, action, (left, right) in zip(flights, flight_actions, pairwise(stays), strict=True):
        if action == f"Fly from {left['city']} to {right['city']}":
            flight.update(source=left["city"], target=right["city"])
    return stays, flights


def _plan_checks(
    stays: list[dict[str, Any]],
    flights: list[dict[str, Any]],
    data: Mapping[str, Any],
) -> dict[str, bool]:
    constraints = {city["city"]: city for city in data["cities"]}
    links = {tuple(link) for link in data["flights"]}
    names = [stay["city"] for stay in stays]
    cities_valid = len(names) == len(constraints) and set(names) == set(constraints)
    duration_valid = all(
        stay["city"] in constraints
        and stay["end_day"] >= stay["start_day"]
        and stay["end_day"] - stay["start_day"] + 1
        == stay["duration"]
        == constraints[stay["city"]]["duration"]
        for stay in stays
    )
    fixed_valid = all(
        stay["city"] in constraints
        and (
            constraints[stay["city"]]["fixed_start"] is None
            or (
                stay["start_day"] == constraints[stay["city"]]["fixed_start"]
                and stay["end_day"] == constraints[stay["city"]]["fixed_end"]
            )
        )
        for stay in stays
    )
    calendar_valid = (
        bool(stays)
        and stays[0]["start_day"] == 1
        and stays[-1]["end_day"] == data["total_days"]
        and all(left["end_day"] == right["start_day"] for left, right in pairwise(stays))
    )
    flights_valid = len(flights) == len(stays) - 1 and all(
        flight["source"] == left["city"]
        and flight["target"] == right["city"]
        and flight["day"] == left["end_day"] == right["start_day"]
        and (flight["source"], flight["target"]) in links
        for flight, (left, right) in zip(flights, pairwise(stays), strict=True)
    )
    return {
        "city_coverage_valid": cities_valid,
        "durations_valid": duration_valid,
        "fixed_schedules_valid": fixed_valid,
        "calendar_valid": calendar_valid,
        "direct_flights_valid": flights_valid,
    }


def score_travel(
    data: Mapping[str, Any], reference: Mapping[str, Any], prediction: str, work: ScoreWork
) -> dict[str, Any]:
    work.use(len(data["cities"]) + len(data["flights"]) + len(reference["stays"]))
    if any("\n" in city["city"] or "\r" in city["city"] for city in data["cities"]):
        return result(
            {"output_grammar_supported": 0.0},
            None,
            None,
            unsupported={"plan_valid": "multiline_city_names_not_supported"},
        )
    expected = [dict(stay) for stay in reference["stays"]]
    gold_stays = [dict(stay, duration=stay["end_day"] - stay["start_day"] + 1) for stay in expected]
    gold_flights = [
        {"source": left["city"], "target": right["city"], "day": left["end_day"]}
        for left, right in pairwise(expected)
    ]
    gold_valid = all(_plan_checks(gold_stays, gold_flights, data).values())
    parsed = _parse_plan(
        tagged(prediction, "Plan"), work, [city["city"] for city in data["cities"]]
    )
    checks = (
        _plan_checks(*parsed, data)
        if parsed is not None
        else dict.fromkeys(
            (
                "city_coverage_valid",
                "durations_valid",
                "fixed_schedules_valid",
                "calendar_valid",
                "direct_flights_valid",
            ),
            False,
        )
    )
    actual = (
        [{key: stay[key] for key in ("city", "start_day", "end_day")} for stay in parsed[0]]
        if parsed is not None
        else None
    )
    metrics = {key: float(value) for key, value in checks.items()}
    metrics.update(
        plan_valid=float(parsed is not None and all(checks.values())),
        format_valid=float(parsed is not None),
        reference_plan_exact_match=float(actual is not None and actual == expected),
        reference_prefix_fraction=prefix_fraction(actual, expected) if actual is not None else 0.0,
        reference_valid=float(gold_valid),
    )
    return result(
        metrics,
        "plan_valid",
        {"stays": parsed[0], "flights": parsed[1]} if parsed is not None else None,
        diagnostics={
            "failures": [key for key, value in checks.items() if not value]
            if parsed is not None
            else ["plan_format"]
        },
        unsupported={"search_procedure_correctness": "search_trace_semantics_not_implemented"},
    )
