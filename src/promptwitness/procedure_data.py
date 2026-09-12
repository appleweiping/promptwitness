"""Immutable six-family procedure cases and local, inert LongProc ingestion.

The adapter reads supplied files, not the reference project's Python code. It does
not render HTML, regenerate search traces, compile code, or call providers.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import _freeze_json, _thaw_json
from .procedure_json_data import (
    MAX_CASE_BYTES,
    ProcedureDataError,
    canonical_bytes,
    integer,
    json_digest,
    load_json,
    object_fields,
    sequence,
    sha256,
    string,
)
from .procedure_schema_data import (
    FAMILIES,
    OUTPUT_BUCKETS,
    relative_path,
    travel_cities,
    travel_stays,
    validate_case,
    validate_source,
)

CASE_FORMAT = "promptwitness.procedure-case/v1"
DATASET_NAMES = tuple(
    f"{family}_{bucket}"
    for family in FAMILIES
    for bucket in OUTPUT_BUCKETS
    if not (family == "pseudo_to_code" and bucket == "8k")
    and not (family == "travel_planning" and bucket == "0.5k")
)


@dataclass(frozen=True)
class ProcedureCase:
    """Closed input/reference separation; hashes are integrity, not authenticity."""

    family: str
    case_id: str
    output_bucket: str
    input: Mapping[str, Any]
    reference: Mapping[str, Any]
    source: Mapping[str, Any]

    def __post_init__(self) -> None:
        canonical_bytes(self._raw())
        if self.family not in FAMILIES or self.output_bucket not in OUTPUT_BUCKETS:
            raise ProcedureDataError("procedure family or output bucket is unsupported")
        string(self.case_id)
        validate_case(self.family, self.input, self.reference)
        validate_source(self.source)
        if self.source["kind"] == "longproc":
            if self.source["dataset"] != f"{self.family}_{self.output_bucket}":
                raise ProcedureDataError("procedure source dataset disagrees with case")
            assets = self.source["assets"]
            if self.family == "html_to_tsv":
                if len(assets) != 1:
                    raise ProcedureDataError("procedure HTML case requires one source asset")
                raw_html = self.input["html"].encode("utf-8")
                if (assets[0]["sha256"], assets[0]["size"]) != (
                    hashlib.sha256(raw_html).hexdigest(),
                    len(raw_html),
                ):
                    raise ProcedureDataError("procedure HTML differs from source asset")
            elif assets:
                raise ProcedureDataError("procedure non-HTML case cannot have source assets")
        for field in ("input", "reference", "source"):
            object.__setattr__(self, field, _freeze_json(getattr(self, field), "procedure case"))

    def _raw(self) -> dict[str, Any]:
        return {
            "format": CASE_FORMAT,
            "family": self.family,
            "case_id": self.case_id,
            "output_bucket": self.output_bucket,
            "input": self.input,
            "reference": self.reference,
            "source": self.source,
        }

    def to_dict(self) -> dict[str, Any]:
        return dict(_thaw_json(self._raw()))

    @classmethod
    def from_dict(cls, value: Any) -> ProcedureCase:
        canonical_bytes(value)
        data = object_fields(value, "format family case_id output_bucket input reference source")
        if data["format"] != CASE_FORMAT:
            raise ProcedureDataError("procedure case format is unsupported")
        return cls(
            *(
                data[key]
                for key in ("family", "case_id", "output_bucket", "input", "reference", "source")
            )
        )

    @property
    def digest(self) -> str:
        return json_digest(self._raw())


@dataclass(frozen=True)
class ProcedureDataLimits:
    """Encoded file/case budgets, not a hard interpreter RSS limit."""

    max_file_bytes: int = 64 * 1024 * 1024
    max_html_bytes: int = 4 * 1024 * 1024
    max_case_bytes: int = MAX_CASE_BYTES
    max_total_read_bytes: int = 256 * 1024 * 1024
    max_total_case_bytes: int = 256 * 1024 * 1024
    max_records: int = 10_000

    def __post_init__(self) -> None:
        for name, ceiling in (
            ("max_file_bytes", 64 * 1024 * 1024),
            ("max_html_bytes", 4 * 1024 * 1024),
            ("max_case_bytes", MAX_CASE_BYTES),
            ("max_total_read_bytes", 512 * 1024 * 1024),
            ("max_total_case_bytes", 512 * 1024 * 1024),
            ("max_records", 100_000),
        ):
            integer(getattr(self, name), minimum=1, maximum=ceiling)


@dataclass(frozen=True)
class ProcedureDataset:
    name: str
    family: str
    output_bucket: str
    cases: tuple[ProcedureCase, ...]
    inventory: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.name not in DATASET_NAMES or self.name != f"{self.family}_{self.output_bucket}":
            raise ProcedureDataError("procedure dataset identity is unsupported")
        sequence(self.cases)
        identities = set()
        for case in self.cases:
            if not isinstance(case, ProcedureCase) or (case.family, case.output_bucket) != (
                self.family,
                self.output_bucket,
            ):
                raise ProcedureDataError("procedure dataset contains a mismatched case")
            if case.case_id in identities:
                raise ProcedureDataError("procedure dataset case IDs are duplicated")
            identities.add(case.case_id)
        canonical_bytes(self.inventory)
        inventory = object_fields(
            self.inventory,
            "files available selected excluded demonstrations html_references html_unique_files "
            "case_digest_sha256 all_record_cases_sha256 read_bytes normalized_case_bytes",
        )
        for name in (
            "available",
            "selected",
            "excluded",
            "demonstrations",
            "html_references",
            "html_unique_files",
            "read_bytes",
            "normalized_case_bytes",
        ):
            integer(inventory[name])
        if inventory["selected"] != len(self.cases) or (
            inventory["selected"] + inventory["excluded"] != inventory["available"]
        ):
            raise ProcedureDataError("procedure inventory counts disagree with cases")
        sha256(inventory["case_digest_sha256"])
        sha256(inventory["all_record_cases_sha256"])
        hash_chain = hashlib.sha256()
        for case in self.cases:
            hash_chain.update(bytes.fromhex(case.digest))
        if hash_chain.hexdigest() != inventory["case_digest_sha256"]:
            raise ProcedureDataError("procedure inventory digest disagrees with cases")
        paths = []
        for item in sequence(inventory["files"]):
            object_fields(item, "path sha256 size kind")
            paths.append(relative_path(item["path"]))
            sha256(item["sha256"])
            integer(item["size"])
            if item["kind"] not in ("records", "prompt", "html", "demonstrations"):
                raise ProcedureDataError("procedure inventory file kind is unsupported")
        if paths != sorted(set(paths)):
            raise ProcedureDataError("procedure inventory files must be sorted and unique")
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "inventory", _freeze_json(self.inventory, "procedure inventory"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "promptwitness.procedure-dataset/v1",
            "name": self.name,
            "family": self.family,
            "output_bucket": self.output_bucket,
            "cases": [case.to_dict() for case in self.cases],
            "inventory": _thaw_json(self.inventory),
        }


def _edges(value: Any) -> list[dict[str, Any]]:
    result = []
    for edge in sequence(value):
        object_fields(edge, "src dst transit")
        result.append({"source": edge["src"], "method": edge["transit"], "target": edge["dst"]})
    return result


def normalize_longproc_record(
    family: str,
    record: Any,
    *,
    case_id: str,
    output_bucket: str,
    source: Mapping[str, Any],
    html: str | None = None,
) -> ProcedureCase:
    """Adapt every declared native field; keep semantic bad gold for audit.

    The caller supplies the inert, already source-verified HTML string. No URL,
    filesystem path, format template or executable content is evaluated here.
    """
    canonical_bytes(record)
    inputs: dict[str, Any]
    reference: dict[str, Any]
    if family == "countdown":
        data = object_fields(
            record,
            "nums target solution search_steps demonstration solution_text num_search_tokens",
        )
        inputs = {
            "numbers": data["nums"],
            "target": data["target"],
            "min_intermediate": 1,
            "max_intermediate": 2000,
        }
        reference = {
            key: data[key]
            for key in (
                "solution",
                "search_steps",
                "demonstration",
                "solution_text",
                "num_search_tokens",
            )
        }
    elif family == "path_traversal":
        data = object_fields(
            record,
            "problem_description context_repr question_repr answer_repr "
            "context_nl question_nl answer_nl",
        )
        question = sequence(data["question_repr"], minimum=2, maximum=2)
        inputs = {
            "edges": _edges(data["context_repr"]),
            "source": question[0],
            "target": question[1],
            **{key: data[key] for key in ("problem_description", "context_nl", "question_nl")},
        }
        reference = {"steps": _edges(data["answer_repr"]), "answer_nl": data["answer_nl"]}
    elif family == "html_to_tsv":
        data = object_fields(
            record,
            "output_length_group task_id website_id html_path task_topic task_description "
            "gt tsv_header filtering_instruction",
        )
        if data["output_length_group"] != output_bucket:
            raise ProcedureDataError("procedure HTML output bucket disagrees with dataset")
        string(data["task_id"])
        relative_path(data["html_path"])
        string(html)
        inputs = {
            "html": html,
            "header": string(data["tsv_header"]).split("\t"),
            **{
                key: data[key]
                for key in ("task_topic", "task_description", "filtering_instruction", "website_id")
            },
        }
        reference = {"tsv": data["gt"]}
    elif family == "tom_tracking":
        data = object_fields(record, "story_components story question solution answer")
        inputs = {key: data[key] for key in ("story_components", "story", "question")}
        solution = string(data["solution"])
        reference = {
            "solution": solution,
            "answer": data["answer"],
            "trace": [line for line in solution.splitlines() if line.strip().startswith("-")],
        }
    elif family == "travel_planning":
        data = object_fields(
            record,
            "ground_truth_cities ground_truth_durations num_cities total_days constraints "
            "connected_cities original_question_text disambig_question_text ground_truth_plan "
            "id estimated_output_tokens solving_procedure",
        )
        string(data["id"])
        inputs = {
            "problem": data["disambig_question_text"],
            "original_question": data["original_question_text"],
            "num_cities": data["num_cities"],
            "total_days": data["total_days"],
            "constraints": data["constraints"],
            "cities": travel_cities(data["constraints"]),
            "flights": data["connected_cities"],
        }
        reference = {
            key: data[key]
            for key in (
                "ground_truth_cities",
                "ground_truth_durations",
                "ground_truth_plan",
                "solving_procedure",
                "estimated_output_tokens",
            )
        }
        reference["stays"] = travel_stays(
            data["ground_truth_cities"], data["ground_truth_durations"]
        )
    elif family == "pseudo_to_code":
        data = object_fields(record, "problem_id pseudocode_lines code_lines testcases")
        string(data["problem_id"])
        inputs = {"pseudocode_lines": data["pseudocode_lines"]}
        reference = {"code_lines": data["code_lines"], "testcases": data["testcases"]}
    else:
        raise ProcedureDataError("procedure family is unsupported")
    return ProcedureCase(family, case_id, output_bucket, inputs, reference, source)


class _Reader:
    def __init__(self, root: Path, limits: ProcedureDataLimits) -> None:
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ProcedureDataError("procedure data root must be a directory")
        self.limits = limits
        self.files: dict[str, dict[str, Any]] = {}
        self.read_bytes = 0

    def read(self, name: str, kind: str) -> bytes:
        relative_path(name)
        path = self.root
        for part in name.split("/"):
            path = path / part
            if path.is_symlink():
                raise ProcedureDataError("procedure source symlinks are not accepted")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(self.root):
            raise ProcedureDataError("procedure source escaped data root")
        if not stat.S_ISREG(resolved.stat().st_mode):
            raise ProcedureDataError("procedure input must be a regular file")
        maximum = self.limits.max_html_bytes if kind == "html" else self.limits.max_file_bytes
        with resolved.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ProcedureDataError("procedure input must be a regular file")
            if (
                before.st_size > maximum
                or self.read_bytes + before.st_size > self.limits.max_total_read_bytes
            ):
                raise ProcedureDataError("procedure input exceeds read budget")
            remaining = min(maximum, self.limits.max_total_read_bytes - self.read_bytes)
            raw = handle.read(remaining + 1)
            after = os.fstat(handle.fileno())
            if len(raw) > remaining:
                raise ProcedureDataError("procedure input exceeds read budget")
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ProcedureDataError("procedure source changed during read")
        self.read_bytes += len(raw)
        item = {
            "path": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
            "kind": kind,
        }
        if name in self.files and self.files[name] != item:
            raise ProcedureDataError("procedure source changed between references")
        self.files[name] = item
        return raw

    def json(self, name: str, kind: str = "records") -> Any:
        return load_json(self.read(name, kind), limit=self.limits.max_file_bytes)


def _load(
    data_root: str | Path, dataset_name: str, limits: ProcedureDataLimits, *, retain: bool
) -> tuple[ProcedureDataset | None, dict[str, Any]]:
    if not isinstance(dataset_name, str) or dataset_name not in DATASET_NAMES:
        raise ProcedureDataError("procedure dataset name is unsupported")
    if not isinstance(data_root, (str, Path)):
        raise ProcedureDataError("procedure data root must be a local path")
    family, bucket = dataset_name.rsplit("_", 1)
    try:
        reader = _Reader(Path(data_root), limits)
        filename = f"{family}/{dataset_name}.json"
        if family == "travel_planning":
            filename = f"{family}/travel_planning_all.json"
        rows = sequence(reader.json(filename), maximum=limits.max_records)
        prompt_name = f"{family}/prompts.yaml"
        reader.read(prompt_name, "prompt").decode("utf-8")
        demonstrations = 0
        if family == "travel_planning":
            demo_rows = sequence(
                reader.json(f"{family}/travel_planning_icl_examples.json", "demonstrations"),
                maximum=limits.max_records,
            )
            for index, demo in enumerate(demo_rows):
                object_fields(
                    demo,
                    "ground_truth_cities ground_truth_durations num_cities total_days constraints "
                    "connected_cities original_question_text disambig_question_text "
                    "ground_truth_plan id",
                )
                normalize_longproc_record(
                    family,
                    {**demo, "estimated_output_tokens": 0, "solving_procedure": ""},
                    case_id=f"demonstration/{index}",
                    output_bucket=bucket,
                    source={"kind": "authored", "label": "demonstration structural check only"},
                )
            demonstrations = len(demo_rows)
        cases = []
        selected = html_references = total_case_bytes = 0
        case_hashes = hashlib.sha256()
        all_hashes = hashlib.sha256()
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ProcedureDataError("procedure source record must be an object")
            canonical_bytes(row, limit=limits.max_case_bytes)
            source = {
                "kind": "longproc",
                "dataset": dataset_name,
                "file": filename,
                "file_sha256": reader.files[filename]["sha256"],
                "row_index": index,
                "record_id": str(row.get("task_id", row.get("problem_id", row.get("id", index)))),
                "record_sha256": json_digest(row),
                "assets": [],
                "prompt_sha256": reader.files[prompt_name]["sha256"],
            }
            html = None
            if family == "html_to_tsv":
                html_name = f"{family}/{relative_path(row.get('html_path'))}"
                html = reader.read(html_name, "html").decode("utf-8")
                source["assets"] = [
                    {key: reader.files[html_name][key] for key in ("path", "sha256", "size")}
                ]
                html_references += 1
            case = normalize_longproc_record(
                family,
                row,
                case_id=f"{dataset_name}/{index}",
                output_bucket=bucket,
                source=source,
                html=html,
            )
            encoded = canonical_bytes(case._raw(), limit=limits.max_case_bytes)
            total_case_bytes += len(encoded)
            if total_case_bytes > limits.max_total_case_bytes:
                raise ProcedureDataError("procedure cases exceed cumulative byte budget")
            case_digest = hashlib.sha256(encoded).digest()
            all_hashes.update(case_digest)
            eligible = True
            if family == "travel_planning":
                estimate = integer(row["estimated_output_tokens"])
                low, high = (0, 2048) if bucket == "2k" else (4096, 8192)
                eligible = low <= estimate < high
            if eligible:
                selected += 1
                case_hashes.update(case_digest)
                if retain:
                    cases.append(case)
        inventory = {
            "files": sorted(reader.files.values(), key=lambda item: item["path"]),
            "available": len(rows),
            "selected": selected,
            "excluded": len(rows) - selected,
            "demonstrations": demonstrations,
            "html_references": html_references,
            "html_unique_files": sum(item["kind"] == "html" for item in reader.files.values()),
            "case_digest_sha256": case_hashes.hexdigest(),
            "all_record_cases_sha256": all_hashes.hexdigest(),
            "read_bytes": reader.read_bytes,
            "normalized_case_bytes": total_case_bytes,
        }
        result = (
            ProcedureDataset(dataset_name, family, bucket, tuple(cases), inventory)
            if retain
            else None
        )
        return result, inventory
    except (OSError, UnicodeError):
        raise ProcedureDataError("procedure source files are unavailable or not UTF-8") from None


def load_longproc_dataset(
    data_root: str | Path,
    dataset_name: str,
    *,
    limits: ProcedureDataLimits | None = None,
) -> ProcedureDataset:
    """Validate every record, then select the named official output-difficulty bucket."""
    actual = limits if limits is not None else ProcedureDataLimits()
    if not isinstance(actual, ProcedureDataLimits):
        raise ProcedureDataError("procedure limits must be ProcedureDataLimits")
    result, _ = _load(data_root, dataset_name, actual, retain=True)
    assert result is not None  # retain=True is local control flow, not untrusted input.
    return result


def audit_longproc_data(
    data_root: str | Path, *, limits: ProcedureDataLimits | None = None
) -> dict[str, Any]:
    """Read all 16 definitions without retaining cases; report hashes and counts only.

    Shared files may be read repeatedly. This verifies schema/inventory, not model
    quality, official evaluator equivalence, reference semantics or authenticity.
    """
    actual = limits if limits is not None else ProcedureDataLimits()
    if not isinstance(actual, ProcedureDataLimits):
        raise ProcedureDataError("procedure limits must be ProcedureDataLimits")
    datasets = []
    files: dict[str, Any] = {}
    for name in DATASET_NAMES:
        _, inventory = _load(data_root, name, actual, retain=False)
        datasets.append({"name": name, **inventory})
        for item in inventory["files"]:
            if item["path"] in files and files[item["path"]] != item:
                raise ProcedureDataError("procedure source changed between datasets")
            files[item["path"]] = item
    return {
        "format": "promptwitness.procedure-data-audit/v1",
        "families": list(FAMILIES),
        "datasets": datasets,
        "files": sorted(files.values(), key=lambda item: item["path"]),
        "schema_valid": True,
        "reference_semantics_verified": False,
        "model_quality_measured": False,
    }
