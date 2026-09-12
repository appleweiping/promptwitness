# Procedure data: six families, separated inputs and references

This development API reads a **local** LongProc data directory into immutable,
closed-schema cases. It does not import the reference project, run its evaluators,
render HTML, interpret YAML, compile submitted C++, regenerate search traces, or
call a model. Existing `TaskSuitePlan` v1 records remain unchanged.

```python
from promptwitness.procedure_data import load_longproc_dataset

dataset = load_longproc_dataset("/local/LongProc/data", "path_traversal_2k")
case = dataset.cases[0]
print(case.case_id, case.output_bucket, case.digest)
# Only case.input belongs in a question. case.reference is private scoring data.
```

These APIs are on the `feat/whole-repository-alignment` development branch; do not
assume the same features exist in a published package or on `main`.

## Case contract

`ProcedureCase(family, case_id, output_bucket, input, reference, source)` validates
and deep-snapshots JSON mappings and arrays. Nested mappings are read-only and
arrays become tuples. `to_dict()` returns a detached JSON value;
`ProcedureCase.from_dict(value)` revalidates the whole contract. Unknown or missing
fields, booleans used as integers, non-finite numbers, nonportable integers, invalid
Unicode, and inconsistent derived fields are rejected with `ProcedureDataError`.

The exact outer keys are `format`, `family`, `case_id`, `output_bucket`, `input`,
`reference`, and `source`; format is `promptwitness.procedure-case/v1`. The digest
is SHA-256 of the complete canonical JSON: sorted object keys, compact separators,
UTF-8, and no Unicode normalization. It includes references and provenance. It is
an integrity identifier, not proof that a source is authentic or a reference is
correct. Python `True == 1` is not used to validate derived JSON fields.

| Family | Input fields | Private reference fields |
| --- | --- | --- |
| `countdown` | `numbers`, `target`, `min_intermediate`, `max_intermediate` | `solution`, `solution_text`, `demonstration`, `search_steps`, `num_search_tokens` |
| `path_traversal` | `edges`, `source`, `target`, `problem_description`, `context_nl`, `question_nl` | `steps`, `answer_nl` |
| `html_to_tsv` | `html`, `header`, `task_topic`, `task_description`, `filtering_instruction`, `website_id` | `tsv` |
| `tom_tracking` | `story_components`, `story`, `question` | `solution`, `answer`, `trace` |
| `travel_planning` | `problem`, `original_question`, `num_cities`, `total_days`, `constraints`, `cities`, `flights` | `ground_truth_cities`, `ground_truth_durations`, `ground_truth_plan`, `solving_procedure`, `estimated_output_tokens`, `stays` |
| `pseudo_to_code` | `pseudocode_lines` | `code_lines`, `testcases` |

Important details:

- Countdown fixes the declared local intermediate policy to integer values
  **1 through 2000 inclusive**, matching the written task rules. The supplied
  `solution` array and `solution_text` can describe different solutions; neither
  is substituted for the other. Import does not silently repair incorrect gold.
- Path edges and reference steps contain exactly `source`, `method`, and `target`.
  Methods are bus/train/plane/ferry. Each input source has at most one outgoing
  edge. Reference edges are not assumed to be valid merely because they are typed.
- HTML is exact decoded UTF-8 text, including scripts and markup, held **inert**.
  No URL is fetched and no browser is involved. Header cells are separate strings;
  TSV references remain original text, including duplicate rows and empty cells.
- ToM `trace` retains the original solution lines whose stripped text starts with
  `-`; the lines themselves are not rewritten. `answer` remains the supplied
  string array. This is a declared trace, not an independently simulated belief
  state or proof of real-world knowledge.
- Travel `constraints` retain the original ordered duration/fixed constraints.
  `cities` derives `{city, duration, fixed_start, fixed_end}` records; absent fixed
  dates are explicit nulls. `flights` are directed pairs. Reference `stays` derives
  `{city, start_day, end_day}` from the original city/duration strings, counting
  arrival and departure days inclusively, so adjacent stays share a travel day.
  A derived schedule is not automatically constraint-satisfying.
- Pseudocode, source code and test cases are data only. Each test case is
  `[stdin_lines, stdout_lines]`. Keeping this sixth family does **not** claim an
  executable-code evaluator or a safe execution sandbox.

`normalize_longproc_record(...)` is a pure adapter for a supplied native record.
For HTML its `html=` argument must be provided separately. Every native key is
required; extra keys are rejected. Native identifiers go into source provenance;
the HTML path becomes an asset inventory entry; the native output group becomes
`output_bucket`. Other native fields are retained or represented by the explicit
renamed/derived fields above. Canonical source-record hashes bind the original
record independently of those transformations. No target-specific demonstration
is inserted into `input`.

## Source identity and deterministic loading

An authored case uses `source={"kind": "authored", "label": "..."}`. A loaded
LongProc case uses these exact source fields:

```text
kind, dataset, file, file_sha256, row_index, record_id,
record_sha256, assets, prompt_sha256
```

Files are named relative to the declared data root. SHA-256 values identify exact
file bytes; `record_sha256` identifies canonical native-record JSON. `row_index`
is zero-based. Case IDs are `<dataset_name>/<row_index>`: repeated native problem
IDs and repeated references to one HTML file are preserved, not deduplicated away.
`assets` contains exactly one `{path, sha256, size}` entry for HTML and none for
other families. A deserialized HTML case must still match its asset byte count
and SHA-256. The referenced `prompts.yaml` is read and hashed as inert UTF-8, **not
parsed, interpolated or executed**; loading does not promise original prompt
equivalence.

`load_longproc_dataset(data_root, dataset_name, limits=None)` returns
`ProcedureDataset` with `name`, `family`, `output_bucket`, tuple `cases`, and an
immutable `inventory`. Files are sorted by relative path; cases retain file order.
Inventory includes file hashes/sizes/kinds, available/selected/excluded counts,
demonstration and HTML reference counts, normalized byte counts, and SHA-256 of
the concatenated 32-byte case digests in order. `case_digest_sha256` covers selected
cases; `all_record_cases_sha256` also covers excluded records (with that requested
dataset identity). The inventory is not a Merkle membership proof or a signed
manifest.

All records are structurally checked **before selection**, including records
outside travel output buckets. Travel's separate four demonstration records are
also structurally checked and their source file inventoried; absent generated
traces are not invented or sent as examples. These records do not become test
cases or silently alter prompts.

`DATASET_NAMES` includes all 16 official definitions. The `0.5k`, `2k` and `8k`
suffixes describe **output difficulty**, not measured input length, a tokenizer
identity, an input budget, or the number of tokens an implementation will produce.
Travel uses `0 <= estimated_output_tokens < 2048` for `2k`, and
`4096 <= estimated_output_tokens < 8192` for `8k`. No `travel_planning_0.5k` or
`pseudo_to_code_8k` definition is invented. Input budgets and provider generation
budgets belong to the separate execution-plan layer.

## Resource and filesystem boundaries

`ProcedureDataLimits` defaults (binary MiB) are:

| Bound | Default | Configurable ceiling |
| --- | ---: | ---: |
| One JSON/prompt file | 64 MiB | 64 MiB |
| One HTML file | 4 MiB | 4 MiB |
| One canonical case | 16 MiB | 16 MiB |
| Cumulative bytes read per dataset call | 256 MiB | 512 MiB |
| Cumulative canonical cases, including excluded rows | 256 MiB | 512 MiB |
| Records per native file | 10,000 | 100,000 |

Repeated HTML reads consume the cumulative read budget even though inventory
entries are unique by path. JSON nesting is limited to 32 levels, case JSON to
one million structural nodes (including object keys), and whole native files to
four million. Native files are parsed within their encoded-file cap; node and
record checks then inspect the parsed values. These are **not** hard RSS/CPU
guarantees. `load_longproc_dataset` retains all selected immutable cases, and
`to_dict()` explicitly allocates a detached copy. Strings are escaped/UTF-8 encoded
in 4096-codepoint chunks so checking the serialized budget does not first encode
one arbitrarily large string. One case may be materialized before the cumulative
case budget rejects it; no partial dataset is returned.

The loader only opens regular local files. Absolute paths, traversal, backslashes,
drive-qualified paths and source symlinks are rejected. HTML must remain beneath
its declared family directory. File metadata is checked around reads, repeated
paths must retain the same hash, and full audit rejects conflicting inventories
across dataset definitions. The directory is **trusted local input**, not a hostile
filesystem sandbox or an atomic filesystem snapshot: concurrent replacement races
and externally shared hard links are not comprehensively isolated. Do not modify
source files during a run. Errors omit record contents and arbitrary source paths.

No partial output files are published by this module; it performs no writes.
Cases and dataset exports contain potentially private input **and complete gold**.
Keep them private unless the source's permissions and intended use allow sharing.

## Reproducible local schema audit

```python
from promptwitness.procedure_data import audit_longproc_data

report = audit_longproc_data("/local/LongProc/data")
assert report["schema_valid"]
assert not report["reference_semantics_verified"]
assert not report["model_quality_measured"]
```

This reads all definitions, checks all referenced HTML, and returns counts and
relative file hashes, not dialogue, HTML, code, questions or reference text. It
does not retain the entire set of cases. Shared files are re-read per definition.

On 2026-09-12, the actual local audit of HELMET
`af609c4d51b97fc35012099380aa889da961c42d` / its LongProc submodule
`d91a0dd1fca8858f5a66db989f2914ae3c2df991` passed all 16 definitions:

| Family | `0.5k` selected | `2k` selected | `8k` selected |
| --- | ---: | ---: | ---: |
| Countdown | 200 | 200 | 200 |
| Path traversal | 200 | 200 | 200 |
| HTML to TSV | 100 | 189 | 120 |
| ToM tracking | 200 | 200 | 200 |
| Travel planning | not defined | 769 | 239 |
| Pseudocode to code | 199 | 200 | not defined |

The 1,600 travel rows were all validated for each definition: 831 are excluded
from `2k`, 1,361 from `8k`, and 592 fall in neither bucket. The separate four travel
demonstrations were checked. All 409 HTML references resolved to 255 unique HTML
files. The audit inventory contains 277 unique files totaling 102,306,627 bytes.
SHA-256 of the complete canonical report is
`7bdb74860d714a4c0c16d760d59933032a865de74db6e94909a1f7d7ac8d9f9d`.
The digest is reproducible with `json.dumps(report, ensure_ascii=False,
sort_keys=True, separators=(",", ":")).encode("utf-8")` followed by SHA-256;
the report intentionally contains no wall-clock timing or absolute host paths.
The [retained complete audit](../benchmarks/results/procedure-data-audit.json)
is that canonical JSON plus one terminal LF, so its file SHA-256 is
`c414cc2a938f292a263001257de3b3fc598c9d9b58dfb3ae9dd36938dd6b2d9f`.
The final source-bound rerun took 139.680 seconds on Python 3.12.13 and reproduced
the earlier canonical digest exactly. All 49 runtime Python files plus `py.typed`
were unchanged before/after; 42 actually imported modules were independently
checked. An earlier attempted binding correctly aborted when an unrelated
task-runner identity repair changed runtime source; it published no accepted
data report. This rerun, not that aborted attempt, supplies the retained audit.

Passing this audit proves the local parser and inventory accepted those bytes.
It does not prove correct gold, correct search procedures, original-evaluator
equivalence, completed code execution, or any model quality. Semantically invalid
but structurally typed references are retained for separate scoring diagnostics,
not fixed, filtered or silently counted as successes. The portable tests use
independently authored fixtures, including intentionally wrong references; no
third-party dataset rows are copied into this repository.

## Attribution

The local adapter was implemented independently against the documented data
contract. The supplied data remain third-party data: see the fixed
[LongProc repository](https://github.com/princeton-pli/LongProc), its README and
Apache-2.0 `LICENSE`, and the
[HELMET LongProc addon](https://github.com/princeton-nlp/HELMET/tree/af609c4d51b97fc35012099380aa889da961c42d/longproc_addon).
Retain the LongProc citation (Ye et al., 2025), HELMET citation (Yen et al., 2024),
and the underlying dataset attributions listed in LongProc's README: Arborist,
SPoC, Stream of Search, and NATURAL PLAN. A repository-level license should not
be presented as independent clearance for all embedded third-party webpage
content. This feature downloads or republishes none of that material.
