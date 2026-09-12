# Safe procedural scoring

This module supplies original, deterministic local scoring definitions for six
procedure families. Five have supported primary metrics. `pseudo_to_code` does
**not** have an executable-correctness score: generated code is never compiled,
executed, imported, sent to a shell, or sent to a model. An isolated execution
sandbox and semantic search-trace verification remain unimplemented. This is not
full LongProc evaluator parity or evidence of model quality.

Use the development checkout containing `procedure_data.py` and
`procedure_scores.py`; older releases do not expose these modules. No additional
dependencies are required. Dataset loading and input/reference separation are
described in [procedure data](procedure-data.md).

```python
from promptwitness.procedure_data import ProcedureCase
from promptwitness.procedure_scores import score_procedure

case = ProcedureCase(
    family="countdown",
    case_id="authored-example",
    output_bucket="2k",  # A difficulty label, not a measured token count.
    input={
        "numbers": [2, 2, 3, 4],
        "target": 24,
        "min_intermediate": 1,
        "max_intermediate": 2000,
    },
    reference={
        "solution": ["2 + 2 = 4", "4 + 4 = 8", "8 * 3 = 24"],
        "solution_text": "2 + 2 = 4\n4 + 4 = 8\n8 * 3 = 24",
        "demonstration": "Authored illustration, not a model output.",
        "search_steps": 0,
        "num_search_tokens": 0,
    },
    source={"kind": "authored", "label": "Documentation example"},
)
report = score_procedure(case, "<Solution>\n2 + 2 = 4\n4 + 4 = 8\n8 * 3 = 24\n</Solution>")
assert report["primary_score"] == 1.0
```

`score_procedure(case, prediction, *, limits=None)` requires a validated immutable
`ProcedureCase` and text. It does no I/O. For provider responses, the caller must
first enforce completion, for example with `task_scores.response_text`; this
function cannot infer whether an arbitrary string was truncated by its producer.
Input, reference and prediction are not mutated.

The JSON-compatible report includes the existing task-score fields `metrics`,
`primary_metric`, `primary_score`, `score_status`, `unsupported_metrics` and
`processed`, plus `scorer_version`, `score_limits`, `work`, and fixed-code
`diagnostics`. `PROCEDURE_SCORER_VERSION` is
`promptwitness.procedure-scores/v1`. All reported metrics are in [0, 1].
`score_status="scored"` means only the primary metric is supported, not that
every metric for the task has been implemented. Unsupported primary scores are
`None`, never a misleading zero or a proxy score. Reports may contain extracted
prediction text; they do not insert reference text into the prediction or errors.

## Definitions and accepted output

| Family | Primary metric | Separate limits/diagnostics |
| --- | --- | --- |
| `countdown` | `full_solution_valid` | Exact integer transitions, operand multiplicities, full reduction, target; valid-step prefix and reference equality are separate. Search-procedure semantics are unsupported. |
| `path_traversal` | `route_valid` | Directed typed edges, connection, final destination, no repeated city or steps after arrival. Reference prefix is not full-route success. |
| `html_to_tsv` | `row_f1` | Exact header and arity, exact-cell row multisets, duplicate multiplicity, no rendering or formula evaluation. |
| `tom_tracking` | `declared_trace_exact_match` | Ordered supplied bullet-trace agreement only; no general belief/entailment judge. |
| `travel_planning` | `plan_valid` | All city, inclusive-day, fixed-schedule and directed-flight constraints; a different valid order can succeed. Search-procedure semantics are unsupported. |
| `pseudo_to_code` | `None` | Only a code-fence presence diagnostic; `executable_test_pass` is explicitly unsupported. |

### Countdown

Exactly one ordered `<Solution>...</Solution>` pair contains three equations for
the four supplied numbers. Each equation is a closed `a OP b = c` form with
ASCII nonnegative decimal integers (no leading zeroes except `0`) of at most ten
digits, and `OP` is one of `+ - * /`. Whitespace is allowed between tokens.
Parentheses, unary signs, powers, floor division, decimals, names, function calls
and arbitrary expressions are not accepted. Neither Python `eval` nor an AST
execution facility is used.

Every step removes its two operands from a multiset and adds the verified result.
Using the same value twice requires two available copies. Division must be exact
and nonzero; every produced value must be in the explicit inclusive range
1..2000. The final inventory must have one value equal to the target. A target
reached early with unused operands is not a complete solution. Correct alternative
reductions can score 1 even when `reference_solution_exact_match` is 0.
`valid_step_fraction` counts successful transitions before the first violation,
divided by three; it is never substituted for `full_solution_valid`.

The local policy follows the task's written positive/inclusive rule. The reviewed
reference search admits zero and rejects 2000, while its final-solution evaluator
does not enforce the same bounds. These differences are intentional and are not
silently hidden in a claimed reference-equivalent score. A structurally valid but
semantically invalid supplied reference is retained and reported by
`reference_valid=0`; constraint-valid predictions can still succeed.

### Paths

Exactly one `<Route>...</Route>` pair contains one edge per line:

```text
<Route>
From A, take a bus to B.
From B, take a train to C.
</Route>
```

Supported methods are `bus`, `train`, `plane`, and `ferry`. The case declares one
outgoing edge per source, including the method; there is no implicit reverse edge
or invented adjacent connection. Known city names are matched as complete typed
fields, even when they contain grammar words. The route must start at the case
source, use connected declared edges, stop on first arrival at the destination,
and not revisit a city. When source equals destination, the empty tagged route is
the valid zero-edge solution.

City names containing LF or CR cannot be represented by this line-oriented output
grammar. Cases containing such names explicitly yield an unsupported primary,
not an accuracy zero; this same restriction applies to requested travel cities.

`reference_prefix_fraction` is the number of initial matching typed edges divided
by the number of reference edges. A matching prefix followed by extra steps may
have prefix 1 but has primary 0. A wrong first edge has prefix 0, not an off-by-one
partial credit. `reference_valid` also checks the declared reference route.

### TSV

Accept raw TSV or exactly one closed `tsv` code fence (only whitespace outside the
fence). LF and CRLF are newline aliases. The first row must exactly match the
declared header in order and spelling. Every subsequent row must have the same
arity; no incomplete trailing row is dropped, padded, or repaired. A local
four-state parser supports quoted fields, tabs/newlines inside quotes, and `""`
as an escaped quote. Quotes inside an unquoted field remain literal text. A
closing quoted field may only be followed by a delimiter, newline, or EOF.
Lone CR outside quotes and unclosed quoting are invalid. An internal blank line
is a row containing one empty field; a final newline alone adds no extra row.
The implementation neither reads nor changes Python's global CSV field limit.

Cells remain exact Unicode strings: no case folding, article deletion,
punctuation stripping, number conversion or spreadsheet formula evaluation.
Row order is ignored, but multiplicity matters. If gold has rows `a,a,b` and a
prediction has `a,b,b,c`, multiset intersection is two rows, precision is 2/4,
recall is 2/3, and F1 is 4/7. In general F1 is `2*intersection/(gold+predicted)`.
Two header-only tables score 1; a nonempty prediction against an empty gold table
scores F1 0. Wrong headers or malformed prediction rows score 0. If the declared
gold table is itself malformed or disagrees with the declared header, primary
scoring is unsupported (`invalid_declared_reference`), not a model failure.

HTML and formulas are inert strings; this module never parses a browser DOM,
fetches URLs, opens external resources or runs scripts. Publishing extracted
cells into a spreadsheet later requires that application's own formula-injection
protections; inert scoring does not sanitize an export for another application.

### Declared ToM traces

Lines whose stripped text starts with `-` form the supplied trace. Remove that
first marker and collapse whitespace runs; preserve case, punctuation,
prepositions, negation and all other semantic tokens. The full ordered bullet
sequence and its length must agree for primary 1. Prose outside bullet lines is
not judged, including any separate final answer. An empty bullet is invalid, and
an empty reference trace yields an unsupported primary. Prefix agreement is
reported independently and may be 1 for an otherwise matching trace with extra
bullets. This checks the *declared trace*, not whether a story implies the trace
or a person's beliefs. Its normalization is deliberately stricter than the
reviewed reference's extensive word deletion.

### Travel plans

Exactly one `<Plan>...</Plan>` pair alternates visit and flight lines, beginning
and ending with visits:

```text
<Plan>
**Day 1-2:** Arriving in A and visit A for 2 days.
**Day 2:** Fly from A to B.
**Day 2-4:** Visit B for 3 days.
</Plan>
```

The first visit may also use `Visit`. An `Arriving in X and visit Y` line requires
the same whole city name twice and is only allowed as the first visit. Day values
are positive ASCII decimal integers with at most six digits. Every requested city
must occur exactly once, with no unknown cities, omissions, or extra suffix.
Check the declared duration against both the inclusive interval `end-start+1`
and the case requirement. Fixed dates must match exactly. The plan starts at day
1 and ends on `total_days`; each next arrival day equals the previous departure
day, so that shared travel days count toward both stays. Each intervening flight
must name exactly those two adjacent cities, on that shared day, and must occur
as a directed connection in the case. One-city plans need no flight. There is no
restriction on the first city from an external starting point.

The primary is the conjunction of all five named constraint diagnostics:
`city_coverage_valid`, `durations_valid`, `fixed_schedules_valid`, `calendar_valid`,
and `direct_flights_valid`. A valid alternative itinerary can have primary 1 and
`reference_plan_exact_match=0`. Reference exactness compares the complete list of
city/start/end triples, not a prefix. `reference_valid` independently checks the
typed reference stays against the case; it does not certify the reference's
search procedure.

## Admission and computational scope

`ProcedureScoreLimits` is immutable and strictly typed: booleans are not integers,
and zero, negative, or larger-than-hard-ceiling budgets are rejected.

| Budget | Default | Hard ceiling |
| --- | ---: | ---: |
| `max_prediction_bytes` | 1 MiB | 4 MiB |
| `max_case_bytes` | 16 MiB | 32 MiB |
| `max_line_bytes` | 64 KiB | 256 KiB |
| `max_lines` | 20,000 | 100,000 |
| `max_work_items` | 500,000 | 2,000,000 |

The complete prediction is admitted before extracting tags. `max_case_bytes`
counts aggregate UTF-8 text/key bytes in scoring input and reference, not the
canonical artifact envelope or provider token count; the data contract separately
bounds its full JSON representation. Encodings are checked in small chunks.
Aggregate work includes input/reference tree nodes and keys, pending children,
and parsed rows/fields/edges/transitions. TSV reserves a conservative delimiter
upper bound before allocating cell objects. Decoded multiline cells have their
own byte check before accumulating oversized content.

Line counts are aggregate over parsing passes: full output admission plus each
extracted section or reference table parsed. Thus a tagged body or table may
count more than once. Limits are declared, reproducible work-admission policies,
not exact token budgets, wall-clock timeouts, memory isolation, or RSS guarantees.
Source/provenance allocation before the scoring call remains the caller/data
loader's responsibility. A `ProcedureScoreLimitError` aborts the score; callers
must retain it as an evaluation limitation, not count it as a correctness zero.

There are no Cartesian row comparisons, graph search, route enumeration or
unbounded arithmetic trees. Expected hash-table operations give approximately
linear work in admitted text, edges, cells and steps; Countdown arithmetic uses
at most ten-digit operands and only three transitions. Typed edge spellings are
built once, not reconstructed for every rejected prediction line. Memory is
bounded by the admitted case/output plus their parsed strings and containers,
with Python-object overhead beyond serialized byte counts.

## Independent tests and fixed-source self-consistency audit

`tests/test_procedure_scores.py` uses authored hand-math cases, exact Fraction
expectations, individual constraint mutations, duplicate counts, Unicode city
names, malformed output, and exact-budget/one-less controls. It does not import
or execute reference evaluators. A separate local read-only audit on 2026-09-12
checked the stored LongProc answers using these original semantic validators;
that audit is **reference self-consistency**, not held-out model performance.

Research provenance was the local HELMET snapshot
`af609c4d51b97fc35012099380aa889da961c42d` and its LongProc snapshot
`d91a0dd1fca8858f5a66db989f2914ae3c2df991`. Relevant inspected files were
`longproc_data.py`, `countdown_evaluator.py`, `travel_planning_evaluator.py`,
`html_to_tsv_evaluator.py`, `tom_tracking_evaluator.py`, `spoc_evaluator.py`, and
the family prompt definitions. No reference implementation was copied, imported,
executed, or substituted for the independent test oracles.

Audit outcomes:

- All 600 path records (200 per difficulty bucket) passed both the typed-edge
  reference check and complete parsing/validation of stored `answer_nl`.
- All 1,600 travel records, including 592 outside the two selected difficulty
  buckets, and all four ICL examples passed typed-stay constraints and complete
  stored-plan validation. ICL examples have a smaller original schema; their
  actual constraints and stays were audited directly without inventing missing
  generated traces or metadata.
- Countdown's supplied `solution` arrays passed 595/600; `solution_text` passed
  593/600. These fields can contain different valid reductions and are not
  required to be identical. Every failed field below produced intermediate zero,
  violating the explicitly chosen positive-result policy. No source values were
  rewritten, and successful alternative predictions remain eligible for primary 1.

| Bucket / zero-based row | Invalid supplied field(s) | Failure |
| --- | --- | --- |
| `countdown_0.5k/81` | `solution` | `intermediate_out_of_range`: 0 |
| `countdown_2k/39`, `/68`, `/148` | `solution_text` | same |
| `countdown_2k/172` | both | same |
| `countdown_8k/1`, `/136`, `/155` | both | same |

Exact audited file SHA-256 values (relative to the supplied LongProc data root):

```text
countdown/countdown_0.5k.json  2f0d1095158bcd87e955970cdabbff217dd1493b3e3d4a6439ffe631008965b8
countdown/countdown_2k.json   5dbc0320e3fc55ba5627b26a90f8c64a6f0bc44e350b065c6c5da620f9675a43
countdown/countdown_8k.json   d026d94ffdc3669396502c8586b812c2f9944365ca538c7ad5dfaad72613938f
path_traversal/path_traversal_0.5k.json  8abd1e2f437b8799d697f33cddbbfdbb1eb80ebe9acbb4736bd5de5d8a28fcb7
path_traversal/path_traversal_2k.json    100ca7c5215f275fd2d8686e95616af2a2edcfaa6e713f262b6f90f086322d61
path_traversal/path_traversal_8k.json    39ae8ec2adc0cc3ab1e3cdd884a4c6783c88ed5b0cb0312382368ca55c39b0a4
travel_planning/travel_planning_all.json  e7d07be7186ea3d964773c872a9116d4eb981ebb3bd4d77826b5c4ba51a2bf5a
travel_planning/travel_planning_icl_examples.json  f8a68e4216d0fac50bafe01dc230170621a554ebfb043adbd0afd8f0271cf8b8
```

No raw dataset rows, HTML, vocabulary, generated code or dialogue are redistributed
by this audit section. Hashes identify observed inputs; they neither authenticate
their publisher nor grant permission to redistribute third-party content.
