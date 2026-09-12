# Input-only procedural baselines

These development-branch algorithms consume typed problem inputs, not reference
answers. They are small, original deterministic solvers, not language models or a
reproduction of the reference project's executable evaluators. Their performance
does not establish instruction-following quality, six-family completion, or whole
repository parity.

## API and semantics

```python
from promptwitness.procedure_baselines import (
    ProcedureBaselineLimits,
    solve_countdown,
    walk_graph_path,
)

result = solve_countdown(
    {
        "numbers": [1, 2, 3, 4],
        "target": 10,
        "min_intermediate": 1,
        "max_intermediate": 2000,
    },
    limits=ProcedureBaselineLimits(),
)
assert result.status == "solved"
assert result.steps == ("1 + 2 = 3", "3 + 3 = 6", "4 + 6 = 10")
```

Both functions accept only a closed input mapping. They reject a `ProcedureCase`
or a mapping containing `reference`, `source`, or extra fields. When using the
dataset adapter, pass `case.input`, never `case`.

`ProcedureBaselineResult` is immutable. It contains an algorithm version,
canonical input SHA-256, admitted states/transitions, and one of:

- `solved`: complete steps and a `<Solution>` or `<Route>` answer.
- `unsatisfiable`: an exhausted *complete* search or directed dead-end/cycle proof.
- `exhausted`: a configured work or output budget prevented completion; this is
  unresolved, not evidence of unsatisfiability. No partial answer is published.
- `unsupported`: a graph city cannot be represented by the single-line/tagged
  route answer format. No edge is silently renamed or discarded.

Countdown requires four positive integer operand occurrences and target in
`1..2000`, with that exact inclusive intermediate-value policy. Every operation
consumes two occurrences and produces one; all four occurrences must ultimately
be consumed in three steps. The deterministic DFS sorts the multiset, enumerates
operand positions in ascending order, and tries addition, multiplication, both
subtraction orders, and both division orders. Division must be exact. Zero,
negative, fractional and above-2000 intermediate values are rejected. Dead
multisets are memoized only after their entire search completes. The memo is sound
because future legality depends on the remaining numeric multiplicities, not on
how they were produced. There is no `eval`, arbitrary-expression interpreter,
reference import, or code execution.

The graph contract has `edges`, `source`, `target`, `problem_description`,
`context_nl`, and `question_nl`. Edges have exactly `source`, `method`, `target`;
methods are `bus`, `train`, `plane`, or `ferry`. There must be at most one outgoing
edge per source. The three prose fields are validated and included in identity
but do not drive the walk. Only directed typed edges are followed, preserving
their labels. The algorithm stops at the first arrival at the destination, even
if the destination has an outgoing edge. Source equals destination is a valid
zero-step route. A reverse edge does not imply a forward edge. This is not a
general branching-graph planner or a natural-language graph extractor.

## Resource policy

Defaults are 50,000 newly expanded states, 250,000 attempted transitions, 100,000
graph input edges, 8 MiB canonical input, and 1 MiB encoded answer. Hard ceilings
bound configured values. Boolean values are not accepted as integers. An exact
search state includes a terminal leaf; a transition includes each proposed
arithmetic operation, including a subsequently rejected one. Graph states count
visited positions including the terminal/cycle-check position; transitions count
actually followed edges. Graph input validation/indexing has its own edge and
encoded input budgets and is not misreported as a search transition.

These are deterministic work and serialized-data budgets, not a process sandbox,
wall-clock timeout or hard interpreter memory limit. Tests compare small integer
reachable sets with an independent labeled-subset dynamic program and check
returned equations with exact rational arithmetic and multiset consumption.

## Published-data benchmark protocol

`benchmarks/benchmark_procedures.py` preregisters all six Countdown and path
definitions (`0.5k`, `2k`, `8k`), 200 records each: 1,200 selected cases, with no
sampling, tuning flag or rejected-case removal. Its eight source SHA-256 pins
cover all six JSON files and the two inert prompt YAML files from the existing
LongProc `d91a0dd1fca8858f5a66db989f2914ae3c2df991` data checkout. Prompts are hashed
but are neither executed nor used by the typed-input baselines. Bucket names
describe published output difficulty, not input token capacity.

From this repository's development checkout with its matching package available:

```powershell
python -m benchmarks.benchmark_procedures D:/path/to/longproc/data --output D:/existing-artifacts/procedures.json
```

The output must be a new file with an existing parent directory, outside the data
directory. Existing files, including hardlink aliases, are never overwritten.
No source data is downloaded or redistributed. LongProc's project license is
Apache-2.0; underlying dataset terms remain separate from the original MIT
implementation here. This script never imports or runs the LongProc Python code.

Every selected case remains in `valid_fraction_all_selected`. Unsolved cases,
solver errors, scoring failures and unsupported metrics have separate counters;
`valid_fraction_scored` additionally exposes the supported scoring denominator.
Invalid gold is a separate diagnostic, not a filter and not a solver input.
For every case the benchmark also scores an empty response, a response with its
closing tag removed, and a semantically altered last equation/edge. If no complete
baseline answer exists, the latter control explicitly uses a malformed fallback.
Error details contain exception types only, never raw problem text.

The aggregate report contains input/result inventory digests, source and dataset
hashes verified before/after execution, actual imported module bindings, protocol
and tool versions, work counts and timings. It contains no case IDs, questions,
equations, city names or copied reference answers. `tracemalloc` measures Python
allocations from delayed imports through data loading/solving/scoring; it excludes
preexisting interpreter allocations, native allocations and total process RSS.
There are no child processes or model/provider calls. A broken stdout after
publication returns failure without falsely claiming the report was not written.

No full-data measurement is asserted by this protocol document alone; measured
results must come from a completed, source-bound report. Primary success on these
typed, mostly procedural/synthetic inputs is not language-model benchmark accuracy.

### Completed measurement: 2026-09-12

The retained [aggregate report](../benchmarks/results/procedure-baselines.json)
is byte-identical to the completed checkout run. Its SHA-256 is
`767cb830c403f89cfb3e100cd75c7013cdc7789b6f176924db439c97ee5c14ec`.
The preregistered protocol digest is
`7c2392adf604a2361b6664e64170f7b0e24959c8c1e05d978e1efd82ee1ab1de`.

| Published output bucket | Selected | Valid input-only answers | Complete-search no solution | Invalid reference answers |
| --- | ---: | ---: | ---: | ---: |
| Countdown 0.5k | 200 | 200 (100%) | 0 | 1 |
| Countdown 2k | 200 | 199 (99.5%) | 1 | 1 |
| Countdown 8k | 200 | 198 (99%) | 2 | 3 |
| Path traversal 0.5k | 200 | 200 (100%) | 0 | 0 |
| Path traversal 2k | 200 | 200 (100%) | 0 | 0 |
| Path traversal 8k | 200 | 200 (100%) | 0 | 0 |

All 1,200 cases were selected and scored. No cases exhausted their configured
budgets, failed execution, had unsupported scoring, or produced a supposedly
solved answer rejected by the scorer. The three Countdown no-solution results
remain in the denominator: they are complete-search outcomes under the exact
positive-integer, at-most-2000 intermediate-value policy, not resource exhaustion.
The five invalid reference answers were not repaired, supplied to the solver, or
used to exclude cases. The difference between reference and baseline counts also
shows why matching the published solution text is not the same as task validity.

Each of the empty, truncated and altered-answer controls was scored for all 1,200
cases: all 3,600 controls were rejected, with no scoring failures or unsupported
control results. The three unsolved cases used the declared malformed fallback
for the altered-answer control, not a fabricated correct solution.

The run used CPython 3.12.13 / Unicode database 15.0.0 on Windows and took
666.348 seconds. Loading and validation accounted for 494.579 seconds, solving
for 56.130 seconds, and scoring plus controls for 112.417 seconds; the remaining
time includes imports, binding checks and report assembly. These measurements
include `tracemalloc` overhead and the shared host's workload, not isolated
production latency. Peak tracked Python allocations were 212,132,931 bytes
(202.305 MiB); this is **not** peak process RSS or native memory.

The report binds all 49 runtime Python source files, `pyproject.toml` and the
benchmark script (51 files), and independently checks the 42 actually imported
module paths against those source hashes. All source and eight input-file
bindings matched before and after execution. In particular, the repaired shared
task runner was bound to
`3d5633791af736d176988033706a1496339093917304c7a09a08e60e782ecf7e`.
This benchmark itself does not invoke the durable task runner, a provider or a
model; its source binding records the frozen checkout, not proof that every bound
module was exercised.

The measured outcome is a useful typed-input solver/scorer check on two families.
It is not learned-model accuracy, a six-family LongProc score, an official
evaluator-equivalence result or evidence that the entire reference repository's
functionality has been matched.

### Run history

The first full-data attempt on 2026-09-12 was stopped before any report was
published because an independent review found a provider-configuration binding
race in the shared durable task runner. This was an implementation-correctness
review, not a data failure or a response to benchmark scores. No intermediate
scores were published or used for tuning. After the runtime repair and regression
checks, the identical preregistered dataset selection, algorithms and budgets were
restarted. Only a completed, unchanged-source run is eligible for retained evidence.

The next completed report (SHA-256
`af494ba77bce0baee34b5649d87aafab310d57ad0f2505ade82c2989d8ca6a55`)
was accepted locally and remains in the history at commit `9532311`. Its first
remote CI run then exposed a separate Python 3.14 POSIX problem: both this
benchmark entry point and the authored demo constructed color-probing argparse
formatters even when stdout was already closed. The fix disables color at both
parser and formatter construction on Python 3.14; earlier versions receive no
unsupported keyword. The existing publication tests now also exercise that probe
on Windows. No algorithm, selected data, budget or scorer changed.

Because the benchmark source changed, the complete fixed protocol was run again,
producing the current report above. All six groups' input/case/result inventory
hashes and every non-timing metric match the previous completed report. The new
script is bound to
`d71d5b5a286f9455b2d7890c5d3abef1981a800a8439acd496f14776b73a7219`.
All 49 runtime Python files remain byte-identical to `9532311`. The earlier
full-data schema audit was not rerun: its unchanged runtime/data bindings remain
separate evidence, not an invented new execution. Neither timing difference is
an isolated performance comparison.
