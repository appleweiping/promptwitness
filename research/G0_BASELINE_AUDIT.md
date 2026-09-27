# G0 baseline and source checks

As of 2026-09-26. These are engineering/source checks, not measurements of
PromptWitness-Delta effectiveness. G0 has not passed all readiness requirements.

## Existing PromptWitness

The local clean starting checkout was
`acdee7653746987b7d4556a33ef4832a1638874c` on
`feat/whole-repository-alignment`. The remote feature branch matched; remote main
was `9606fbe0980ef07a16be1883ab8f04ba55ba1754`. Work is isolated on
`research/promptwitness-delta-v1`. Other project repositories were not modified.

The actual full baseline run produced 1,434 passed and 3 skipped tests. Statement
coverage was 8,559/8,994 (95.1634%); branch coverage was 3,157/3,536 (89.2817%).
The combined coverage metric was 93.5036%. The existing code therefore passes
its configured combined 90% gate; it has not reached a 90% branch-only target.
There is no new incremental research package yet.

Ruff lint/format, strict mypy (49 source modules), and Bandit passed on that
baseline. This is distinct from the historical successful
[CI run 34699696102](https://github.com/appleweiping/promptwitness/actions/runs/34699696102).
New research work has not been committed or verified by remote CI.

Evidence retained outside the source checkout:

| Artifact | SHA-256 |
| --- | --- |
| `build-verification-promptwitness-delta-20260926-baseline.xml` | `f6e5c2273e010b4c1d1114476d6d634939b407148ecb8bda60524a9c5f2b7eae` |
| `build-verification-promptwitness-delta-20260926-baseline-coverage.json` | `8d57279dd8ae22faa590924ad3b1b7f30d962c57ea8e236ea8fb34184149ae52` |

## Official comparison implementations

Source copies are pinned, unmodified upstream implementations. Installing them
in temporary local environments does not modify PromptWitness dependencies.

| Implementation | Source commit | Executed check | Actual result |
| --- | --- | --- | --- |
| GEPA | `d771eb21b5dd3228bc3f567293d2ccfc423fc900` | Data-loader, candidate-selector, module-selector, evaluation-cache and validation-cache tests | Initial run: 47 passed, 4 failed because optional `datasets` was absent. Isolated core rerun: 47 passed, those 4 explicitly deselected. Not a complete upstream-suite pass. |
| ProEval | `4bb4fcc4a097a710ee0935098d0caa7270842208` | Entire upstream `tests/` in a minimal CPU environment | 115 passed, 8 skipped. Skips are optional tuned-prompt-feature tests requiring Torch. Not a reproduction of the paper's experiments. |
| DSPy / MIPROv2 | `da1736e21ffda8cc4b86379d4748b011764d507c` | Bootstrap, random-search and teleprompt-utility upstream unit tests on Python 3.12 | 13 passed. These are mechanical unit checks, not MIPROv2 optimization or model-performance experiments. |

The GEPA paths checked include `core/adapter.py`, `core/engine.py` and
`strategies/eval_policy.py`. Native selection/frontier updates use actual
per-instance outcomes. Surviving candidates cannot be passed predicted values
for unqueried examples. Native minibatch screening remains in the controls.

ProEval's score-feature `BQPriorSampler.plan()` has target-free acquisition and
requires at least two authorized historical score columns. `SamplingPlan`
keeps item alignment immutable and refuses missing, extra or non-finite scores.
The tested plan matches the legacy native score-feature implementation on its
upstream fixture. Its native acquisition is not stratified SRSWOR; a shared
hypergeometric certifier needs a separately labeled randomized adaptation.
Historical score columns must come from the allowed training lineages, not the
unqueried candidate or held-out oracle. Neural-feature variants are not yet run.

DSPy source inspection includes bootstrapped demonstrations, grounded instruction
proposal, Optuna search and periodic complete evaluation. Demonstration or
proposal preparation is not free and must remain in call/token accounting.
The default chat adapter can make an additional JSON-fallback model call on a
parse failure. A one-episode/one-request forecast cannot silently assume that
fallback is free. A task-specific single-response adapter needs explicit tests
and symmetric use in the paired controls. Its configuration is a disclosed
deviation from the default adapter, not a change to native MIPRO selection.
An initial dependency command inadvertently chose Python 3.14 and attempted to
build NumPy 1.26.4; that owned process tree was stopped. The successful test
execution used the existing Python 3.12 interpreter and a temporary environment.

| Upstream test artifact | SHA-256 |
| --- | --- |
| `gepa-core-source-tests.xml` (initial, includes failures) | `c9c01dc7b62e78d53e2de183cfd1fc4d5cee19c3c854dd600b36481ea07db568` |
| `gepa-isolated-core-source-tests.xml` | `94e68d7bda20f0df2bbd96a2a08c28a30b75a1d347520f7173f7d11c9c423e54` |
| `proeval-source-tests.xml` | `d8bcc78e823720c06ede4912981a071cadc43c194aedbdea22e2b8ec5c8e66c0` |
| `dspy-python312-core-source-tests.xml` | `10dd99fa2235401e79315e4ff848070da5edacb54287fece86eac2252be867e0` |

Pending G0 evidence: finalized grouped pools and licenses/scorers; representative
full-protocol resource forecast; third-model feasibility; complete relevant
literature and implementation audits; native budget boundaries. No research
result, baseline victory, novelty proof or confirmatory launch is established.

## Additional G0 checks (not method experiments)

The native MIPRO mechanical compile in
`results/summary/g0/mipro-native-mechanical-v4.json` exercises unmodified pinned
DSPy bootstrap, proposal and Optuna paths using synthetic completions. It records
25 synthetic task calls (including seven training bootstrap calls) and four
synthetic proposer calls. Its six initial search requests match the canonical
demo-free renderer. There are zero real model calls in this check; its scripted
scores are not model accuracy. The preceding two-candidate fixture did not reach
the bootstrap candidate-set path and is retained as superseded evidence.
Pinned `BootstrapFewShot._train` subtracts the bootstrapped demo count from the
labeled-demo limit before concatenation. With both default limits set to four,
the total candidate context contains at most four demonstrations, not eight.
The eight-demo G0 profile is a stress envelope, not the default native context.

Training-pool proposals meet 512/256/256 for each family, with no shared group
identifiers across these pools. Grouping is a disclosed proxy: BFCL uses exact
function names, HotpotQA selects complete document-disjoint rows and excludes
bridges, and instruction data uses verifier IDs rather than the meaningless
wrapper descriptor `constraint_type=multi`. It is not a semantic-family proof.
The five BFCL categories leave 214 derived final units with an imbalanced category
distribution; this is not an official leaderboard split or a frozen final pool.

The initial instruction scorer-construction audit found four selected annotations
whose paragraph index exceeds the declared paragraph count. Pinned open-instruct
`ParagraphFirstWordCheck.build_description` replaces such an index with a random
one; this changes the declared task and cannot be accepted silently. Proposal v2
applies a general out-of-range metadata exclusion to the entire training corpus
before formal freeze, without reading candidate scores or rewriting requirements.
All 1,072 excluded training rows are retained in the proposal. The replacement 1024-unit pool
passes construction for 1612 constraint instances under both audited seeds. This
does not validate every checker's semantic correctness or authorize changing an
IFBench final-test requirement.

Pinned official scorer sources are available in isolated external checkouts:
BFCL `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`, IFBench
`1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d`, and IF training verifiers
`99b1ee970490a2d0d5664663eb0b1acc410b944c`. The BFCL data README declares
Apache-2.0. IFBench code is Apache-2.0 and its released data is ODC-BY-1.0 with
third-party terms. HotpotQA's official website declares CC BY-SA 4.0. Dataset
attribution/source notices and full task scoring remain implementation work.

The completed 400-request cost probe consumed 0.790188309 allocated GPU hours,
279292 input tokens and 33242 output tokens, with no failed inference requests.
One zero-request startup failure is included. A separately frozen 300-request
supplement adds the third model, diversified function tasks, larger output caps
and context/proposer stress proxies. It is not a native optimizer-effectiveness
experiment. Two additional zero-request startup failures (temporary directory
and an incorrect snapshot path) were preserved and corrected using the verified
task-owned directories. The supplement is complete: 700 cumulative real requests,
three retained zero-request startup failures, 2.400758047 allocated GPU hours,
1369522 input tokens and 150845 output tokens. The eight-demo scenario is only a
pressure envelope; it cannot by itself establish native optimizer feasibility or
failure. No statistical certifier, predictor or pilot result is implemented.

The cost-context audit checks exactly the four previously fixed, fit-reserved
instruction responses with all training constraints. Two pass; two fail.
It issues no model calls and reads no final data. This is an acceptance check of
the stress-context ingredients, not a baseline accuracy estimate or optimizer
comparison. The NLTK 3.9.2 sentence resource is owned, external and hash-recorded.
The failed first CPU check (missing `punkt_tab`) generated no model requests;
installing that resource does not modify the declared instruction requirements.

The completed Qwen envelope contributes 16 distinct base-profile fit responses.
The subsequent training-only acceptance audit checks all 16 in a frozen hash
order, with seven passing the conjunction of their original verifiers. Its v2
record fixes the language-detector seed at 11; the first four passing responses
are used once each for a proposed bounded cost context. Failed responses and
their inference cost remain retained. This is not the native randomized bootstrap
sequence or a measured optimization gain.

A final calibration of at most 210 requests was frozen at 2026-09-27T01:09:23Z.
It uses 70 requests per model: 48 base task calls, 16 instruction tasks with four
distinct accepted fit demonstrations, two official-signature JSON proposer cost
fixtures, and four identical-request repeats. Single-unit deterministic generation
is intended to avoid batch composition changing the recorded response unit.
Four repeats cannot prove universal determinism; source audits and statistical
assumptions remain prerequisites to any certificate. Shared-server workloads
appeared during Mistral's envelope measurement and were left untouched; measured
allocated time is not exclusive-device throughput.

Its first complete main-model boundary contains 70 Qwen requests. Strict ledger
and raw-output validation retains 770 cumulative calls, 1,448,879 input tokens,
178,514 output tokens and 3.253816169 allocated GPU hours, including the three
earlier zero-request startup failures. All four identical-request diagnostics
match text and token counts; this does not establish universal determinism.
The ledger's atexit elapsed time is authoritative, rather than the slightly
earlier stdout elapsed snapshot.

The completed-model decision order, recorded after cost observations but before
the boundary, was applied without changing the scientific matrix or worker.
`model-boundary-budget-v1.json` forecasts 1,858.228 GPU hours for measured Qwen
task stages, or 2,229.873 with 20% reserve. Other models, proposer calls, reloads,
pilot, future failures and consumed G0 are explicitly unpriced in this subset,
not assigned zero cost. This sample-mean forecast of maximum allowances is not a
certified lower bound: early stopping, shared load and a different authorized
execution protocol could change actual costs.

The subset already exceeds the 1,000-hour protective ceiling. The remaining
140 optional Olmo/Mistral native-context calls were not started. This adopted
protocol ends RESOURCE_BLOCKED before G1; no pilot, method result or complete
native-context calibration is claimed. See `research/DECISION.md` and
`research/FINAL_STATUS.json` for completed/missing work and budget headroom.
