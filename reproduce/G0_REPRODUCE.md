# G0 evidence reproduction

This is cost and engineering evidence only. There is no implemented Delta
predictor, certificate, eligibility gate, research CLI or optimization-result
matrix. Recomputing these files does not authorize new model calls.
The adopted protocol ended RESOURCE_BLOCKED at its first completed native-context
model boundary. See `research/DECISION.md`; the remaining 140 optional cost calls
were not executed. No full 910-call calibration or G0 approval is claimed.

## Inputs and isolation

The source checkout is `research/promptwitness-delta-v1`, based on
`acdee7653746987b7d4556a33ef4832a1638874c`. Research changes are currently
uncommitted and have no same-SHA remote CI result. Do not confuse the successful
historical baseline CI with verification of this sidecar.

Private connection settings and keys are not stored here. Archived raw JSONL,
SQLite ledgers, upstream checkouts and model snapshots are outside the repository
under the task-owned artifact directory. Dataset and model revisions are recorded
in the configuration files. Inference snapshots were checked against official
metadata; a mirror URL alone is not an integrity check. No remote custom model
code is enabled, and inference uses offline pinned snapshots.

The IFBench scorer checkout bundles public final data and upstream outputs. See
`research/ACCESS_AUDIT.md`; do not inspect its data/evaluation directories or call
those artifacts our measured results. Split access is not yet a process sandbox.

## CPU checks

From the repository root, with the existing Python 3.12 development environment:

```powershell
.venv/Scripts/python.exe -m pytest --no-cov tests/reproduce
.venv/Scripts/ruff.exe check reproduce tests/reproduce
.venv/Scripts/ruff.exe format --check reproduce tests/reproduce
```

These fixtures test accounting, provenance, isolation and input preservation.
They do not test statistical coverage or demonstrate method benefits. The full
pre-research baseline separately passed 1,434 tests with three skips. It has a
93.5036% combined coverage metric, not 90% branch-only coverage.

The new helper Bandit scan returned six findings (four medium, two low), not a
zero-finding pass. `results/summary/g0/helper-security-review.json` records manual
triage for local-only pinned HF snapshots, fixed HTTPS HEAD probes and a fixed
no-shell GPU inventory command. The frozen workers are preserved byte-for-byte.
This source review is not an independent security audit.

## Recompute completed cost evidence

Use the archived 700-request ledger, not a live or later expanded ledger. Output
paths are exclusive-create: choose a fresh verification directory rather than
overwriting the retained summaries. Substitute your artifact root and paths.

```powershell
python -m reproduce.summarize_cost_envelope `
  --ledger <artifacts>/envelope-v1/preflight.sqlite `
  --records <artifacts>/envelope-v1/qwen-envelope-v1.jsonl `
  --records <artifacts>/envelope-v1/olmo-envelope-v1.jsonl `
  --records <artifacts>/envelope-v1/mistral-envelope-v1.jsonl `
  --plan configs/research/preflight-cost-envelope-v1.json `
  --prior-summary results/summary/g0/preflight-infra1.json `
  --startup-failure results/raw/g0/olmo-envelope-startup-failure.json `
  --startup-failure results/raw/g0/olmo-envelope-path-startup-failure.json `
  --output <verification>/cost-envelope-v1.json
```

The summarizer retains all first-400 calls/time/tokens, compares archived runs
against the ledger and includes three zero-request startup failures. It requires
every inference attempt in that phase to be reported, rather than discarding
failed records or using generation-only time as allocated GPU time.

## Bounded final calibration

`preflight-native-context-v1.json` freezes at most 210 additional calls before
execution. The worker SHA-256 is
`d084772b96abb8ca7894a3ee75c1e6eacb445344c7efec9151d9846679688192`.
It requires single-unit generation, CPU threads four, BF16/SDPA, compilation off,
deterministic algorithms, TF32 off and cuBLAS workspace `:4096:8` set before Torch
import. All previous calls remain in the same ledger. There are no request
retries or silent replays. A task-owned writable temporary directory is necessary;
the server root/home storage was full. Preserve unrelated GPU jobs.

The frozen inputs contain 48 base task calls, 16 instruction tasks with four
distinct accepted fit-side demonstrations, two official-signature JSON proposer
cost fixtures and four identical-request repeats per model. They are not native
optimizer runs. Four successful repeats would be a diagnostic, not proof of all
response-table assumptions. If a phase fails or remains incomplete, report it;
do not manufacture a successful 910-call summary.

When all planned calibration records exist, `summarize_native_context.py` checks
the expanded ledger against the byte-hashed 700-request archive and retains all
previous consumption. `forecast_calibrated_campaign.py` uses its measured costs
with at least 20% reserve and labels every proxy/unverified assumption. It never
changes `g0_passed` to true merely because a scenario falls below the ceilings.

`campaign-cost-counts.proposed-v2.json` retains all 180 runs and 144 mechanism
pairs. BFCL final width is 214 because the legal derived source is insufficient
for 256 after the required training pools and exposed-component exclusions. The
earlier count plan is retained unchanged. This preformal data-width adjustment
is not a baseline/seed reduction and does not create or borrow final examples.

No paper result, final-test score, certification guarantee or submission-ready
claim may be inferred from these cost files.

## Recompute the terminal completed-model boundary

The expanded 770-request ledger was copied only after the Qwen worker exited.
Keep the separate 700-request archive unchanged. With fresh output paths:

```powershell
python -m reproduce.summarize_native_context `
  --ledger <artifacts>/native-boundary-v1/preflight.sqlite `
  --archived-ledger <artifacts>/envelope-v1/preflight.sqlite `
  --records <artifacts>/native-boundary-v1/qwen-native-context-v1.jsonl `
  --plan configs/research/preflight-native-context-v1.json `
  --requests <artifacts>/native-context-requests-v1.jsonl `
  --prior-summary results/summary/g0/cost-envelope-v1.json `
  --completed-model-boundary-only `
  --output <verification>/native-context-qwen-boundary-v1.json

python -m reproduce.check_model_boundary_budget `
  configs/research/campaign-cost-counts.proposed-v2.json `
  <verification>/native-context-qwen-boundary-v1.json `
  <verification>/model-boundary-budget-v1.json
```

The first command refuses incomplete started runs, changed archived consumption,
missing raw outputs, altered frozen requests and mismatched execution provenance.
It marks the full calibration incomplete and the two remaining model profiles
unmeasured. The second preserves matrix allowances, prices only observed-model
task stages, discloses every excluded cost and never approves G0 from a subset.
Use `native-context-resource-decision-order-v1.json` for the preboundary decision
order. Cost observations preceded that decision-order record; neither artifact
is scientific preregistration or a statistical lower bound.

The research helpers and reports are uncommitted local sidecars. Their last
verified committed base remains `acdee7653746987b7d4556a33ef4832a1638874c`.
Historical CI verifies that base only, not these new files.
