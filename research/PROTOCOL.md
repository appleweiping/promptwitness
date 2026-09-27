# PromptWitness-Delta v1: research protocol draft

Status: G0 readiness has not been established. This document is a design, not a
preregistration, statistical implementation, experimental result or submission.
The adopted charter and protective ceilings are in [scope.lock.json](scope.lock.json).
Only PromptWitness is in scope; expansion of the other four repositories is frozen.

## Research question and attribution

Test whether prompt-edit structure adds predictive information about paired
correctness transitions beyond the same text features, parent scores and allowed
parent traces. Test whether that information reduces evaluation cost with matched
decision quality, and improves GEPA and MIPROv2 under the same total resource budget.
Structure search, active evaluation, stratified estimation and error-budget
allocation are existing methods. The proposed contribution requires experiments
that isolate the additional value of edit information.

## Fixed reference and randomization

Each optimization run first evaluates its seed prompt on the complete search
pool. That reference vector remains fixed throughout the run and its actual calls
are charged. Candidate ancestry remains a separate predictor feature. Replacing
the fixed reference requires a new, explicitly costed protocol.

For one candidate and a fixed finite suite of N binary scored units, define
`d_i = new_i - old_i`, `Delta = sum(d_i)/N` and
`R = count(old_i == 1 and new_i == 0)/N`. R is unconditional; its denominator is N.
For nondeterministic execution, define the unit as a recorded sample/replicate
under pinned execution semantics. Temperature zero alone does not establish a
fixed response table. Unsupported execution assumptions must be reported as
empirical evaluation rather than formal certification.

Before reading any current-candidate outcome, freeze at most eight nonempty
strata, their complete member identities, allocation schedule and random seed.
Each stratum is homogeneous in the known reference correctness. Risk bins and
metadata may further subdivide those groups. Use a separate uniformly random
permutation without replacement within each stratum. Candidate-specific outcomes
cannot alter the frozen allocation or predictor used to establish it.

Observation sizes are the deduplicated values
`min(N, ceil(N*f))`, for f in `[1/16, 1/8, 1/4, 1/2, 3/4, 1]`.
The monotone per-stratum integer allocations must sum to each specified size,
never exceed a stratum's size, and reach every unit at the final size. The exact
allocation algorithm remains an implementation prerequisite. If a nonempty
stratum has zero observations, its count interval is `[0, N_h]` and the overall
point estimate is unavailable. It is not imputed as zero.

At a fully represented fixed look, use the design-weighted paired estimate
`Delta_hat = sum_h (N_h/N) * mean_h(d_i)`. Under stratified simple random sampling
with fixed positive sample counts, each stratum mean is unbiased for its finite
mean; linearity gives design unbiasedness. Optional stopping does not make the
stopped point estimate unbiased. Decisions use simultaneous intervals instead.

## Count intervals and error budget

For an old-correct stratum, K_h is its total number of regressions. For an
old-incorrect stratum, K_h is its total number of improvements. Exactly one
unknown binary transition count is needed per stratum. At a fixed look,
`X_h ~ Hypergeom(N_h, K_h, n_h)`. Invert the two single-sided tails over feasible
integer counts: retain K when `P_K(X >= observed) > delta` for the lower bound
and `P_K(X <= observed) > delta` for the upper bound. A census gives the exact
observed count. Empty strata are omitted; a completely empty suite is invalid.
Floating-point tail evaluation must be checked against exact small-population
rational calculations and a maintained scientific library before use.

Reserve `delta = 0.05 / (32 * 6 * 8 * 2)` per count direction, candidate slot,
look and stratum. The factor two is justified by homogeneous reference scores;
mixed-reference strata are unsupported in this version. Two-sided count
intervals imply simultaneous bounds for both Delta and R, so those metrics do
not receive a second independent allocation. The union bound totals at most
0.05 in one run. Unused slots are not recycled after observing outcomes.

Let G be old-incorrect strata and B old-correct strata. From simultaneous integer
bounds L_h/U_h derive:

```
Delta_L = (sum_G L_h - sum_B U_h) / N
Delta_U = (sum_G U_h - sum_B L_h) / N
R_L = sum_B L_h / N
R_U = sum_B U_h / N
```

Return ELIGIBLE only when `Delta_L >= -0.01` and `R_U <= 0.05`.
Return INELIGIBLE when `Delta_U < -0.01` or `R_L > 0.05`.
Otherwise query the next frozen look, complete the suite if affordable, or return
INCONCLUSIVE. Comparisons at a census should use integer/rational thresholds.
Eligibility is a property of this finite suite and protocol, not future safety
or proof of superiority. A failed or unsupported contract check is a separate
static decision.

Conditional on the history before each candidate's random audit, that candidate
and its allocation are fixed. The two tail inversions each have error at most
delta. A union bound across the reserved slots, without assuming independence,
controls incorrect certificates within one optimization run. Persist the slot
allocation and plan before querying; crash recovery must retain both. Separate
optimization runs do not inherit a single global 5% guarantee.
This argument is a proof outline; implementation tests, execution-assumption
checks and independent statistical review remain required.

## Predictor and optimizer integration

Reuse PromptDocument, message IDs, diff categories, rendering and providers.
Create a research sidecar for allowed edits, source locations, immutable external
interfaces and supported schema keywords. Unknown schema semantics must return
UNSUPPORTED. The existing bounded argument checker is not a behavior certificate.

The first predictor uses frozen text features and regularized logistic transition
heads with explicit structure/input/allowed-parent-trace interactions. Fit on
authorized prior records only. Report missingness and estimated cost explicitly.
Candidate responses, unqueried oracle outcomes and final-test labels are excluded.

Keep the same proposer, models, constraints, scorer and budget across evaluator
controls. Native GEPA uses per-instance scores and already screens proposals on
minibatches. Inspect a pinned official implementation before integration. An
early rejected candidate may avoid later evaluations; a surviving candidate must
receive every actual score required by the native selector. Predicted or missing
scores cannot fill its score vector. Any changed selector must be labeled adapted
and shared by the corresponding controls.

## Experiment and resource freeze

The planned families are BFCL-derived pinned offline categories, HotpotQA
distractor and IFTrain-to-IFBench. Use two independent main model families and
one third family for frozen-prompt transfer only. Respect official splits and
isolate predictor-fit, search, selection and final-test access. Group related
lineages, documents, tool families and constraint families across splits.

The planned mechanism matrix contains 144 pairs, 24 per family/model cell, with
up to 512 legal training-side units per pair. The planned optimization matrix
contains 180 runs: three families, two models, five seeds and six configurations
listed in the scope lock. Counts are plans, not completed experiments. Preserve
all failed, inconclusive and early-stopped runs in the denominators.

Before choosing checkpoints or starting real inference, establish the existing
hardware allocation, permitted usage and trusted connection. Measure 200 real
requests, then forecast all roles, full-reference evaluation, selection, final
testing, transfer, predictor work, retries and failures. Check every protective
ceiling and any lower actual quota with at least 20% forecast reserve. Episode
and call accounting boundaries must be frozen to prevent double counting or
free validation. The charter's limits are ceilings, not a grant of GPU time.
Paid API spending is zero and new cloud rental is unavailable by default.

Cache reuse requires the complete request, prompt/contract, model revision,
tokenizer/template, decoding/seed, scorer, tool environment, data version and
recorded random unit to match. Account for cold-start and amortized costs
separately. A changed scorer may rescore a recorded output; it does not create a
new model response. Raw private endpoints and credentials are excluded from
public artifacts.

## Gates and finite delivery

G0 must establish a runnable baseline, pinned resources/data/models/baselines and
a feasible measured forecast. G1 implements and verifies the minimal method.
G2 runs a pilot with at most two method revision cycles. G3 freezes hashes and
runs one confirmatory campaign. G4 regenerates the paper from raw evidence and
checks reproducibility and same-commit CI.

Adopt all numerical readiness targets from the scope lock without changing them
after results: 30% cost reduction against full evaluation, 15% against the
strongest valid generic evaluator, 10% structural gain against text-only under
the same certifier, 95% clear-eligible retention, and two-point macro improvement
with a positive adjusted lower bound and family noninferiority. Efficiency alone
does not satisfy the complete optimization claim.

Deliver one of SUBMISSION_PACKAGE_READY, NEGATIVE_RESULT_COMPLETE,
RESOURCE_BLOCKED, NOVELTY_BLOCKED or INTEGRITY_BLOCKED with explicit evidence and
missing work. A terminal blocked report requires a new authorized, versioned
protocol before restarting. This draft has no preregistration hash; data splits,
licenses, model/tokenizer revisions, baseline commits, cost forecast, allocation
implementation, statistical tests and independent review remain incomplete.
