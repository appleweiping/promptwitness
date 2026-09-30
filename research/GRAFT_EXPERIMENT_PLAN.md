# GRAFT experiment plan (preregistered 2026-09-28, before any optimizer result)

Fixed before running `run_graft.py` on any task. Changes after results must be
logged here with a date and reason; the original text stays.

## Models and protocol

* Task = optimizer model (self-optimization, as in GReaTer): Llama-3-8B-Instruct,
  Gemma-2-9B-it (GReaTer's models), OLMo-3-7B-Instruct (fully open, extra check).
* GReaTer data (commit 42a22d9) with GReaTer split sizes; seeded splits in
  `graft_tasks.py` (`SPLIT_SEED=20260928`). Train drives search, dev selects among
  accepted prompts, test is read once per run.
* Shared reader for every prompt: greedy reasoning (search 384 tokens, evaluation 512),
  task-typed extractor, 8 greedy answer tokens, typed parser.
* Starting prompt for every search method: GReaTer's initial instruction as typed
  blocks (strategy, empty procedure slot, output).

## Stage A — fidelity (C1). Status: running

OLMo-3 (pilot prompt, logical_deduction_three_objects), Llama-3 and Gemma-2 on
date_understanding (mc), formal_fallacies (binary), object_counting (integer);
8 train rows, 8 label-free proposals per slot, seed 11. Estimators: renormalized
patching, raw gate (incumbent, no offset, hole, centroid). Targets: exact
fixed-reasoning loss change; fresh-reasoning loss/accuracy change.
Pass criterion for using patching as GRAFT's scorer: mean Spearman >= 0.5 across the
seven runs and clearly above raw gates. Otherwise GRAFT scores edits exactly and the
estimator claim is dropped.

## Stage B — smoke. One task, Llama-3, patch and exact scorers, 3 rounds: runtime,
memory, cost ledger, no crash. Not reported as a result.

## Stage C — controlled comparison (C2, C3, C5)

Tasks (fixed now, chosen for answer-type diversity, not by pilot outcome):
date_understanding, formal_fallacies, object_counting, navigate,
tracking_shuffled_objects_five_objects, movie_recommendation, FOLIO.
Models: Llama-3-8B, Gemma-2-9B. Seeds: 1, 2, 3.
Search methods (same proposals/verification/selection budget: 12 rounds, batch 8,
K=6 proposals per slot, shortlist 3):
* GRAFT (`patch`), exact fixed-reasoning scoring (`exact`), random shortlist
  (`random`), textual gradients with the same model (`textgrad`): 3 seeds;
* raw gate ablation (`gate`): seed 1 only.
Fixed prompts: ZS-CoT, GReaTer initial prompt, GReaTer published prompts (pooled
final beams from the official logs, <= 12 selected on dev).
Report: per-task test accuracy (mean ± sd over seeds), average, paired sign test
over task×seed, and search cost (GPU seconds, forward/backward tokens, calls).

## Stage D — breadth (C2)

All 21 GReaTer BBH tasks + GSM8K + FOLIO, both models, GRAFT seed 1 versus ZS-CoT,
GReaTer initial and GReaTer published prompts. GSM8K test is the full 1319 set.

## Stage E — non-brittleness (C4), evaluation only

* Transfer: prompts optimized on one model evaluated on the other two.
* Fluency: perplexity of final prompts under a third model; PromptWitness structural
  validation of every final prompt.
* Perturbation sensitivity: accuracy change under one-token deletion and under a
  meaning-preserving paraphrase of the final prompt (same perturbation procedure
  for all methods).

## Stage F — cost reference

Official GReaTer code on one or two tasks with Llama-3 for wall-clock and token cost
under our hardware, if its environment can be installed without changing shared
Python; otherwise report GReaTer's published cost with the hardware difference stated.

## Claims map

C1 Stage A; C2 Stages C and D; C3 cost ledgers of Stage C and Stage F; C4 Stage E;
C5 Stage C ablations (gate, random, exact) plus fidelity estimator ablations.

## Amendments

* 2026-09-28, before any optimizer result: dev selection is restricted to the
  incumbents at checkpoint rounds 0, 4, 8, 12 (deduplicated) instead of every
  accepted prompt, for all methods alike. Reason: dev re-evaluation of every accepted
  prompt dominated projected run time on one GPU.
* 2026-09-29, after validity v1 and before any Stage C run: Stage C is deferred until
  the decision study selects the search objective (answer loss vs fork margin vs fresh
  minibatch accuracy). Reason: validity v1 shows the answer objective does not predict
  dev accuracy of block edits. The third model is Qwen3-8B instead of OLMo-3 (advisor
  request); Llama-3-8B and Gemma-2-9B remain the main models for comparison with
  GReaTer.
* 2026-09-29, frozen after the first decision-study result (Gemma-2 date) and the GPT-6
  round-2 review, before inspecting any other decision-study result:
  **Hypothesis H-fork.** For LLM-proposed whole-block edits, the change in decision margin
  at self-generated reasoning forks (CE of the correct-branch token minus CE of the
  incorrect-branch token, given the shared prefix) predicts held-out accuracy change better
  than (a) GReaTer's answer loss on fixed greedy reasoning and (b) fresh greedy accuracy on
  the same train rows, at lower measured end-to-end cost; the one-pass renormalized estimate
  preserves this.
  **Validation protocol (replication, v3).** Independent seeds (new train rows, proposal
  pools, dev slices) on >= 2 models x >= 3 tasks, plus a non-initial prompt state. Report
  fork coverage; results on all rows (operational policy) and conditional on fork-eligible
  rows; a paired hierarchical bootstrap (rows, dev questions, pools) of the *difference*
  in Spearman and in best-5 regret between predictors; costs measured end to end (no
  extrapolation). **Causal check:** forcing the correct vs incorrect fork token and
  sampling continuations must change success probability more than position-matched
  random divergences on the shared prefix. The decision-study runs launched before this
  entry (Llama-3, Gemma-2, Qwen3 on date/fallacies/navigate/counting, seed 11) are
  exploratory and are reported as such. A failed replication means H-fork is not claimed;
  the paper then reports the exact calculus and estimator with the negative objective
  finding.
* 2026-09-29, before any Stage C/D result: evaluation is separated from search. Search
  runs stop after search (`run_graft.py --defer-eval`) and record their checkpoint
  prompts as typed blocks; dev selection among checkpoints and the single test read are
  done by `select_and_test.py` with a vLLM reader that renders the same token ids and
  uses the same stop tokens, extractor and parser. Fixed-prompt baselines
  (`evaluate_prompts.py --engine vllm`) use the same engine, so every method in a
  comparison is read identically (greedy decoding in different kernels can differ on
  near-ties, so engines are never mixed within a comparison). Reason: evaluation
  dominated run time on shared GPUs. Qwen3-8B joins Llama-3-8B and Gemma-2-9B in
  Stage C and D; GReaTer published prompts exist only for the latter two, so Qwen3
  comparisons with GReaTer use an official GReaTer rerun if its code can be adapted,
  otherwise only the other baselines.
* 2026-09-29, before any Stage C result, while the v3 replication is still running:
  the Stage C variant set is fixed independently of the H-fork outcome. Preregistered:
  `patch` (GRAFT, GReaTer's answer objective, patching shortlist, fresh verification),
  `exact`, `random`, `textgrad` with seeds 1-3 and `gate` with seed 1. Exploratory
  (labeled as such unless H-fork replicates): `fork-patch-fresh` (fork objective,
  seeds 1-3) and `fork-patch-margin` (fork objective, generation-free acceptance,
  seed 1). Models: Llama-3-8B, Qwen3-8B, Gemma-2-9B. Search-time generation runs on a
  vLLM server beside the HF scorer (`--gen-server`), identically for every variant of a
  model; dev selection and the test read use `select_and_test.py`. The official
  GReaTer implementation shortlists by the gradient and then selects by fresh-reasoning
  errors plus a prompt-perplexity penalty, so `patch` vs `random` isolates the value of
  the gradient shortlist exactly as in GReaTer.
* 2026-09-29 19:10 UTC, outcome (not an amendment): the H-fork replication failed its
  preregistered criteria (fork vs fresh-8 accuracy P(diff>0) = 0.017; causal check
  negative; fork vs answer loss not established). H-fork is not claimed. Per the Stage C
  amendment, the fork variants remain exploratory and GRAFT's reported method uses the
  preregistered configuration.
* 2026-09-29 ~22:50 UTC, before any Stage C result and before any baseline is reported:
  reader fix and budgets. Qwen3 writes answers after the extractor in LaTeX
  (`$$ \boxed{-4} $$`); the parser did not strip `\boxed{`, and 8 answer tokens cut long
  numbers (observed on 18 Qwen3 GSM8K/arithmetic reads: reasoning correct, parse failed).
  The parser now strips LaTeX/markdown wrappers and Unicode minus; answers are read with
  16 tokens (`ANSWER_TOKENS`). Qwen3 also exceeded the 512-token evaluation budget on
  causal judgement (60 of 69 test answers truncated), so reasoning budgets are 1024
  tokens for search and evaluation (GReaTer's own selection uses up to 1024). All fixed
  prompts and all Stage C runs use the fixed reader; the v3 and token-level studies (all
  multiple-choice tasks, whose extractor ends in "(") are unaffected and keep their
  preregistered settings.
* 2026-09-29 ~23:00 UTC, after cross-family review round 3 (`.aris/traces/review-r3/final.md`),
  before any Stage C result and before any TS3 data exist:
  1. **Stage C is paused** until the phase-2 controls (done except one token run) and one
     instrumented real-model smoke (2 rounds, `patch`, object_counting, generation server)
     are recorded; it is then launched manually. The exploratory fork variants (84 runs)
     are dropped (H-fork failed). Stage C analysis is prespecified as task-level paired
     comparisons with seeds nested within tasks, plus accuracy versus measured total
     search-and-selection compute; no speed claim is made from pass counts.
  2. **Split provenance.** GReaTer optimized its published prompts on rows 0-49 of each
     released file and monitored rows 50-99; 41% of our BBH test rows fall in those
     ranges. Published-prompt comparisons are reported on the full test split and on the
     clean subset (file rows >= 100), for all fixed prompts alike.
  3. **H-cond, prospective (the only new objective allowed as confirmatory).** Score an
     edit by the change in GReaTer's answer loss averaged only over training rows whose
     incumbent greedy extracted answer is correct ("answer-conditioned"; exact and
     patched); if no row qualifies, the ranking is random. Prior evidence, disclosed: on the
     already-inspected v3 runs it beats the all-row answer loss (d_rho +0.18, P 0.985) but not
     a random ranking (+0.03) and loses to fresh-8 (-0.24). Test: `decision_study.py` block
     mode on the untouched task tracking_shuffled_objects_three_objects, seed 303, 24 rows,
     200 held-out questions, Llama-3-8B, Qwen3-8B and Gemma-2-9B. Primary metric: best-3
     regret (`analyze_decision.py --regret-k 3`); secondary: Spearman. H-cond is supported
     only if answer_cond_patch beats both answer_patch and random on best-3 regret with
     P(gain > 0) >= 0.95 in the pooled paired bootstrap; otherwise it is retired.
* 2026-09-30 ~01:15 UTC, before any ablation run: **token-level shortlist ablation of the
  official GReaTer code.** The fixed-reader baselines show real gains of GReaTer's published
  prompts over its initial prompt on Llama-3 (+4.7 points full test, +3.4 [+0.5, +6.4] on
  the clean subset), while the token-level control shows single-token edits cannot be
  ranked by any signal at n = 200. The framing rules (`GRAFT_FRAMING.md`, case "not
  predictive / real gains") call for testing whether GReaTer's gains come from its
  gradient shortlist or from its fresh-reasoning selection. Test: official GReaTer
  (`greater_rerun.sh`, Llama-3-8B, our 50 train rows, the official hyperparameters) with
  the gradient shortlist vs. the same loop with the shortlist drawn uniformly from the
  same model-proposed candidates (`GREATER_SHORTLIST=random`,
  `scripts/research/graft/greater_random_shortlist.patch`). Tasks, chosen because the
  published prompts gain most there (so a null for the gradient is informative):
  tracking_shuffled_objects_five_objects and geometric_shapes first, then object_counting
  and movie_recommendation. One seed each (low power; reported as such). Outcome: test
  accuracy of the final prompt with the shared reader (full and clean subsets), plus wall
  time; compared with each other and with the initial prompt.
