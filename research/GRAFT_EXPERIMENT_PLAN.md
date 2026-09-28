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
