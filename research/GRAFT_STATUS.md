# GRAFT research status (living document)

Branch `research/graft`. ARIS stage: Workflow 1 done (idea + cross-model review),
Workflow 1.5 (experiment bridge) in progress. Target: ACL Rolling Review (Oct 12 or
Dec 20, 2026 cycle), GReaTer-level protocol and writing quality.

## Locations

| What | Where |
|---|---|
| Code (local) | `D:\Company\nlp-original-projects\promptwitness-graft` |
| Code (server) | `/media/lenovo/data2/promptwitness-graft`, a git checkout of `origin/research/graft`; `.aris/sync.sh` pushes, fast-forwards the server and checks local = GitHub = server |
| Runs (server) | `/media/lenovo/data2/promptwitness-graft-runtime/` |
| Data (private) | `D:\Company\research-artifacts\graft-data\greater-42a22d9` (+ server copy, SHA256SUMS verified) |
| Models (server) | OLMo-3-7B-Instruct (pilot cache); `graft-hf-cache/`: Llama-3-8B-Instruct (NousResearch re-host), Gemma-2-9B-it (unsloth re-host) |
| Reviewer traces | `.aris/traces/` (not committed) |

Server connection details are kept outside the repository (`.aris/`). Only GPU 0 is usable (GPU 1 hosts another user's vLLM service). Root disk is 99%
full; everything lives on `/media/lenovo/data2`.

## Timeline of findings

1. **2026-09-28 idea review (GPT-6 via Codex, xhigh).** PROCEED WITH CAUTION,
   novelty 6/10, potential 5/10 as proposed. Main objections: (i) "exact vertex" was
   false for length-changing edits; (ii) exact fixed-reasoning loss scoring
   (PMPO-like, one forward per edit) is the real competitor; (iii) run parity and
   fidelity before any benchmark campaign. Full text: `.aris/traces/idea-review/final_r1.md`.
2. **Exact vertices repaired.** Per-slot RoPE offsets (extra query/key rotation by
   the cumulative length change of preceding slots) make every replacement,
   deletion, insertion and multi-slot combined vertex equal the real edited prompt;
   fp32 tests on Llama, Gemma-2 (softcapping) and OLMo-3 architectures pass, with
   finite-difference checks of gate and offset gradients (`tests/test_superposed_gates.py`).
3. **Fidelity v1 (OLMo-3, pilot prompt, logical_deduction, 8 rows, 22 edits): negative
   for raw gate derivatives.** Exact fixed-reasoning loss change predicts fresh-reasoning
   loss change (Spearman 0.90), but first-order gate estimates do not rank edits
   (-0.16 by edit mean; hole point 0.25; candidate length 0.25). Cause: attention
   saturation; estimates reach hundreds of nats for edits whose true effect is < 2.5.
4. **Renormalized superposed patching** (`superposed_patching.py`): exact attention
   renormalization per layer with exact RoPE rotation, linear only through later
   layers; same single forward/backward. Self-consistency check and tiny-model
   correlation tests pass. Fidelity v2 (OLMo-3, Llama-3-8B, Gemma-2-9B; 3 answer
   types) running.
5. **Frozen pilot, logical_deduction validation (32 items, 3 seeds):** A 22.0,
   token-GReaTer B 24.7, random block C 24.3, gradient-norm block D 27.0. D ran
   partly concurrently with GRAFT probes on the same GPU (wall-clock budgets were not
   binding: 1030–1287 s of 1800 s). Holdout and date_understanding pending.

6. **GPU sharing with the frozen pilot.** The pilot batch starts a run only when GPU 0
   uses < 2 GB, to keep its wall-clock budgets fair; GRAFT jobs made it wait
   (`waiting_for_gpu_0` after 14/24 runs). Decision: finish the queued GRAFT fidelity
   and smoke jobs, then launch nothing on GPU 0 until the pilot's 10 remaining runs and
   holdout finish, then start Stage C.
7. **Fidelity v2, trimmed rendering (OLMo-3, pilot prompt):** patching rho 0.79; raw
   gate variants 0.14-0.37. Untrimmed first run: 0.87 vs -0.21-0.26.

8. **Fidelity v2 on OLMo-3 (4 runs: logical deduction pilot prompt x2 renderings,
   date_understanding, formal_fallacies under the GReaTer protocol):** renormalized
   patching mean Spearman 0.78 (0.63-0.87), normalized regret 0.17; raw gate variants
   0.12-0.38, regret 0.87-1.37. Cost per example: patching 1.8-4.5 s for all edits,
   per-edit exact vertex evaluation 7.8-21 s. Exact fixed-reasoning change vs fresh
   loss change: 0.84-0.90 (logical deduction) but 0.19-0.36 (date, fallacies):
   GReaTer's answer-only objective misses reasoning changes on some tasks.
9. **Verified-reasoning objective** (`graft_runtime.verified_targets`): score a
   gold-consistent self-generated reasoning plus the answer. Being measured on
   Llama-3/Gemma-2 (12 rows) and OLMo date_understanding.
10. **Paper** moved to the ICLR 2027 template (user request): 9-page main text,
    required AI-use statement drafted for the authors to verify.

11. **Stage B smoke (Llama-3, formal_fallacies, 3 rounds; not a result).** The loop
    runs end to end (search ~2 min/round, peak 18-20 GB). Found and fixed: answers
    such as `**invalid**` were unparsed (accuracy far below chance), and proposer
    meta-text ("Here is the revised block: reasoning_policy:") leaked into an
    accepted prompt (`clean_proposal`). Cost at ~20 edits/round: exact scoring 40 s
    vs renormalized patching 46 s per 3 rounds, so the efficiency claim needs larger
    pools (`bench_scoring.py`, K = 4-32 per slot) and a cheaper kernel.
12. **Compute so far (GRAFT, GPU 0):** about 4 GPU-hours (probes, fidelity v1-v2,
    smoke); well inside the 1000 GPU-hour cumulative cap.

13. **Validity study v1 (8 train rows -> dev accuracy of 26 edits, 60 dev examples;
    Llama-3 date/formal_fallacies, Gemma-2 date).** No fixed-reasoning objective reliably
    predicts held-out accuracy of whole-block edits: GReaTer's answer loss rho -0.04/-0.03/0.03,
    verified reasoning 0.25/-0.16/-0.02, contrastive margin 0.13/-0.22/-0.35, policy gradient
    -0.30/-0.49 (patching tracks the exact values). Only the fork (decision) margin is
    consistently positive on Llama-3 (0.26-0.30; best-5 above mean, worst-5 below) but it
    rests on few forks. On formal_fallacies the average proposal adds +12 points: the
    proposal pool, not the answer-loss ranking, drives gains. The objective, not the
    estimator, is the bottleneck.
14. **Decision study (v2)** running: forks between greedy and self-samples on 24 rows,
    fork margin scored exactly and by one-pass patching (logit-margin loss), against fresh
    accuracy on 8 and 24 train rows, with measured cost; Llama-3 (GPU 0), Gemma-2
    (GPU 1), then Qwen3-8B. Both GPUs are shared with another project's training jobs.
15. **Models:** the advisor asked for a stronger model than OLMo; Qwen3-8B (pure
    softmax attention, non-thinking mode) replaces OLMo-3. Qwen3.5-9B is unsuitable: 24
    of 32 layers are linear-attention recurrences where slots and gates are undefined.

16. **Decision study, Gemma-2 date_understanding (24 rows, 12 forks on 7 rows, 26 edits,
    60 dev examples).** Spearman with dev-accuracy change: fork margin exact 0.48, one-pass
    patching 0.46 (best-5 edits +0.057 vs mean +0.042); GReaTer answer loss -0.26 / 0.08;
    fresh accuracy on 8 train rows 0.13 (483 s), on all 24 rows 0.16 (1449 s). The fork
    estimate costs 78 s plus 380 s of shared sampling that does not grow with the number
    of candidates. Consequence: the runner gains `--objective fork` and
    `--accept margin` (shortlist by one-pass estimate, exact re-score, no generation).

17. **Exploratory decision study, Llama-3 date_understanding** (pre-freeze, seed 11):
    fork margin rho -0.18 (exact) / -0.20 (patch), answer loss 0.05 / -0.21, fresh-8 0.14,
    fresh-24 0.14; edits barely move accuracy on this task (mean -2.2, best +5 points on 60
    questions), so the target is mostly noise. Exploratory formal_fallacies runs were
    stopped by mistake after 5-16 minutes and are not rerun.
18. **GPT-6 round 2** (`.aris/traces/review-r2/final.md`): PROCEED WITH CAUTION, novelty
    7/10, potential 7/10 if replicated (about 4/10 now). Requirements adopted: frozen
    hypothesis H-fork and replication protocol (plan amendment), measured costs, fork
    coverage, eligible-row comparisons, hierarchical bootstrap, causal fork check, cite
    ContraPrompt (textual contrast of traces with a strong LLM), Global Forking Tokens,
    Forking Paths, Critical Tokens, Phi-4 pivotal tokens, decision-token restoration.
19. **Replication v3 launched 2026-09-29 07:51 UTC** on validation-only BBH tasks
    (logical deduction 3/7, tracking shuffled 7; 200-question targets; seed 101;
    non-initial prompt state with seed 202): Llama-3 on GPU 0, Qwen3-8B on GPU 1.
    Disk on data2 is at 98% (shared); vLLM is being installed in its own venv for
    later evaluation phases.

20. **v3 interim (logical_deduction_seven_objects, Llama-3 and Qwen3, 20 edits, 200 dev
    questions).** Spearman with dev change: answer loss -0.55 / -0.25, fork margin exact
    -0.14 / +0.18, patch +0.17 / +0.11, fresh-8 +0.49 / +0.18, fresh-24 +0.58 / +0.05.
    Two diagnostics change how every rho must be read. (i) The dev target itself is
    noisy: split-half reliability (Spearman-Brown) is 0.32 / 0.38, so no predictor can
    exceed rho of about 0.56 / 0.61; true edit effects vary by about 1.5 points while
    proposals cost 5.3 points on average for Llama. (ii) The answer loss is anti-predictive
    only on rows whose greedy reasoning is wrong (-0.61 / -0.30 there, -0.04 / +0.08 on
    right rows): lowering the gold answer's loss after wrong reasoning rewards prompts
    that make the answer ignore the reasoning. Pooled bootstrap so far: fork - answer
    +0.32 (P>0 0.95), fork - fresh-8 -0.12 (P>0 0.27). Search and evaluation are now
    separated (`run_graft --defer-eval`, `select_and_test.py`, vLLM reader) so that every
    method is read by one engine.

## Next

* Fidelity v2 decides the scorer: renormalized patching must beat raw gates and
  approach exact scoring at lower cost; otherwise GRAFT uses exact-vertex scoring and
  the contribution shifts to exact structural edit calculus + structure-level search.
* Pilot optimizer runs (`run_graft.py`, scorers patch/gate/exact/random) on a few
  tasks; then the preregistered subset (4–6 BBH tasks + GSM8K or FOLIO, Llama-3 and
  Gemma-2, 3 seeds) and the published-GReaTer / ZS-CoT baselines
  (`evaluate_prompts.py`).
