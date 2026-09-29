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

21. **Fork causal check, logical_deduction_seven_objects (v3, 32 rows, 6 continuations per
    arm).** Forcing the correct vs the incorrect first-divergence token changes success
    probability by +0.085 (Llama-3, 41 forks) and -0.069 (Qwen3, 17 forks); position-matched
    random divergences change it by |0.19| and |0.14|, and |dP_fork| is 0.20 / 0.19. On this
    task first divergences between greedy and sampled reasoning are no more decisive than
    random positions (mostly wording), which fails H-fork's causal premise here and explains
    the weak fork-margin validity.
22. **Evaluation engine ready.** vLLM 0.10.2 in its own venv (`graft-vllm`, torch 2.8 cu128
    on driver 550, transformers pinned to 4.56.2 because 5.x breaks vLLM's tokenizer
    wrapper). Rendered prompt token ids are identical to the HF venv for Llama-3, Gemma-2
    and Qwen3 (`render_parity.py`); a GPU smoke test of `VllmReader` passes. Chained queues
    (`phase2_queue.sh`) start fixed-prompt baselines on all 23 tasks and the token-level
    validity control (GReaTer-style single-token edits, same seed/tasks/target as the block
    study) as soon as each GPU's v3 queue finishes.

23. **Bug (Gemma-2 sampling) found and fixed.** `sample_reasonings`, label-free proposals
    and textual-gradient feedback stopped at the generation config's EOS ids; Gemma-2's
    config lists only `<eos>`, so sampled text could run past `<end_of_turn>` (and sampled
    drafts kept it before the extractor, which can misjudge sample correctness). Fix:
    `sampling_stops` = generation-config EOS + chat end-of-turn tokens. Llama-3 and Qwen3
    are unaffected (identical stop sets and random draws), so v3 stands. Affected: the
    exploratory Gemma-2 date decision run (fork rho 0.46-0.48) and Gemma-2 samples in
    validity v1; they are marked exploratory and are not used as evidence.

24. **Search-time generation service.** `graft_genserver.py` runs a vLLM engine beside the
    HF scorer on the same GPU; `run_graft --gen-server` sends greedy reasoning, self-samples,
    proposals and textual feedback there (HF only computes losses and gradients). All eight
    Stage C variants and `select_and_test.py` run end to end on a small test model (with
    head_dim 128: vLLM's FlexAttention fallback for head_dim 16 crashed, FlashAttention is
    what real models use). Found and fixed on the way: `build_targets` used
    `dataclasses.replace` without importing it, which would have crashed every Stage C run
    with the answer or fork objective.

25. **GReaTer's loop, as implemented.** The official code shortlists token candidates by
    the fixed-reasoning gradient, then regenerates reasoning under every shortlisted
    candidate and keeps the one with the fewest answer errors on the minibatch plus a
    prompt-perplexity penalty (`Total = Mistakes + 0.02 * ControlLoss` in its logs). The
    gradient therefore only decides which candidates are tried; `patch` vs `random`
    shortlists in Stage C isolate exactly that contribution at the block level.
26. **Queued pipeline (server, chained):** v3 -> phase 2 (fixed-prompt baselines on 23
    tasks with vLLM; token-level validity control on LD7/TS7) -> phase 3 (one real-model
    smoke run with the generation server, then Stage C: 357 runs, Llama-3 on GPU 0,
    Qwen3 on GPU 1, then Gemma-2 shared). Stage F (official GReaTer on our train split,
    Llama-3, timed) is scripted (`greater_rerun.sh`) and waits for free GPU time.

27. **v3 interim, 5 of 7 decision runs (LD7 L/Q, LD3 L, TS7 L/Q), 2026-09-29 14:30 UTC.**
    Pooled Spearman with dev change: GReaTer answer loss -0.37 [-0.37, -0.13], fork margin
    -0.12 (exact) / -0.11 (patch), fresh-8 +0.22, fresh-24 +0.39 [+0.12, +0.45]. Paired
    differences: fork - answer +0.25 (P>0 0.95); fork - fresh-8 -0.34 (P>0 0.013); fork -
    fresh-24 -0.50 (P>0 0.000); same on fork-eligible rows. Causal check on 4 runs: mean
    |dP_fork| 0.16-0.22 vs |dP_ctrl| 0.14-0.23 (forks not more decisive than random
    positions). H-fork is failing its preregistered criteria; the answer-loss result is
    robust. Remaining: LD3 Qwen, state-variation runs, LD3 causal checks.

28. **v3 replication complete (2026-09-29 19:04 UTC): H-fork is not supported.** Eight
    decision runs (Llama-3 and Qwen3 on LD7, LD3, TS7, plus a non-initial prompt state on
    LD7 for each model; 20-26 edits, 200 held-out questions) and six causal checks.
    Pooled Spearman with held-out change: GReaTer answer loss -0.31 [-0.31, -0.10] (exact),
    -0.20 [-0.28, -0.07] (patching); fork margin -0.13 / -0.14; fresh accuracy on 8 rows
    +0.22 [+0.01, +0.26], on 24 rows +0.30 [+0.08, +0.33]. Preregistered criteria:
    (a) fork - answer +0.17 [-0.06, +0.32], P>0 0.90 (not established); (b) fork - fresh-8
    -0.36 [-0.41, -0.01], P>0 0.017 (fails; fresh accuracy is better); (c) causal:
    signed dP_fork - |dP_ctrl| -0.16 [-0.20, -0.11], and even |dP_fork| - |dP_ctrl| is
    -0.01 [-0.04, +0.03] (first divergences are not more decisive than random positions).
    Robust finding: GReaTer's objective ranks block edits in the wrong direction, driven by
    rows with wrong reasoning (answer loss on wrong rows -0.61, -0.48, -0.29, -0.52 for
    Llama-3). Dev reliability varies from 0.01 (Qwen3 state run: no signal) to 0.90.

## Next

* Fidelity v2 decides the scorer: renormalized patching must beat raw gates and
  approach exact scoring at lower cost; otherwise GRAFT uses exact-vertex scoring and
  the contribution shifts to exact structural edit calculus + structure-level search.
* Pilot optimizer runs (`run_graft.py`, scorers patch/gate/exact/random) on a few
  tasks; then the preregistered subset (4–6 BBH tasks + GSM8K or FOLIO, Llama-3 and
  Gemma-2, 3 seeds) and the published-GReaTer / ZS-CoT baselines
  (`evaluate_prompts.py`).
