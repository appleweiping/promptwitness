# GRAFT research status (living document)

Branch `research/graft`. ARIS stage: Workflow 1 done (idea + cross-model review),
Workflow 1.5 (experiment bridge) in progress. Target (user instruction of 2026-10-01): an
ICLR main-track submission in the ICLR template, GReaTer-level protocol, scale and writing
quality, high novelty. The ICLR 2027 deadline (Sep 25, 2026) has passed and nothing was
submitted, so the goal is a submission-ready paper for the next ICLR-class deadline.
From 2026-10-01 the ARIS executor and reviewer are both Claude Opus 5.5 (user instruction);
reviews run in fresh-context subagents that read the repository and raw results directly,
recorded as same-family reviews.

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

29. **Mechanism probe (own-answer loss, 3 usable runs; suggestive only).** On wrong-reasoning
    rows, the change in the gold answer's loss and in the model's own answer's loss are
    anti-correlated across edits (Llama-3 LD7 state -0.91, Qwen3 LD7 state -0.33, Llama-3
    LD3 -0.13): GReaTer's objective favors edits that move probability from the answer the
    reasoning supports to the gold answer, and such decoupling predicts lower held-out
    accuracy on the Llama-3 state run (-0.40). Too few wrong rows elsewhere for a claim.

30. **Exploratory, negative: gradient-guided evaluation rows.** Choosing the 8 (or 12)
    training rows on which the answer-loss change varies most across edits, then ranking
    edits by fresh accuracy on those rows, does not beat random rows on the v3 data
    (mean rho +0.15/+0.21 for exact/patch top-8 vs +0.19 random-8; +0.19/+0.27 vs +0.24
    for 12 rows). Not pursued. Literature review (`GRAFT_LITERATURE.md`) added: GEPA (ICLR
    2026 oral), p1 (variance decomposition), HbBoPs, OPRO replication, CoT faithfulness.

31. **Token-level control (3 of 4 runs).** GReaTer-style single-token edits change
    held-out accuracy by about +-2 points (SD 0.020-0.024) while flipping the greedy answer
    on 16-22% of the 200 questions; the target's split-half reliability is 0.05-0.11, so no
    predictor (gradient or evaluation) can be validated at this granularity. Block edits
    flip 6-31% with SD 0.02-0.07. Greedy reasoning is non-local in the prompt: even one
    token re-rolls a fifth of the answers, which is the regime where fixed-reasoning
    first-order signals cannot work.
32. **Reader bug (Qwen3) fixed before any baseline or Stage C result:** LaTeX-wrapped
    answers (`oxed{}`) were unparsed and 8 answer tokens truncated numbers; Qwen3
    reasoning exceeded 512 tokens on causal judgement (60/69). Parser fixed and tested,
    16 answer tokens, 1024-token budgets (plan amendment). Old-reader baselines archived;
    re-running. Killing a vLLM run must include its `VLLM::EngineCore` child (an orphan
    held 30 GB on GPU 0 for 40 minutes).
33. **Cross-family review round 3 (GPT-6):** C1 yes (conditional), C2 partial (fidelity; no
    speed claim), C3 partial (association moderate, mechanism low), C4 no, C5 pending;
    best framing: re-examination with an exact instrument (borderline to weak accept if
    clean); novelty 6, significance 5. Adopted: Stage C paused behind a real-model smoke,
    fork variants dropped, split-provenance audit (41% of BBH test rows were seen by
    GReaTer's optimization), prospective H-cond test on untouched TS3 (plan amendment).

34. **Fixed-reader baselines (vLLM, 23 tasks; dev selects among <= 12 published prompts).**
    Llama-3: ZS-CoT 60.2, GReaTer initial 61.7, GReaTer published 65.6 (21 tasks);
    published - initial +4.7 [+2.5, +7.0] on the full test split, +3.4 [+0.5, +6.4] on the
    clean subset GReaTer never saw; largest gains tracking-5 (+32), geometric shapes (+22),
    object counting (+18). Qwen3 (no published prompts): ZS-CoT 82.3, initial 81.6, near
    ceiling on many tasks. Gemma-2 running. Split audit: 41% of BBH test rows fall in
    GReaTer's train/monitor rows.
35. **Real-model smoke (Qwen3, object_counting, 2 rounds, generation server):** ~60 s per
    round, HF peak 18.6 GB + server 18.9 GB; accepted readable procedural edits. Stage C
    (273 runs) armed per GPU after the TS3 test; GPU 0 first runs the official GReaTer
    gradient-vs-random shortlist ablation (plan amendment).

36. **Gemma-2 baselines (fixed reader, 21 BBH tasks so far):** ZS-CoT 71.4, initial 70.9,
    published 73.9; published - initial +3.3 [+1.2, +5.2] full, +3.3 [+0.6, +6.1] clean,
    13-14 wins of 20. GReaTer's published gains replicate on both of its models on data it
    never saw. Paper restructured to the re-examination framing (validity, mechanism,
    forks, token non-locality written with v3 numbers; end-to-end pending); SEPO (typed
    structural editing with evaluation feedback, Aug 2026) cited.

37. **TS3 prospective test (untouched task, 3 models incl. Gemma-2 with the sampling fix).**
    H-cond fails (no gain over all-row answer loss; worse than random, P 0.001) and is
    retired. The main result replicates out of sample with reliable targets (ceilings
    0.90-0.93): answer loss -0.13 / -0.20, fork ~0, fresh-8 +0.43, fresh-24 +0.39; best-3
    regret of the patched answer-loss shortlist 0.114 vs 0.057 for a random ranking.
38. **Compute incidents.** Official GReaTer crashed at import (`np.infty`, NumPy 2); fixed
    in our copy (semantically identical `np.inf`). Stage C `patch` runs OOM beside the vLLM
    server on long-reasoning tasks (fp32 gated attention is quadratic in length):
    candidates per superposed pass 32 -> 8 (exact per candidate) and server share 0.40 ->
    0.36; the two failed Qwen3 patch runs are retried at the end. GPU 0 was taken by
    another user's job (35 GB) after we freed it; our GPU-0 schedule waits for it.

39. **Stage C v2 running (140 runs; plan amendment).** Measured ~20-33 min per search run
    (fresh verification dominated); the shortlist is now verified in one batched call.
    GPU 1 runs Qwen3, then Llama-3 (shared through run claims), then Gemma-2 HF-only;
    GPU 0 waits for another user's job to finish, then runs the official GReaTer
    gradient-vs-random ablation and joins Llama-3. Memory: the vLLM server needs a 0.40
    share (0.36 leaves 0.3 GB KV cache for Qwen3); the scorer uses 8 candidates per pass.
    Always kill a server's `VLLM::EngineCore` child with it.

40. **ARIS auto-review loop, round 1 (GPT-6 via Codex; `review-stage/`, not committed):
    5/10, not ready.** Verified numbers match the files. Found and fixed: the "random
    ranking" comparator was a fixed ordering (now the exact uniform-random expectation;
    the patched answer-loss shortlist is still worse than random, by 0.024 [0.006, 0.033]
    best-3 regret over eleven runs); the random scorer shared the minibatch RNG (separate
    streams; two Qwen3 random runs re-run); overclaims on attribution, token "re-rolls"
    and the decoupling mechanism (own-answer probe inconsistent: -0.91, -0.33, -0.13,
    +0.04, +0.86) rewritten as measured facts and hypotheses; Stage C analysis now pairs
    seeds and gives conditional and task-level intervals. Launched a direct re-roll
    measurement (`reroll_study.py`) through the Qwen3 server; GReaTer ablation gets two
    repetitions per variant.

41. **Re-roll measured directly (Qwen3, 200 held-out questions, `reroll_study.py`).**
    Single-token edits (24 per task) change the greedy reasoning on 92% (LD7) / 78% (TS7) of
    questions (min 64% for any edit), first divergence after a median of 53 / 40 tokens;
    answers change on 27% / 36%; correctness flips on 14.5%. Block edits: reasoning changed
    on 99% / 97% after 9 / 14 tokens; answers 33% / 54%; correctness 17% / 23%. Token edits
    are somewhat more local but not local. Llama-3 measurement pending (needs its server).

42. **Stage C interim, Qwen3 (seeds 1-2 complete; seed 3 and two re-run random runs
    pending; evaluated through the Qwen3 vLLM server, same path for all methods and
    fixed prompts).** Mean test accuracy over 7 tasks: patch 88.6, textgrad 88.3, exact
    87.6, random 86.0, GReaTer initial 83.5, ZS-CoT 83.5. patch - random +2.5
    (conditional [+0.4, +4.5], task-level [-0.7, +6.6], W/T/L 4/0/3); patch - initial +5.1
    (movie recommendation 38 -> 66). Search+selection minutes per run: patch 32, exact 30,
    textgrad 27, random 20. Interim only; no decision is taken on it.

43. **Scheduling with one GPU (operational, no change to any analysis).** GPU 0 has been
    held by another user's job for ~22 h. GPU 1 order: Qwen3 Stage C -> its re-runs ->
    official GReaTer 2-step smoke -> GReaTer gradient vs random on tracking-5 -> Llama-3
    seeds 1-2 -> GReaTer on geometric shapes -> Llama-3 seed 3 -> Gemma-2 (HF only) ->
    GReaTer repetition 2 and two more tasks. GPU 0, if freed, draws from the same pools
    (run claims for Stage C and GReaTer). Server-backed evaluation blocks the search
    queue while it runs (6 h for 42 Qwen3 runs).

44. **2026-10-01: two bugs found and fixed; no completed number changes except as stated.**
    (i) `delete_block` could produce an unordered block partition (zero-width block tied in
    offset with a non-empty block earlier in tuple order) after a sequence "empty a block,
    fill the neighbouring empty slot (which moves the emptied block behind it), refill, empty
    again"; the validator then raised, which crashed `qwen3-movie_recommendation-patch-s3`.
    Blocks are now ordered by (start, end), which only changes previously failing cases.
    (ii) Run records serialized checkpoint blocks in tuple order, and `select_and_test.py`
    rebuilt prompts by concatenating them, so a reordered prompt was rebuilt with two blocks
    swapped. Audit over all 63 Stage C run records (`audit_checkpoint_order.py`): 4 of 244
    checkpoints affected (qwen3 navigate patch s2 r12, navigate patch s3 r8, tracking-5
    random s1 r4/r12). Records are now written in message order and the evaluator rebuilds
    from the recorded text (`prompt_from_record`); the affected runs are re-evaluated and the
    movie patch s3 run is re-run before any Stage C number is final.
45. **Official GReaTer smoke (2 steps) crashed on one GPU:** the official transfer configs load
    the same model twice (gradient worker and a reasoning "test worker" on a second GPU); our
    one-copy config had no test worker (`IndexError` in `morph_control`), and the spawned
    worker kept the process alive holding 37 GB on GPU 1. Fix
    (`scripts/research/graft/greater_single_gpu.patch`): the single model serves both roles
    (identical weights, so the algorithm is unchanged), and workers are stopped in a
    `finally` block. The runtime script's claim name now includes the steps suffix, so a smoke
    run no longer claims the full run.
46. **Opus 5.5 strategic review (ARIS research-review, 2026-10-01):** as is, 4-5/10 at ICLR
    (negative headline, no positive method; strawman risk; instrument without a customer;
    missing citations incl. "Textual Gradients are a Flawed Metaphor", Library of Babel,
    Faster-GCG gradient/loss concordance, AttriBoT, Block-Attention/EPIC, coupled generation).
    Recommended direction: gradients over the *distribution* of reasoning (exact likelihood
    ratios of self-generated reasoning under each edit; GReaTer as the fixed-reasoning
    special case), gated by a preregistered kill test before any further investment.

47. **Opus 5.5 auto-review round 1 (nightmare, 3/10, `review-stage/`).** Validity numbers verified
    exactly and robust at pool level (answer-exact rho negative in 8/11 pools); found: vLLM reader
    nondeterminism (prefix caching; same-prompt re-reads flip 2-14% of answers), a truncation
    artifact (selected movie_recommendation prompts exhaust 1,024 tokens on 90-100% of test
    questions), stale H-cond number (-0.057 should be -0.029 [-0.048, -0.002]), the TS3 *exact*
    objective is null (rho -0.126; regret 0.080 vs 0.085 random), Qwen3 wrong-row values omitted,
    biased percentile intervals, truncated proposals accepted, testability overclaim. Fixes and
    the H-dist kill test were preregistered (plan amendment 2026-10-01) before any new data.
48. **Pipeline regret (zero GPU, `pipeline_regret.py`, 11 block pools, one round, shortlist 3,
    verification by fresh greedy accuracy on training rows 0-7, accept if better than the
    incumbent).** Mean realized held-out change of the accepted edit: answer-loss shortlist -3.27
    points (exact) / -3.23 (patched), random shortlist -1.34, fresh-24 shortlist -0.18,
    verify-all -2.27; the best edit per pool would give +5.50. Verifying all candidates on 8 rows
    is worse than a random shortlist (winner's curse); verify-all on 16 rows -0.77, on 24 rows
    +1.14 (margins of 1-3 net rows do not fix 8-row verification: -1.18 to -0.27). The
    bottleneck of shortlist-then-verify self-optimization is verification noise, not the
    shortlist signal. Rank-normalizing each row's loss changes leaves the Llama-3 wrong-row
    anti-correlation (-0.25 to -0.61) but not Qwen3's (+0.25, -0.04, +0.07): the wrong-row
    concentration is a Llama-3 finding, not a general one.
49. **Queue v3 on GPU 1 (user-approved restart, 12:46 UTC):** the hung GReaTer smoke and our old
    master queue were stopped; queue: fixed GReaTer smoke -> Llama-3 (reader null, H-dist pools,
    re-roll with null) -> Qwen3 (same) -> Qwen3 null with prefix caching on -> Gemma-2 H-dist
    (HF only) -> Llama-3 Stage C seeds 1-2. Queue file `cc/gpu1.queue` (editable while running).

50. **H-dist interim, 4 of 11 pools (Llama-3; exploratory reading, no decision):** primary
    (tempered SNIS, candidate reads) rho +0.54, +0.47, +0.19, **-0.67** (LD7, LD7 state, LD3,
    TS7); incumbent-read variant similar; GReaTer objective -0.55, -0.48, -0.41, -0.66; fresh-8
    +0.48, -0.07, +0.27, +0.48. On TS7 the score favours edits that impose elaborate procedures,
    which shift the reasoning distribution by 35-50 nats (mean log w) and are the worst edits on
    held-out (-10 to -17 points): off-policy evaluation fails where a candidate's reasoning lies
    outside the incumbent's samples, and self-normalization hides it. A label-free coverage gate
    (route edits whose samples carry less than one unit of the candidate's mass, median log
    sum_s w_s < 0, to fresh-8) helps on TS7/LD3 and hurts on LD7/LD7-state; not preregistered,
    reported as exploratory only. Read-off flips 0.2-3.2%, |A| << |D| on all four pools.
    Wrong-direction finding for the fixed-trace objective holds on all four.
51. **H-dist interim, pool 5 (Llama-3 TS3, untouched task):** primary -0.39, incumbent-read -0.49,
    fresh-8 +0.36, GReaTer objective +0.08. On Llama-3 the importance score is positive on the
    three logical-deduction pools and negative on both tracking pools; H-inc (pools 2-5 so far:
    +0.47, +0.19, -0.67, -0.49) is unlikely to pass. The registered test is completed on all
    pools regardless; read-off flips 0.3%, |A| 0.001 vs |D| 0.069 on this pool.
52. **Reader nondeterminism, second source (2026-10-01 16:45 UTC).** Without prefix caching the
    first null check (3 reads, 200 questions) was token-identical, but the null inside the
    Llama-3 token re-roll run changed the greedy reasoning on 54.5% of questions in one re-read
    and 0% in the other. Cause (most likely): with a 0.40 memory share the KV cache holds far
    fewer tokens than 200 concurrent sequences need, so vLLM preempts and recomputes sequences,
    which changes numerics. Fix: `--max-num-seqs` cap so that no preemption can occur
    (`MAXSEQS` in `ops/with_server.sh`), verified by same-prompt probes before any evaluation
    job uses a configuration; re-roll studies are re-run under a verified configuration. Search
    runs may stay nondeterministic (noise shared by all arms); every dev/test read must be
    reproducible. The H-dist samples and reads used the uncapped server: numerical noise only
    (it can only inflate the measured read-off flips, which is conservative for |A| << |D|).
54. **Independent second opinion on H-dist (Opus 5.5, 5 pools):** the importance score is noise
    (split-half reliability -0.31..+0.26; agreement with the on-policy estimate of the same
    quantity +0.03/-0.32/+0.10/-0.52/-0.22; per-pool percentile CIs all include 0, so the
    deduction-vs-tracking split is not an effect), explained by coverage: KL 4.5-14.6 nats per
    trace vs ~exp(KL) samples needed (Chatterjee & Diaconis 2018); optimistic bias growing with
    distance on the tracking pools. Corrections adopted: |A| re-based on on-policy totals (the
    SNIS D was at its permutation null); the "e^20" statement replaced by KL. Amendment added
    before the remaining pools (reliability, on-policy A, KL gate at 0.5 log S, bf16 null).
    Framing: analysis paper with a prescriptive conclusion -- edits act on the reasoning
    distribution (KL ~10 nats), neither fixed-trace gradients nor off-policy reweighting can see
    it, the bottleneck is on-policy verification noise; headline experiments are Stage C v2 and
    the official GReaTer gradient-vs-random ablation.
55. **GReaTer smoke 2 (18:16 UTC):** single-GPU fixes work through step 1 (regeneration chunks
    11-15 s; ~10-13 min per step, so ~20 h per 106-step run); OOM in step 2 because the spawned
    gradient worker keeps ~20 GB of cached blocks. Fix: the worker empties its CUDA cache after
    each task (`greater_worker_cache.patch`); smoke re-queued.
53. **Queue (operational):** Stage C v2 Llama-3 seed 1 (28 runs) now precedes the remaining H-dist
    pools; the importance-score arm is not in it (H-dist/H-inc unresolved, Llama-3 pools mixed).

## Next

* Fidelity v2 decides the scorer: renormalized patching must beat raw gates and
  approach exact scoring at lower cost; otherwise GRAFT uses exact-vertex scoring and
  the contribution shifts to exact structural edit calculus + structure-level search.
* Pilot optimizer runs (`run_graft.py`, scorers patch/gate/exact/random) on a few
  tasks; then the preregistered subset (4–6 BBH tasks + GSM8K or FOLIO, Llama-3 and
  Gemma-2, 3 seeds) and the published-GReaTer / ZS-CoT baselines
  (`evaluate_prompts.py`).
