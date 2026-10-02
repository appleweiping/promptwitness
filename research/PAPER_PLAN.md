# Paper plan (ICLR format), v2 2026-10-01 — analysis framing

Supersedes the v1 plan (distributional scoring as the method), which was conditional on H-dist;
the interim H-dist evidence (5 Llama-3 pools: split-half reliability -0.31..+0.26, KL 4.5-14.6
nats per trace) makes a positive importance-sampling method untenable. H-dist is still completed
on all 11 pools and reported as registered.

Working title: **"What Gradients over Reasoning Miss: Prompt Edits Act on the Reasoning, Not on
the Read-Off"** (alt.: "Prompt Edits Rewrite the Reasoning: Limits of Gradient-over-Reasoning
Prompt Optimization").

## One-sentence contribution

Gradient-over-reasoning prompt optimization scores an edit by how it changes the answer read off
a *fixed* reasoning; with an exact calculus for structural edits and preregistered tests on three
models we show that edits act almost entirely by changing the reasoning itself, so the
fixed-reasoning signal ranks structural edits in the wrong direction, reweighting the model's
own samples cannot recover the missing term, and what self-optimization achieves is decided by
how candidates are verified on fresh reasoning.

## Causal spine

gap (GReaTer's signal is a gradient over a fixed reasoning; prompts are edited as structures) ->
question (does the signal predict what structural edits do?) -> finding (no: wrong direction) ->
insight (exact decomposition D + A; GReaTer sees only A; A is small, D dominant; KL ~10 nats) ->
consequence 1 (off-policy reweighting cannot cover the shift: preregistered failure) ->
consequence 2 (the only reliable signal is fresh reasoning; verification noise then dominates:
winner's curse) -> end-to-end test (shortlist signal vs verification size, structural and
official GReaTer) -> implication (spend compute on verification, not on the shortlist signal).

## Claims-evidence matrix

| Claim | Evidence | Status |
|---|---|---|
| C1 The fixed-reasoning objective ranks LLM-proposed block edits in the wrong direction | 11 pools x 3 models, pool sign-flip test (8/11 negative, p=.017; patched 9/11, p=.037); fresh positive 9-10/11; best-3 regret vs exact random expectation | done (Table 1) |
| C2 Edits act on the reasoning distribution; read-off term is small and unpredictive | readoff_stats on-policy A vs held-out; KL per trace; flips 0.2-3.2% | 5 Llama-3 pools done; Qwen3 5 + Gemma 1 queued |
| C3 Reweighting own samples (tempered SNIS, beta without labels) does not rescue the signal | H-dist preregistered kill test, H-inc; reliability, KL gate, numeric null; beta figure | 5/11 pools; queued |
| C4 Verification on small minibatches chases noise | pipeline_regret (8/16/24 rows, margins) + Stage C v2 random-v8 vs random-v50 | simulation done; Stage C v2 running |
| C5 With full verification, the shortlist signal does not matter (structural) | Stage C v2: patch-v50 vs random-v50 (paired by seed), textgrad-v50 as different class | seed 1 queued (resumes after H-dist pools) |
| C6 Same question inside official GReaTer (token level) | gradient vs uniformly random shortlist from the same candidates, official code | smoke 4 queued |
| C7 Even single-token edits rewrite the reasoning | re-roll study under a verified deterministic reader | needs determinism probe (in-process engine) |
| C8 Structural self-optimization with full verification is competitive at GReaTer's scale and cost | breadth: 21 BBH + GSM8K + FOLIO, Llama-3 + Gemma-2, vs ZS-CoT, GReaTer initial/published (clean subset) | after Stage C v2 |

Outcome branches: if C5/C6 show the gradient shortlist *beats* random, the implication becomes
"the gradient helps only as a cheap pre-filter before full verification" (still consistent with
C1-C4 since the shortlist is verified on fresh reasoning). If C8 is below GReaTer published
numbers on the clean subset, the breadth table is reported as a cost/accuracy tradeoff, not a win.

## Sections

1. Introduction (1.5 p): hook = self-optimization for small models; gap; question; findings
   C1-C3 with numbers; prescription C4-C6; contributions list; Figure 1.
2. Background (0.5 p): GReaTer objective and loop; notation.
3. Structural edits and an exact calculus (1 p): typed blocks; superposed gated slots; exactness
   (proofs in appendix); role = makes the lifted objective computable for every block edit in one
   pass (patched vs exact rho 0.79).
4. Diagnosis (1.5 p): C1 table; token-level C7; wrong-row analysis (Llama-3 only).
5. Why: decomposition (1.5 p): Eq. D + A; Proposition (faithful reading); C2 table; C3 with the
   coverage argument (Chatterjee-Diaconis) and the registered result; beta figure.
6. Verification is the bottleneck (2 p): C4 simulation; Stage C v2 (C5); official GReaTer
   ablation (C6); breadth (C8).
7. Related work (1 p). 8. Limitations + conclusion (0.5 p).
Appendix: proofs, preregistration log with dates, per-pool tables, reader determinism, costs,
prompts, GReaTer patches.

## Figures/tables

F1 hero: (a) fixed trace vs distribution (an edit moves pi_P to pi_P', KL ~10 nats; GReaTer's
gradient lives on one trace of pi_P); (b) the decomposition bars per pool (|A| vs |D|).
F2 validity per signal (pool dots, Table 1 companion). F3 beta curve. F4 verification noise grid.
T1 validity; T2 read-off; T3 Stage C v2; T4 GReaTer ablation; T5 breadth; T6 costs (appendix).

## GPU order (one A6000; GPU 0 belongs to another user)

1. Determinism probes (in-process engine) -> GReaTer smoke 4 -> Qwen3 H-dist 5 pools ->
   Qwen3 prefix-cache null -> Gemma-2 H-dist (HF) -> Stage C v2 seed 1 (resume, ~19 h) ->
   re-reads (deterministic config) -> Qwen3 token pools.
2. Then: GReaTer ablation (~20 h per run; 2 tasks x 2 arms first), Stage C v2 seeds 2-3
   (49 runs, ~30 h), breadth (~46 runs + evaluation), re-roll studies.
