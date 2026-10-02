# Results digest (single source of numbers for the paper), 2026-10-02

Every number below is read from a recorded artifact. Local copies of server outputs live in
`D:\Company\research-artifacts\graft-runs\` (not in git). Server runtime:
`/media/lenovo/data2/promptwitness-graft-runtime/`. Status log: `research/GRAFT_STATUS.md`
(items 47-58); plan with dated amendments: `research/GRAFT_EXPERIMENT_PLAN.md`.

## Framing (PAPER_PLAN.md v2)

Analysis paper. One sentence: gradient-over-reasoning prompt optimization (GReaTer) scores an
edit by how it changes the answer read off a *fixed* reasoning; with an exact calculus for
structural edits and preregistered tests on three models we show that edits act almost entirely
by changing the reasoning itself, so (i) the fixed-reasoning signal ranks structural edits in the
wrong direction, (ii) reweighting the model's own samples cannot recover the missing term
(registered test failed), (iii) single-token edits move accuracy like numerical re-rolls of greedy
decoding, and (iv) what self-optimization achieves is decided by the verification budget on fresh
reasoning. End-to-end evidence (Stage C v2, official GReaTer gradient-vs-random ablation, breadth
table) is still running: keep `\todo{}` markers for it, never invent numbers.

## A. Validity of the fixed-reasoning objective (Table 1, `paper/tables/validity_block.tex`)

11 pools (Llama-3-8B L, Qwen3-8B Q, Gemma-2-9B G; LD7/LD3 logical deduction, TS7/TS3 tracking
shuffled objects; * = non-initial prompt state), 20-21 LLM-proposed block edits per pool, 24
training questions, target = change of greedy accuracy on 200 held-out questions (384-token
reasoning budget). GReaTer objective (exact) negative in 8/11 pools, pooled Spearman -0.26,
sign-flip p = 0.017; patched (one-pass) estimate negative 9/11, -0.20, p = 0.037; exact vs patched
mean Spearman 0.79. Fresh greedy accuracy positive in 9/11 (8 rows, +0.28, p = 0.010) and 10/11
(24 rows, +0.33, p = 0.004). Best-3 regret: patched 0.091 vs random 0.067 (worse in 8/11,
p = 0.022); exact 0.075 (7/11, p = 0.14). Wrong-row rank-normalized analysis: negative for Llama-3
(-0.25..-0.61, 5/5 pools), not for Qwen3 (+0.25, -0.04, +0.07).
Pipeline simulation (`pipeline_regret.py`): verify-all on 8 rows -2.27 points, 16 rows -0.77,
24 rows +1.14; best edit per pool +5.50; answer-loss shortlist -3.27 (exact) / -3.23 (patched),
random shortlist -1.34.

## B. Registered H-dist / H-inc (`dist-final/analysis_11pools.{json,log}`, `analysis_hinc10.*`)

S = 16 incumbent samples per training question, tau = 0.7, no top-k/p truncation; exact tempered
log-ratios under every candidate; beta chosen without labels (largest of {1, 1/2, 1/4} with median
ESS >= S/4: beta* = 1/2 in 8 pools, 1/4 in 2, 1 in 1). Kill criterion (fixed before the data):
beat a random shortlist in pooled best-3 regret with bootstrap P >= 0.9 and in >= 7/11 pools, and
not fall > 0.05 below fresh-8 in pooled Spearman.
- primary (SNIS, candidate reads): pooled Spearman +0.034 [BC -0.065, +0.190]; regret 0.075 vs
  random 0.067 (gain -0.009 [-0.027, +0.008], P>0 0.21, lower regret in 6/11 pools); vs fresh-8
  d_rho -0.243 (P>0 0.019). **H-dist NOT SUPPORTED.**
- H-inc (incumbent reads = distribution term only; confirmatory on pools 2-11): Spearman -0.024,
  regret 0.083 vs random 0.069 (6/10 pools, P>0 0.18); vs fresh-8 d_rho -0.280. **NOT SUPPORTED.**
- 11-pool predictors (Spearman [BC], regret): answer_exact -0.258 [-0.310, -0.243], 0.075;
  snis_hard beta 0 / 1/4 / 1/2 / 1: +0.018 / +0.018 / +0.058 / +0.018, regret 0.080 / 0.072 /
  0.073 / 0.062; snis_soft beta 0..1: -0.016 / +0.008 / -0.002 / -0.011, regret 0.072 / 0.068 /
  0.063 / 0.057; inc_primary +0.023; score_fn (first-order) -0.098, 0.089; kl_gate +0.218, 0.052;
  fresh8 +0.277, 0.052; fresh_all (24 rows) +0.325, 0.043; fresh_dist (coupled sampled, 24 rows x
  4) +0.413, 0.035; random 0.067.
- Verify-all realized held-out change (points; pools improved/worsened): fresh8 -2.27 (+3/-4),
  fresh_all +1.14 (+5/-4), primary -2.18 (+5/-6), fresh_dist +1.00 (+5/-5), score_fn -8.45 (+2/-9),
  answer_exact -5.00 (+1/-10).
- Per pool: split-half reliability of primary -0.31..+0.26 (L LD7 -0.31, L LD7* +0.06, L LD3
  -0.26, L TS7 -0.31, Q LD7 -0.12, Q LD7* -0.24, Q LD3 -0.12, Q TS7 +0.14, L TS3 +0.26, Q TS3
  +0.20, G TS3 -0.29). KL median per trace (nats): L 10.2, 4.5, 10.6, 14.6, 8.9 (TS3); Q 3.9, 1.1,
  8.4, 4.9, 6.7 (TS3); G 10.4. Fraction of edits under the KL gate (<= 0.5 log 16 = 1.39 nats):
  0.00-0.52 (median 0.10). Agreement of primary with the on-policy estimate (fresh_dist):
  -0.52..+0.36.
- Chatterjee & Diaconis (2018): importance sampling needs a sample size ~exp(KL).

## C. Read-off term (`paper/tables/readoff.tex` <- `dist-final/readoff_11pools.tex`)

Hard read of the same incumbent sample under the candidate vs the incumbent: flips 0.0-3.2% of
(candidate, question, sample) triples; |A| 0.0-1.2 accuracy points per edit vs mean |held-out
change| 2.1-10.9 points and mean |fresh change| 2.1-10.7; Spearman(A, held-out) -0.31..+0.51
(G TS3 undefined, A = 0); within-question sd of log w 3.4-9.2.

## D. Verification at matched cost (exploratory; `dist-final/verify_cost.{json,log}`)

Same 11 pools; predictor = mean over rows of (candidate - incumbent correctness); 400 random
row/sample subsets per cell; G = generations per candidate. Spearman / regret / verify-all (pts):
G=4: greedy 4x1 +0.178/0.056/-1.71, coupled 4x1 +0.150/0.055/-1.37, indep 4x1 +0.148/0.056/-1.29;
G=8: greedy +0.234/0.052/-1.45, coupled 8x1 +0.195/0.052/-1.19, indep +0.199/0.052/-1.02;
G=12: greedy +0.264/0.049/-1.05, coupled 12x1 +0.235/0.049/-0.79;
G=16: greedy +0.291/0.047/-0.71, coupled 16x1 +0.253/0.048/-0.74;
G=24: greedy +0.325/0.044/+0.41, coupled 24x1 +0.295/0.044/-0.38, indep 24x1 +0.294/0.044/-0.58,
coupled 12x2 +0.281/0.047/-0.33; G=48: coupled 24x2 +0.347/0.042/+0.88, indep 24x2
+0.346/0.042/+0.74; G=96: coupled 24x4 +0.413/0.034/+1.44. Random regret 0.067.
Conclusions: at equal G greedy >= sampled; common-random-number coupling adds nothing over
independent seeds; quality and realized gain rise monotonically with G; verify-all on <= 16
generations lowers held-out accuracy on average.

## E. Determinism and numerical re-rolls (`reroll3/*.json`; deterministic in-process vLLM engine)

- Engine: with prefix caching, same-prompt re-reads changed 2-14% of answers; without prefix
  caching but multiprocess engine core, one of two re-reads changed the reasoning on 55-66% of
  questions (capped concurrency); with the engine core in the calling process, 4/4 re-reads x 200
  questions token-identical at the evaluation (4,096 tok.) and search (384 tok., KV cache for ~5
  sequences of 4,096 tokens, i.e. heavy preemption) configurations.
- Single-token edits (GReaTer-style substitutions; 24 per pool; 384-token budget; 200 held-out
  questions). Reasoning changed / answer changed / correctness flipped / first divergence (median
  tokens) / accuracy over edits mean +- sd (base accuracy):
  L LD7: 87.5% / 40.7% / 20.3% / 110 / 0.462 +- 0.022 (base 0.505)
  L TS7: 83.7% / 44.2% / 19.0% / 164 / 0.319 +- 0.020 (base 0.275)
  Q LD7: 92.9% / -- / 14.6% / -- / 0.521 +- 0.020 (base 0.520)
  Q TS7: 79.2% / -- / 16.1% / -- / 0.273 +- 0.020 (base 0.280)
  Same-prompt null re-reads: 0% everything.
- Numerical re-rolls: the unchanged incumbent read in separate requests of 16/25/40/64/100
  questions (each deterministic, different batch shapes): reasoning changed L LD7 67-71%, L TS7
  56-61%, Q LD7 83-86%, Q TS7 65-73%; correctness flipped 13-16%, 8-13%, 14-19%, 10-13%; first
  divergence median ~135-155, ~175-190, ~67-77, ~41-56 tokens; accuracy over re-rolls incl. base:
  L LD7 0.467 +- 0.018, L TS7 0.301 +- 0.022, Q LD7 0.517 +- 0.009, Q TS7 0.271 +- 0.009.
- Reading: mean accuracy over token edits equals mean over re-rolls in all four pools; the spread
  is similar for Llama-3 and about twice the re-roll spread for Qwen3 (needs the formal
  exchangeability test, `reroll_analysis.py`). Greedy accuracy on 200 questions carries a
  numerical noise floor of ~1-2 points (sd) even with a deterministic engine; block edits move
  held-out accuracy by 2.1-10.9 points on average.
- Truncation: at the validity budget (384 tokens) incumbent samples hit the budget on 97% (L
  LD7*) and 100% (Q LD7) of samples, 0.3% (L LD3), 14% (L TS7); the fixed-reasoning objective is
  negative on low-truncation pools too (L LD3 -0.41, L TS7 -0.66).

## F. Official GReaTer on one GPU (appendix)

Official code (`greater-official`), Llama-3-8B, our 50 training rows, 106 steps, topk 40, topq 6,
batch 64. Memory-only patches (repo `scripts/research/graft/greater_*.patch`): the single model
serves gradient and reasoning workers; regenerate 1/8 of rows per call (all rows still evaluated);
candidate logits in batches of 9 and freed; the gradient worker's weights frozen (GReaTer's backward
also computed ~16 GB of unused weight gradients; `one_hot.grad` unchanged). Smoke (2 steps): 31
min, peak 33 GB of 48 GB. Shortlist ablation: `GREATER_SHORTLIST=random` draws the shortlist
uniformly from the same model-proposed candidates. Runs queued (TS5, geometric shapes; gradient
and random).

## G. Pending (keep \todo)

Stage C v2 (Llama-3, 7 tasks, patch-v50 / random-v50 / random-v8 / textgrad-v50, seeds 1-3,
50-row verification, deterministic reader, dev selection, 4,096-token test); official GReaTer
gradient vs random; breadth (21 BBH + GSM8K + FOLIO, Llama-3 + Gemma-2); block re-reads of Table 1
targets under the deterministic reader (with numerical re-rolls and truncation).
