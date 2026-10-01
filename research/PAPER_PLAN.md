# Paper plan (ICLR format), 2026-10-01 — conditional on H-dist / H-inc

Working title: **"Score Prompt Edits by How They Reshape Reasoning: Distributional
Self-Optimization of Structured Prompts"** (alt.: "What Gradients over Reasoning Miss").

## One-paragraph story

GReaTer lets a small model optimize its own prompt with the gradient of the answer loss
computed over its own, *fixed*, reasoning. We lift this signal from tokens to the structural
edits practitioners make (typed blocks; an exact calculus makes every structural vertex a
real prompt) and find, in preregistered experiments on three models, that it ranks block
edits in the wrong direction (pool level). The cause: an edit reshapes the *distribution* of
the model's reasoning (log-likelihood ratios of the model's own samples spread by tens of
nats; even a single-token edit rewrites the greedy reasoning on most questions, against a 0%
deterministic-reader null), and the fixed-trace objective only sees the answer read-off. We
decompose an edit's effect on expected accuracy into a reasoning-distribution term and a
read-off term; GReaTer's objective is the read-off term on one trace. Scoring edits by exact
likelihood ratios of the model's own sampled reasoning (tempered self-normalized importance
sampling, beta chosen without labels) recovers the missing term without generating under any
candidate; it [ranks edits as well as or better than fresh evaluation; pending all pools] and
[end-to-end results pending]. Verification noise explains why gains of self-optimization are
fragile: greedy minibatch verification on 8 rows chases noise (winner's curse).

## Sections -> evidence

1. Introduction (contributions: diagnosis, decomposition, distributional scoring, end-to-end).
2. Background: GReaTer's objective and its shortlist-then-verify loop.
3. Structural edits: typed blocks; exact superposed calculus (short; details/proofs in appendix);
   its role: exact vertices + one-pass estimates of likelihood ratios (trace_fidelity.py).
4. Diagnosis (validity study, 11 pools, 3 models; pool-level sign-flip tests; BC intervals):
   GReaTer objective negative 8/11 (p=.017); fresh positive 9-10/11; token level; re-roll with
   deterministic null; wrong-row analysis (Llama-3 only after rank normalization).
5. Theory: decomposition (D + A), GReaTer = A on one trace, first-order = score function,
   wrong-row proposition, verification-noise model.
6. Method: distributional scores (SNIS family, ESS rule, incumbent reads), cost, one-pass weights.
7. Experiments:
   7.1 Validity of distributional scores (H-dist primary; H-inc on 10 unseen pools); beta figure.
   7.2 Pipeline: verification noise (8/16/24 rows), verifiers compared (pipeline_regret, analyze_dist).
   7.3 End-to-end Stage C v2 (Llama-3 primary; patch/random/dist/textgrad at 50-row verification;
       random-v8 control); accuracy vs measured compute; truncation reported.
   7.4 Official GReaTer: gradient vs random shortlist (token level), 2+ tasks x 2 reps.
   7.5 Breadth (GReaTer-scale table): best configuration on 21 BBH + GSM8K + FOLIO, Llama-3 and
       Gemma-2, vs ZS-CoT, GReaTer initial/published (clean subset), TextGrad-style.
   7.6 Transfer/readability (if time).
8. Related work (GReaTer & token-gradient methods incl. Faster-GCG mismatch; structural APO incl.
   SEPO, SAPO, PCO, GEPA; evaluation noise & prompt sensitivity incl. FormatSpread, nondeterminism;
   off-policy / coupled evaluation; block-wise KV/positional encoding as prior art for the calculus).
9. Limitations (MC-heavy validity tasks; 8-9B models; one GPU; same-family reviewer).
Appendix: proofs; preregistration log with dates; per-pool tables; reader determinism; costs.

## Figures/tables

F1 overview (token vs block; fixed trace vs distribution). F2 validity per signal (pool dots).
F3 beta curve (dist_figure.py). F4 verification noise (pipeline grid). T1 validity table
(per pool). T2 end-to-end Stage C v2. T3 breadth table. T4 GReaTer ablation. T5 costs.

## Open experiments (queue order)

1. H-dist/H-inc pools (running); re-roll with null; Qwen3 prefix-cache null.
2. Re-reads: baselines (23 tasks x 3 models) and Stage C v1 Qwen3 (4,096 tokens, deterministic).
3. Stage C v2 Llama-3 (dist arm only if supported).
4. Official GReaTer gradient vs random (needs the 1/8-fraction fix verified by the smoke).
5. trace_fidelity (one-pass weights) on pools with stored traces.
6. Breadth with the best configuration.
