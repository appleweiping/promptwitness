# Theory notes for the distributional re-examination (draft, 2026-10-01)

Status: working notes for the paper rewrite; claims here are mathematical, not empirical.

## Setup

Model f, prompt P, question x with gold answer y. Reasoning r ~ pi_P(. | x) (temperature tau),
then an extractor and a greedy answer read; c_P(x, r) in {0, 1} is the correctness of that read,
q_P(y | x, r) the gold-answer probability after the extractor (soft read). Expected accuracy

    J(P) = E_x E_{r ~ pi_P(.|x)} c_P(x, r).

## Decomposition of an edit's effect

For an edit P -> P' and w(r) = pi_P'(r | x) / pi_P(r | x),

    J(P') - J(P) = E_x E_{r ~ pi_P} [ (w(r) - 1) c_P'(x, r) ]      (D: reasoning distribution)
                 + E_x E_{r ~ pi_P} [ c_P'(x, r) - c_P(x, r) ]       (A: answer read-off)

(exact; it only uses E_{pi_P}[w g] = E_{pi_P'}[g]). GReaTer's objective,
-log q_P'(y | x, r_greedy), is a soft, single-trace version of A on the incumbent's greedy trace;
its token gradient is the first-order change of that quantity. It contains no estimate of D.

First order in a small edit: E[(w - 1) c] ~ E[(log w) c] = Cov_{pi_P}(log w, c) (since
E[log w] ~ 0 to first order) -- the score-function ("REINFORCE") term: an edit helps if it
makes the model's own correct reasoning more likely relative to its incorrect reasoning.

## Estimators (shared samples, no generation under candidates)

Draw r_1..r_S ~ pi_P^tau once per question. For every candidate, exact teacher-forced
log w_s (or a one-pass superposed estimate). Tempered self-normalized importance sampling

    J_beta(P' | x) = sum_s w_s^beta g_s / sum_s w_s^beta,   g_s in {c_P'(x, r_s), c_P(x, r_s)}

beta = 0 is the fixed-reasoning family (no reweighting; with soft reads, GReaTer's objective
averaged over samples); beta = 1 is consistent for J(P') (as S -> infinity); intermediate beta
trades bias for variance. With g_s = c_P(x, r_s) (incumbent reads), the estimate isolates D;
for small beta it is ~ beta Cov_s(log w_s, c_s) + mean c. beta is chosen without targets
from the effective sample size ESS = (sum w^beta)^2 / sum w^(2 beta).

## Why the read-off term mis-ranks edits on wrong rows (proposition sketch)

Let r be a trace whose reasoning concludes a != y and whose read is faithful under P:
q_P(a | x, r) >= 1 - eps. For any P' with q_P'(y | x, r) > q_P(y | x, r) + delta,

    q_P'(a | x, r) <= 1 - q_P'(y | x, r) < 1 - q_P(y | x, r) - delta <= ... ,

i.e. lowering GReaTer's loss on r requires moving probability away from the answer the
reasoning supports (decoupling the answer from the reasoning). Such an edit changes A on the
fixed trace but not the probability that the model produces correct reasoning (D), and
decoupling is not rewarded on held-out questions where correct answers come from correct
reasoning. Empirically (rank-normalized): the wrong-row anti-correlation holds for Llama-3
(-0.25 to -0.61) but not for Qwen3 -- the proposition says when the fixed-trace signal *can*
mislead, not that it must.

## Why fixed traces are fragile under edits (re-roll)

KL(pi_P || pi_P') = -E_{pi_P} log w has a median of 4.5-14.6 nats per trace across the Llama-3
pools (within-question sd of log w 4.6-9.2 nats; the pooled sd across rows and edits, ~20 nats,
mixes between-question variation and is not the relevant quantity). The greedy trace under P is
typically not a likely trace under P'. A first-order signal on one fixed trace cannot represent
this change -- and neither can self-normalized importance sampling with S = 16 samples, which needs
about exp(KL) samples (Chatterjee & Diaconis 2018): on the first five pools its split-half
reliability is ~0 and it does not agree with the on-policy estimate of the same quantity.
(Corrected 2026-10-01: an earlier version compared |A| with an SNIS estimate of D, which is at
its permutation null; A is now compared with on-policy totals: |A| 0.1-1.2 points vs held-out
effects of 3.4-10.9 points, rho(A, held-out) -0.31 to +0.28.)

## Verification noise (pipeline)

Greedy correctness is a {0,1} quantization of a per-question success probability p_x(P). A
paired comparison on n rows has variance ~ f / n with flip rate f ~ 0.16-0.22 per edit (token
edits) while true effects differ by ~1-2 points: at n = 8 the standard deviation is ~15
points, at n = 50 ~6 points. Selecting the best of many candidates on such noise realizes a
winner's curse (verify-all on 8 rows: -2.27 points; on 24 rows: +1.14). Distributional
estimates average over S samples per row and avoid the quantization.
