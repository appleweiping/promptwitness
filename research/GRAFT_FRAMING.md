# Paper framing: decision rules fixed before the remaining evidence (2026-09-29)

Written while v3 (block-level validity replication), the token-level validity control and
the fixed-prompt baselines are still running, so that the paper's narrative follows from
pre-stated outcomes instead of being chosen after seeing them.

## Evidence that is already stable

* Exact structural edit calculus: every replacement, deletion, insertion and combined
  vertex equals the real edited prompt (fp32 tests, four architectures).
* One-pass renormalized patching tracks exact fixed-reasoning losses (mean Spearman about
  0.8); raw gate derivatives saturate and do not rank edits.
* For whole-block edits, GReaTer's objective (gold-answer loss after fixed greedy
  reasoning) does not predict held-out accuracy and is anti-predictive on several runs;
  the anti-correlation comes from rows whose reasoning is wrong.
* Held-out accuracy differences between proposals are small (true spread about 1.5
  points); a 200-question target has reliability about 0.3-0.4, so rho is bounded near 0.6.

## Open questions and what each answer implies

Q1. Does H-fork replicate (v3, frozen protocol)? Interim: fork > answer (P about 0.95),
fork < fresh-8, causal premise fails on LD7.
Q2. Is GReaTer's objective valid at its native token level (token-level control: same
tasks, seed, rows and 200-question target; edits are single-token substitutions from the
model's own top-k)?
Q3. Do GReaTer's published prompts beat its initial prompt and ZS-CoT on fresh splits
with the shared reader (all 21 BBH tasks, Llama-3 and Gemma-2)?

| Q2 (token-level answer loss) | Q3 (published GReaTer gains) | Framing |
|---|---|---|
| predictive (pooled rho > 0, CI above 0) | real gains | **Locality**: reasoning-conditioned gradients are valid for local (token) edits and fail for structural edits because structural edits change the reasoning. Contribution: exact structural calculus + diagnosis + a structural optimizer that scores with fresh evidence where locality fails. |
| not predictive | no reliable gains | **Re-examination**: the objective behind gradient-over-reasoning prompt optimization does not predict held-out accuracy at either granularity, and published gains do not survive fresh splits; the exact calculus is the instrument that makes this testable for structures. |
| not predictive | real gains | Gains come from GReaTer's search/selection, not from its gradient signal; test by a random-candidate ablation of the token optimizer if compute allows. |
| predictive | no reliable gains | Valid local signal but gains too small to survive test noise; report effect sizes with power analysis. |

H-fork (Q1) is reported as preregistered whatever the framing: a failed replication is a
reported negative result, not dropped.

## Stage C consequences

* If H-fork replicates: GRAFT's main objective is the fork margin (patching + margin
  acceptance), with answer-loss GRAFT as the GReaTer-objective ablation.
* Otherwise: Stage C still runs the preregistered comparison (GRAFT with GReaTer's answer
  objective, exact scoring, random shortlist, textual gradients; fresh verification and
  engine-shared dev selection for all), because it measures what structural search with
  gradient scoring achieves end to end; the fork objective is included as an exploratory
  variant and labeled as such.
* Every framing reports readability, transfer and perturbation robustness (Stage E) for
  structural vs token-level prompts, which does not depend on the objective question.
