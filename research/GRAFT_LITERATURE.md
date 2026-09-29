# GRAFT literature review (ARIS `research-lit`, web source; 2026-09-29)

Scope: prompt optimization with small open models, gradient vs. textual vs. search
signals, evaluation reliability, and reasoning faithfulness, read against what GRAFT
has established (exact structural calculus; faithful one-pass estimator; GReaTer's
fixed-reasoning objective anti-predicts held-out accuracy of block edits; H-fork failed).
Zotero/Obsidian are not configured and no local paper library exists; sources are the
web, the papers' own pages, and (where released) their code.

## Table

| Paper | Venue | Method | Key result | Relevance to GRAFT |
|---|---|---|---|---|
| GReaTer (Das et al.) | ICLR 2025 | One-hot token gradients of the gold-answer loss after fixed self-generated reasoning shortlist candidates; selection regenerates reasoning and keeps the fewest errors + perplexity penalty (read in the official code) | Small models (Llama-3-8B, Gemma-2-9B) beat LLM-driven optimizers on BBH/GSM8K/FOLIO | Our template and main comparison; we test its signal beyond single tokens and find it misaligned |
| GEPA (Agrawal et al.) | ICLR 2026 (oral) | Reflective natural-language mutation of prompts from sampled trajectories; Pareto-front candidate selection | +6 pts over GRPO with up to 35x fewer rollouts; +10 over MIPROv2; Qwen3-8B | Exemplar framework (budget-centric tables, selection ablation, generalization gap, prompt length, transfer); a strong evaluation-based baseline class |
| OPRO (Yang et al.) | ICLR 2024 | LLM generates instructions from a trajectory of scored prompts | Gains on GSM8K/BBH with large optimizers | Baseline family; shows reliance on evaluation scores |
| Revisiting OPRO (Zhang et al.) | Findings ACL 2024 | Replication with small models | Small models are ineffective OPRO optimizers | Motivates self-optimization with small models via gradients (GReaTer's niche) |
| EvoPrompt (Guo et al.) | ICLR 2024 | GA/DE operators implemented by LLMs over prompt populations | Gains on 31 datasets | Evaluation-driven structural search; exemplar ablations of operators |
| APE (Zhou et al.) | ICLR 2023 | Generate-and-score instruction candidates | Human-level instructions | Earliest generate-then-evaluate baseline |
| ProTeGi (Pryzant et al.) | EMNLP 2023 | Textual "gradients" (critiques) + beam search with bandit selection | Gains on classification tasks | Textual-gradient family; our `textgrad` baseline mirrors it |
| TextGrad (Yuksekgonul et al.) | Nature 2025 | Backpropagation of natural-language feedback through computation graphs | Broad gains with strong LLM critics | Textual-gradient baseline; needs a strong critic |
| MIPRO/MIPROv2 (Opsahl-Ong et al.) | EMNLP 2024 | Bayesian optimization over instructions and demos for LM programs | Strong program optimizer | Structured (module-level) search, evaluation-based |
| Modular prompt optimization (Sharma et al.) | 2026 preprint | Section-local textual gradients on structured prompts | Gains over flat textual gradients | Closest structural textual-gradient competitor |
| GMPO / GradPO / PMPO | 2025-26 | Gradients or masking locate influential segments; rewrites scored by evaluation or rationale likelihood | Gains with small models | Segment-level use of gradients as locators, not scorers |
| HbBoPs (Schneider et al.) | ICML 2025 | Deep-kernel GP + Hyperband for prompt selection | Efficient black-box selection | Multi-fidelity evaluation: the right tool when signals are noisy |
| p1 (Gao et al.) | 2026 preprint | Decomposes reward variance into response noise vs. prompt-quality variance; trains on high-variance user prompts | Two AIME prompts suffice; beats GEPA | Closest to our reliability analysis; we add a split-half ceiling for *ranking* edits and a mechanism for a specific signal's failure |
| Optimization before evaluation (Sadjoli et al.) | ACL 2025 Industry | Compare model rankings with/without per-model PO | Rankings change | Evaluation methodology context |
| Composing policy gradients and PO (Ziems et al.) | CAIS 2026 | Module-level GRPO combined with prompt optimization | +11 pts | Weight-space alternative; out of scope (frozen model) |
| Lanham et al. | 2023 preprint (Anthropic) | Early answering, adding mistakes to CoT | Faithfulness varies with task/size | Our mechanism (answer decoupled from reasoning) is a faithfulness phenomenon |
| Turpin et al. | NeurIPS 2023 | Biasing features change answers without being mentioned in CoT | Up to 36-pt drops on BBH | Same: answers can move independently of stated reasoning |
| Forking paths / critical tokens / decision tokens (Bigelow; Lin; Shen) | ICLR 2025; 2024; ICML 2026 | Tokens where sampled reasoning branches decide outcomes | Few tokens matter | Basis of H-fork; our causal check finds first divergences are not such tokens |
| ContraPrompt; contrastive reflection | 2026 preprints | Textual contrast of successful vs failed traces via a reflector LLM | Gains over GEPA (ContraPrompt) | Trace contrast done in text; our fork objective did it with gradients and failed |

## Landscape

Two families dominate. Evaluation-driven optimizers (APE, OPRO, EvoPrompt, ProTeGi,
MIPRO, GEPA) generate candidates with an LLM and decide by measured task scores; their
recent progress comes from better use of trajectories (GEPA's reflection, Pareto
selection) and from sample efficiency. Gradient-driven optimizers (AutoPrompt, GCG,
PEZ, GReaTer) use a model's own gradients, which lets a small model optimize itself
without a stronger critic; GReaTer's key move is to take the gradient of the answer
loss *after the model's own reasoning*. Structure has entered both families only as
locations (GMPO, GradPO) or as textual section feedback (modular PO); no work scores
structural rewrites with model gradients, because length-changing edits break the
one-hot relaxation. GRAFT's exact calculus removes that obstacle.

Reliability of the optimization signal is an emerging theme: p1 shows that response
variance can swamp prompt-quality variance and selects user prompts to fix it;
HbBoPs spends evaluation budget adaptively; OPRO's replication shows small models are
weak optimizers. None of these asks whether the *gradient* signal used by GReaTer-style
methods is valid for the edits it ranks.

CoT faithfulness work shows answers can be decoupled from stated reasoning. GRAFT's
mechanism result connects the two: on rows whose reasoning is wrong, lowering the gold
answer's loss rewards prompts that decouple the answer from the reasoning, which is why
the objective ranks edits in the wrong direction.

## Gaps GRAFT can fill (honest reading after v3)

1. No exact first-order machinery for structural (variable-length, insertion, deletion)
   prompt edits existed; GRAFT provides it, verified numerically, with a faithful
   one-pass estimator.
2. No study tests whether reasoning-conditioned gradient objectives predict held-out
   accuracy, at any granularity, against a reliability ceiling. GRAFT's validity study
   does this for structures (and, in phase 2, for GReaTer's own token edits).
3. The mechanism (decoupling on failed rows) links prompt-optimization objectives to CoT
   faithfulness; it suggests objective designs that keep answers tied to reasoning.

## Framework lessons from the exemplars (to imitate, not copy)

* GReaTer: one clear idea (gradients *over reasoning*), a figure contrasting it with
  text-feedback optimizers, main tables on BBH/GSM8K/FOLIO for two small models with
  strong baselines, transfer to other models, an ablation removing reasoning, and
  qualitative prompt examples.
* GEPA: a problem statement centered on budget (rollouts), main tables per task model
  with an aggregate row and a budget column, a candidate-selection ablation on an
  identical harness, generalization-gap and prompt-length analyses, cross-model
  transfer, and extended applications.
* OPRO / EvoPrompt: operator and meta-prompt ablations, starting-point sensitivity,
  optimization curves, and explicit overfitting analysis (train vs. test).
* p1: variance decomposition as the analytic core, then a method derived from it.

For GRAFT this means: (i) a crisp statement of the question (can gradient feedback over
reasoning move from tokens to structures?), (ii) an exact method, (iii) an analysis
section that is the paper's scientific core (validity with ceilings, mechanism, causal
test), (iv) budget-centric main tables against GReaTer/GEPA-style baselines on GReaTer's
benchmarks, and (v) transfer, prompt length and readability.
