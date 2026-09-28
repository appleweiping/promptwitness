# GRAFT: gradients over superposed prompt blocks

> 中文摘要：导师建议从 GReaTer（ICLR 2025，token 级真实梯度 + 推理链上的梯度）出发，
> 把梯度反馈从脆弱的 token 级迁移到 PromptWitness 所操作的结构单元（带类型的消息块）。
> 本文提出 GRAFT：把每个可编辑块的候选改写并行"叠加"在同一位置槽中，用可微的注意力门控
> 连接下游 token，一次前向/反向即可对 *变长、整块* 的替换/删除/插入给出一阶损失估计；
> 门控端点恰好是离散提示本身。这是 GReaTer 在 one-hot 词表单纯形上求导的结构级类比。

Date: 2026-09-28. Stage: ARIS Workflow 1 (idea) for the `research/graft` branch.
Supersedes nothing in history; the frozen four-arm pilot (`research/structured-gradient-pilot`,
commit 8eb0cd4) keeps running and becomes preliminary evidence and ablation arms.

## 1. Problem

GReaTer (Das et al., ICLR 2025) lets a small model optimize its own prompt with true
gradients: generate reasoning under the current prompt, append an extractor, take the
answer cross-entropy, differentiate with respect to a one-hot token indicator, and swap
one token for the candidate with the most negative gradient. Two properties make the
token coordinate brittle:

1. Its optimized prompts are frequently ungrammatical. Example from the official
   Gemma-2 date_understanding log: `. Use a for a or  e or whatever I need help to. or understanding.`
2. A token swap cannot express a change of *length* or *structure*: it cannot delete a
   harmful instruction, insert a missing procedure, or replace a sentence with a
   differently sized one, and each step moves one token (105 steps, ~5 h/task on 2xA100).

PromptWitness already represents prompts as typed, versioned structures (messages,
typed blocks, required literals, variables, tool contracts) with policy gates and
structural diffs. The question: **can true gradient feedback operate directly on those
structural units?** The obstacle is that alternative blocks have different token
lengths, so one-hot / embedding linearizations (GReaTer, GCG, AutoPrompt) are undefined,
and the pilot's aligned-position dot product almost never applies.

## 2. Closest work (verified this session)

| Work | Unit | Signal used to *score alternatives* | Gap GRAFT addresses |
|---|---|---|---|
| GReaTer (ICLR'25) | token | one-hot gradient over reasoning | fixed length, brittle text |
| GCG / AutoPrompt | token | one-hot gradient | same |
| GradPO (Chakma+ 2606.29639) | 1-5 token spans | gradients only *locate* spans; candidates scored by explicit loss | alternatives never scored by gradient |
| GMPO (Zhao+ ACL'26) | fixed windows | directional derivative toward a mask *embedding* locates segments; black-box generator + explicit multi-judge CE selects | off-manifold mask path; no gradient scoring of rewrites; no insertion |
| PMPO (Zhao+ 2025) | semantic units | masking ablations + explicit CE | many forward passes |
| MPO (Sharma & Henley 2601.04055) | fixed sections | section-local *textual* gradients from a critic LM | no numerical gradient |
| SAMMO (EMNLP-F'24), SPRIG | program graph / components | black-box search / GA | no gradient |
| Superposition prompting (ICML'24) | RAG documents | Bayesian saliency for pruning paths | inference speed, not optimization |
| Linear-time demo selection (EMNLP'25) | demonstrations | first-order output approximation | demos only, embedding linearization |
| AtP / AtP* | model components | attribution patching | interpretability, not prompt search |

Our structural pilot arm D (mean token-gradient norm picks the block, then explicit
evaluation) is essentially the GradPO/GMPO "gradients locate, evaluation decides" recipe.

## 3. Method

### 3.1 Structured prompt and objective
Prompt `P = (b_1..b_m)`; each block has a type (task, strategy, constraint, output,
input, example), an editable flag, required literals (e.g. the answer format), and
PromptWitness policy constraints (no new template variables, no label leakage,
length bound). For a fit example `(x, y)`: greedy reasoning `r ~ f(P(x))`, extractor `e`,
loss `L(P) = -log p_f(y | P(x), r, e)` (GReaTer's gradient-over-reasoning objective;
`r` is held fixed during differentiation, as in GReaTer).

### 3.2 Superposed, gated forward pass
For each editable block `b_i` the same frozen model proposes `K` label-free candidates
`c_{i,1..K}` (type-conditioned rewrites and prefix-conditioned block continuations,
using unlabeled fit inputs only — the analogue of GReaTer's top-k next-token proposals).
We build one augmented sequence in which every candidate occupies a **parallel slot**:

* candidate tokens reuse the position ids of `b_i`'s slot and may attend only to the
  prefix before `b_i` and to themselves (never to `b_i` or sibling candidates);
* every downstream token (later blocks, reasoning, extractor, answer) may attend to the
  incumbent and to all candidates, with attention weights multiplied by a gate:
  `a_tu = G_tu exp(s_tu) / sum_v G_tv exp(s_tv)`, `G_tu = g_{seg(u)}` for gated segments.

Gate base point `g0`: incumbents 1, candidates 0, which reproduces the original forward
pass exactly (Prop. 1), so `L(g0) = L(P)`. Setting incumbent 0 and one candidate 1
reproduces the rewritten prompt up to a constant RoPE offset downstream (Prop. 2).
Gates are applied multiplicatively after the max-shift, so derivatives at `g = 0` are
exact (no epsilon bias). One forward + one backward gives, for all slots at once:

* deletion effect `D_i = -dL/dg_{b_i}`,
* insertion effect `I_{i,k} = dL/dg_{c_{i,k}}` (also for new empty slots: INSERT op),
* replacement estimate `R_{i,k} = dL/dg_{c_{i,k}} - dL/dg_{b_i}` = directional derivative
  along the gate-simplex edge from the incumbent vertex to the candidate vertex.

This is the block-level counterpart of GReaTer/GCG: there the one-hot vector over the
vocabulary is relaxed and `grad . (e_v - e_x)` scores a token swap; here the per-slot gate
simplex over {incumbent, candidates} is relaxed and `grad . (delta_c - delta_b)` scores a
whole-block swap of **arbitrary length**. The relaxation lives in attention space, whose
vertices are real prompts, unlike embedding interpolation toward a mask token (GMPO).

### 3.3 Search loop (per round)
1. fit minibatch; greedy reasoning under the incumbent (reused from last acceptance);
2. label-free proposals per editable block + DELETE for non-required blocks + INSERT slots;
   PromptWitness policy filter;
3. superposed forward/backward per example, average scores;
4. shortlist top-mu edits (and the per-slot-best combination);
5. exact check with fresh reasoning on the minibatch; accept if accuracy (then loss)
   improves; PromptWitness structural diff records each accepted edit with predicted
   and measured effect ("edit witness");
6. dev-set checkpoint selection; test once.

Variant GRAFT-EG: a few exponentiated-gradient steps on the per-slot simplices before
rounding (a soft, superposed prompt that is still anchored to real text).

## 4. Claims and required evidence

* **C1 fidelity.** Gate derivatives rank actual loss changes of whole-block edits
  better than token-gradient norm (pilot D / GradPO), mask-embedding directional
  derivative (GMPO), and random. Metric: Spearman / top-1 hit over candidate pools,
  per task and model; endpoint exactness check (Prop. 1/2) with measured RoPE-offset error.
* **C2 efficacy.** On GReaTer's protocol (21 BBH tasks 50/100/100, GSM8K, FOLIO;
  Llama-3-8B-Instruct and Gemma-2-9B-it as both task and optimizer model) GRAFT is at
  least on par with GReaTer's own published prompts and above ZS-CoT, textual-gradient
  block optimization with the same small model, and gradient-localization baselines.
* **C3 cost.** GRAFT reaches its accuracy with far fewer model operations/GPU-seconds
  than token-level GReaTer (measured on a subset with the official code) and than
  evaluate-every-candidate search.
* **C4 non-brittleness.** GRAFT prompts are fluent (reference-LM perplexity), pass
  PromptWitness policies, are less sensitive to paraphrase/token perturbations, and
  transfer better across models (Llama <-> Gemma, OLMo-3) than token-level prompts.
* **C5 ablations.** no-reasoning gradient; no insertion/deletion; random shortlist;
  K scaling; first-order vs EG; block granularity (message vs sentence).

## 5. Risks and kill criteria

* First-order gate estimates may be poorly calibrated (attention saturation, AtP*).
  Kill/reframe if Spearman on held-out candidate pools is not clearly above the norm
  heuristic on >= 2 models. Mitigation: EG / midpoint (integrated) estimate.
* Fixed-reasoning approximation: a rewrite changes the reasoning itself. Measured by
  exact check; same limitation as GReaTer.
* Custom attention must reproduce the model exactly (Gemma-2 softcapping, sliding
  windows, GQA, QK-norm). Unit test: max |logit diff| vs stock model at g0.
* Compute: single available A6000 (GPU1 is occupied by another user's service).

## 5b. Revision after cross-model review and fidelity v1/v2 (2026-09-28)

The GPT-6 review (PROCEED WITH CAUTION, novelty 6/10) found Prop. 2 false for
length-changing edits and asked for parity and fidelity evidence first. Changes:

1. **Exact vertices.** Candidates start at their incumbent's first position; each slot
   has a continuous offset `w_j` that moves every later token by an extra RoPE
   rotation. Then replacement, deletion, insertion (empty slots) and multi-slot
   combined vertices equal the real edited prompts (fp32 tests on three
   architectures; gate and offset gradients match finite differences).
2. **Raw gate derivatives fail** (fidelity v1/v2 on OLMo-3: Spearman -0.2 to 0.26
   against exact fixed-reasoning loss changes). Cause: attention saturation; the
   derivative at `g=0` scales with raw candidate mass `S_c/Z_0`.
3. **Renormalized superposed patching** keeps each layer's softmax renormalization and
   RoPE rotation exact for the swapped slot and linearizes only propagation through
   later layers: `dL ~ sum_l <dL/do_l, o_l(c) - o_l(b)>`. This is the AtP* insight
   (Kramár et al. 2024: exact attention, linear elsewhere) carried from
   interpretability to structural prompt search, made possible because the
   superposed pass already holds every candidate's exact keys/values. Same single
   forward/backward per example; Spearman 0.87 on the same OLMo-3 pool.

The contribution is therefore framed as: (a) an exact structural edit calculus over
typed prompt blocks (every vertex is a real prompt, length changes included), and
(b) a faithful one-pass estimator for all single-slot edits, used as the gradient
signal in a GReaTer-style self-optimization loop.

## 6. Pilot evidence so far (frozen pilot, OLMo-3-7B, logical_deduction, 32-item val)
Baseline A 22/32 (x3); token arm B 26, 23, 25; random-block C 23, 24, (seed 3 pending);
gradient-norm block D pending. Too small for conclusions; motivates C1 before C2.
