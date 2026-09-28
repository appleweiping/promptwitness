# Structured-Gradient Prompt Search: implementation comparison

This is a development pilot, not a paper result. The hypothesis is whether an
answer-loss gradient helps choose *whole semantic blocks* to rewrite, relative
to token editing and gradient-free block selection under matched compute.

| Component | GReaTer (ICLR 2025) | This pilot |
| --- | --- | --- |
| Search coordinate | One prompt token at a time | One complete, typed `PromptDocument` block at a time |
| Proposal | LM top-k next-token candidates from task prefixes | Same local LM proposes at most three complete block rewrites per chosen block |
| Reasoning and loss | Generate reasoning, append answer extractor, compute answer cross-entropy | Generate reasoning without a gold label; freeze its sampled tokens for a second differentiable forward; supervise every gold-answer token only |
| Gradient | Autograd through a reduced one-hot candidate embedding at the current token | Autograd `dL/de` over actual prompt input embeddings; mean token gradient norm ranks editable blocks |
| Candidate choice | Negative token gradient shortlist, then actual loss forward | Gradient selects two blocks; candidate ordering uses a first-order dot product only when full rendered token positions align; otherwise use actual evaluation |
| Acceptance | Track lower-loss prompt | Keep parent unless a rewritten prompt improves fresh task scoring on the fixed selection batch |

The official code examined at commit `42a22d9211e55528e8894c89ca7d13ae817366d2`
uses `llm_opt/gcg/greater_opt.py` for reduced one-hot input gradients and token
candidate selection, and `llm_opt/base/attack_manager.py` for generated reasoning
and extractor construction. Its published Algorithm 1 is the method reference;
this pilot uses a different coordinate and a contemporary local model, so a
future token baseline must be called an adapted reproduction.

**Gradient interpretation.** For fixed sampled reasoning `r`, the code measures
`∂ CE(answer | prompt, input, r, extractor) / ∂ input_embedding`. It does not
differentiate the discrete generation of `r`. Block sensitivity is the arithmetic
mean of the L2 norms of its mapped token gradients; this is a search heuristic,
not causal attribution. Tokens crossing a block boundary are excluded. No detached
KV cache is used for the differentiable forward. The model weights are frozen.

The first tasks are BBH `logical_deduction_three_objects` and `date_understanding`.
The local GReaTer CSV bytes are pinned by SHA-256 in
`configs/research/structured_gradient_splits.json`, with disjoint 32 fit, 32
validation, and 100 holdout row IDs per task. The [BBH repository](https://github.com/suzgunmirac/BIG-Bench-Hard)
publishes an MIT license; this project commits IDs and hashes, not copied task
rows. The first row of each task was viewed during setup, so developer blindness
to those public examples is not claimed. Holdout rows must not drive search.

The development gate is one actual model sample: reasoning generation, full
answer loss, finite nonzero mapped gradient, frozen weight check, finite
difference direction, and one fresh rewritten-prompt score. The later comparison
must measure all four methods, all failures, and GPU time before drawing a
research conclusion.

Sources: [paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/18a42aad2fa8aa871e2ee20d425c208d-Paper-Conference.pdf),
[official code](https://github.com/psunlpgroup/GreaTer).
