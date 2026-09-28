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

## Next

* Fidelity v2 decides the scorer: renormalized patching must beat raw gates and
  approach exact scoring at lower cost; otherwise GRAFT uses exact-vertex scoring and
  the contribution shifts to exact structural edit calculus + structure-level search.
* Pilot optimizer runs (`run_graft.py`, scorers patch/gate/exact/random) on a few
  tasks; then the preregistered subset (4–6 BBH tasks + GSM8K or FOLIO, Llama-3 and
  Gemma-2, 3 seeds) and the published-GReaTer / ZS-CoT baselines
  (`evaluate_prompts.py`).
