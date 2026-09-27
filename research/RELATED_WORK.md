# Related work: initial method review

Status: selected method sections reviewed on 2026-09-26; the complete literature
and implementation audit required by G0 is unfinished. No novelty conclusion or
experimental comparison is established. The full proposed bibliography is
retained in `scope.lock.json` as supplied planning references.

| Source | Sections checked | Consequence for this research |
| --- | --- | --- |
| [SAMMO](https://aclanthology.org/2024.findings-emnlp.37.pdf), Findings of EMNLP 2024 | Main methods, experiments, related work and limitations; Appendix A and B model versions | Symbolic search is prior work; include upfront optimization costs and isolate additional edit information. |
| [GEPA v2](https://arxiv.org/pdf/2507.19457v2), ICLR 2026 Oral as stated on the paper | Main algorithm and experimental observations; Appendix E.2–E.4 inference and optimizer budgets | Preserve native minibatch/Pareto screening. Our bounded local-model setup is not a numerical replication of its paper settings. |
| [ProEval v2](https://arxiv.org/html/2604.23099v2), ICML 2026 as listed on its arXiv record | Main methods/results/discussion; Appendix B proof, C controls/feature ablations and E source selection | GP-prior estimation is a strong control; posterior uncertainty does not establish finite-suite frequentist coverage. Source selection cannot inspect unqueried target scores. |
| [Active Testing](https://proceedings.mlr.press/v139/kossen21a/kossen21a.pdf), ICML 2021 | Main Sections 2–6, including acquisition, LURE, surrogate/retraining and unweighted-estimator diagnostics | Distinguish adaptive weighted acquisition from frozen SRSWOR; surrogate/training costs and calibration affect efficiency. |
| [Waudby-Smith and Ramdas v4](https://arxiv.org/html/2006.04347v4) | Main Sections 1–4 and Appendices A–E, including martingale proofs and hypergeometric extension | Fixed outcome assumptions are necessary; the planned six-look union-bound inversion is not their anytime PPR algorithm. |

Official implementation entry points identified:
[GEPA](https://github.com/gepa-ai/gepa),
[ProEval](https://github.com/google-deepmind/proeval).
Source heads checked before model outcomes: GEPA
`d771eb21b5dd3228bc3f567293d2ccfc423fc900`, ProEval
`4bb4fcc4a097a710ee0935098d0caa7270842208`, and DSPy
`da1736e21ffda8cc4b86379d4748b011764d507c`.
GEPA's `src/gepa/core/engine.py` evaluation and selection paths confirm that
survivors pass actual per-instance validation scores to state/frontier updates.
DSPy's `dspy/teleprompt/mipro_optimizer_v2.py` initialization, proposal and full
evaluation paths use bootstrapped demonstration candidates, grounded proposals,
Optuna search and periodic full validation. These costs cannot be discarded.
Mechanical official-source tests are recorded in `G0_BASELINE_AUDIT.md`. Complete
adapter/runtime audits and performance experiments remain pending; none of the
papers' experimental results has been numerically reproduced or defeated.

## Additional primary-record checks

The following primary records and selected method sections were checked on
2026-09-26/27. The reading scope is explicit; these are not implementation
reproductions or empirical comparisons. No listed result is our measured result.

| Source / reading scope | Existing topic | Boundary for PromptWitness-Delta |
| --- | --- | --- |
| [MPO v1](https://arxiv.org/abs/2601.04055v1), main method read | Fixed sections, local critiques and LLM consolidation | Local editing alone is not an edit-conditioned statistical evaluator. |
| [ADOPT v2](https://arxiv.org/html/2512.24933v2), Sections 3.1–3.3 | Dependency-conditioned local gradients, joint Bayesian prompt selection, and Kernel-SHAP allocation using already evaluated coalitions | Optimizer allocation is not a correction of unqueried paired transition counts. |
| [SERO v1](https://arxiv.org/html/2605.28433v1), Sections 3.1–3.4 | Typed role contracts, leave-one-out credit, masked controller actions, and score-gated commitment | Structural admissibility and its single-task score gate do not establish the planned finite-suite simultaneous intervals. |
| [CARE](https://journals.sagepub.com/doi/10.3233/FAIA251321), selected Sections 3.1–3.2 | Analyzer dependency extraction and refiner template/constraint-preservation directives | Constrained prompt transformation is prior work; semantic guarantees asserted in the paper are not independently verified here. |
| [RLMOpt v1](https://arxiv.org/html/2608.10471v1), Sections 3.1–3.7 | Model-directed component search, per-field regression floors, paired normal-approximation gate and Pareto finalization | Regression checks already exist. Seed, polish and diagnosis costs outside its search counter must still be charged in our all-role ledger. |
| [ESPO v1](https://arxiv.org/html/2609.04197v1), Sections 3.1–3.5 | Error clusters, diverse strategies/cross-pollination, and bootstrap win-count selection | Repeated resampling of fixed validation outputs is not additional independent evidence or our exact sequential certificate. Its informal bound is not validated here. |
| [STEVE v1](https://arxiv.org/html/2609.23716v1), Sections 2.1–2.4 | Error-focused updates, seed-correct preservation pool and fresh shared verification minibatches | Regression rejection is not original; the score/regularization objective is different from an unconditional regression-rate certificate. |
| [RECAP v4](https://arxiv.org/html/2606.06698v4), selected main protocol/results/limitations | Proactive specification-only continual adaptation without evaluation feedback | Our feedback-access optimization protocol is not proactive RECAP evaluation. |
| [PPI v4](https://arxiv.org/html/2301.09633v4), main convex-estimation theorem and mean algorithm | Prediction-assisted rectification; valid intervals require explicit rectifier/imputed-mean confidence sets | Accurate predictions alone are not uncertainty estimates. Its displayed normal-approximation mean algorithm is not the planned finite-population exact inversion. |
| [PPI++ v2](https://arxiv.org/html/2311.01453v2), selected Sections 3 and 6 | Weighted prediction correction, asymptotic GLM covariance and power tuning | Efficiency is prior work; asymptotic inference does not automatically certify small stratified finite suites with repeated looks. |
| [tinyBenchmarks v2](https://arxiv.org/html/2402.14992v2), Sections 2–4 | Historical correctness/IRT anchor selection and model-assisted whole-benchmark estimates | Curated static subsets are not a paired sequential certificate; histories must have authorized item alignment. |
| [Function-calling attribution v2](https://arxiv.org/pdf/2607.02595v2), PDF pages 1–8 including Section 3 and selected Section 4 diagnostics | Canonical same-output rescoring, format-only controls, induction sensitivity and cross-contract scope | Hold interface contracts fixed. Wrapper de-aliasing is not argument repair; format-only sufficiency is not proof of absent procedure. Its induction changes are not single-factor causal interventions. |

Official task documentation was also checked for BFCL V4, HotpotQA distractor,
and IFBench. The current BFCL source is `ShishirPatil/gorilla`, not the stale
`gorilla-llm/gorilla` mirror. IFBench commonly reports loose prompt accuracy;
this charter selects strict prompt accuracy and must disclose that difference.
The official IFBench README links IF-RLVR training data
`allenai/IF_multi_constraints_upto5`; no invented "IFTrain" repository is needed.

Before G1/G3, finish reading the five core papers beyond these selected sections,
audit their pinned official code and complete the comparison matrix for MPO,
ADOPT, SERO, CARE, RLMOpt, STEVE, ESPO, RECAP, PPI/PPI++ and tinyBenchmarks.
Imported titles and publication assertions require primary-source checks before
being used in the paper. A differently named combination is insufficient proof
of novelty. Keep model budgets, label access, randomization assumptions, scorer
and selector deviations visible in every comparison.
