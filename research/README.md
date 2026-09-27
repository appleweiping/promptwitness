# PromptWitness-Delta research

This directory contains the adopted finite research scope and current readiness
evidence. The project is evaluating whether prompt-edit information can improve
paired evaluation efficiency and optimizer outcomes. Those benefits are untested.

Current terminal status: **RESOURCE_BLOCKED before G1**. See the
[decision report](DECISION.md) and [machine-readable status](FINAL_STATUS.json).
The retained 770 real requests are cost probes, not optimization experiments.
The measured Qwen task subset forecasts 2,229.873 GPU hours with 20% reserve at
the unchanged matrix allowances, exceeding the 1,000-hour protective ceiling.
Actual cumulative consumption is 3.253816 allocated GPU hours. This is a failed
planning gate for the frozen execution protocol, not a scientific negative result
or proof that every alternative protocol is infeasible.

- [Protocol draft](PROTOCOL.md): method definitions, statistical assumptions,
  experimental controls, budgets and finite stopping rules.
- [Scope lock](scope.lock.json): the supplied charter and actual readiness state.
- [Related work](RELATED_WORK.md): source sections checked and remaining audit.
- [Claim/evidence table](CLAIM_EVIDENCE.csv): engineering evidence and untested
  research claims.

The research API and `promptwitness delta plan/run/report` are planned, not
implemented commands. G0 resource and baseline checks precede implementation and
the real-model pilot. The current code remains the existing PromptWitness API.
The remaining optional calibration was not launched. Restart requires a new
user-authorized protocol version; no automatic restart, submission or main merge.
