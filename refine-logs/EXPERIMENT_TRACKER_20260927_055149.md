# PromptWitness-Delta 接续实验 tracker

日期：2026-09-27 05:51:49 UTC。状态对应真实已执行工作；确认矩阵不缩减。

| ID | 范围 | 状态 | 实际证据/限制 |
|---|---|---|---|
| S0-code | strict scoring adapters、异常语义、source-only 检查 | VERIFIED_MECHANICS | 39 tests；一次 blocker 修复和复审 |
| S0-cpu | Hotpot/IFTrain/IFBench 官方实现 | PARTIAL_VERIFIED | fresh-agent-follow-doc exit 0；13/13 authored fixture 匹配 |
| S0-fit | 已有 fit-only 历史输出 | COMPLETED_LIMITED_AUDIT | Hotpot 4；IFTrain 5；不是准确率基准/Pilot |
| S0-BFCL | 真实官方 AST checker import 与 fit 集成 | NOT_QUALIFIED | SDK/registry 引入依赖未解决；Mock 不算通过 |
| S1 | 实际进程 split-access；在线有限总体/随机执行单元 | NOT_IMPLEMENTED | 既往预览披露保留；batch 输出差异未清 |
| S2 | 原生 GEPA/MIPRO early rejection 与全角色成本 | NOT_IMPLEMENTED | 接口 smoke 非原生优化收益 |
| G2 | 真实 M1、训练/验证侧 Pilot | NOT_RUN | 各前置门禁及自身成本尚缺 |
| G3 | 144 pairs / 180 runs / 36 transfers | NOT_ADMITTED | 完成仍 0/0/0；没有 Pilot GO |
| G4 | 统计/审查/叙述交付 | NOT_READY | auto-review-loop 默认 4 未改，尚未开始科学轮 |

新模型调用和 allocated GPU-hours 均为零；历史成本没有清零。
持续 active goal，无 heartbeat/cron/watchdog 进程。
下一步按原计划解决 BFCL runtime、split isolation 和 native search；不重选题或放宽目标。
