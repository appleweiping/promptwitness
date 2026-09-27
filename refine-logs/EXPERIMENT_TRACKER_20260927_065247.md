# PromptWitness-Delta 接续实验 tracker

日期：2026-09-27 06:52:47 UTC。完整研究仍未完成，固定矩阵不缩减。

| ID | 范围 | 状态 | 实际证据/限制 |
|---|---|---|---|
| S0-code | 官方评分适配器与 BFCL 原版绑定 | VERIFIED_MECHANICS | 55 targeted tests；本次 blocker 修复后一次复审 |
| S0-cpu | Hotpot/IFTrain/IFBench 官方实现 | PARTIAL_VERIFIED | 前轮 13/13 authored fixtures、fit archive 限定交集；未在本轮重跑 |
| S0-BFCL-runtime | 官方 AST matcher、三个 scorer-only 模型配置 | AUTHORED_RUNTIME_VERIFIED | fresh-doc 原样 exit 0；40 预定用例 × 3 配置；99 分数检查、21 预期错误；不是 120 个模型样本 |
| S0-BFCL-GT | BFCL 真实 annotation 与已有 fit 输出 | PENDING | 尚未接入 fit-only auditor，authored fixtures 不替代 real GT |
| S1 | 进程级 split-access；在线有限总体/随机执行单元 | NOT_IMPLEMENTED | 既往预览披露与 batch 输出差异保留 |
| S2 | 原生 GEPA/MIPRO early rejection、全角色成本 | NOT_IMPLEMENTED | 旧接口 smoke 非原生优化收益 |
| G2 | 真实 M1、训练/验证侧 Pilot | NOT_RUN | 科学准入及自身成本尚缺 |
| G3 | 144 pairs / 180 runs / 36 transfers | NOT_ADMITTED | 实际完成仍 0/0/0；没有 Pilot GO |
| G4 | 统计、默认审查、研究叙述 | NOT_READY | auto-review-loop 默认 4 未改，科学轮尚未开始 |

本轮 148 helper tests（44.84 秒），55 targeted tests（10.83 秒）；新模型调用、
allocated GPU-hours、付费均为零。历史消耗不清零；不把 CPU wall time 说成完整 core-hour。
新 source 尚未 commit/push，exact-SHA CI 待交付后检查。已有 622668b 的 CI
36298664057 为历史成功，不代替这次新代码的证明。
持续 active goal，无 heartbeat/cron/watchdog；其他四仓库冻结。下一步落实 S1
并接入 BFCL fit-only real GT，然后按原计划继续 S2 与真实研究准入。
