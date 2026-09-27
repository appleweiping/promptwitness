# PromptWitness-Delta 接续实验 tracker

日期：2026-09-27 07:36:11 UTC。目标 active；完整研究尚未完成。

| ID | 范围 | 状态 | 实际证据/限制 |
|---|---|---|---|
| S0-cpu | 官方 Hotpot/IF 和 BFCL authored runtime | PARTIAL_VERIFIED | 先前 13 fixtures、fit 限定交集及 BFCL 99 分数/21 错误检查；未在本轮重跑 |
| S0-BFCL-GT | 真实 annotation 与 fit 输出 | PENDING | 真实 fit-only store/审计评分尚未连通 |
| S1-files | Linux 角色读取机制 | AUTHORED_NATIVE_VERIFIED | ABI1，6 worker；19 allowed/47 denied reads；66 append-open denied；非真实数据隔离结论 |
| S1-pipeline | 真实 store、角色接入、消息/阶段 receipts | NOT_COMPLETE | runtime 必须排除 bundled data；不能靠路径名字推断真实文件语义 |
| S1-online | 有限总体与随机执行单元 | NOT_VALIDATED | 原 batch 输出差异与 preview 披露保持 |
| S2 | 原生 GEPA/MIPRO early rejection、全角色成本 | NOT_IMPLEMENTED | 旧接口 smoke 不是原生搜索实验 |
| G2 | 真实 M1 与训练/验证侧 Pilot | NOT_RUN | S0/S1/S2 准入仍缺 |
| G3 | 144 pairs / 180 runs / 36 transfers | NOT_ADMITTED | 完成仍为 0/0/0；没有 Pilot GO 或完整成本准入 |
| G4 | 统计、默认审查与研究叙述 | NOT_READY | 默认科学审查最多 4 轮未改，未开始 |

本轮 172 helper tests passed、1 Windows native skip；Linux 文档原样命令
exit0、4.7744489 秒，私有 artifact 9099 bytes。新模型调用/GPU/付费=0；
历史账本不清零。22/170 是添加环境 witness 前的测试记录，不冒充最终 24/172。
当前新增源码尚未 commit/push，exact-SHA CI 待推送；先前 3acaa0b/36301672235
成功 15/15、Ubuntu3.12 1687/94.03% 只覆盖那次 S0 源码。
持续 active goal，无 heartbeat/cron/watchdog，其他四仓库冻结。
