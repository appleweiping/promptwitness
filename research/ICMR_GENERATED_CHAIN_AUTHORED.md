# 生成描述链路的 Linux 端到端 authored 回归

记录：2026-09-28 06:14:38 UTC。状态：Windows 回归 `PASS`；
Linux 实受限进程与精确 SHA CI `PENDING`；独立审查 `UNAVAILABLE_LOCAL_ONLY`。

此前单元测试分别证明模型收据与目标描述，以及目标描述到受限 ranker；
没有一次将两者接在同一个冻结 `AuditPlan` 内。本次在
`tests/reproduce/test_retrieval_generated_description.py` 增加 Linux-only
自制数据用例，连续执行 64 查询：

1. 自制 input-only CIRR 形状的查询、完整 11 图 ID 库和参考 caption；
   scorer 独持自制 gold，四个 gold leaf 均置 sentinel。
2. 原 `PersistentModel.metered` 和 `ResourceLedger` 每查询预留、结算一次
   authored `_execute` 响应；输出固定为 `authored target first`。
   这不是 Qwen/OLMo 推理，也没有启动 GPU child。
3. `RestrictedRankerSession` 使用原 Landlock search-only 角色及
   authored ranker 替身，经描述 IPC 返回完整排序；独立受限 scorer
   读取自制真值。原冻结 gate 先取合法随机前缀，合格后把 survivor
   补成 64 个实际分数。
4. 测试检查模型账本 64 calls / 1088 input / 256 output authored tokens，
   外层 rank+score 128 attempts、ranker 内部 64 attempts、无 unresolved。
   这些是自制响应的测试计数，不是实际研究消耗或实测模型 token。

本地定向 12 passed、7 skipped（Windows 上 Linux 用例按条件跳过）；
全套 2059 passed、20 skipped、124 warnings、覆盖率 94.04%，exit 0。
Ruff check/format 和 `git diff --check` 通过。
fresh gpt-6-astra/xhigh 只读审查请求返回 `agent thread limit reached`；
按 `experiment-bridge` fallback 做了本地逐项审查，未把它称为独立 verdict。

这项回归若在 Linux CI 通过，只证明自制替身、账本类、受限 IPC、scorer
及 gate 在单次进程链中可组合；不证明真实模型文件边界、CLIP 语义、
正式图库/许可、captioner 来源、ONLINE_PINNED、全角色费用、M1/Pilot
或 C1–C3。服务器 SSH 指纹仍未获新确认，未登录/部署/读取正式数据。
ARIS 默认4轮、历史成本、无定时器和其他四仓冻结不变。
