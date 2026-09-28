# ICMR 检索冻结审计桥，2026-09-28 00:08:54 UTC

状态：`AUTHORED_LOGICAL_BRIDGE_NOT_M1_ADMITTED`。本增量接续同一 ARIS
experiment-bridge，不开新科学轮次、预算或定时器。

`reproduce/retrieval_audit_bridge.py` 将已有 `AuditPlan`、`AuditJournal`、
`evaluate_candidate` 和检索专用受限 scorer 接成一条 search primary-hit 路径。
原 gate 在每次 query 排名之前预留逻辑 episode；固定随机前缀才会调用排名器；
worker 失败不补零、失败尝试不静默重放。只有本计划的 ELIGIBLE 结果可调用
`complete_survivor()`，其余单位须再取得实际受限评分，才向 native selector 返回
完整向量。自审发现随机审计顺序不等于 selector 顺序，现要求调用者显式传入
冻结 `unit_ids`，核对完整人口并按该顺序输出；混合 0/1 fixture 核对 ID 对齐。

这不是完整 CIR controller。调用者仍须冻结 candidate、图像/文本 encoder、
store、排序回调和 selector 顺序，并在每次真实图像、文本、LLM、GPU 操作之前
接上物理费用账本。桥本身不认证回调、不证明 ONLINE_PINNED、不接 selection
或 sealed final，也不阻止有人绕过它直接调用 scorer；逐查询反馈的可推断风险
继续按 `ICMR_RESTRICTED_SCORING.md` 披露。原 gate 的 2048 logical episodes 与
32 candidate slots 不变，不新增用户未授权的小额度。

本机最后六项定向 `5 passed, 1 skipped`；混合 0/1 测试改动后重新运行完整
`tests/reproduce` 为 `486 passed, 8 skipped`（119.85 秒）；Ruff、format、
mypy、Bandit、新文件 `git diff --check` 通过。fresh-context code review 判定
无本增量 BLOCKING，same-family/provisional；其非阻断混合分数建议已落实。
这不构成跨家族科学接受。

复审者 WSL Ubuntu 实测六项 `3 passed, 3 failed`：两项系统 Python 缺 SciPy；
真实 worker 在既有 `process_access.py` 初始化读取项目包时 PermissionError，
未进入本次 bridge。独立单项诊断仍 `1 failed`；虽检测到 Landlock ABI 7 和
预期 runtime root，该 WSL/DrvFS 布局未资格化，不抹掉失败或推断根因。
应以本次新源码精确 SHA 的已配置 Linux CI 做窄 authored 实进程检查；旧
`fda4464` scorer CI 不覆盖桥。没有 benchmark 图像/标注访问或新模型推理；
新增生成调用、GPU、付费 API 为零，CPU 开销未完整计量。M1/Pilot/ICMR
方法效果与论文均未准入，历史账本及 default4 不变。
