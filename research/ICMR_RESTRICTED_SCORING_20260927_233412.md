# ICMR 检索受限评分入口，2026-09-27 23:34:12 UTC

状态：`AUTHORED_SCORER_PATH_NOT_M1_ADMITTED`。同一 ARIS run 的 M1 工程增量；
不是 CIRR/FashionIQ 正式数据运行、Pilot、方法收益或最终集准入。

`reproduce/retrieval_role_scoring.py` 提供 CIRR/FashionIQ fit、search、selection
的受限评分入口。可信控制端只发 query ID 与完整候选排名，不发 target、subset；
`retrieval_{stage}_scorer` 子进程从本阶段 scorer-only gold leaf 读取标签，复用
现有检索评分核并只返回 primary hit、recall、subset recall。search 可对非空查询
子集评分；fit/selection 要求其阶段输入人口全部覆盖。final 故意不开放；无有效
Linux Landlock、worker 失败、缺失标注或不完整排名都不能补成零分。父进程把
store 先解析为绝对路径，避免子进程切换到 scratch 后丢失相对路径。

fresh-context 源码审查为 same-family/provisional。首次指出相对 store 路径的
真实阻断；修复及回归后复审确认原阻断关闭、无新阻断或 target/subset 泄漏。
审查本地 6 passed/3 skipped；跳过 Linux 实进程项，不替代 CI。作者 Windows
定向 71 passed/4 skipped；加入 fit 断言后完整 `tests/reproduce` 为
481 passed/7 skipped，新测试 6 passed/3 skipped；Ruff、format、mypy、
Bandit 与 `git diff --check` 均通过。Linux 实进程资格须由本提交精确 SHA CI
再次验证，不借上一提交的 CI 通过。

本轮所有输入、图库和 gold 均为自制 fixture；无正式 benchmark 图像或标注
访问，无新模型推理/GPU/付费 API。代码让一条 restricted scoring 路径实际可调，
但完整 CIR controller 尚未强制所有 proposer/predictor/optimizer 走新角色；
合法完整数据、真实近重复分组、sealed final、官方全图库 parity 与所有角色费用
仍缺。因此 M1/Pilot/论文结论保持 `NOT_ADMITTED`，历史账本/default4不变。
