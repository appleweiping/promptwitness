# Selection 阶段 input-only 排名隔离

记录：2026-09-28 08:27:46 UTC。状态：`PASS_AUTHORED_LOCAL_LINUX`；
精确 SHA CI 待提交后核验；正式检索实验仍 `NOT_ADMITTED`。

这次为已存在的受限 `selection` 检索 scorer 补上持久 ranker。新
`retrieval_selection_ranker` 只获 `selection/inputs` 读取权限，不获
`search`、`fit`、`final` 输入或任何 gold；`final` ranker 入口仍明确拒绝。
`RestrictedRankerSession` 默认 `search` 行为不变，选 `selection` 时绑定
对应 store/worker/计费阶段。两阶段的排名须为类别图库的完整 ID 排列；
缺失、重复、额外和异类别 ID 均在 ranker 账本内失败，worker 发帧前还
校验一次，不能以不完整排名向 scorer 冒充成功。

验证记录：

- Windows 原配置全套：2070 passed、26 skipped、440.39 秒、覆盖率
  94.03%，exit 0；格式修正后，相关定向回归再次 exit 0。
- WSL ext4 fresh clone 中实际运行 Landlock 定向回归：修复测试路由后
  67 passed、1 skipped；最后仅将该路由改写为等价 `if/elif`，复拷该文件
  后同一组再次 exit 0。实际测试覆盖 15 个角色 × 11 个 split leaf 的
  read/write 拒绝矩阵、selection ranker 到独立 scorer 的自制端到端链路、
  gold sentinel 隔离及不完整排名失败。第一次 WSL 定向运行曾因测试
  脚本把新角色误路由到 `search` 而 1 failed，已修复并保留该失败事实。
- 项目级 Ruff check/format、mypy（61 个源文件）、配置内 Bandit，
  sdist/wheel build、Twine 两包检查以及 `git diff --check` 均通过。
- 新上下文只读 `gpt-6-astra`/xhigh 审查先提示 worker 端完整性缺口，
  修改后第二次静态审查未发现 BLOCKING 或实质 NON-BLOCKING；属于
  **same-family/provisional**，没有独立执行测试。详见
  `refine-logs/EXPERIMENT_CODE_REVIEW.md`。审查后仅把测试脚本阶段路由
  改成等价 `if/elif` 以满足 formatter，该机械改写未经重新审查。

本次证据仅是自制输入/真值与受限进程边界，不是 CIRR/FashionIQ 正式
数据结果；没有真实 Qwen/OLMo 推理、正式图库/许可、在线优化、
sealed final、全角色成本、M1/Pilot 或 C1–C3 收益。没有新增 GPU、
付费 API 或服务器调用；CPU 验证开销不记作零。此前历史预算和 ARIS
默认 4 轮未改，无定时任务，另外四个仓库仍冻结。SSH 当前指纹与先前
核验值不同，未连接 `8.133.245.52:33123`。
