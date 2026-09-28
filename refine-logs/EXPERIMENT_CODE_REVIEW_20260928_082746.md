# Selection 受限 ranker 的同系列静态审查

记录：2026-09-28 08:27:46 UTC。两次 `codex exec` 均为新上下文、
只读 `gpt-6-astra`/xhigh，审查对象是相对 `293aeff` 的 8 个未提交
实现/测试文件。请求、回复和元数据保存在忽略目录
`.aris/traces/experiment-bridge/2026-09-28_run01/002-selection-ranker-code-review*`
与 `003-selection-ranker-followup-review*`；不提交 trace 或私有数据。

初审：0 BLOCKING；1 NON-BLOCKING——worker 在 scorer 之前仍可能
输出不完整排名。后续增加类别图库完整排列校验：ranker 的两个实际
方法在账本内校验，worker 发送结果前再次校验；失败结算为 failed，
不产生成绩。再次审查：0 BLOCKING、无实质 NON-BLOCKING。复审指出
search 默认路径、合法排列的原顺序、CIRR 参考图的 scorer 排除以及
FashionIQ 路径未被新规则改变。复审后仅将测试脚本中的阶段路由写成
等价 `if/elif` 以满足 Ruff format，未再取得第三次静态审查。

两次审查都未运行测试，也非跨模型/跨机构独立科研验收，故标记
`same-family/provisional`、`independence_verified=false`。实际本机和
WSL 测试单独记录在 `research/ICMR_SELECTION_RANK_ISOLATION.md`。
正式数据、模型、许可、全角色预算、M1/Pilot 及 C1–C3 仍未准入。
