# Selection 真实 CLIP 资格脚本代码审查

记录：2026-09-28 09:10:03 UTC。新上下文、只读 `codex exec`、
`gpt-6-astra`/xhigh 两次；因代理入口 `agent thread limit reached`，
改用 CLI 新会话。请求/回复/元数据保存在忽略的
`.aris/traces/experiment-bridge/2026-09-28_run01/004-*` 与 `005-*`。
同属 OpenAI 系列，`same-family/provisional`、非独立科学验收；审查没
运行测试、模型或联网。

初审：1 BLOCKING——selection 描述评分仅提交单查询，真实 scorer
要求完整人口；替身测试错误容忍。1 NON-BLOCKING——scorer 失败
stderr 未留存。修复为两条 description 排名及一次完整 selection 评分，
测试约束同真实 scorer；正常非零退出的 stderr 写到私有 scratch。
唯一 follow-up：BLOCKING 无；确认 search 单查询诊断仍合法、selection
完整人口及 10/12 内层操作、6/7 forward 计数一致。

Follow-up 另报两项非阻断：第二条描述排名失败时第一条成功结果原先
没有写入报告；已在审查后补逐次写入并加失败回归，**未再复审**。
`launch_role` 超时或零退出后 JSON/身份验证失败的 stderr 不受新增
留存分支覆盖；不得宣称所有失败都持有 stderr。真实 WSL/CLIP 资格
与本地测试另见 `research/ICMR_SELECTION_REAL_CLIP.md`，不是 reviewer
亲自执行的证据。正式数据、许可、M1/Pilot 和科学主张仍未准入。
