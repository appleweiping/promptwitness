# Experiment bridge 代码审查：SEARLE 自制 CPU tie 后续

记录：2026-09-28 07:30:48 UTC。审查对象是已提交的 `b6faa72` 相对
`5831a1e` 的增量。使用 `codex exec` 新上下文、只读模式、
`gpt-6-astra`/xhigh；原始请求、回复与元数据留在忽略的
`.aris/traces/experiment-bridge/2026-09-28_run01/001-searle-tie-code-review.*`。
它与实施者同属 OpenAI 模型系列，所以标记 **same-family/provisional**，
`independence_verified=false`；审查是静态只读，没有复跑测试，也不构成
独立科学验收。

审查结论：**BLOCKING：无。** 两项 **NON-BLOCKING**：

1. 初版全并列 CIRR subset 目标名次为 `1,5,5,5,5`，R@1/2/3 无法
   区分。后续未提交 diff 改为五个不同目标名次 `1,2,3,4,5`；私有固定
   源码再次运行后，subset R@1/2/3 分别是 20/40/60%，原函数体结果
   与独立 scorer 一致。该后续修改由作者验证，**未由上述 reviewer 复审**。
2. 原有三个自动测试均在进入 tied 检查前失败。后续新增一个可选的
   Linux/private-source 回归：在 `PW_SEARLE_VALIDATE_PY` 指定事先核验的
   固定源码时运行正向并列检查，再注入词典序故障，验证
   `FAILED_RETAINED` 且 tied/scientific 标志保持 false。WSL 实测
   `4 passed`；普通跨平台 CI 在没有私有第三方源码时跳过此可选测试，
   **不把 CI 跳过写成已执行**。该后续修改同样未被 reviewer 复审。

Reviewer 另确认：上游文件 SHA 与 pin 一致、没有将上游函数体纳入 MIT
仓库、四次完整 55-ID 排列检查及 17 项指标比较语义明确、成功与失败
收据边界合理；它没有发现本增量新增 benchmark 数据访问。固定源码执行
仍是可信代码执行而非沙箱。对照只使用自制 CPU 样例；真实图库、GPU、
完整官方 CLI/服务器 parity、正式图像/标注许可、真实双模型与 M1/Pilot
科学结果都 **NOT_ADMITTED**。SSH 指纹未由用户从可信渠道重新核验，
未连接服务器。
