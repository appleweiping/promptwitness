# ICMR 目标描述：生成模型到受限检索器的工程连接

记录：2026-09-28 05:45:24 UTC。状态：`PASS_AUTHORED_TRANSPORT_ONLY`，
独立代码审查 `UNAVAILABLE_LOCAL_ONLY`，真实双模型和正式数据 `NOT_RUN`。
这是 M2 的前置接线，不是 M1、Pilot、C1–C3 或 ICMR 论文结果。

## 本次实现的闭环

`reproduce/retrieval_generated_description.py` 从 input-only 查询文件读取
query ID、参考图 ID、修改文本及类别；调用者须为每个参考图提供一条已固定 caption。
候选 prompt 只允许文本消息，既可以同时绑定 `reference_caption` 与
`modification`，也可以作为原有 instruction/demo 前缀并附加固定输入消息。
未知/不完整变量、工具和多模态块会拒绝；不读取或传递 target/subset 真值。

`GeneratedDescriptionRanker.rank_one()` 先通过现有 `PersistentModel.metered`
以 `purpose=search` 预留并结算真实模型请求，再检查响应身份、状态和非空输出，
最后将原样输出交给现有 `RestrictedRankerSession.rank_description()`。
模型私有 wire 收据、LLM token/GPU 物理账本、CLIP text/rank 操作账本及外层
`RetrievalAuditBridge`/受限 scorer 是不同层次的收据；本模块没有合并或免计费用。
重复 attempt/query ID 会在物理账本预留处拒绝，不能作为免费缓存命中。

`TorchRuntime` 新增仅 task 角色可用的 `cir_description` family，输出上限
256 tokens。旧 `TASK_CAPS` 的三个文本任务、旧角色上限及解码设置不变；
`SETTINGS` 中显式增加新上限，因此新进程的 backend profile digest 会变化。
这是有意的新执行身份，不能拿旧冻结身份继续本次请求；旧历史账本不清零。

## 实测与审查边界

Windows 定向 46 passed；`tests/reproduce` 522 passed、16 skipped；
全套 2059 passed、19 skipped、124 warnings、覆盖率 94.04%，exit 0。
新文件 Ruff check/format、`mypy src` 61 文件、配置 Bandit `src`、
sdist/wheel build 与新 0.2.0 两包 Twine check 均 exit 0。
独立辅助目录 Bandit 扫描仍报既存本地 `from_pretrained` 未显式 revision 的
B615 medium/high-confidence；不称全树 Bandit 通过。

测试验证了两个输入字段的精确渲染、拒绝未声明字段/多模态、候选对象不被修改、
空完成输出不进入检索、实际 `ResourceLedger` 在 authored `_execute` 之前预留，
以及同一请求重放不增加模型调用。模型和 ranker 均由测试替身承载；
**没有**因此实际执行 Qwen/OLMo、真实 captioner、正式图像、正式评分或 GPU。
现有受限 CLIP 自写图运行是上一增量的独立证据，不能嫁接成此次双模型结果。

fresh gpt-6-astra/xhigh 审查调用因 agent thread limit 被拒绝；按
`experiment-bridge` fallback 做了本地逐项审查，记录在
`refine-logs/EXPERIMENT_CODE_REVIEW.md`，标为 `[local-only]`，不伪称
same-family provisional 或跨家族接受。没有部署新代码；服务器当前 SSH
ED25519 指纹与用户先前核验值不一致，等待可信渠道确认。未连接服务器，
未消费新模型调用/token/GPU/付费 API，也未读取新的官方 CIRR/FashionIQ 数据。

## 下一准入步骤

先核验服务器身份，再在已授权服务器上对两个原生成模型使用同一固定
caption+modification 做最小真实 sanity；冻结并验证 caption 来源、执行身份、
两进程数据权限和连续物理账本。随后才可对真实训练侧数据与完整图库做官方
评分/隔离、M1、Pilot 和成本预测。此模块只实现 search；fit、reference、
selection、final、原生 GEPA/MIPRO、完整全角色预算和科学证据仍未完成。
ARIS 默认 4 轮、总资源上限、无定时器和其余四仓冻结均不变。
