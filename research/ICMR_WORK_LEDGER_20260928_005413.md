# ICMR 检索物理操作账本增量，2026-09-28 00:54:13 UTC

状态：`AUTHORED_DURABLE_OPERATION_ACCOUNTING_NOT_M1_ADMITTED`。接续同一 ARIS
experiment-bridge，不启动新科学轮次或定时器，不改变 default4、用户跨版本总预算
及既有生成模型/GPU 物理账本。

## 做了什么

`reproduce/retrieval_work_ledger.py` 新增单写入 SQLite 物理操作收据。编码器加载、
图像/文本编码、完整排名回调、受限评分器在实际运行前持久预留 attempt；成功和
异常分别结算。进程中断留下 `reserved` 未知成本，恢复时阻止继续及静默重放；
没有假定失败为零费用。`MeteredClipEncoder` 读取现有 `ClipCPUEncoder` 的实际
forward 计数，预处理失败可与 forward 后失败区分。没有计数器、进程中断或
无法读取时保持 NULL/未测，而非记为零。嵌套 callback 与 encoder 各有收据。

`RetrievalAuditBridge` 现在强制传入该账本，并逐查询记录排名回调与受限评分
进程；评分 scratch 建立也在受限评分收据内。原 `AuditJournal` 的固定随机前缀、
逻辑 episode、失败不补零及 survivor 实际补全不变。`check_retrieval_composed_cpu`
将原 authored CLIP 基线的加载、6 次图像、4 次文本编码与 4 次完整排名接到
新账本，保留原自制图像、遍历顺序、batch1、0.5 示范融合权重和 bitwise 检查。
新 fake-encoder 接线测试验证 15 次操作、10 次已测 forward、5 次未测 forward
类别；它不证明新源码已经用真实 CLIP 再运行。

这份账本是生成模型 `ResourceLedger` 的补充，不替代 token/GPU 计费。排名回调
可能含未接入的模型、captioner、图像读取和缓存费用；现有 callback 只测外层
耗时，不能认证内部成本。汇总的 settled operation 时长会因嵌套而重叠，明确
标为非可加值，不能当 CPU/GPU allocated time 或端到端运行时间下界。完整
CIR controller、来源/权重冻结、所有物理角色、ONLINE_PINNED、selection、
sealed final 和全矩阵预算预测仍未完成。

## 审查与验证

fresh-context source review 为同模型家族 `provisional`。初审发现嵌套耗时
12 秒外层 + 10 秒内层被误称 22 秒“下界”的真实 BLOCKING；已改字段名和
说明，并加入确定性重叠回归。唯一 follow-up 复审确认该 blocker 关闭，未见
新的本增量 blocker。这不是跨家族科学审查。

最终源码本机：定向 `18 passed, 2 skipped`；`tests/reproduce` 为
`492 passed, 8 skipped`（81.68 秒）。Ruff/format、Bandit、新账本独立
strict mypy（`--explicit-package-bases --follow-imports=silent`）、
`git diff --check` 通过。仓库全套 `2029 passed, 11 skipped`（443.97 秒）
启动于最后两处小幅源码调整前，不用作最终源码字节级证明；待精确 SHA CI。
Windows 跳过 Linux Landlock worker 和本机缺 Torch 的数值项，不能由本地
通过宣称这些环境已执行。上轮 WSL/DrvFS 失败仍原样保留。

本轮没有读取或下载官方 CIRR/FashionIQ 图像、标注或 final 成员，没有新真实
CLIP/生成模型调用、GPU allocation 或付费 API；本地测试 CPU 开销非零，未
完整计量。此前真实 CLIP authored 资格是旧源码的独立证据，不能替代这次新
账本的真实模型运行。M1/Pilot/ICMR 方法收益、论文和来源许可均未准入。
