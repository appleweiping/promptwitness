# ICMR 目标描述检索通道：精确源码 CI 终态

记录：2026-09-28 05:11:08 UTC。此次增量提交
`42ba46f4563a78dbf944fd9c0c6282074bb53b76` 已推送至
`research/promptwitness-delta-v1`，`git ls-remote` 读回相同 SHA。
[GitHub Actions run 36380294671](https://github.com/appleweiping/promptwitness/actions/runs/36380294671)
的 `headSha` 与之相同，终态 `completed/success`、**15/15 jobs success**。

CI 包含增量核心 branch coverage、Ruff/格式/mypy/配置 Bandit、分发包
构建、Ubuntu/macOS/Windows 多 Python 版本测试和三平台 wheel smoke。
Ubuntu Python 3.10 为 2064 passed/2 skipped、coverage 94.04%；Ubuntu
Python 3.14 为 2064 passed/2 skipped、coverage 94.05%，保留 128 warnings。
其他 job 的 success 仅按终态记录，不推断相同测试计数。Windows 本地
Python 3.12 原配置全套为 2046 passed/20 skipped、94.03%；另一次
无覆盖率全套同为 2046 passed/20 skipped。sdist/wheel 构建与新包
Twine check 本地 exit0。新增 helper 的 medium+ Bandit 定向扫描 exit0；
扩大到整个历史 `src reproduce` 树的扫描 exit1，10 medium/39 low，
不能以 CI 配置扫描通过掩盖它。

私有 WSL CPU、第一方 CLIP ViT-L/14 在 Landlock search-only ranker 中
执行一次冷索引、两次直接查询和一次自写目标描述查询，两个独立 scorer
处理自写真值；Linux focused 18 passed/1 skipped。这是与公开 CI **独立**
的实运行资格，权重、三张自制图和完整输出不上传。详情、源码/权重/收据
SHA-256、失败尝试与费用边界见
`research/ICMR_DESCRIPTION_TRANSPORT_20260928_043725.md`。

CI 没有运行私有真实 CLIP 权重、正式 CIRR/FashionIQ 图像或标注、
Qwen/OLMo 生成，也没有官方完整图库/tie parity、sealed final 或在线
优化器实验。该代码只打通 *caller-supplied description → 受限检索 →
独立评分*，不证明目标描述由真实模型产生，不证明方法提升。科学门禁
`M1/Pilot/C1–C3/scientific_admission=NOT_ADMITTED`；正式数据许可、
全角色 token/GPU/CPU 费用、captioner/双生成模型、官方评价对齐仍待完成。
ARIS 默认四轮、历史预算、零付费 API/云租赁、无定时器、其他四仓冻结
以及不上传私有数据/审查 trace 的约束不变。本轮状态为可审计工程进展，
不是论文或目标终态。
