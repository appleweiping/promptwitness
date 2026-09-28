# SEARLE 自制 CPU tie 差分：审查后强化

记录：2026-09-28 07:30:48 UTC。本文件补充此前
`ICMR_SEARLE_TIE_PARITY_20260928_071120.md`；保留其原始运行、测试
与限制记录，不改写历史收据。

对已推送 `b6faa72` 的新上下文只读代码审查给出 **无 BLOCKING、
两项 NON-BLOCKING**。它属于 same-family/provisional，未复跑实验；
审查原文保留于忽略的 ARIS trace，公开摘要见
`refine-logs/EXPERIMENT_CODE_REVIEW_20260928_073048.md`。

本地随后调整全并列自制 CIRR subset 的 distractor，使五个目标名次
覆盖 `1,2,3,4,5`，显式断言 R@1 < R@2 < R@3。私有 WSL Ubuntu
Python 3.12.3/Torch 2.7.1+cpu 使用事先核验的固定 SEARLE `validate.py`
（SHA-256 `83eef3dd2448e82ba9ef59708df8dfa3a88a8f17ec1d4c0f152676dc5dde28fc`）
再次运行退出码 0；新 checker SHA-256
`37a8c602514bb1748607eed9c7528b5e220befaecf5a99a477a06f951a4f2e79`。
私有正向收据 SHA-256
`c0fe5710136ccba1d27e32da47df15411c41dfd225779b3168230e7defb8d037`，
其中 subset R@1/2/3 的独立 scorer 值分别为 20/40/60%，固定上游函数体
浮点结果分别为约 20.0000003/40.0000006/60.0000024%。报告状态仅
`PASS_PINNED_SEARLE_METRIC_BODY_AUTHORED_CPU_TIE_FIXTURE`；
`ties_qualified=false`、`official_full_gallery_parity_established=false`、
`scientific_result=false`。

可选私有源码测试在 WSL 的实际运行是 `4 passed`，同时覆盖词典序故障
导致的 `FAILED_RETAINED` 和 false 标志；Windows 定向为
`43 passed, 1 skipped`，因为 CI/Windows 没有该私有源码。Ruff 检查、
格式和 `git diff --check` 已通过。完整原配置套件及新 SHA CI 正在
重新运行，结果另记，不在此预称通过。上述后续修改未经 reviewer
二次复审，不把 same-family 静态审查称为独立验证。

没有下载或运行官方 CIRR/FashionIQ 图像、标注或 test server，没有真实
Qwen/OLMo、GPU 或新的付费模型调用。完整图库/GPU tie parity、官方
评分、原图许可、FashionIQ 准确许可、captioner、双模型、全角色成本、
C1–C3/M1/Pilot 与论文效果仍 NOT_ADMITTED。与此前相同的 ARIS run、
默认 4 轮、历史账本、无定时器和其他四仓冻结均未改变。
