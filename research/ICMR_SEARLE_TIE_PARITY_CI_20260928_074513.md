# SEARLE 自制 CPU tie 审查后修正：精确提交 CI

记录：2026-09-28 07:45:13 UTC。源码和报告提交
`5bee65a88e41cd1295c0fb12a11892a0ffe12946` 已推送到
`origin/research/promptwitness-delta-v1`。
[CI run 36392794136](https://github.com/appleweiping/promptwitness/actions/runs/36392794136)
的 `headSha` 精确等于该提交，终态 `completed/success`，15/15 jobs
全部 success。当前工作树在提交后干净。本机同一代码的原配置完整套件
2058 passed、22 skipped、coverage 94.03%，项目级 Ruff、format、
mypy、配置 Bandit、build 和 Twine 通过；详见
`research/ICMR_SEARLE_TIE_PARITY_20260928_073737.md`。

CI 没有下载或执行私有 pinned SEARLE `validate.py`，因而新增可选测试
在公开矩阵中按设计跳过；私有 WSL 环境另有正向与故障注入的 4 passed
记录。CI 不访问 CIRR/FashionIQ 官方图像、标注或 test server，
不运行 Qwen/OLMo、真实 CLIP/GPU。它证明跨平台工程检查通过，
**不证明**正式图库/GPU tie parity、数据使用许可、真实模型成本或
M1/Pilot/C1–C3 方法收益。审查仅 same-family/provisional，审查后的
修正未经 reviewer 二次复审。SSH 主机指纹仍未从可信渠道重新确认，
未登录服务器。ARIS 默认4轮、历史账本、无定时器及其他四仓冻结不变。
