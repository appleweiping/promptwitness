# 检索原生 GEPA 自制资格的精确源码 CI

记录：2026-09-28 11:58:22 UTC。实现与资格摘要已推送到
`research/promptwitness-delta-v1`，源码 commit
`d70d317b3039cf2db8c7c1a3971b93ec3673ffc7`。GitHub Actions
[run 36418174783](https://github.com/appleweiping/promptwitness/actions/runs/36418174783)
的 `headSha` 精确等于该值，终态 `completed/success`，15/15 jobs 成功。

可核查的测试日志包括：Ubuntu Python 3.10 为 2130 passed、3 skipped、
coverage 94.04%；Ubuntu Python 3.14 为 2130 passed、3 skipped、
coverage 94.05%；Windows Python 3.10 为 2110 passed、23 skipped、
coverage 94.04%。本地 Windows Python 3.12.13 原配置另有
2105 passed、28 skipped、coverage 94.01%；版本/可选环境差异不把
两组数字混写。其余 CI 质量、构建及平台 jobs 亦 success。

CI 仅覆盖仓库代码与测试，不安装私有固定 GEPA 研究环境，也不执行
私有完整原生 qualification、官方 CIRR/FashionIQ 图像或真实 Qwen/OLMo。
私有 authored 原生正反场景的 exit 0 与 SHA-256 收据另见
`research/ICMR_NATIVE_GEPA_RETRIEVAL.md/json`。本 CI 不授予真实
ONLINE_PINNED、受限 scorer 同次隔离、全角色预算、正式数据许可、
M1/Pilot、C1–C3、ICMR 投稿或科学效果。ARIS 默认4轮、既有历史成本、
无定时器和其他四仓冻结不变；SSH 指纹冲突，未登录服务器。
