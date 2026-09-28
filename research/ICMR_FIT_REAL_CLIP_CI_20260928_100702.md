# Fit 受限 CLIP 资格代码的精确 SHA CI

记录：2026-09-28 10:07:02 UTC。源码与报告提交
`c99aff83ea88064b5a6b86e80d98e2f33a2e241f` 已推送到
`origin/research/promptwitness-delta-v1`；[CI run 36406953262](https://github.com/appleweiping/promptwitness/actions/runs/36406953262)
的 `headSha` 与其逐字相同，终态 `completed/success`，15/15 job 成功。

包括 Quality、incremental core branch coverage、各平台测试、发行包构建和
三平台 wheel smoke。Ubuntu 3.10 与 3.14 各为 2107 passed/3 skipped，
coverage 94.04%/94.05%；后者保留 127 warnings。Windows 3.10 为
2087 passed/23 skipped、coverage 94.04%。这次 CI 包含审查后补入的
fit 缺查询失败测试；与本地“全套在补丁前收集”的限制分开记录。

CI 不下载私有 CLIP 权重或正式 CIRR/FashionIQ 图像/标注，也不运行
真实 Qwen/OLMo。自制真实 CLIP/Landlock CPU 资格证据另见
`ICMR_FIT_REAL_CLIP.md/json`，不应把 CI 通过解释成官方数据实验、
完整全角色费用、M1/Pilot、C1–C3 效果或 ICMR 稿件准入。
SSH 当前提供的 ED25519 指纹与此前用户核验值不一致，未登录。
本增量保持同一 ARIS run、默认 4 轮、历史预算、无定时器和其他四仓冻结。
