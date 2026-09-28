# ICMR 检索 GEPA 受限 scorer：精确源码 CI

记录：2026-09-28 12:52:26 UTC。源码提交
`5f6c7249e84692a4b7e5d358269dc78e62c29fe1` 已推送至
`research/promptwitness-delta-v1`；[GitHub CI run 36423756566](https://github.com/appleweiping/promptwitness/actions/runs/36423756566)
的 `headSha` 精确相等、终态 `completed/success`，15/15 jobs 均完成且
成功，包括 Quality、incremental branch coverage、跨平台测试、
distribution build 和 wheel smoke。

代表性测试收据：Ubuntu Python 3.10 为 2135 passed、3 skipped、
coverage 94.04%；Ubuntu Python 3.14 为 2135 passed、3 skipped、
coverage 94.05%；Windows Python 3.10 为 2115 passed、23 skipped、
coverage 94.04%。本地当前源码另有 Windows Python 3.12 全套
2110 passed、28 skipped、94.03%；两者环境与跳过数不同，不能混算。

CI 不安装私有固定 GEPA/WSL ext4 实验环境，不复跑 Landlock 原生
qualification、真实 CLIP/Qwen/OLMo 或官方 CIRR/FashionIQ 数据；
自制受限评分器运行仍以 `ICMR_NATIVE_GEPA_RESTRICTED.md/json` 的
私有哈希收据为准。CI 是工程可复现性证据，不是优化器对 gold 的
进程隔离、全角色成本、M1/Pilot、C1–C3 或论文结果的科学验收。
SSH 当前指纹冲突未登录；goal active、ARIS default 4、无定时器、
其他四个 NLP 仓库冻结及历史账本不清零。
