# 检索 MIPRO 自制资格的精确源码 CI

记录：2026-09-28。源码提交
`22733777c526e1b6d5f4d7061ddf632ed5f60ed3` 已推送到
`research/promptwitness-delta-v1`；[GitHub CI run 36412935014](https://github.com/appleweiping/promptwitness/actions/runs/36412935014)
返回相同 `headSha`、`completed/success`、15/15 jobs 成功。

- Ubuntu Python 3.10：2118 passed、3 skipped，coverage 94.04%。
- Ubuntu Python 3.14：2118 passed、3 skipped、127 warnings，coverage 94.05%。
- Windows Python 3.10：2098 passed、23 skipped，coverage 94.04%。
- 其余矩阵测试、增量核心分支覆盖、质量、构建及三系统 wheel smoke
  同属该次 15/15 成功；这里不虚报未逐项提取的测试数。

本地 Windows Python 3.12 的 2093 passed、28 skipped/94.03% 与上面
CI 平台数不同，不能互相替代。CI 安装的普通开发依赖不包含私有固定
DSPy/Optuna 检查环境，因此**未在 CI 运行完整 native MIPRO 编译资格**；
该资格由本地隔离 CPU 环境独立执行并在
`ICMR_NATIVE_MIPRO_RETRIEVAL.md/json` 记录私有结果哈希。
CI 也未下载第一方 CLIP 权重、正式 CIRR/FashionIQ 图像/标注、
真实 Qwen/OLMo 权重，未运行 GPU、Pilot、正式数据评测或论文统计。
15/15 只证明精确提交的工程检查成功，不表示 ONLINE_PINNED、
全角色成本、C1–C3、M1/Pilot 或 ICMR 投稿包通过。
