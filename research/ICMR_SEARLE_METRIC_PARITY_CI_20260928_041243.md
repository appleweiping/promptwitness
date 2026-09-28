# 固定 SEARLE 指标函数体差分：精确提交 CI

记录：2026-09-28 04:12:43 UTC。源码提交
`040313894f74a70d15c6ff8e66eba8718625087a` 已推送至
`research/promptwitness-delta-v1`，远端分支读回同 SHA。
[GitHub Actions run 36376271977](https://github.com/appleweiping/promptwitness/actions/runs/36376271977)
终态 `completed/success`，**15/15 jobs success**。Ubuntu Python 3.10：
2060 passed、2 skipped、coverage 94.04%；Ubuntu Python 3.14：
2060 passed、2 skipped、127 warnings、coverage 94.05%。Windows/macOS
测试、质量、构建和三平台 wheel smoke 也都成功。警告和 skip 不写成零。

同一源码本地 Windows 全套 `2043 passed, 19 skipped`（379.78 s）；
三个新失败收据回归 `3 passed`，全仓 Ruff/format、61 文件 source mypy、
配置的 source Bandit、`git diff --check` 通过。单独扫描新 helper 的
Bandit B102 仍为 Medium/High-confidence、exit 1：它执行固定 SHA 的可信
上游函数体，不是任意文件安全沙箱，未用 `nosec` 隐藏。Windows helper
mypy 因缺 Torch 模块直接失败，显式忽略缺失导入后通过；正向 Torch
资格由私有 WSL CPU 实际执行，不归功于 CI。

CI 不下载/运行私有 SEARLE 源码，也不含 benchmark 数据、正式图像或 CLIP
权重；它验证的是自写 negative receipts、包和既有工程回归。
独立 WSL 正向运行的固定源码 SHA、17 项窄差分和私有收据见
`ICMR_SEARLE_METRIC_PARITY_20260928_035638.md`。两项证据不能合成
官方服务器、完整图库、tie 行为、数据许可、M1/Pilot 或 C1–C3 方法效果。
已有 all-tie 反例、CIRR NLVR2 与 FashionIQ 来源/许可门槛继续有效；
ARIS 默认 4 轮、历史资源账本、无定时器和其他四仓冻结不变。
