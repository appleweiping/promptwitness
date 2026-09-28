# ICMR 检索冻结审计桥：精确源码 CI，2026-09-28 00:22:04 UTC

源码提交：`b1609b75e0276fd93275cf6c9f49e62d35e30b82`，已推送到
`origin/research/promptwitness-delta-v1`。
[GitHub Actions run 36361517661](https://github.com/appleweiping/promptwitness/actions/runs/36361517661)
终态 `completed/success`，15/15 job 成功。Ubuntu Python 3.10 完整测试为
`2033 passed, 1 skipped`（184.96 秒）；Ubuntu Python 3.14 为
`2033 passed, 1 skipped, 128 warnings`（90.53 秒）。CI 工作流在各测试矩阵
安装 `.[dev,research]` 后运行 `pytest`；新桥测试中的 Linux 实 worker 用例
只在非 Linux 跳过，因此本提交的 Ubuntu 测试包含该用例。质量、构建和三平台
wheel smoke 也成功。

这只关闭了“新桥尚无精确 SHA Linux CI”工程待验项。先前 WSL/DrvFS 六项
`3 passed, 3 failed` 的环境失败原样保留，不推断它已修复。上述测试均为
authored fixture；没有访问 CIRR/FashionIQ 官方图像/标注、真实 encoder 或
模型推理。完整 CIR controller、物理资源账本、ONLINE_PINNED、selection、
sealed final、M1/Pilot 和 ICMR 方法效果仍为 `NOT_ADMITTED`。不能把 CI
成功当作论文结果或跨家族科学审查。原 ARIS default4、历史账本与无 timer
状态不变。
