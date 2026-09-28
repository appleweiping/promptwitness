# ICMR 检索操作账本：精确源码 CI，2026-09-28 01:03:59 UTC

源码提交 `83e574a9f615aa24ae5a8851830903a941ec5846` 已推送到
`origin/research/promptwitness-delta-v1`。
[GitHub Actions run 36364121379](https://github.com/appleweiping/promptwitness/actions/runs/36364121379)
终态 `completed/success`，15/15 job 成功。Ubuntu Python 3.10 全套为
`2039 passed, 1 skipped`（181.92 秒）；Ubuntu Python 3.14 为
`2039 passed, 1 skipped, 127 warnings`（83.21 秒）。工作流测试矩阵
安装 `.[dev,research]` 后执行 `pytest`；新增检索账本、桥接和 checker 接线
测试在此精确源码树中，Linux-only scorer worker 测试不因平台条件跳过。
Quality、构建和三平台 wheel smoke 同样通过。

这关闭本增量精确 SHA Linux CI 待验，不是新的真实 CLIP qualification，
也不是 CIRR/FashionIQ 原图/标注、全图库模型排名、物理全角色成本、
ONLINE_PINNED、M1/Pilot 或论文效果证据。Windows 本机跳过项、先前 WSL
失败及审查的同家族 provisional 边界均保留。该 CI 不验证后续未提交的
`EXPERIMENT_PLAN.md` claim map 文档修正；文档须另行提交并核对新 SHA。
