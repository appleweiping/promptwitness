# Selection 受限 ranker：精确提交 CI 终态

记录：2026-09-28 08:37:02 UTC。源码及本地证据提交
`e0c6daeb561ea7b9c80234012ed8c09621cea2ee` 已推送到
`research/promptwitness-delta-v1`。GitHub [run 36397811924](https://github.com/appleweiping/promptwitness/actions/runs/36397811924)
的 `headSha` 与该提交完全一致，终态 `completed/success`，15/15 job
成功，无失败或待运行 job。Ubuntu Python 3.10 日志为 2093 passed、
3 skipped、coverage 94.04%；Python 3.14 为 2093 passed、3 skipped、
coverage 94.05%。这两个 Linux job 运行了公开仓库的实际测试矩阵；
私有模型、数据和服务器均未由 CI 触及。

通过状态只授予这次 input-only selection ranker 的工程验证：
受限 worker 可与独立 selection scorer 在自制 fixture 上组合，gold
隔离和完整排名失败路径受测试覆盖。它不授予正式 CIRR/FashionIQ
数据许可、真实模型效果、完整官方 parity、sealed final、全角色
费用、M1/Pilot、C1–C3 或 ICMR 稿件资格。ARIS 默认 4 轮及旧账本未改；
没有新定时器或服务器 SSH 连接，四个非 PromptWitness 仓库仍冻结。
