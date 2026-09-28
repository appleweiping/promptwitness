# ICMR ranker 隔离精确 SHA CI，2026-09-28 02:34:43 UTC

源提交 `769b05a736a9c7d8dc7e97188a81cd316efda440` 已推送到
`research/promptwitness-delta-v1`。GitHub Actions
[run 36369807411](https://github.com/appleweiping/promptwitness/actions/runs/36369807411)
终态 `success`，15/15 jobs 成功。Ubuntu Python 3.10 与 3.14 各
`2052 passed, 2 skipped`；3.14 保留 127 warnings。质量、打包、wheel smoke
和其他平台矩阵也终态成功。新增 Linux-only persistent Landlock ranker、
gold sentinel、受限 scorer/gate、半帧与背压、超时后 outer failed 与 inner
unresolved 的 authored 测试已被该源码 SHA 的 Ubuntu 全套执行。

这个 CI 只提升本增量为 `ENGINEERING_PASS_AUTHORED`：替身 ranker、操作者
自写输入/标签/虚拟 checkpoint，不能替代受限进程内真实 CLIP 权重和图像编码、
合法 CIRR/FashionIQ 数据、官方 evaluator parity、全角色资源预测、
fit/selection/final 或 ONLINE_PINNED。历史 WSL drvfs 本机失败保留，不以 CI
倒写成 WSL 本地通过。M1、Pilot、科学效果和 ICMR 稿件仍 `NOT_ADMITTED`。
ARIS 默认四轮、历史账本、无 timer、其他四仓冻结不变。
