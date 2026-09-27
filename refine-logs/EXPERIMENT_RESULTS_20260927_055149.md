# 接续初始记录：官方 scorer CPU slice

日期：2026-09-27 05:51:49 UTC。此记录不是研究桥完整结束，不能进入真实 Pilot 或 auto-review-loop 效果接受。

## 真正执行的检查
- 原生 authored fixture：Hotpot 3、IFTrain 4、IFBench 6（含 2 OOD）；
  13/13 与预定 expected 匹配，未加载 final prompt。
- 旧 fit-only archive 交集：Hotpot 4 个中 4 个 EM；IFTrain 5 个中 3 个满足
  raw-text/all-constraints。这是固定 membership 与指定 archive 的交集，不是
  新生成调用，不替换 ACCESS_AUDIT 中历史另一次 16 个的审计，也非模型准确率估计。
- 39 新机械 tests；全部 research helpers 132 tests，46.08 秒；
  incremental core 102 tests，134.66 秒，statement 98.9785%、branch 98.3696%。
  后两时间为 pytest wall time，不是 GPU 或精确 CPU core-hour。
- Ruff/format、mypy src 61 files、adapter strict mypy、configured Bandit src 通过。
  新 helper Bandit 保留 5 LOW（B404/B603/B607 的 import/subprocess/git PATH）；
  argv list、shell=false、明确本任务 source 路径，未 widening ignore。
- 全套只 collect 到 1671；尚未将 collect 当成 passed。新提交 CI 要核对 exact SHA。
  旧 SHA `475c96d769bf95894d4f2c428b3b038b54a21a77` 的 run 36296675360 是历史通过，
  不能冒充尚未提交 source 的 CI。

## 完整科学状态
all-scorers PARTIAL_NOT_PASSED；BFCL runtime NOT_QUALIFIED；process split、online
execution、native optimizer early rejection 仍未通过。真实 M1 UNFITTED，Pilot NOT_RUN，
144 pairs/180 runs/36 transfers 完成 0。没有测量方法效果或科学负结果。

固定统计假设/门限与完整矩阵未修改；旧 v1/v1.1 状态和消耗保留。
新模型调用=0、allocated GPU-hours=0、付费=0；未租云或访问其他任务 GPU。
CPU setup、下载、扫描、构建耗时未伪装成完全计量的 CPU 总额。

## ARIS 接续
experiment-bridge 继续 running，默认 auto-review-loop 最多 4 轮未动。
实施审查与文档验收为 same-family/provisional，不是跨模型接受。
有效真实实验和研究叙述尚未交付，active goal 不结束。定时任务不重建。
