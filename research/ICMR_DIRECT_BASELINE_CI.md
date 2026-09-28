# ICMR direct CLIP baseline 精确 SHA 工程 CI，2026-09-28 01:43:04 UTC

源码 `c08f5c654dd8600ad1ff5ee2cc1afb15380a0183` 已推送到
`research/promptwitness-delta-v1`；[GitHub Actions run 36366650505](https://github.com/appleweiping/promptwitness/actions/runs/36366650505)
对应同一 head SHA，终态 `success`，15/15 jobs 成功。Quality、incremental core
coverage、分发构建、三平台 wheel smoke 与 Python 3.10–3.14 测试矩阵均成功。
Ubuntu 3.10：2045 passed、1 skipped（190.22 秒）；Ubuntu 3.14：2045 passed、
1 skipped、127 warnings（69.48 秒）。Linux-only authored 集成测试没有平台跳过，
包括 input-only 元数据→假图像解码/编码→完整图库排名→真实 Landlock 受限 scorer
→固定 AuditPlan/gate→survivor 分数补全。该测试运行实际 ranker 类与真实隔离
scorer 进程，但 CLIP 编码器和融合排序函数是测试替身；CI 不包含官方图像、真实 CLIP 权重或
Qwen/OLMo 调用。

本机最终源码 `tests/reproduce` exit 0，新文件定向 5 passed/1 skipped，Ruff、
format、Bandit 通过；辅助代码可选依赖缺失时使用放宽 untyped-import 规则的
targeted mypy 检查通过。早一次全仓测试 exit0、源包覆盖率 94.01%，但其启动
早于最后的 FashionIQ 回归和计数器类型小改动，不能当最终字节级本机全套证明；
本次精确 SHA CI 才是新提交的全套证据。

科学状态仍为 `DESIGN_ALIGNED_EMPIRICAL_PENDING`、M1/Pilot 未准入。真实数据
许可/版本、原图、训练分组和 final 封存、模型/权重/融合冻结、caption/LLM、
ranker 进程级输入隔离、全角色费用 forecast、原评测器 parity、native 在线优化、
统计和英文稿均待完成。历史 1170 次生成调用与 3.8074371029887377 allocated
GPU-hours 的已知快照不清零；本增量没有新增真实生成模型调用、GPU allocation 或
付费 API，本机与 CI CPU 工作量非零且未完整计量。ARIS default4、同一 run、
无 timer、其他四仓冻结不变。
