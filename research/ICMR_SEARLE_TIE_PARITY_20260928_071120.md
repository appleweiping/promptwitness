# SEARLE 固定函数体：自制 CPU 并列排序差分验收

记录：2026-09-28 07:02:02 UTC。状态仅为
`PASS_PINNED_SEARLE_METRIC_BODY_AUTHORED_CPU_TIE_FIXTURE`，不是正式
CIRR/FashionIQ 数据、完整图库、GPU、官方测试服务器、M1、Pilot 或论文效果。
原无并列验收 `research/ICMR_SEARLE_METRIC_PARITY.md` 和所有旧收据保留。

改动在 `reproduce/check_retrieval_searle_metric_parity.py`：原 55 候选
严格排序/11 查询/17 项指标对照不删；新增 55 个向量完全相同的自制
图库，5 个 CIRR 和 4 个 FashionIQ 查询。该样例中 Torch 2.7.1+cpu
的 `argsort` 完整顺序与词典序不同。生产用单查询
`rank_float32_cpu` 的完整 55-ID 排列与固定 SEARLE 函数体所用的
批量距离/排序表达式逐项相同；再由固定原函数体实际计算 CIRR
全库 R@1/5/10/50、subset R@1/2/3、FashionIQ 各类 R@10/50，
与本项目独立 `summarize_rankings` 对照。后者加上 macro/micro
共 17 项并列指标；全部绝对差小于 `1e-4` 百分点。两组共 34 项
指标对照，且并列样例有 4 次完整排列检查。上游函数体来源为
[固定 SEARLE revision](https://github.com/miccunifi/SEARLE/blob/a9c314ba4b6e6e14be5a9c3fdf6b66e6a0c23e37/src/validate.py)。

原文件 SHA-256 `83eef3dd2448e82ba9ef59708df8dfa3a88a8f17ec1d4c0f152676dc5dde28fc`
先核验；它只在本机私有研究目录，未复制入 MIT 仓库。最终 checker
SHA-256 `e4066c10cd81d44ec447a0d568de94edfa14937c7d92dd5499dad1aa8b852d81`。
WSL Ubuntu Python 3.12.3、Torch 2.7.1+cpu、`CUDA_VISIBLE_DEVICES=""`，
最终一次实际运行 exit 0，checker wall 3.3013 秒；私有
`qualification.json` SHA-256
`cd98afae7fbb2fed84f507afc831e4a9add79028fa757dfb44af6e8c61b749c8`。
报告格式升至 v2；原 v1 收据不改。`authored_cpu_tie_fixture_checked=true`
只表示该固定自制样例通过；顶层 `ties_qualified=false`、
`official_full_gallery_parity_established=false`、`scientific_result=false`
保持不变。批量排序表达式的排列检查与原函数体的指标比较是不同证据，
不能将前者写成完整原 CLI/服务器的运行。

本地 Windows 定向 43 passed；Ruff 检查/格式与 `git diff --check`
通过。隔离 mypy 检查通过；不隔离的单文件 mypy 因既有导入文件
`retrieval_cpu_ranking.py` 的未标注参数而失败，未掩盖。单文件 Bandit
保留既有可信固定 AST `exec` 的 B102 Medium/High-confidence 一项，
无 `nosec`；该 SHA 核验不把 `exec` 变成任意源码沙箱。
fresh gpt-6-astra/xhigh 审查请求因 agent thread limit 被拒，只有
`[local-only]` diff/失败收据/正式表达式核对，不声称独立审查通过。
精确源码 CI 待提交后核验。

2026-09-28 07:11:20 UTC 增量：提交前本机原配置完整套件 exit 0，
2058 passed、21 skipped、372.73 秒，coverage 94.03%（门槛 90%）。
与上轮 2059/20 记录相比有一项由 passed 变为 skipped，原因未在
本次套件输出中说明，不把该项宣称为本次执行通过。WSL 固定源码
正向检查是单独运行，不在跨平台 CI 套件中。


本次不读取任何 benchmark annotation/image，不生成真实 caption 或模型
输出，不新增真实模型调用、GPU allocation、付费 API；CPU 开销非零。
正式图库 tie、GPU/批大小数值、原图许可、FashionIQ 明确许可、封存
final、双真实模型、全角色预算和 C1–C3/M1/Pilot 仍未准入。
同一 ARIS run、默认 4 轮、历史账本、无定时器和其他四仓冻结不变。
