# ICMR 指标函数体差分验收（仅自制无并列输入）

记录：2026-09-28 03:56:38 UTC。状态为
`PASS_PINNED_SEARLE_METRIC_BODY_AUTHORED_NO_TIES`，属于 M1 前置的窄工程资格，
**不是** CIRR 官方测试服务器、FashionIQ 正式评测、完整图库、真实图像、M1、
Pilot、方法收益或 ICMR 投稿结果。原科学门禁保持 `NOT_ADMITTED`。

`reproduce/check_retrieval_searle_metric_parity.py` 在本地先核对私有上游
`validate.py` 的 SHA-256
`83eef3dd2448e82ba9ef59708df8dfa3a88a8f17ec1d4c0f152676dc5dde28fc`，
对应 [SEARLE 固定提交](https://github.com/miccunifi/SEARLE/blob/a9c314ba4b6e6e14be5a9c3fdf6b66e6a0c23e37/src/validate.py)。
它只编译原 `cirr_compute_val_metrics` 与 `fiq_compute_val_metrics` 的 AST
函数定义，保留原函数体/decorator，以自制 CPU 特征替换预测生成器，
不运行其顶层模型/数据代码。固定源码属于可信代码执行，`exec` **不是** sandbox；
上游 CC BY-NC 4.0 源码只存在私有研究目录，未复制进本项目 MIT 仓库。

在原生 WSL CPU Torch 2.7.1+cpu、`CUDA_VISIBLE_DEVICES=""` 下，主作者
执行最新 checker SHA-256
`32b4bdedca9c8639c6ce67da1dc3ae2068d60bee04d2edfa5e5abcaa6c812f3f`。
自制 55 候选严格排序：CIRR 5 查询、FashionIQ 三类别共 6 查询。
原 CIRR 函数执行一次、FashionIQ 函数每类别一次；与本项目独立
`summarize_rankings` 在百分数尺度比对 13 项原函数直接输出：CIRR 全库
R@1/5/10/50、subset R@1/2/3，FashionIQ 每类 R@10/50，
另由原函数三类输出按明确权重计算 macro R@10/50 和查询等权
micro R@10/50 并比对 4 项，共 17 项。全部差值在
`1e-4` 百分点以内。类别人数故意为 1/2/3，使 R@10 的 macro
44.4444% 与 micro 33.3333% 不同，避免混同两个 estimand。
本次 `qualification.json` 私有收据 SHA-256 为
`a601ff37053bfbf7fd94ec30fbcb10abf0078002ef89c9970bb09fd8228b9af7`；
checker wall 8.925882023 s，不代表完整 CPU allocation。
本次 0 新模型 forward、0 GPU allocation、0 付费 API；CPU 开销非零。

三个负向回归检验错误 SHA/缺失函数在 Torch 导入前失败，以及函数体已编译
但首次输入构造失败时保留 `FAILED_RETAINED` 收据且不误报已执行。
fresh-context gpt-6-astra/xhigh 同家族 provisional
审查发现旧版在真正执行前即把 `source_metric_function_bodies_executed`
写为 true；故障注入实证后，已改为两函数和比较都完成才置 true，
独立复测关闭此收据缺陷。审查者还用同一固定源码在独立 WSL 输出目录
完成正向对照；它不等于跨家族独立科学审查。主作者本地两个负面测试通过；
主作者最新源码本地定向 `3 passed`；完整套件与精确提交 CI 另行记录，
不能由先前 SHA 的 CI 代替。Ruff 全仓、格式、新 helper mypy
（忽略本机缺 Torch 的导入）及配置的 source Bandit 通过。
单文件 Bandit 实际 exit 1，唯一项是可信固定 AST 执行的
`B102:exec_used`（Medium/High-confidence）；未加 `nosec` 隐藏它，也不把
SHA 核验说成可安全运行任意第三方源码。Windows 本地 helper mypy 在未装
Torch 时直接报告缺失模块；仅使用 `--ignore-missing-imports` 后通过。

**未关闭的准入项：** 已有 all-tie 反例表明本项目 lexical tie 与 Torch
排序不同；本次特意无并列，不消除该反例。55 候选不代表 CIRR/FashionIQ
完整库。固定 SEARLE 方法源码不是 CIRR 官方服务器或 FashionIQ 官方
独立 scorer 的等价证明。正式图像/标注来源和许可、真实图库数值及 tie
行为、训练分组与 sealed final、captioner/双生成模型、全角色成本和
C1–C3 效果仍待通过各自门禁。CIRR 的 NLVR2 条款与 FashionIQ exact
CDLA/image-source 尚未解决，不借助镜像绕过，也未下载官方图像/标注。
历史成本、ARIS 默认 4 轮、无定时器、其他四仓冻结均未改变。
