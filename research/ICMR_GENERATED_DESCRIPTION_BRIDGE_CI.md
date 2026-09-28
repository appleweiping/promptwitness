# 生成描述接线：精确源码 CI

记录：2026-09-28 05:56:28 UTC。源码提交
`da9a4ba4b10ad9cca701e7db48d68755bd841231` 已推送，远端分支
`research/promptwitness-delta-v1` 读回同一 SHA。
[CI run 36383516624](https://github.com/appleweiping/promptwitness/actions/runs/36383516624)
的 `headSha` 同一，终态 `completed/success`，15/15 jobs completed/success。
Ubuntu 3.10：2076 passed、2 skipped、coverage 94.04%；Ubuntu 3.14：
2076 passed、2 skipped、127 warnings、coverage 94.05%。

本地 Windows 原配置全套为 2059 passed、19 skipped、124 warnings、
coverage 94.04%，exit 0；`tests/reproduce` 522 passed、16 skipped。
新文件 Ruff/format、`mypy src` 61 文件、配置 Bandit、0.2.0 sdist/wheel
与 Twine 均过。辅助目录扩展 Bandit 的旧 B615 告警未清零。

这些证据只关闭当前源码的跨平台 authored 工程测试和构建。
CI 不加载私有 Qwen/OLMo、CLIP 权重、正式 CIRR/FashionIQ 图像/标注，
不证明 caption 来源、服务器身份、全角色物理成本、M1/Pilot 或 C1–C3。
fresh review 仍因 agent thread limit 未完成，仅本地审查；SSH 新指纹
未获可信渠道确认，未连接服务器。本轮新增真实模型调用、token、GPU、
付费 API 均为零；历史实际费用继续保留。
