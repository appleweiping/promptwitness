# 生成描述端到端 authored 链路：精确源码 CI

记录：2026-09-28 06:26:24 UTC。源码提交
`a9707c441694e63646f62feaca1b4fc862d3c279` 已推送并从远端分支读回。
[CI run 36385678817](https://github.com/appleweiping/promptwitness/actions/runs/36385678817)
的 `headSha` 与之相同，终态 `completed/success`，15/15 jobs success。
Ubuntu 3.10 与 3.14 均为 2077 passed、2 skipped；覆盖率分别
94.04%/94.05%，3.14 保留 127 warnings。前一提交同两环境均为
2076 passed、2 skipped，新增 Linux-only 用例使通过数各加一、
跳过数不增。测试源码实际启动受限 ranker 与独立 scorer，并断言
模型替身账本64次、外层128次、内层64次和完整64项 survivor。

本地 Windows 原配置全套2059 passed、20 skipped、124 warnings、
coverage94.04%；多出的跳过项是 Linux-only 测试。Ruff/format 和
`git diff --check` 已过。远端 CI 也覆盖质量、构建、wheel smoke。

状态提升仅为 `PASS_AUTHORED_LINUX_END_TO_END_CHAIN`。
`PersistentModel._execute` 在该测试被替身替换，受限 ranker 使用
authored 排名而非 CLIP 权重；输入图像/annotation 均为自制 ID/字节。
CI 不加载私有 Qwen/OLMo 或真实 CLIP，不访问正式 CIRR/FashionIQ
数据、服务器、封存 final 或外部测试服务。64 次是测试账本模拟值，
不是新增真实模型费用。服务器现行 SSH ED25519 指纹仍与已核验值
不同，未登录。fresh 代码审查被 agent thread limit 阻止，只有
`[local-only]` 检查。M1/Pilot/全角色成本/C1–C3/论文结果仍未准入。
