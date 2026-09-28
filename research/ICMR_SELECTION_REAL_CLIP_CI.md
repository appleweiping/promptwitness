# Selection 真实 CLIP 自制资格：精确源码 CI

记录：2026-09-28 09:25:01 UTC。源码与自制资格报告提交
`ec5e64d7f18317c47e91118a9bb9000bd9974444` 已推送至
`research/promptwitness-delta-v1`。[GitHub CI run 36402372673](https://github.com/appleweiping/promptwitness/actions/runs/36402372673)
的 `headSha` 精确匹配，终态 `completed/success`、15/15 job 成功。
Ubuntu Python 3.10 日志为 2098 passed、3 skipped、coverage 94.04%；
Python 3.14 为 2098 passed、3 skipped、coverage 94.05%。

CI 执行公开自制 fixture、Linux Landlock worker、代码质量、构建和
跨平台测试；它**没有**下载/运行私有 ViT-L/14 权重、官方 CIRR/FashionIQ
图像与标注、Qwen/OLMo 或 GPU 服务。真实 CPU 权重资格另以私有 WSL
输出哈希和操作账本记录于 `research/ICMR_SELECTION_REAL_CLIP.md/json`，
不能与 CI 合并推断正式 benchmark 效果。数据许可/封存、完整官方评分
对照、captioner/fusion 科学冻结、全角色成本、在线 native 引擎、M1/
Pilot 和 C1–C3 仍 `NOT_ADMITTED`。服务器 SSH 指纹冲突，未连接；
ARIS 默认 4 轮、历史账本、无定时器和四个冻结仓库不变。
