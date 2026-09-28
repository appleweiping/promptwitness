# Fit 阶段真实 CLIP 受限排名资格

记录：2026-09-28 09:55:11 UTC。结论仅为
`PASS_AUTHORED_REAL_CLIP_FIT_AND_SCORER`：现有 input-only ranker 增加
`fit` 角色，只能读 `fit/inputs`，与独立 `retrieval_fit_scorer` 配合完整人口评分。
`search` 默认行为与 `selection` 保持；`final` ranker 仍不准入。

私有 WSL2 ext4 工作区从当前本地分支复制 9 个改动文件并逐字节比较，然后
复用此前核验的第一方 OpenAI CLIP ViT-L/14 权重（SHA-256
`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`）。
`CUDA_VISIBLE_DEVICES=""`，3 张本地生成图、2 条自写修改和 2 条自写目标描述；
无 CIRR/FashionIQ 正式原图或标注。权重只以硬链接置于私有 `fit/inputs`，
未复制入仓库。该资格脚本的融合权重 0.5 尚未科学冻结。

原命令 `python -m reproduce.check_retrieval_rank_real_clip <owned checkpoint>
<new private output> --stage fit` exit 0；`qualification.json` 状态为
`PASS_AUTHORED_REAL_CLIP_DESCRIPTION_RESTRICTED_FIT_AND_SCORER`。
2 条直接查询与 2 条目标描述都交给完整人口 scorer，ranker PID 与两个
scorer PID 不同。内层收据 12 attempts/12 completed/7 known forwards，
外层 6/6；failed=unresolved=0。checker wall 35.472 s。
自制形状的两个 `primary_hit=1` 由自写目标刻意构造，**不是**真实检索性能。
嵌套操作 wall 不可相加为 CPU allocation；全角色成本仍未测量。

私有输出仅留 WSL，公开哈希供复核：

| 私有文件 | SHA-256 |
|---|---|
| qualification.json | `f77cc37519ea350cd27b537a8d30b7bfc4fa24688126de11720521ea7db333d3` |
| inner ranker-work.sqlite | `f1fb19bd29ce399fbf62ce1e254a6d34013fb55f9d3d6d8478da11fa1f8c3da0` |
| outer-work.sqlite | `4158999084d01133f39a8e65a085dd43462538f0a18656aa8215d600e64afedd` |
| checker source | `62ee8696cf24488b1dbbee7a3e36aef11bd148ec8f9012d03a07ce6e6c6bbc67` |
| scorer source | `f9a03747bccab8dfba6a8957c3d46e2802cb42204d50662dd78d3021d0ea9820` |

Linux 原生 Landlock 定向组 94 passed、1 skipped，涵盖 16 角色×11 叶
读/写 sentinel 和 fit ranker→scorer；新上下文审查建议的“少一条 fit 查询”
失败路径另在受限进程中补回归，最终文件 2 passed/10 deselected，外层账本
记 failed、无补零。Windows 定向最终 76 passed/19 skipped；完整原配置
2082 passed/28 skipped、覆盖率 94.01%，但全套在最后这条 Linux 测试
补丁前收集，不能说该全套执行了补丁。Ruff check/format、mypy 61 个包源码、
配置内 Bandit、sdist/wheel build 和 Twine 两包通过。

新上下文 `gpt-6-astra` 只读审查 0 BLOCKING、1 NON-BLOCKING；指出的
缺查询测试缺口已补，后续测试补丁未再次交给 reviewer 审查，仅由主作者
核对并在 Linux 执行。审查为同模型家族 provisional，原始 trace 私有保存于
`.aris/traces/experiment-bridge/2026-09-28_run01/006-*`。

未连接 SSH 指纹变化的服务器；未调用 Qwen/OLMo、GPU 或付费 API。
正式数据许可、sealed final、完整图库/官方评分、captioner 与融合冻结、
全角色物理成本、native online、M1/Pilot、C1–C3 效果和英文稿均未准入。
本轮不改变 ARIS 默认 4 轮、历史费用或其他四仓冻结状态。精确新 SHA CI
待提交推送后核验。
