# Selection 阶段真实 CLIP 受限进程资格

记录：2026-09-28 09:10:03 UTC。结论只为
`PASS_AUTHORED_REAL_CLIP_SELECTION_AND_SEARCH_COMPATIBILITY`：这是自制输入的
工程资格，不是 CIRR/FashionIQ 正式成绩、M1/Pilot 或 C1–C3 的效果证据。

WSL2 ext4 私有工作区使用此前取得并核验的第一方 OpenAI CLIP ViT-L/14
权重（SHA-256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`）。
`CUDA_VISIBLE_DEVICES=""`；三张本地生成的红、蓝、灰图、两条自写修改文本
与自写真值。权重以硬链接放进各自的 input-only leaf，未重新下载，也不提交。
融合权重 0.5 只是资格脚本固定值，尚未在训练侧科学冻结。
运行前后核对私有工作副本与待提交工作树的 6 个相关源文件逐字节相同；
checker SHA-256 为 `e33a865d3b45a59531236c7304ae393bc988ee79eae35473a3990cce81ba44bb`，
scorer 源码为 `f9a03747bccab8dfba6a8957c3d46e2802cb42204d50662dd78d3021d0ea9820`。

| 私有运行 | ranker / scorer | 全排名与真值评分 | 内层操作 / encoder forward | 外层操作 | checker wall |
|---|---|---|---:|---:|---:|
| `selection` | `retrieval_selection_ranker` / 独立 selection scorer | 两条直接查询、两条自写目标描述；两次完整人口评分 | 12 / 7 | 6 | 37.657 s |
| 默认 `search` | search 角色 / 独立 search scorer | 两条直接查询、单条描述诊断 | 10 / 6 | 5 | 33.168 s |

两次均 exit 0，全部操作 completed、failed 0、unresolved 0；ranker 与两个
scorer PID 不同。私有 SQLite 核对：selection 内层 4 个 shared 操作
（3 image forwards）和 8 个 selection 操作（4 text forwards）；search 为
4 shared（3 forwards）和 6 search（3 forwards）。自制 CIRR 形状评分先去
参考图，两查询的 `primary_hit` 恰为 1；这个刻意构造的结果**不能**估计
真实检索质量。操作 wall 秒数彼此嵌套，不可相加为 CPU allocation；完整
CPU 分配成本、caption/cache/index/生成器/优化器/selection/final 全角色成本
仍 `UNMEASURED_NOT_ZERO`。

私有收据 SHA-256（文件留在 WSL，不提交）：

| 阶段 | `qualification.json` | 内层 SQLite | 外层 SQLite |
|---|---|---|---|
| selection | `3d890d5f8b8fdc9591ac96646ee1ead976a9bb0bbd45fc9b01b0a5df444c07bc` | `40c87558baa30c7f6921a4aab5cd914ff061b0e453c1de3076e2cc7897367ba4` | `050b5bdec8e9def8b662195907a24d496fccd253216473ee15d8995fdff2d42c` |
| search | `0f1376b4e3a7856b7295e95bfe73996cac6a4e13b18776fcb27cac1ec3dee3fd` | `dbc8aec21e50458f527322a19e4e5696701fd90421a30ef0eba9a91ee24ca4c1` | `20d7711101f60147a9c514f0ec377f58905f84e46431163b1884f2120fa82c46` |

代码审查初次发现 selection 描述评分只提交一条查询，违反 scorer 完整
人口契约；修复为两条描述排名及一次完整评分，替身测试也按真实契约拒绝
不完整人口。私有 scorer 在正常非零退出时保存 stderr。第二次同系列静态
审查未发现 BLOCKING；其后又补上“第二条描述失败时保留第一条成功结果”
回归，**这项后续补丁没有再获 reviewer 复审**。stderr 留存仅覆盖正常
非零退出，不涵盖 `launch_role` 超时或零退出后的身份/JSON 拒绝。

最终源码测试：Linux 实 Landlock 组合 85 passed、1 skipped；Windows
完整原配置 2075 passed、26 skipped、453.43 s、coverage 94.01%。项目级
Ruff check/format、mypy 61 源文件、配置内 Bandit、sdist/wheel build、
Twine 两包和 diff whitespace 检查均通过。精确新 SHA CI 待提交后核验。
未运行真实 Qwen/OLMo 描述生成、官方完整图库和评价程序、正式数据许可/
封存、在线优化、全角色成本或新确认矩阵。ARIS 默认4轮、历史账本、无定时器
及其他四仓冻结不变；SSH 指纹冲突，未连接服务器。
