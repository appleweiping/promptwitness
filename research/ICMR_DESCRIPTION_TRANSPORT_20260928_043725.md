# ICMR 目标描述进入受限 CLIP 检索器：自制输入真实资格

记录：2026-09-28 04:37:25 UTC。状态
`PASS_AUTHORED_REAL_CLIP_DESCRIPTION_RESTRICTED_SEARCH_AND_SCORER`。
这是组合图像检索 M1 前置的工程资格，**不是**正式 CIRR/FashionIQ 实验、
真实 Qwen/OLMo 生成、方法收益、Pilot 或论文结果。

## 本轮闭合的具体断点

此前受限 search ranker 只能按 query ID 使用数据集原始 modification text。
现在新 `rank_description(query_id, request_id, target_description)` 把未来
生成模型输出的目标描述作为独立 JSON 数据帧送入同一 Linux Landlock
search-only 子进程。参考图像、类别和完整图库仍取自 input-only 文件；
融合权重沿用本次传入的 0.5，尚未科学冻结。gold 不进入 ranker；
原 direct 输入路径不变。ranker
按不同 request ID 分别持久预留 text encoding 与 full rank 操作；
重复 request ID 在再次编码前拒绝，返回帧绑定 query ID 和 request ID，
身份不一致立即关闭子进程。这是输入与物理操作接线，不认证描述来源，
也不替代生成模型的 token/GPU 账本。

新 checker 用私有 SHA-256 已核对的第一方 OpenAI CLIP ViT-L/14 权重
`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`
和三张本地自制图像，在原生 WSL CPU、`CUDA_VISIBLE_DEVICES=""` 下
执行一次冷建索引、两次原始 modification 查询和一次自写目标描述查询。
后者的文本为 checker 自写的 `a blue square on a white background`，
**不是** Qwen/OLMo 输出。三个完整 3-ID 排名通过同一受限 ranker；
两个独立的受限 scorer 调用读取仅在 gold 侧的自写真值。
本地自制 primary hit 不衡量视觉语义、生成质量或检索性能。

私有 `qualification.json` SHA-256：
`d0f3cb2dc156a1c35ced154f93e3177a32a0678c5da89d49a06ab969a82fda1e`。
最终复跑 ranker PID 924，两个 scorer PID 1327/1328，进程不同。
内部物理操作 10/10 completed、0 failed/unresolved、6 次已知
CLIP encoder forward（3 图、3 文本）；外部操作 5/5 completed，
checker wall 91.77802198 s（与 Windows 全套并行，不能作为独占速度基准）。
内外操作时长为嵌套非可加总和，不是
全进程 CPU allocation。无新生成 LLM 调用/token、GPU allocation 或
付费 API；CPU 和图像预处理费用非零且未完整计量。三个本次源码与
WSL 实际执行副本 SHA-256 分别为
`dc6c65d370688d36fd3dfee6b9be4e223175e8dedfb6ab2ec0b306e83103703e`、
`34c91772d680192c1171531298a4b7986f575ceadfc34f897233ee5765ffb067`、
`8485f7c44213285f05e88ac273690f77ce3a0abb0e604c0d37c6ce91ed23bcad`。
私有原模型权重和自制图像均未提交。

## 审查、失败和未准入项

部署前 fresh-context gpt-6-astra/xhigh 只读审查为同家族 provisional，
无 BLOCKING；两处已过时的 docstring 已修复。审查的首个定向命令继承
全仓 coverage 门槛而 exit 1，无断言失败；正确关闭定向 coverage 后
Windows 12 passed/7 Linux skips。作者实际 WSL 测试首命令因私有 venv
无 pytest-cov 而不识别 `--no-cov`，没有执行测试；去掉该参数后
**18 passed、1 skipped**，含真实 Landlock ranker/scorer、重复请求拒绝、
身份错误以及两种 scorer 失败收据回归。Windows 全仓无覆盖率运行
2046 passed/20 skipped；原配置覆盖率运行同样 2046 passed/20 skipped、
94.03%（超过 90% 门槛），两次均 exit0。全项目 Ruff check/format、
`mypy src`（61 source files）、配置 Bandit、sdist/wheel 构建与新包
Twine check 均 exit0。精确 SHA 远端 CI 另行记录，不以旧 CI 冒充。
审查完整记录保留在忽略的 `.aris/traces/`，
仓库只保存概括，不上传审查 trace。

扩大到整个历史 `src reproduce` 树的 Bandit 扫描 exit1（10 medium、
39 low，含原有未钉 revision 的本地 `from_pretrained` 调用）；这不是
本增量四个 helper 的定向检查。新增四个 helper 的 medium+ 定向扫描
exit0，CI 配置的 `bandit -q -c pyproject.toml -r src` 本地 exit0；
全历史辅助程序告警留待独立核查，不能记为安全全绿。

仍需接入两个实际生成模型，并让其同输入 caption+modification 经过
token/GPU 物理账本、fit/search/selection/final 角色与冻结提示词控制；
之后才可测试在线审计器及同预算原生 GEPA/MIPRO。正式图像/标注许可、
训练分组、sealed final、完整库官方 scorer/tie 对齐、captioner、全角色
预算预测均未通过。`CIRR` 原图 NLVR2 用户条款和 FashionIQ exact CDLA/
image-source 问题保持原样；不从镜像或公开示例绕过。先前公开样例接触
披露也继续有效。`M1/Pilot/C1–C3/scientific_admission=NOT_ADMITTED`，
ARIS 默认 4 轮、历史成本、无定时器和其余四仓冻结不变。
