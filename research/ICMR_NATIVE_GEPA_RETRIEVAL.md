# ICMR 检索原生 GEPA 搜索接入：仅自制资格

记录：2026-09-28 11:48:28 UTC。结论限定为
`PASS_AUTHORED_NATIVE_GEPA_RETRIEVAL_SEARCH`，不是 CIRR/FashionIQ 正式
成绩、真实模型优化、M1/Pilot 或 ICMR 方法效果。固定原 GEPA 源码 commit
`d771eb21b5dd3228bc3f567293d2ccfc423fc900`；检查器核对该 checkout
HEAD，以及已安装原生 API、adapter、engine、reflective proposer、acceptance
与 selection 源文件逐字节一致。

## 接入的实际边界

`RetrievalGEPAGateController` 作为原生 `custom_candidate_proposer` 和
`acceptance_criterion` 的定向适配。proposer 返回新文本**之前**，对确切
parent 完整 search 向量冻结、持久化 candidate-bound plan；原生 child
minibatch 随后运行。acceptance 先检查原生 indices、前后分数及两个
`EvaluationBatch` 的 scores/outputs 全部完整且二元，才允许严格改善判断。
审计返回 INELIGIBLE 时拒绝；INCONCLUSIVE/UNSUPPORTED 与评分失败不补 0；
ELIGIBLE 后才完成 64 条实际 search 分数并将 child 暴露给 GEPA 的
后续完整 selection 评估。`gepa_prompt` 保留每个原生组件名称与原文，
不是普通字符串拼接导致结构丢失。

本路径只支持**新运行**、单 parent、`AllImprovements`、`use_merge=False`。
固定上游不同 seed 的 resume、merge 或自定义 selector 可以绕过该 hook；
控制器本身不强制全局配置。上游也可能捕获并重试 proposal-hook 异常，
即使设置 `raise_on_exception=True` 也不是全局 fail-stop。外部 ranker、
scorer、proposal callback 的来源与身份由调用者负责；本适配不授予
`ONLINE_PINNED`、受限进程隔离或科学准入。

## 自制原生执行与计数

`reproduce/check_retrieval_gepa_search.py` 用固定 GEPA 原始 reflective engine、
Pareto/selection 流程，自写 64 个 search query ID 与独立 16 个 selection
query ID、自制 12 项图库、目标位置规则和项目 `score_ranking` 真正指标。
原生例子只给 ID；reflective 反馈给已生成排名和 hit，不给目标真值。
提案是脚本化 callback，审计 scorer 是**进程内自制替身**，没有真实 LM、
官方图像、受限 scorer worker 或 GPU。正例的按规则给定 64 个 reference
分数不是新执行的 64 次物理评分。

| 场景 | 原生 metric attempts | 审计外层 rank+score attempts | 逻辑 episodes | 原生结果 |
|---|---:|---:|---:|---|
| 自制保留候选 | 48 | 128 | 128（含预置 reference 64） | 完整 search survivor 64，进入 selection |
| 自制退化候选（固定首批8） | 32 | 8 | 另一个独立 journal | minibatch 0→8，审计仅 rank 4 条后 INELIGIBLE；未进入完整 selection |

正例冻结发生在第一条 child 原生评分之前；退化候选没有可查询的完整
survivor 向量。原生 attempts **不含**审计 attempts，二者不可相加为
模型调用或真实总成本。完整检查器 wall 8.378 秒、正例 wall 7.442 秒
仅是该 CPU 自制运行的观测，不是完整角色 CPU allocation。该刻意
构造的胜者与拒绝案例不能估计真实质量或节省。首次复用旧环境失败因
SciPy 缺失；随后在独立私有 Python 3.12.13 / 固定 GEPA 环境运行，
最终 `retrieval-gepa-native-20260928-04/qualification.json` exit 0。

私有最终 qualification SHA-256：
`f95d734fd40120a045053000c1eb6f7a8d8a77610557b6b00f73d6e230bf0818`。
公开 checker 源 SHA-256：
`9af2f71487be3173ead4af0e57e032a604a5bd0f20e2e3cee1c6892a55b2d4ce`；
controller 源 SHA-256：
`25c3ef755127c35d5ba407e49d846a43b1310c981855158aef2e6c8b3c1075dd`。
原始私有运行目录、SQLite、trace 和环境不提交。

## 测试、审查与尚未解决项

新增直接回归 12 passed，包含冻结时序、实际完整 survivor、真实回退
拒绝、原生截短 minibatch 与缺失输出拒绝。Windows 原配置全套
`2105 passed, 28 skipped`，433.28 秒、coverage 94.01%。Ruff check/
format、`mypy src` 61 文件、按 `pyproject.toml` 配置的 Bandit、sdist/
wheel build 与两包 Twine 均 exit 0。一次**未加载项目配置**的 Bandit
扫描因既有 B101 assert 返回 1；重新按 CI 使用配置后通过，不改旧代码。
精确新 SHA CI 待提交推送后核验。

同家族 GPT-6-Astra/xhigh 只读审查初轮发现 1 BLOCKING：原先可以
截短原生 minibatch 或用全 `None` outputs 通过；已修并补两条回归。
复审对当前源码确认该 blocker 关闭、0 新 BLOCKING、1 NON-BLOCKING：
建议将 before 侧、单项缺失、outputs 长度、eval scores 不一致和
不改善但不完整等内存已证伪的边界再固化为仓库回归。复审自行执行了
从原方法提取的 41 个纯内存畸形输入测试；**未**自行重跑完整 pytest
或原生 checker。审查 trace 在忽略的 `.aris/traces/experiment-bridge/
2026-09-28_run01/009-*`、`010-*`，`same-family/provisional` 而非
跨家族科学接受。

完整输出形状不认证真实输出或查询对应关系；selection 端完整性、
真实 Qwen/OLMo、官方 CIRR/FashionIQ 数据许可和封存、真实受限 scorer
同次在线运行、全角色资源预算、M1/Pilot、C1–C3 仍 `NOT_ADMITTED`。
跨版本历史真实模型账本保持 1170 calls、2,136,046 输入 token、
195,222 输出 token、3.8074371029887377 allocated GPU-hours；本次
新增真实模型/GPU/付费 API 为 0，全角色 CPU 成本 `UNMEASURED_NOT_ZERO`。
SSH 端点指纹与上次用户确认值冲突，未连接。ARIS 默认 4 轮、无定时器、
其他四个 NLP 仓库冻结；研究目标继续进行中。
