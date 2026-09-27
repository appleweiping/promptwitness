# 原生 MIPRO 严格边界：实际诊断与未解决的选择问题

记录：2026-09-27T17:13:03Z。目标 active、无 timer，原 ARIS 默认4轮/科学0不变。

固定源码的 helper 会把 `TrialPruned` 捕获成零分。新增 `mipro_search.py`
在一个 owned synchronous compile 内恢复严格异常/实际结果语义，保留原
minibatch sampler，接现有 incremental controller 的冻结计划和实际响应。
通过候选先补齐原 search census，未观察或失败分数不填零。

主执行者在独立 CPU 环境实际执行原 `MIPROv2.compile` 两条路径，原67910
终态exit0。安装的三个选定 DSPy 源码文件与固定 checkout bytes相同。

| authored 路径 | COMPLETE | PRUNED | 进入 native helper 的完整评分次数 |
|---|---:|---:|---:|
| full | 8 | 5 | 8 |
| minibatch | 10 | 5 | 10 |

所有 PRUNED value 为 None；两条路径都从75分 seed 选到完整实际100分非seed。
minibatch 在 trial3/11 实际执行 post-baseline full evaluation；trial7 被拒绝时
跳过了到期 full block。保留原日志的 expected upstream error、字段忽略 warnings
和 Optuna multivariate ExperimentalWarning，不声称零警告。总命令 walltime
未独立测量，不将工具 poll 时长相加冒充实际总运行时间。

这不是研究收益：64单位、12 trials、scripted LM/proposer、固定 fixture seed、
in-process authored exact-match scorer。candidate fixture ledger每条路径280calls；
它不包含 reference/bootstrap/proposer，另报 fixture LM task351/proposer4。
不据此估算全角色成本、方法效果或默认科研轮数。

独立 source review 发现真实的 cadence/selector 缺陷：原 objective 的提前退出
可省略对更好候选的周期完整评测。保留的 authored 源码控制流反例中，seed50，
两次good minibatch100，bad0恰逢full边界：不prune返回good，prune仍返回seed。
这是原方法提取/fake Study 检查，不是安装版 compile。诊断成功也没有解决它。
状态为 `DIAGNOSTIC_GATE_BOUNDARY_OK_CADENCE_UNQUALIFIED`，科学部署仍阻断。

source043及唯一followup044为gpt-6-astra/xhigh same-family/provisional，原全文私有。
followup仅允许该诊断。fresh agent-follows-doc因代理 thread limit 未运行；
没有以作者自己的执行替代独立验收或宣称环境已资格化。

本机68针对测试；全套1917passed/6skipped/695.56s/94.04%src覆盖，Ruff329、
mypy61、configured src Bandit、isolated build/Twine均通过。新增helper Bandit
保留一项LOW B404：仅构造authored CompletedProcess，无subprocess执行，不加ignore。
新提交CI待commit/push后核验；上次bc6fdab的36333077196成功不替代新source证据。

实际物理DB26a7e04b... bytes未变且重算1170calls/2136046input/195222output/
3.8074371029887377allocatedGPUh，historical unknown count未编码保留。新真实call/GPU/
modeltoken/paid均0；未触碰服务器/共享环境/其他任务，其他四仓冻结。

接续优先解决/共享到期full cadence适配及strict native控制，而不是开始Pilot或继续
堆fixture；随后真实backend/GEPA/fullroleforecast、在线统计、M1、Pilot和原确认矩阵。
以前访问预览、失败与历史状态不改写；未完成研究不能冒充负结果或完成。
