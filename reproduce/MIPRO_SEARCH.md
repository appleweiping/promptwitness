# MIPRO 原生搜索接入：严格边界与机械验证

此实现是 experiment-bridge 的一个工程组件，不是完整研究准入、真实模型优化
收益、ONLINE_PINNED 或 Pilot。其他四仓库冻结，默认 ARIS 轮数未改变。
目前仅允许 authored CPU 诊断；完整原生搜索科学部署仍被 cadence/selector 问题阻断。

## 固定源码中的实际问题

固定 DSPy `da1736e21ffda8cc4b86379d4748b011764d507c` 的
`dspy/teleprompt/utils.py::eval_candidate_program` 捕获所有 `Exception`，
返回 `Prediction(score=0.0, results=[])`。Optuna 的 `TrialPruned` 也是
`Exception`：直接从评分器抛出它，原 helper 会把拒绝变成完成的零分 trial。

`mipro_search.strict_mipro_search` 仅在一个同步、单进程 owned compile 中
替换 MIPRO 模块使用的 helper 和 Evaluate；结束或异常时恢复原符号。
原 bootstrap、proposer、TPE、trial 数、minibatch RNG、full evaluation
与 selector 的源码不替换；采样仍调用原版 `utils.create_minibatch`。
失败直接传播，INELIGIBLE 以 `TrialPruned` 到达原 Optuna study；
INCONCLUSIVE/UNSUPPORTED 中止执行，不伪造零分或 partial best 结果。

这是明确的 adapted boundary：相应 native 控制也必须使用同一严格异常边界，
`max_errors=1` 和 `failure_score=NaN`。不能将它说成完全未适配的原版行为。
不修改固定上游 checkout。非同步并行 compile 不在这个实现的支持范围。

重要的未解决交互：`TrialPruned` 在原 objective 中提前退出，会跳过该次
原计划的周期 full-evaluation block 及其中的 study trial 登记。源码未替换
不代表可达执行顺序未改变；若最后一个相应边界被拒绝，不能假定完整候选已被
原周期 selector 考虑。这轮机械验收会保留此事实。科学控制组与完整原生搜索
等价性仍未获准，后续必须明确解决/共享相应 cadence 适配，不能借通过小 fixture
自动放行真实实验。当前实现只验证严格传播与 complete survivor 接口。

## 与现有实际 controller 的连接

`MIPROGateEvaluator(driver, execute, resolve)` 复用 `IncrementalPipeline`、
`PipelineController`、逻辑 journal 和实际 resource ledger：

- `resolve(program)` 在首个当前候选结果之前忠实保存 instruction 与全部 demos，
  注册/复用原 PromptDocument，并冻结随机单元、层和计划。
- `mipro_prompt` 使用既有单一 `task_input -> task_output` 文本接口，
  不等同于 DSPy 默认 ChatAdapter。对应控制与 actual model executor 必须同样渲染。
- seed 复用原完整 reference；拒绝候选不进入 selector 的完整分数集合。
- survivor 必须先补齐原 search census 的实际响应/官方评分，再给原 helper 返回
  它请求的 batch。复用已经观察的输出，不生成预测分数，不把缺失或失败填零。
- 原生 `total_eval_calls` 是 nominal batch 请求计数，不再等于实际 prefix/census
  调用数；成本以 journal 和 actual physical ledger 为准，两者不能混用。

此处未授予在线确定性假设；真实随机执行须先完成独立统计与执行审核。
真实 LM bootstrap/proposal 路由、fit/demo 边界、GEPA、全角色实测 forecast、
真实 M1、Pilot GO 和原确认矩阵仍为后续工作。

## 原版完整 compile 的 CPU 机械验收

在已安装固定 DSPy 的本地隔离环境，从仓库根目录原样执行：

```powershell
& 'D:/Company/research-artifacts/promptwitness-delta-20260926/mipro-search-venv-aris/Scripts/python.exe' -m reproduce.check_mipro_search 'D:/Company/research-artifacts/promptwitness-delta-20260926/dspy' 'D:/Company/nlp-original-projects/promptwitness/.aris/compute/mipro-search-native-doc'
```

输出目录必须原先不存在；不能覆盖失败记录，也不能重新运行到同一个目录。
`qualification.json` 保存可解析汇总，子目录保留原 controller 事件与 fixture
ledger。命令先比较安装的 MIPRO、utils、Evaluate 源码与固定 checkout 的原 bytes，
随后执行 full 与 minibatch 两条原 `MIPROv2.compile` 路径。

它明确使用 authored 64-unit 数据、脚本化 LM/proposer、fixture seed、12 trials，
原 bootstrap 默认 demo 上限保留。原 Optuna study 被观察而不是替换；
验收要求同时有 COMPLETE/PRUNED、pruned value 为 None、每个进入 selector 的
实际 survivor vector 完整，原生 selector 从75分 seed 选择100分非seed；
minibatch 路径实际经历 post-baseline full evaluation，并记录周期边界拒绝跳过
full evaluation 的既有交互。退出后原符号恢复。
汇总状态是 `DIAGNOSTIC_GATE_BOUNDARY_OK_CADENCE_UNQUALIFIED`，不是全搜索
资格通过；对应 `evaluator=None` strict native 控制 compile 也仍未资格化。

scorer 使用 authored in-process transport 与 authored exact match，不是 Linux
进程隔离或官方真实数据评分证明。candidate fixture ledger 不含 reference、
bootstrap/proposer；脚本另报所有 fixture LM 调用，不能当完整科学成本计量。
新增真实模型调用和 GPU 分配均为零。不能从这个小 fixture 的轮数、耗时或
prune 比例推算研究默认设置、真实节省或准入。执行异常须保留并按原 traceback
诊断；按 ARIS fresh CODE_REVIEW 后再执行，fresh agent 原样按文档验收。
