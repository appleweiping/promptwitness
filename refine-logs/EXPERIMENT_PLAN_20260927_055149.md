# PromptWitness-Delta 接续实施与实验计划

从既有协议整理，未添加新实验矩阵，也未通过 Pilot/确认资源门禁。
研究合同见 `idea-stage/docs/research_contract.md`；最新授权见 `research/ARIS_GOAL.md`。

## Milestones

| Milestone | 必做工作 | 准入/完成证据 | 当前状态 |
|---|---|---|---|
| S0 | 官方评分器 CPU sanity；边界机械测试；仅已有 fit 侧输出 rescore | 原生源码、环境实际执行、输出原始记录、新上下文审查 | PARTIAL_VERIFIED；39 tests、13/13 native fixtures；BFCL 未 qualified |
| S1 | 进程级 fit/search/selection/final 控制；在线执行随机单元 | 对实际进程的允许/拒绝读路径证据，未窥视 final | NOT_IMPLEMENTED |
| S2 | GEPA/MIPRO 原生搜索与 early rejection、全角色计费 | 实际 selectors 的完整 survivor vectors、异常语义和成本日志 | NOT_IMPLEMENTED（接口 smoke 已有） |
| G2 | 真实 M1 和至少两任务训练/验证侧 Pilot | S0/S1/S2、统计/执行核验及自身实测成本准入 | NOT_RUN |
| G3 | 144 pairs、180 runs、36 transfers | 真实 Pilot GO、完整角色成本及 20% 预测余量、科学 freeze | NOT_ADMITTED |
| G4 | 统计结果、局限、复现、默认 ARIS 审查、研究叙述 | 原始证据、审查记忆、exact-SHA CI | NOT_READY |

S0 的新 fixture 与 fit-response audit 属于机械集成，不计为 Pilot，也不作为 M1
已有效、原生 optimizer 已复现或方法已节省资源的证据。
初始 BFCL module import 失败于缺失 vendor SDK；未把它替换成简化参数匹配器。
该缺口保留为未完成的官方运行时资格，而不是一次零分评测。

## 固定实验参数和完整范围

沿用 `scope.lock.json` 的三个任务、两个主模型、种子 11/23/37/53/71、六个
配置、32 候选/128 proposer/2048 episodes 约束及 512/1024/2048 累计检查点。
保留六个机制基线及所有消融，不新增弱基线替代强控制。
原始 readiness targets 全部保留，参见协议与 scope lock；尚未达成任何效果门限。
总资源上限和累计消耗不归零；全角色 GPU/token 预测未知，不能填写零或先猜收益。

## 执行顺序

先按 ARIS 做新上下文代码审查，再运行最低成本的 S0 official CPU sanity，
保留 mismatch/exception 并修复后复审（按 skill 默认）。
已完成该 slice 的一次修复复审及 fresh-agent-follow-doc CPU witness；
当前仍须补 BFCL 资格，不能用这次部分结果清所有 scorer gate。
没有 S1/S2 和实际科学准入，不启动真正 Pilot 或确认矩阵。
S0 不需要 GPU；未来模型实验只用当时已分配可用的单张 GPU，完整环境须另做
kernel witness 与 fresh-agent-follow-doc，CPU scorer 通过不能证明 GPU 环境 ready。
达到 G3 时才根据真正 job manifest 使用 `experiment-queue`，不提前启动空调度器。
