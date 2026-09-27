# MIPRO 共享到期评测适配：实际 CPU 验收

记录：2026-09-27T18:29:47Z。目标 active，无 timer；ARIS 默认4轮/科学0不变。
这是工程组件验收，不是科学 native-equivalence、真实模型收益或 Pilot。

固定 DSPy 原 helper 会把 TrialPruned 变成零分；原 objective 的提前退出还会
省略到期 full evaluation。新 `mipro_cadence.py` 从固定方法派生、保留 MIT 声明，
在严格 native 和 gated 路径共同使用：拒绝后先处理到期的实际 survivor，再重抛
原 prune；以实际 objective 次数调度，空池/已全评测组合明确跳过，不制造数值 trial。
原 bootstrap、proposer、TPE、minibatch sampler 和 full-selector 方法仍被调用，
但 objective 是明确的共享适配，不说成未修改的原版行为或相同候选序列。

source 唯一 followup045 用固定原 full-selector/helper 验证了 good/good/bad、
bad/bad/bad/good、good/bad/bad/bad/bad/bad、全拒绝的控制流。两个新 blocker
及报告含 Program 对象的序列化错误均实际修正；最终无阻止限定 CPU 诊断的已证实
blocker。该 review 是 gpt-6-astra/xhigh、same-family/provisional。初次请求在
压缩前未落盘，完整旧 verdict 由同一 reviewer 原样重附保留；不声称初次 trace 完整。

主作者原73744与 fresh-doc 原91776各执行一次安装版四条 compile，均终态 exit0。
doc047 原样执行最新文档，未修改代码/环境；继承模型的 canonical family 未知、
independence unverified，不宣称跨家族接受。其 dispatch 到终态观察266.221s，
tool wait170.0801052s，不是 OS process lifetime。主作者未独立测量整条命令耗时。

| authored 路径 | COMPLETE | PRUNED | winner | fixture task/proposer 调用 |
|---|---:|---:|---:|---:|
| gated full | 8 | 5 | 100 | 351/4 |
| gated minibatch | 12 | 5 | 100 | 351/4 |
| strict native full | 13 | 0 | 100 | 839/4 |
| strict native minibatch | 17 | 0 | 100 | 423/4 |

均从75分 scripted seed选到100分 scripted非seed；PRUNED值均null。
minibatch objective3/6/9/12均实际 full；objective6/native trial7被拒绝仍完成
survivor promotion。gated candidate ledger各280calls，不包含所有角色；表中都是
authored LM调用，不能算真实资源消耗或推算节省。安装原源码 bytes 匹配固定 checkout，
源码 checkout未改，三项 runtime symbols恢复。保留16 proposer字段警告、2 Optuna
实验功能警告及上游吞异常 witness traceback；物理trial标签不是objective次数。

最新 source CI 22214f6/run36337808916实际14/15通过、macOS3.14并发初始化失败。
本地确定性复现相同 preflight 错误；加最小 in-process RLock串行 constructor，
保留外部数据库 header/sidecar拒绝与SQLite writer recheck。76passed1skip，
不声称解决跨进程初始化。fresh journal review因thread limit未创建，trace046
保存完整失败请求/响应，修复评判为 local-only，不自称独立审查通过。
旧 a77b622 CI首轮超时及唯一重试15/15成功均保留，不替代新source证据。

本机完整套件原67909：1924passed6skip/1116.10s/94.03%。收集阶段出现 Windows
WMI fatal0x8007000e日志，随后继续并终态exit0；不称干净环境。运行中仅澄清了
helper docstring，行为代码未再改，随后另17tests2.12s及Ruff通过。
74 component tests55.50s、Ruff343、mypy61、configured src Bandit、isolated
sdist/wheel/Twine均通过。helper Bandit保留1LOW B404：仅构造authored
CompletedProcess，没有subprocess launch，不扩大忽略项。原本地失败全部私有保留。
交付 commit/push后核对新 exact-SHA CI；本记录不预先宣称其通过。

实际物理 DB26a7e04b... bytes多次核对未变；沿用先前 closed-history实测1170calls/
2136046input/195222output/3.8074371029887377allocatedGPUh。historicalunknowncount
原摘要未编码，不报告已证明为0；本轮新realcall/GPU/modeltoken/paid均0，无重置。

接续真实 backend/all-role cost与固定 GEPA原engine；其默认异常处理会终止运行，
不能直接照搬MIPRO TrialPruned。统计/在线随机单元、其余实际vectors/selection、
M1、PilotGO、144pairs180runs36transfers和论文证据仍缺。ICMR规格已登记，
主题匹配未建立，稿件未开始；其他四仓冻结，未完成研究不报告完成。
