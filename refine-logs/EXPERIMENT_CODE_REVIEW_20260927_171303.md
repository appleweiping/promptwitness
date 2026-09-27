# MIPRO source review：拒绝边界与真实反例

记录：2026-09-27T17:13:03Z。私有原始prompt/完整response：source043与唯一followup044。
fresh gpt-6-astra/xhigh、same-family/provisional，不宣称跨家族接受。

第一review发现两个blocker：

1. 原objective在TrialPruned提前退出，跳过周期full及study.add_trial，可能改变winner。
   作者没有修复此语义；当前明确限制为CPU诊断，完整科研部署仍被阻断。
2. 原100分baseline/100分候选与泛COMPLETE检查不能证明nonseed survivor或selector推进。
   改为75seed/100survivor，要求nonseed完整vector、实际winner、post-baseline full、
   到期拒绝不执行full的定向证据；运行是否达标由实际diagnostic判定。

followup确认这些修正/import顺序/闭包绑定正确，没有阻止限定CPU诊断的新blocker。
对应evaluator=None strict native control依然未资格化，不能掩盖原cadence缺陷。
完整counterexample命令/output保留私有；它运行选定原源码method/fake Study，
不是安装版native compile。正确性表述不扩大到M1/online/Pilot或完整科学准入。

第一review实跑68authoredtests；followup只读语法与callback检查，未冒充又跑了68项。
主作者随后原67910一次CPU诊断exit0；fresh doc spawn返回thread limit reached未运行。
没有自我签字环境资格、复开审查run重刷verdict或改默认4轮。

Bandit新增helper保留一项LOW B404 import，仅构造authored CompletedProcess；
真实process launcher仍是原controller，fixture mock不执行subprocess，不增加suppressions。
依赖安装为task-owned新CPUenv，原interface环境未改；真实历史DB未变。
