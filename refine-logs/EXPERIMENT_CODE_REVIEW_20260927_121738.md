# 固定前缀逐单元接入：审查与实际验收

记录：2026-09-27T12:17:38Z。目标 ACTIVE、无定时任务；默认4轮、原144pairs180runs36transfers不改。

Fresh gpt6astra/xhigh 初审131passed30.29s/RuffPASS，最初无blocker，漏检条件head选择。
实际本地 SYNTHETIC 单head 回归在旧代码1failed1passed7.12s；reference0有improvement却
错误取regressionNone。最小修复按固定reference取所需head，2passed7.74s；唯一follow-up
承认遗漏，133passed24.06s/RuffPASS，无剩余blocker。不改变统计、预算或selector。

原uniform源码fresh-doc一次exit0/114.6229545s、78workers264官方手写checks，保留不升级。
最终源码新独占路径fresh-doc一次exit0/107.7728659s/stderr空。医生filter省略ABI值与scalars，
作者单独下载aggregate JSON，核验78distinctPIDs、每个ABI1、actualscores64/4/0、
logicalepisodes128/68/65、真实input/outputtokens0。不是从数组长度推断。
eligible4early补60；rejected仅4分数；failed1attempt/0分数/保留未知token预留、不重放。
原seed/callerchoice保留，final不获授权。decoderexit未单独测量；confinement未重做攻击审计。

独立Python3.10.15/NumPy2.2.6/SciPy1.15.3环境按有序phase构建exit0、hypergeom导入exit0，
最终warm-reuse。源码往返bytes相同、native code/resource副本bytes相同，无共享环境/权限扩大。
最终本地helper+incremental382passed1WindowsLinuxskip41.03s、Ruff283PASS、mypy61srcPASS，
configuredBanditsrc无问题；helpers一个LOW/HIGHconfidenceB404保留、不忽略。
完整本机package/build本轮未跑，历史timeout1800s/wheelWinError127未抹掉。
直接script--help import失败保留，文档module入口成功。完整private trace023–026不提交。

Canned executor/AUTHORED模拟计费与真实账本分离，new real calls/GPU/tokens/paid0，历史不重置。
真实backend/wire、wholeGPUallocation、ONLINE_PINNED、全角色physicalcost、原生GEPA/MIPRO
引擎尚未验收；真实M1UNFITTED/PilotNOT_RUN。工程same-family/provisional不构成科学接受。
当前SHA CI待commit/push。初次报告patch重复同一alias操作被原子拒绝，未改变任何报告或实验；
现按timestamp-first再mechanical-copy固定alias，并比较bytes，保留错误证据。
