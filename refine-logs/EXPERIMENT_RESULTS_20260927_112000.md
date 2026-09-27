# 实际角色 IPC/controller 的 Linux 验收

日期：2026-09-27 11:20:00 UTC。目标 ACTIVE，定时任务已删除且无替代，ARIS 默认4轮不变。

此次接通已有 PromptDocument/features/predictor/official scorer，不是另搭占位平台。
launcher 第一行只含 policy；应用消息和已知 package 导入在 Landlock 后。增加 grant
仅已知包目录，无 data/repo parent。scorer 读取正确阶段 gold；predictor 不接当前结果。
controller 持久化 literal freeze，reference 完整实际评分后才发布，候选先冻结后查询，
失败/缺失不记零，search/selection 依实际向量和结束 receipt 推进，final 不开放。

新上下文只凭文档/provider/invocation 各执行一次，三个 SSH/评分 exit均0、stderr空：
BFCL33 authored units ×4实际向量=132检查、SSHwall26.196983s；Hotpot3×4=12，
6.520261s；IFTrain3×4=12，6.353936s。共15实际独立 worker，全部 ABI1、PID不同于
controller；UNFITTED predictor 保留 missing probabilities；native caller 对 seed 的
选择原样保留。BFCL原live session只轮询未重启；decoder exit 未单独观察。无doc
runtime divergence、安装/修复/重试/推理。上述 PASS_AUTHORED_ONLY 不证研究效果。

public2080953包源与 reviewed helpers 往返 bytes相同，native code/resource-only
副本与已验收 stage diff-qr无差异；warm-reuse BFCL1c3ad127/text27f974aa，未改 spec/
pip/sharedCUDA。CPU worker wall合计28.009603969752788s，不称exact core-hours。

审查首次发现原 seed 无法保留的P1；旧代码2回归失败后修复，单次 follow-up无功能
blocker。首次修复断言/checker用了错误receipt key，2个KeyError保留，改为既有
frozen_prompt。Ruff单断言格式 exit1 已mechanical formatter修正。trace020/021/022
私有，same-family/provisional不是科学接受。新helper Bandit一LOW B404 exit1保留。

本地268helperspassed1WindowsnativeSkip18.78s；Ruff/checkformat276、mypy61src、
configuredBanditsrcPASS；本轮full localpackage/build未跑，旧timeout/WinError127不改。
上一2080953 exactCI36312943226 success15/15，1789passed157.44s94.03%；本轮新SHA
CI仍待提交推送核验，不能继承它或用CI替代真实验收。

完整training store三任务512/256/256与58旧fit响应不改，真实reference/parent/selection
和科学配置冻结未生成。online random-unit/nativeearlyreject/all-role cost尚未接通，
M1UNFITTED、PilotNOT_RUN、确认0/1440/1800/36。新calls/GPU/tokens/paid0，历史成本
不归零。下一步已有AuditPlan/journal的实际incremental/native接入，不重复已过fixture。
