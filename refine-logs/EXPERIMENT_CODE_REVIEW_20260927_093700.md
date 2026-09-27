# 文本任务 fit 链路审查与具体失败

日期：2026-09-27 09:37:00 UTC。experiment-bridge 工程审查，不是科学审查一轮。

新上下文检查 stage/preparation/scoring/policy/doc/spec/tests 和 literal 原版
scorer imports。首审29passed/16.27秒未发现 blocker，但 authored fixture错误地
复制了BFCL的final_derived；实际auditor调用因此exit1。保留这一漏检，不把
后续测试反写为首审通过了真实数据。主执行者只读 metadata 名称/数量和
plan_training_pools 代码，未看真实样例/gold/output，确认正确三池结构。

具体修复：在数据读取之前要求 fit/search/selection，不再访问不存在的
final_derived。官方Hotpot dev/IFBench final 独立，未为排除表读取其内容。
ID-token fit 筛选、完整context/messages/约束及原始评分语义不变；新fixture
与真实结构一致，并覆盖意外池名。唯一复审30passed/22.47秒，无剩余发现，
明确承认首审漏检。traces015/016私有，same-family/provisional，不称跨家族。

auditor第二次修正代码后的原样命令exit0，512fit/gold每family，两模型各4/5
旧响应，messages与旧请求相同，无成绩选样。Linux env实际build exit0；
传输、native import 和 fresh-agent doc 仍待实际完成，当前无worker评分verdict。

本机helper236passed1skip；Ruff/format、mypy src和configuredBandit src通过。
新增helper扫描9LOW/1MEDIUM/0HIGH提示保留，其中固定官方HTTPS urlopen触发
B310泛化提示，不接受用户/响应构造URL，无 file scheme，不抑制告警。
实际资格仍需独立按文档执行一次；不以CI或这些测试替代M1/Pilot/科学准入。
