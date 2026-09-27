# 共享 MIPRO 到期评测组件审查

记录：2026-09-27T18:29:47Z。保留旧 source043/044及先前全部反例。

本组件 fresh reviewer 的第一判定发现两条实际新反例：空池省略 synthetic trial
后末轮不再到期；非空池但组合都已 full 时原 selector 抛 ValueError并覆盖 prune。
改为 objective_calls独立计数及明确未full组合检查。唯一followup045复验固定原
full-selector/helper，确认3/4/6轮及全拒绝行为，拒绝仍PRUNED/None。
followup中又复现报告含DSPy Program导致JSON失败；改为四个简单字段投影，
同一运行中的reviewer重新读取并实际验证JSON往返。最终没有阻止限定CPU诊断
的已证实blocker，17tests实际通过；不更改默认科学4轮/当前0轮。

source为gpt-6-astra/xhigh same-family/provisional。初次完整请求未在压缩前
保存，原完整verdict已由同一reviewer原样重附，明确保留trace缺口而不重建假请求。
唯一followup的完整请求、in-turn通知、响应保存在私有trace045。

主作者73744/fresh-doc91776均实际四路径compile exit0；fresh-doc047继承设置
的canonical family未知/independence unverified，不称跨家族接受。控制编译的
机械通过不授予科学控制、真实LM、官方数据评分、ONLINE_PINNED、M1或Pilot。

最新CI实际暴露constructor预检查并发错误，已确定性复现及最小in-process
mutex修复；外部hot rollback/WAL/header保护未绕过。独立journal review两次
尝试均thread limit，最新完整失败请求/响应trace046；本项只作local-only审查。
不扩大跨进程协调承诺，不修改原测试timeout或忽略foreign-data保护。

最终本机1924passed6skip/1116.10s/94.03%，WindowsWMI异常日志保留；Ruff/mypy/
configured src Bandit/build/Twine通过，helper1LOW B404已解释未隐藏。
具体新证据、失败及未完成研究见research/ARIS_MIPRO_SEARCH.md。
