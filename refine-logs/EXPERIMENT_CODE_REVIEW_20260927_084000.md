# BFCL fit 数据与进程应用审查

日期：2026-09-27 08:40:00 UTC；完整trace私有，不提交。
这是experiment-bridge实现/环境闭环，不是科学auto-review-loop一轮。

新上下文直接读取原版binding、code-only stage、fit准备/评分、进程policy、
doc/tests/spec。首审112passed1skip，发现namespace discovery BLOCKING和
timeout日志NON-BLOCKING。显式绑定已授权helper目录，不放宽父目录权限；
超时独占保留原始partial bytes后继续抛错，不返回零分/完成状态。
Linux sentinel现在限制后走同样helper import路径。一次复审113passed1skip，
无未解决问题；同家族/provisional，不称跨家族接受或科学准入。

独立data auditor原样准备512fit输入/annotation、两个主模型各20旧响应，
未修改membership/按分数挑选。原输入与先前成本语料相同。主开发者仅见
schema/count/provenance及错误。五类别store不代表旧响应覆盖五类别：全部
40响应只是simple_python。非fit字节流与IDtoken访问、旧preview披露保留。

首次fresh-doc实际exit1：vendor SDK Path.home() 在干净env与隔离权限下
无法确定HOME。主执行者读原始import traceback；仅把HOME设为已有获权
scratch，没有新passwd/home授权。保留原始失败路径，环境无依赖重建。
该修复经过launcher-env单测与新上下文实际文档验收，不写成额外完整代码
审查。新fresh-doc当前命令一次exit0/10.388639s：官方real-GT40条，独立
worker/controller PID、ABI1，12/20与2/20；无新的inference或final访问。
011/012为source review/复审；013失败与014成功的环境trace分别保留。

静态Ruff/format、mypy61src/strict4helper、configuredBandit src通过。
新增helper保留8LOW（结构化Git subprocess），无中高、不扩大ignore。
本机sdist/Twine通过但wheel遇WinError127失败，准确源码CI仍待推送。
完整IPC/stage/其他任务进程、online/native optimizer成本与M1/Pilot仍未完成。
