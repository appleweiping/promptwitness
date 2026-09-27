# S0 scorer slice 代码审查

原始完整 request/response/meta 保存在私有 `.aris/traces/experiment-bridge/2026-09-27_run01/`，
不提交 reviewer trace。这里是脱敏摘要，不是跨模型接受或科学结果审查。

## 第一轮：BLOCKING
新上下文 reviewer 指出 `git diff HEAD -- evaluation_lib.py ifbench` 的目录
pathspec 可能检查到 `ifbench/data/`，违反 ACCESS_AUDIT 的 source-only 限制。
这是实际命令选路径缺陷，不是已经窥视 final 内容的证据。
修复：只列出必要 scorer Python 文件；增加 pathspec 和 revision mismatch 两项机械测试。

## 一次修复后复审
同一 reviewer 独立执行 39 项机械测试通过，确认上述 blocker 已消除；
该 slice 未报告其他有证据的 blocking/non-blocking 缺陷。
review route 为 same-family/provisional；未把它标成 cross-family accepted。
reviewer 未执行 native fit-response audit，也未读数据/原始响应/final/eval。

## 文档与环境验收
另一个全新上下文只收到 scorer 文档、CPU ledger 和原样命令。
文档命令首次退出 0，无 repair/retry；48.674 秒是提交到观察完成的 wall time，
含轮询开销，不是纯 CPU time。13/13 官方机械 fixture 匹配；只 audit 冻结 fit
membership 与指定历史 archive 的交集，Hotpot 4 个、IFTrain 5 个。
CPU sanity 成功不能清 BFCL、split-isolation、online、native-search 或 Pilot 门禁。
