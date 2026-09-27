# S0 scorer slice 代码审查

完整原始 request/response/meta 位于私有 `.aris/traces/experiment-bridge/2026-09-27_run01/`，
不提交 raw trace。以下是脱敏摘要，全部同家族质量审查为 same-family/provisional。

## 先前 Hotpot/IF slice
先前 reviewer 发现目录 pathspec 可能检查到 `ifbench/data/`，已改成必要 Python
文件并补测试，一次复审确认解决。另一个 fresh-doc 执行者原样命令 exit 0，
13/13 authored fixtures；只审计冻结 fit membership 与既有 archive 的交集。
这段保留历史，不表示本轮重新执行或 BFCL/科学门禁已过。

## 本次 BFCL 第一轮：BLOCKING
新上下文直接阅读实现、文档、计划和原版 AST/source 配置；未读 benchmark/final。
发现 `[{}]` annotation 会在空调用/非法 JSON 时提前转成零分，绕过 native checker。
修复：于格式零分分支前拒绝空答案记录。合法零参数答案 `{"function": {}}` 不拒绝。
增加两个 mock 回归用例和对应 authored-runtime 用例；没有引入 schema 框架。

## 本次一次复审
同一 reviewer 直接读取修改后的文件并实际运行 55 mock tests（14.48 秒），
确认原 blocker 解决，未报告新 blocking/non-blocking。它还用纯 mock 检查合法
零参数委托；首次手写命令行 JSON 传参失败保留，改为 json.dumps 的探针通过。
reviewer 未验证新增依赖或执行 native scorer，不能将它的 verdict 当成运行事实。
Canonical helper 保存完整 trace；004 为预写 request/response，005 为同次调用的
helper 镜像，不是另一轮审查；006 为这唯一复审。

## 环境与原样文档执行
独立 CPU 环境首次 import 失败：Qwen-Agent 工具模块直接导入缺失的 soundfile。
保留旧 spec 快照，按原始 traceback/安装源码证据在第三 pip phase 补齐依赖，
原版 source/matcher 不改。修复后 BFCL_IMPORT_WITNESS exit 0；慢导入的诊断
栈在 Anthropic SDK typing 初始化，不是模型推理或 GPU kernel timeout。

另一个全新上下文只获 BFCL 文档、ledger 和 invocation，原样执行一次，未修复/
重试/读实现或数据。Exit 0；313.959 秒包含轮询。输出 23,802 bytes，3 个模型
配置各 40 预定用例；主执行者随后机械核验所有 expected/actual/error 一致。
99 个分数检查加 21 个预期错误，不是 120 个真实样本或独立性能估计。
007 是该 fresh-doc trace。BFCL real-GT、split、online、native-search、Pilot 门禁未清。
