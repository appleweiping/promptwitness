# 接续初始记录：BFCL 原版 runtime slice

日期：2026-09-27 06:52:47 UTC。不是完整 experiment-bridge、Pilot 或方法效果。

## 实际执行

- 原版 BFCL `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` AST matcher，
  `Language.PYTHON`、冻结五类别、明确 JSON-only/exact-name 适配；未改原版 matcher。
- 三个固定任务模型只使用 scorer-only 元数据，不实例化 SDK handler。
  fresh-doc 原样命令 exit 0，313.959 秒包含轮询。40 用例 × 3 配置全部匹配：
  每配置 33 个观测分数和 7 个预期错误，共 99+21；不称为真实数据/model accuracy。
- 初次环境 import 的 `ModuleNotFoundError: soundfile` 保留；旧 spec hash
  f55fe7f9，新 spec 3b2487d1 在第三 pip phase 补齐该实际依赖。未修改共享环境。
  修复后 import/version witness exit 0；SDK 初始化慢，诊断栈保留，不抹掉失败。
- 同家族新上下文代码审查发现空 annotation 的格式零分捷径；最小修复+两回归，
  一次复审通过其限定范围。55 targeted tests，10.83 秒；全部 helpers 148，44.84 秒。
- Ruff/format、strict mypy 三个 adapter/binding 文件、mypy src 61、configured
  Bandit src 通过。新 BFCL helper 的 Bandit 保留 5 LOW，0 MEDIUM/HIGH；
  B404/B603/B607 为结构化 git argv 与 PATH，未新增 ignore，也不代替历史 helper findings。
- 本轮完整 suite 未在本地跑；sdist/wheel/Twine 在本快照时仍运行，未写成通过。
  推送后需检查 exact-SHA CI。历史 622668b/run 36298664057 成功 15/15，
  Ubuntu Python3.12 日志 1671 passed/94.03%；不冒充此次改动的 CI。

## 未完成项与资源

本次未读任何 BFCL dataset/gold/final/历史模型响应；所有 inputs/answers 是 authored
fixtures。BFCL real-GT end-to-end 仍 pending，all-scorers PARTIAL_NOT_PASSED；
S1/S2 未实施，真实 M1 UNFITTED，Pilot NOT_RUN，144/180/36 完成全为 0。
没有建立方法收益或科学负结果；其他四仓库不动，协议门限与矩阵不改。

新模型调用=0、allocated GPU-hours=0、付费 API/cloud=0。CPU setup/扫描/构建
不是 GPU 成本，未宣称精确完整 CPU 总额；跨版本历史账本保持原值。
Active goal 继续，默认科学 auto-review-loop 最多 4 轮不动，不重建定时任务。
下一步为实际进程访问隔离、BFCL fit-only 真实 GT、在线执行与原生 optimizer 集成。
