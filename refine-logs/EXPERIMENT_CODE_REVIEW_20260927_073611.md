# S1 进程访问机制代码审查

日期：2026-09-27 07:36:11 UTC。完整 raw trace 留在私有 `.aris/`，不提交。
本次为 experiment-bridge 新实现审查，不是科学 auto-review-loop 的一轮。
默认科学审查最多 4 轮未修改，仍未开始。

## 当前审查与修复

新上下文直接读 role policy、ctypes Landlock、launcher、独立哨兵矩阵、
文档和测试。首次没有 blocking；发现一个 non-blocking：独立 CLI 没有
主动设置环境哨兵，“未继承”可能是空泛观测。添加 private_env_witness
上下文，进入时设置 authored 哨兵，finally 恢复原值，并测试有/无旧值及异常。
同一 reviewer 只复审一次，24 passed、1 skipped（6.72 秒）；原问题解决，
没有剩余 blocking/non-blocking。全部质量意见为 same-family/provisional，
不是跨家族接受或科学准入。008/009 是实际 reviewer trace。

## 真实环境文档验收

另一个全新上下文仅得到文档、ledger 和传输/原样 invocation，在已登记
服务器执行两行 Bash 命令一次。Exit0、stderr 空、SSH 执行 4.7744489 秒；
metadata-only stat 为普通文件 9099 bytes。未安装、修复、重试或读实现/数据。
010 为该 fresh-doc trace，同家族文档 assurance 仍为 provisional。

主执行者取回 authored JSON 后机械核验：controller 与六个 worker PID
不同、六 PID 互异；19 个允许读取、47 个拒绝读取、66 个 append-open
拒绝、6 个 scratch 写入、6 个环境哨兵未继承，全部与独立预定矩阵相符。
不是 benchmark 样本或评分；ABI1 不约束网络/stat/truncate，不宣称 air gap
或所有写操作不可变，也不能限制同用户未沙箱进程。

## 静态检查与历史

本机新测试 24 passed、1 skipped（7.04 秒）；全部 helpers 172 passed、
1 skipped（28.30 秒）。Windows 跳过 Linux native 测试，不能替代服务器证据。
Ruff/format、strict mypy 两 helper、mypy src 61 files、configured Bandit src 通过。
新 process helper Bandit 保留 2 LOW（B404/B603），0 MEDIUM/HIGH：解释器与
当前源文件的结构化 argv、无 shell、close_fds/干净 env；未扩大忽略范围。

先前 S0 Hotpot/IF/BFCL 审查、空 annotation 修复、一次复审与 CPU 运行时
验收保留在 EXPERIMENT_CODE_REVIEW_20260927_065247.md；未在本轮重跑。
3acaa0b 已推送且 exact-SHA CI 36301672235 全 15 项通过，Ubuntu Python3.12
实查 1687 passed/94.03%。该 CI 是先前 S0 提交，不证明本次未提交代码。

## 未覆盖

真实 store 与各角色实际流水线、授权消息接口、selection/final 阶段 receipts
未完成；没有真实 BFCL fit-GT、在线采样或 native optimizer 实验。
完整 S1/S0/S2 和科学准入依旧未过，不能由 132 次访问检查清门禁。
