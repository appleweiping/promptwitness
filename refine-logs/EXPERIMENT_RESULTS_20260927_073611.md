# 接续初始记录：S1 实际进程访问机制

日期：2026-09-27 07:36:11 UTC。不是完整 bridge、Pilot 或方法性能结果。

## 本轮实际执行

Linux 5.15.0-139-generic、x86_64、已登记 venv Python3.10.15，不安装/修改
共享包。ABI VERSION 实际返回 1；标准库 import witness exit0。规格
configs/research/process-access-env.json，canonical hash 2b4137bc，CPU1/GPU0。
早期 Windows SSH 引号和 CRLF heredoc 探针失败保留；修复传输为编码的
原命令，不修改内核/共享环境。能力探针本身不等于实际权限通过。

新上下文代码审查提出环境探针 non-blocking，最小修复并一次复审。
Fresh-doc 在服务器原样运行一次，exit0、stderr 空、SSH 执行 4.7744489 秒。
取回 9099-byte authored JSON 后验证 6 个独立 worker 和全部读/append-open：
19 allowed reads、47 denied reads、66 denied append opens、6 scratch 写入、
6 环境哨兵拒绝继承。六角色的完整矩阵见 reproduce/PROCESS_ACCESS.md。
所有应用前已执行 no_new_privs + Landlock；没有无约束降级路径。

机械检查：24 本机新测试 passed/1 skip；全部 helper 172 passed/1 skip，
strict mypy 两文件、mypy src 61、Ruff/format、configured Bandit src 通过。
新 helper 2 LOW findings 保留；未放宽忽略。没有本机完整 suite 运行成绩。

先前 S0 提交 3acaa0bd0d231091e1c78cd2a62160120cd89cd0 已实际推送；CI
36301672235 success15/15，Ubuntu Python3.12 日志 1687 passed/157.95 秒、94.03%。
额外本机构建先遇隔离 pip 镜像 access violation；失败不抹掉。切换明确的
official PyPI uv build 后 sdist/wheel 与 Twine 均 exit0。该证据只覆盖那次 S0。
当前 S1 源码的构建/CI 留待本次提交后检验，不引用旧绿灯冒充新验证。

## 访问与结果边界

本轮不读取 benchmark gold、最终内容或模型响应；所有 sentinel 是自编写。
下一步 BFCL 真实 GT 的路径已由原版 Python 常量和 git 文件名 metadata 确认：
bfcl_eval/data/possible_answer。其位置在 scorer package 内，实际 runtime
代码 staging 必须排除 bundled data；不能直接把完整 package 根授权为源码。
没有打开这些 gold 内容，也没有为了通过门禁变更池 membership。

Landlock ABI1 不限制网络/stat/truncate；不声称 air gap、所有写操作不可变
或对未沙箱同用户进程的保护。真实 store 内容、各角色调用/消息、阶段
receipts 和在线执行仍未接好。因此完整 S1、real-GT S0、S2、M1/Pilot/确认
尚未完成。之前 final-card preview/bundled download 披露保留，不能称完全盲态。

## 原目标与消耗

144 pairs、180 runs、36 transfers 完成仍为 0；真实 M1 UNFITTED、Pilot NOT_RUN。
新 model calls、allocated GPU-hours、付费均为零。没有方法收益或有效科学负结果。
跨版本历史消耗仍 898 calls/3.4812718904634763 GPU-hours；CPU 操作不伪称 GPU
成本，也未声称掌握完整精确 CPU core-hours。目标 active，默认科学 4 轮不动，
无定时任务，其他四仓库冻结。
