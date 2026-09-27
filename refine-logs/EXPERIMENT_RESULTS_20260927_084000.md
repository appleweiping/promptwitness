# S0 BFCL 真实 fit 与进程评分记录

日期：2026-09-27 08:40:00 UTC。目标 active，完整研究未完成。
机器可核验证据：research/ARIS_BFCL_FIT.json。不是优化效果或 Pilot。

## 实际执行与结果

固定 BFCL 官方 revision，controller 只取 110 个原版 Python code blobs。
本机 stage 和传输回读的每个字节均对照 Git 原物通过；没有向 runtime
复制 `.git`、bundled data、gold、raw archive 或真实响应。

新建独立 Linux Python3.10.15 CPU1/GPU0 环境；规格 hash1c3ad127。
按三阶段安装 exit0/81.5452437500935 秒；原版 AST import exit0/4.654732085997239秒。
Data auditor 原样命令一次 exit0/22.3575514 秒，准备固定 proposed-v2 fit512条，
其中473条原版 possible_answer，39条按类别的 irrelevance absence 记录。
所有 question/function 与旧成本探针语料完全相同，membership/group 未改。
两主模型各20条既有 base/task fit 响应全部纳入，未按成绩挑选。

第一次 fresh-doc exit1/8.9585247 秒：原版 SDK import 的 Path.home() 无法
确定 HOME。保留空 result 和原始 stderr；只把 launcher HOME 指向本来就
获写权限的 worker scratch，不继承真实 user home，不授予 passwd/home。
新上下文按修正后的独占输出路径执行一次，exit0/10.388639 秒；ABI1，
controller3679586 与 worker3679588 不同，未观察到 stderr。

Qwen **12/20**、OLMo **2/20**。这是40条旧成本探针的 simple_python 响应
对真实官方 gold 的复核；其他四类别没有旧真实响应，不能由512条已准备
记录推出全池准确率，也不能把本结果当模型总体比较、M1效果或优化收益。
原版 JSON interface 适配继续对所有控制共享；无 inference handler。
逐 ID scores 留在私有 scratch，不公开 raw data/response。

## 审查、测试与失败

新上下文代码审查发现 namespace 导入 blocker 和 timeout 日志遗漏；修复
后同一 reviewer 一次复审，无剩余问题，113passed1skip/16.47秒。
显式 namespace 绑定不扩目录权限；原生 sentinel 限制后导入 helper 并
验证六 worker/66读取/66append-open。HOME 是后续实际部署错误的具体修复，
并非额外完整审查 verdict。质量意见与 fresh-doc assurance 均 provisional。

全部 helper 在 HOME 修复前206passed1skip/32.15秒；HOME 针对性41passed1skip。
最终 helper rerun 正在运行，尚不写成通过；Windows native skip 不替代 Linux。
Ruff/format、strict mypy4helper、mypy61src、configured Bandit src 通过。
新增 helper Bandit8 LOW（结构化 Git argv/无 shell/可信固定路径），无中高风险；
没有扩大忽略。原 verifier/SCP顺序失败和 Windows find quoting 失败保留。
本机 uv sdist 成功，wheel 因 WinError127 元数据文件复制失败；Twine 仅检查
sdist 通过，不声称 wheel 或整个 build 通过。

先前 dd17ee2 exact-SHA CI36303954121 实查success15/15，Ubuntu3.12
1712passed/94.03%；它不是当前源码 CI。本轮待 commit/push 后核验新 SHA。

## 访问、成本与未完成项

auditor 的流经过非 fit 原始字节/ID tokens，但只 decode fit bodies；不是
物理空气隔离。主开发者未打开输入/gold/响应/per-ID scores，仅诊断 import
traceback。此前 final-card preview/bundled download 披露不删除，不称完全盲态。
其他8个 leaves 空置。当前仅实际 fit 应用链路，不代表 S1角色/IPC/stage结束
receipts 或 online 随机单元已接通。ABI1 网络/stat/truncate 等限制仍存在。

新模型调用、GPU allocation、输入/输出 tokens、付费均为零；旧响应既有成本
不重算、不归零。记录各组件 CPU wall，但不伪称精确全部 CPU core-hours。
跨版本仍898calls/3.4812718904634763GPUh，其他四仓库冻结、无定时任务。
M1 UNFITTED/Pilot NOT_RUN/确认0/144、0/180、0/36；S2 native early rejection
与完整角色成本未完成。继续实际流水线，不由本次单项清全部科学门禁。
