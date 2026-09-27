# HotpotQA / IFTrain fit 进程评分实现快照

日期：2026-09-27 09:37:00 UTC。目标 active；科学研究未完成。
机器状态：research/ARIS_TEXT_FIT.json。本快照真实 Linux worker 尚未运行。

已实现固定官方代码/资源 stage、真实 fit preparation、独立受限 worker。
11 个原版 code files、四类 NLTK resources 的 155 个文件本机逐字节对照；
不把源码 checkout/Git/bundled final 或数据加入 runtime。官方导入的空 NLTK
目录在 controller 准备，worker 缺资源立即失败，不自动下载。

首次 auditor 原样准备 exit1/12.3659875秒：错误假设文本分池有 BFCL 的
final_derived。实际生成函数/metadata 只有 fit/search/selection，正式 final
来自独立官方数据，不读取它们来造排除表。失败及初审漏检保留；修复要求
准确三池结构，第二次改变代码后的单次准备 exit0/10.2915773秒。
两任务各512 fit inputs/gold；两个固定主模型各 Hotpot4、IFTrain5 旧响应；
旧请求 messages 完全相同，无成绩选样。9文件/3964676B，其他8 leaf为空。
非fit物理字节/ID tokens 的 auditor 流访问披露保留，不称空气隔离。

新 Linux CPU1/GPU0 Python3.10.15 env27f974aa，venv与原样pip phase exit0，
wall13.956878877012059秒。分散文件与单个压缩包传输仍在各自原始 handle
进行，没有杀进程重启；压缩副本使用新路径，须对实际解出的字节再核对。
此时没有 native import/fresh-doc 成功证据，不公开拟造分数。

同家族新上下文首审未发现问题，但真实准备发现漏检；同 reviewer 仅一次
复审确认修复，无剩余问题，30passed/22.47秒。traces015/016私有，provisional。
本机全部 helper236passed1skip/56.31秒，skip不替代Linux；Ruff/format256、
mypy61src、configured Bandit src通过。helper Bandit9LOW/1MEDIUM/0HIGH保留；
B310是固定官方HTTPS URL提示，不接受operator/response URL，未扩大ignore。
full local suite未跑；本机build未重复旧WinError127失败，当前SHA CI待推送。

上一源码60b9914的CI36307630987已核实15/15成功：Ubuntu3.12 1746passed，
94.03%，CI build/三OS wheel smoke过。它不代表本次新源码CI或研究效果。
BFCL既有真实40fit评分证据与所有历史账本/preview/失败均不改。

本轮 CPU 准备/评分实现未生成任何新calls/GPU/tokens/paid；历史响应成本不重算。
M1仍UNFITTED、Pilot NOT_RUN、矩阵0/144、0/180、0/36。原生engine early reject、
全角色成本、在线随机单元、真实IPC/stage receipts仍是后续工作，不由测试替代。
