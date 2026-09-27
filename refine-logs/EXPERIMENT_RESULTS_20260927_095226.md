# HotpotQA / IFTrain 真实 fit worker 记录

日期：2026-09-27 09:52:26 UTC。目标 active，完整研究未完成。
证据：research/ARIS_TEXT_FIT.json。本记录补充093700未运行快照，不改历史。

11个固定原版code files与155个四类NLTK资源文件完成controller原物核对；
实际Linux解包stage回读后再核对original bytes通过。原始逐文件传输也结束，
167文件与已验证packaged副本逐字节一致，均保留，不覆盖。stage无benchmark
数据/Git/archive/gold/响应。CPU1/GPU0 Python3.10.15 env27f974aa实际build
exit0/13.956878877012059秒；原版54 checker classes import exit0/0.4734686401207秒。

真实auditor准备首次exit1/12.3659875秒，错误复制BFCL final_derived字段。
实际文本metadata只有fit/search/selection；对应修复/fixture/唯一复审均留存。
第二次修正后准备exit0/10.2915773秒：两项512 fit inputs/gold，两个固定主模型
各Hotpot4/IFTrain5既有base/task响应，原请求messages相同，没有按成绩选样。
其他8leaves空。Parquet非fit物理字节和JSONL IDtokens流经过auditor但仅fit正文
decode；不是空气隔离，旧final-card preview与bundled-download披露不改。

fresh context按当前TEXT_FIT.md两行Bash命令原样执行一次：实际SSH/final scorer
exit0，SSH wall4.895834秒、工具wall6.7972974秒。ABI1，controller3855532与
worker3855533不同，状态COMPLETED_FIT_AUDIT_ONLY。真实fit18条及13个手写
native机械检查完成；逐ID结果留在私有scratch，主开发者仅下载aggregate。

| 旧fit-only响应 | Qwen | OLMo |
|---|---:|---:|
| Hotpot官方answer EM | 4/4 | 3/4 |
| IFTrain raw-text全约束 | 3/5 | 4/5 |

这些很小的旧成本探针响应只证明native fit-rescore链路，不是全512准确率、
模型优劣比较、M1、Pilot或优化收益；IFBench只手写checks，未读取final。

fresh-doc stderr有base64“输入无效”诊断。与Windows pipe尾部CR一致，但原因是
推断，decoder退出码未独立测量；Bash仍执行完scorer并返回0。保留全部原始
stdout/stderr与一次调用，无修复重试，不写成clean transport或整个pipeline无异常。
Code/doc assurance同家族/provisional，traces015/016/017私有。首审漏检也不删除。

本机helper236passed1skip/56.31秒，Windows skip不替代Linux；Ruff/format、mypy61src
及configuredBandit src过。helper Bandit9LOW/1MEDIUM/0HIGH提示保留，B310针对
固定官方HTTPS URL，未扩大ignore。full local suite未跑，旧WinError127 wheel失败
保留，未重复本机build。0286bdc exactCI36310149312 success15/15，Ubuntu3.12
1776passed120.41秒94.03%，CI build/三OSwheel smoke过；本次doc交付新SHA仍待CI。

新model calls/GPU/tokens/paid均0；历史账本898calls/3.4812718904634763GPUh不清零。
BFCL加文本真实fit链路已完成；完整S1搜索/selection/IPC/stage receipts、online
随机单元、native GEPA/MIPRO early reject、全角色成本未完。M1 UNFITTED、Pilot
NOT_RUN、确认0/144、0/180、0/36，不由本次小项清除scientific admission门禁。
