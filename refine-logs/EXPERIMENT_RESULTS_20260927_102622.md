# 三任务完整 training-store 实际准备

日期：2026-09-27 10:26:22 UTC。目标 active，完整研究未完成。
证据：research/ARIS_TRAINING_STORE.json；上一窄 fit worker 成绩不变。

新上下文审查实际核对 proposed-v2：三个任务各512fit/256search/256selection，
BFCL 另有214final-derived metadata。其五类均保留；Hotpot全部distractors、
IFTrain全部约束和原 user/system messages 不改。先做跨池 ID/group 校验，
再从固定 Git objects/已有 training Parquet 读取选中 training rows。

fresh agent 按 TRAINING_STORE.md 两行 PowerShell 原样一次：实际exit0，
toolwall7.93127秒，未返回stderr，无doc/runtime divergence。三任务总3072
inputs及3072真实annotations分叶；全部fit inputs/gold与旧已验收store相同。
仅旧fit58响应复制，无按正确率/输出筛选，无search/selection响应进入fit。
main只读取aggregate receipt及文件metadata；25files9241430B。
search/reference、search/parent、final/inputs、final/gold实际entry counts均0。

BFCL final-derived只读ID/group metadata，其正文/gold未decode；不读Hotpotdev/
IFBenchfinal或其metadata。Parquet非选中物理row-group字节可能经过auditor，
不是空气隔离；既有final preview/bundled下载披露完整保留。

此次 PREPARED_NOT_SCIENTIFIC_FREEZE 只证明实际准备。未生成reference、parent、
候选结果、selection分数或任何freeze；尚无application IPC/controller。
M1 UNFITTED、Pilot NOT_RUN、确认0/1440/1800/36，科学准入未通过。
下一步真正接入角色消息和依实际完整reference/候选/配置/阶段结束receipt推进
的controller，不能用任意stage字符串或目录存在作为selection/final许可。

复用CPU env c90d1dd7：Python3.12.13/PyArrow25.0.1，未build/install，GPU0。
单Python worker；CPU线程数量未独立认证，wall不是exact core-hours。
源码审查和doc验收同家族/provisional，018/019trace私有。审查首次局部pytest
继承全包覆盖率门限退出1，13测试通过但不能写whole-command success；随后
明确--no-cov局部验证退出0。主作者初次F401 lint失败修复，保留这些事实。

本机13targetedpassed2.36秒；wholehelpers249passed1Windowsnative skip16.70秒；
Ruff/checkformat267files、mypy61src、configuredBandit src及新helperBandit0issues过。
本轮未跑full local package/build，旧WinError127及历史全包timeout不改。
上一670891e exactCI36311348600 success15/15、1776passed94.03%，
本轮新源码CI仍须提交推送后核验；不以它替代实际准备或科研效果。
新model/GPU/tokens/paid均0；历史898calls/3.4812718904634763GPUh不归零。
