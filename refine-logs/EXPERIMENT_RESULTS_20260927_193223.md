# ICMR 方向 M0 初始机械结果

记录：2026-09-27 19:18:39 UTC；详见 `research/ARIS_ICMR_RETRIEVAL.md`。

只有 authored CPU M0 完成：全库排名→单目标hit→原paired gate三路径，
eligible4早停但最终64实际score、rejected4实际score、failed1attempt/0score。
主作者32.2339146s/fresh-doc25.0974021s，均一次exit0；逻辑episodes不是模型call。
新33tests及原incremental合计135passed/134.50s，reviewer另测33passed104.64s。
source048/049限定same-family/provisional；doc050身份unknown限制明确保留。

这不是图片/encoder/模型试验、实际节省或检索效果，真实费用增加0。
最新物理历史1170calls/2136046input/195222output/3.8074371029887377GPUh保留。
新方向fixture不初始化或清零真实账本；新矩阵forecast未测量非0。

Ruff355格式/mypy61/src+newhelperBandit通过；fullsuite39297、build/Twine40858
尚无终态。本次不按预期数字宣称通过，WMI异常日志保留。
下一步许可/权重/原图访问及新data隔离→nativeparity→actualcost→M1/Pilot。
研究桥仍running；不移交科学auto-review/paper-writing，不结束activegoal。

19:32:23 UTC原运行终态：full39297 exit0，1957passed6skip/2013.77s/94.03%；
Ruff final366格式化通过。实现27f90c5已push且CI36344286371success，源码审计
e95bd09已push/CIqueued；localbuild40858仍未终态。没有新真实模型/image结果，
不把全部软件测试通过转写为科学效果或Pilot。新科学准入与数据权限仍pending。
