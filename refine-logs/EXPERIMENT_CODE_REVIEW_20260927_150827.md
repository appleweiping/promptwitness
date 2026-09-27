# 推理文件边界：源审查、实际失败与新文档窄验收

记录：2026-09-27T15:08:27Z。Goal ACTIVE，无timer；ARIS默认4轮/科学0轮不变。

initial boundary source CODE_REVIEW031/032通过，之后original fresh-doc033实际Qwen加载失败304，
两次CPUprobe通过但OLMo未开始；原退出1不能被后续SSH读取退出0取代。原账本及source保留。

日志先读，再运行限定CUDA seededkernel诊断；rescue034指出只等tracerleader会漏ownedgroup，
唯一followup035确认修复：保留原exit、只信号本次newgroup、等待整组终态后关闭计费，未终态
保留allocation。authored检查最初WindowsSIGKILLfixture错误保留，修正fixture后4case通过。
初诊断失败且未推理、未读data；native分配仍实际计费，不把观察等待记成0成本。

review036审查精确只读cpuinfo/mmap两文件；实际第一次metadata repair两读成功仍CUDA304，
trace暴露ownnewthreadcomm O_WRONLY拒绝。唯一followup037审查第二处具体own-task file write：
只在推理child单线程自身解析procself/task，WRITE_FILE和ABI支持时TRUNCATE，无create/remove/
reparent/execute；其他角色默认空，不扩到整个proc或其他PID，data/scratch overlap failclosed。
不是仅按comm名过滤的授权；ABI1网络/stat/truncate/ioctl和合作操作者限制明确披露。

实际第二seededkernel finite8x8/RTXA6000，原exit0/wholegroupterminal/完整allocationclosed；
NVIDIA对其他GPU/非必要proc/devchar路径尝试仍denied，没有增加授权，不据kernel自认证模型。
fresh-doc038仅四文档，literal一次，两CPU+Qwen/OLMo共6实际wires，268.0048149s/原exit0。
合并stdout/stderr没有warning/traceback观察，但未分别instrument stderr；decoderexit未另测量。
这是当次实际机械验收，不是源码review新轮次/科学审查/全阶段接受。

当前helpers+incremental443passed3skip59.84s，最终Ruff311、mypy61src、configuredBanditsrc通过。
新helperBandit4LOW保留（固定absolute解释器、shellfalse的B404/B603），无新增ignore；
no-isolation BackendUnavailable失败与旧WMI/timeout/WinError127保留；isolated build与Twine通过。
全本机package/安装wheel smoke本轮未跑；前b67de274CI15/15不能冒充当前未提交SHA。
完整prompts/responses/native aggregate保持private，未上传trace、DB、rawresponses或用户端点。

当前仅模型进程访问/真实wire/cost资格：M1UNFITTED/PilotNOT_RUN/ONLINE_PINNED未验证，
真实reference与selection/fullnativeengine/fullroleforecast/独立统计准入缺失。不能称桥完成。
