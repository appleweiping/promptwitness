# 实验桥实际结果（不是科研效果）

记录：2026-09-27T15:08:27Z。Goal ACTIVE、无定时任务，科学审查0轮、默认4轮不变。

修复后的推理文件边界已通过两模型实际窄验收：fresh context 按四份文档原样一次，
原 session36246/SSHexit0，实测268.0048149s；Qwen与OLMo各3条原固定input-only fit wires。
两次CPU probe各11read/11append denied，ABI1、先限制后应用导入、零benchmark读授权；
实际模型随后加载并生成，不能仅靠CPUprobe声称模型可运行。

Qwen增加2204input/108output/0.05194203197956071 GPUh；
OLMo增加2095input/94output/0.020981736646758176 GPUh。
六次共4299input/202output/0.07292376862631889 GPUh，完整cold/idle/exit计费。
跨版本实际914calls/1784463input/193444output/3.6819725114061215 GPUh，paid0。
新DB36350670...本机字节核对并closed_history重算，reserved/openallocation0；
旧base未知成本计数未编码，历史保守token不归零。释放后GPU466MiB/0%，其他任务原样保留。

首次两CPU通过后Qwen CUDA304失败、OLMo未到；原seededkernel失败，两个系统metadata读修复
仍失败，actualthreadcomm写拒绝。随后只给本child自己的proc-task文件写授权，第二kernel
通过，再执行本次新模型资格。每个旧source/失败/原退出/诊断分配成本完整保留。
ABI1合作文件边界，不封闭network/stat/truncate/ioctl，不是airgap；own-task写不是comm名匹配。
模型接收input-only fit rows；可信controller解析原十行源含此前披露的unused fit-gold fixtures，
不得声称controller没加载这些标签。未打开新goldstore/final，未执行scorer或原生optimizer。

本地443passed3Windows限制skip59.84s；最终Ruff311/mypy61src/configuredBanditsrc通过。
新helperBandit4LOW0MEDIUM0HIGH保留；no-isolation build缺setuptools backend失败保留，
改用本地isolated uv builder实产sdist+wheel，Twine两包通过。全本机package suite及安装wheel
smoke本轮未跑，历史WMI/timeout/WinError127不抹掉。前b67de274 exactCI15/15成功，
1860passed160.32s94.06%；当前提交CI需另查。

完整code/rescue/doc traces031–038私有，same-family/provisional，不是独立统计接受。
真实reference/parent/selection/configfreeze、ONLINE_PINNED、原生GEPA/MIPRO和fullrole forecast
未完成；M1UNFITTED、PilotNOT_RUN、确认0/144pairs0/180runs0/36transfers，方法效果未测。
记录见research/ARIS_MODEL_ACCESS.json；旧ARIS_MODEL_TRANSPORT.json对应旧部署，不改写。
