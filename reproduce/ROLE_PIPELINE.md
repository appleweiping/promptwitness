# 真实角色 IPC 与阶段控制器

这是实际应用路径，复用 PromptDocument、frozen features、TransitionPredictor 和固定官方 scorer；
不是独立的假评分 demo。controller 只在完整实际 reference 评分成功后发布 reference。
每个候选先持久化 parent/candidate/configuration/execution/predictor/reference 的 literal freeze，
然后才分发角色消息。scorer 接实际响应并打开其 granted gold；predictor 只有输入、完整
reference、已观察的 parent trace 和固定模型系数，没有当前响应/分数字段。

launch_role 的第一行只有启动 policy。worker 在 Landlock restriction 之后才导入应用并读取
余下的 application JSON。只读 runtime 增加已知 src/promptwitness 包目录，不授予其 repo
parent，也不授予原始 checkout/data/Git/archive。新 package import 绑定发生在 restriction 后。
网络/stat/truncate/其他同用户进程局限仍见 PROCESS_ACCESS.md；不是恶意本地操作者沙箱。

controller 的全分 route 接受完整真实 score vectors，不准 predicted/missing fill。
search 结束记录要求 native survivors 的完整向量；selection 不能通过任意 stage 参数越级，
必须等待 search_ended 且使用先前冻结的 survivors。原生选择规则由 caller 保留，本模块只
记录其实际 chosen prompt；selection_ended 不开放 final。events/request/stdout/stderr 为私有。
失败和超时保留实际日志/CPU wall，不计零；unresolved attempt 不自动重放。若完整 reference
结果/文件已落盘而 receipt 尚未写入，resume 只能使用相同原始记录完成 publication。
返回 freeze 是 detached copy，修改 literal run configuration 会被拒绝。
原始 seed 的 freeze 来自 run_frozen，search 向量复用完整实际 reference；可作为
native incumbent 与新候选一同进入 selection，仍要求其独立实际 selection 向量。
不重新评分 seed search、不伪造候选记录，也不替换原生选择规则。

当前还不是完整 S1/scientific gate：真实训练 store 的 reference/parent 尚未生成，
真实 M1 UNFITTED，online finite-population 与 native GEPA/MIPRO early rejection 尚未接通。
全分 route 是必须保留的实际 native-selector 路径；后续须把已有 AuditPlan/journal 和
实际 inference/cost ledger 接到这条路径，不能用下面 authored 验收代替原 144/180/36 矩阵。

## authored native CPU 验收

checker 仅用固定手写案例，BFCL 复用已有 authored fixtures（含五种 offline 类别），
Hotpot/IFTrain 使用手写正常/错误/空文本。不读取任何真实 corpus、响应 archive 或 final。
为三个任务各启动五个受限独立 worker，实际传 application JSON，调用原版 scorer，
predictor 明确 UNFITTED、返回 missing probabilities；验证真实阶段顺序与重开记录。
同时验证原始 seed 与新候选共同存活，并原样保留 caller 对 seed 的选择。
假 execution tokens 明确标 AUTHORED，不作科研随机单元或配置冻结。模型调用/GPU/token 为零。

controller 已在新的独占 code stage 放入当前 reviewed helpers 和已知 package source；
两个 runtime 只从既有逐字节核对的 code/resource-only stage 复制并比较全部 bytes。
warm-reuse 原 CPU 环境：BFCL@1c3ad127 与 text@27f974aa，不修改 spec/pip/shared CUDA。
fresh context 只获本文、provider ledger 和下列 invocation，逐行原样执行一次，
不读 implementation/data/gold/per-ID records，不安装/修复/重试/覆盖/清理。
若某一步失败，保留原现场，不将其他步骤或汇总当全通过。

```bash
cd /media/lenovo/data2/promptwitness-delta-20260926/aris-pipeline-20260927/code
/media/lenovo/data2/promptwitness-delta-20260926/aris-bfcl-fit-20260927/venv/bin/python -m reproduce.check_role_pipeline /media/lenovo/data2/promptwitness-delta-20260926/aris-pipeline-20260927/follow-doc-bfcl bfcl /media/lenovo/data2/promptwitness-delta-20260926/aris-pipeline-20260927/code/reproduce/bfcl-runtime /media/lenovo/data2/promptwitness-delta-20260926/aris-pipeline-20260927/code/reproduce/text-runtime
/media/lenovo/data2/promptwitness-delta-20260926/aris-text-fit-20260927/venv/bin/python -m reproduce.check_role_pipeline /media/lenovo/data2/promptwitness-delta-20260926/aris-pipeline-20260927/follow-doc-hotpotqa hotpotqa /media/lenovo/data2/promptwitness-delta-20260926/aris-pipeline-20260927/code/reproduce/bfcl-runtime /media/lenovo/data2/promptwitness-delta-20260926/aris-pipeline-20260927/code/reproduce/text-runtime
/media/lenovo/data2/promptwitness-delta-20260926/aris-text-fit-20260927/venv/bin/python -m reproduce.check_role_pipeline /media/lenovo/data2/promptwitness-delta-20260926/aris-pipeline-20260927/follow-doc-iftrain instruction_following /media/lenovo/data2/promptwitness-delta-20260926/aris-pipeline-20260927/code/reproduce/bfcl-runtime /media/lenovo/data2/promptwitness-delta-20260926/aris-pipeline-20260927/code/reproduce/text-runtime
```

记录每个真实 process exit、wall、汇总及 stderr。PASS_AUTHORED_ONLY 只证实际 IPC/stage 的
机械闭环，不能表示真实 reference、M1、online、native optimizer savings、Pilot 或科学准入。
