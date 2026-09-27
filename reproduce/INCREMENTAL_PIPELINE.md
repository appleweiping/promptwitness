# 固定前缀逐单元路径

`IncrementalPipeline` 接入已有 `AuditPlan`、`AuditJournal`、certifier、
`complete_survivor_scores` 和 `PipelineController`，不另建统计方法或预算。
run 的 literal configuration 必须保留原 2048 evaluation episodes；完整原 reference
即使物理复用也仍计入逻辑账本。资源 ledger 由实际 backend 提供，保留历史消耗和原总上限，
本模块不增设阶段小额度，不重置真实历史。

候选先冻结 parent/prompt/predictor/configuration/execution/reference，再冻结完整随机单元映射、
采样 permutation/looks、执行假设和 token 预留配置。risk 模式只能使用已冻结 predictor 的
实际概率：reference 答对用 regression，答错用 improvement；缺少所需 head 或 UNFITTED
不可冒充 M1。uniform 模式不使用风险预测。resume 使用原计划，不能重新
抽样或更换 replicate/caps。默认 `make_plan` 的 fresh SystemRandom 与明确标记的 seeded
mechanical fixture 分开；这里不会证明 `ONLINE_PINNED`，该假设必须先由真实 backend 验证。

每个查询先预留逻辑 episode，再记录 generation_request 并预留实际调用/token 资源，然后
调用 injected executor。executor 只收到 frozen prompt/configuration/execution、实际 input 和
replicate，没有 gold 或当前/未查询 score。它必须实际执行该请求，返回实际响应和 token 数。
官方 scorer 通过既有受限 worker 逐单元评分，累积的 parent traces 只含已查询项。
executor callback 的具体 backend、实际 wire request、分配窗口和原生优化器引擎仍须单独
接入和验收；本接口不把未验证 callback 或传入 scope 字符串当作证明。

completed model call 与 scorer success 分开记账：scorer 失败不能把已成功调用的 token 消掉。
已知 token 失败/超额保留实际数；未知成本保留预留额而不是零。CPU worker wall 或单次响应
wall 不用来推算 GPU-hours；真实 backend 应记录完整 cold/idle/exit allocation lifetime。
reference、selection、proposer、reflection 等其他角色的物理调用尚未由这条 search 路径统一接入，
因此不能声称 all-role costs 已完成。

原 designated prefix 的 bounds 决定 early rejection，不预测填分。INELIGIBLE/failed 候选的
未查询项仍未知；ELIGIBLE 候选必须按实际查询补齐全向量后才能进入未改动的 native selector。
原 seed/reference 仍可作为 incumbent，selection 必须等待 search 结束、保留 caller 原选择规则。
未决 generation/worker/logical reservation 均 fail closed，不自动重放；跨账本落盘边界可能留下
未决 episode，需检查原记录，不将其伪称为已恢复或重新生成。final 未授权。

## authored Linux CPU 验收

下面 checker 只验证 Hotpot 官方 scorer 的实际逐单元 IPC，不访问真实 corpus/final。
三个独占案例各有 64 手写 inputs：eligible early prefix 后补齐64实际 scores，rejected
保留部分 scores，failed 保留失败逻辑 episode 和模拟 token reservation、不自动重试。
executor 是 canned answer，不调用模型；`AUTHORED_NOT_REAL_COST.sqlite` 中的数是假定的机械
计费 witness，绝不导入实际历史账本，也不报告成真实 model/token 消耗。
这种 baseline/fixture 和 seeded plan 不是研究实验，更不替换原 144/180/36 矩阵。

新独立环境按 `configs/research/incremental-linux-env.json` 的有序阶段构建，CPU1/GPU0，
不修改旧 qualified env 或共享 Python/CUDA。原 text code/resource-only runtime 复制后比较 bytes；
worker 不获 data/Git/repo ancestor。当前 reviewed helper/source archive 在独占 code stage。
fresh context 仅获得本文、provider ledger 和下列调用，逐行原样执行一次，不修复、重试、安装、
覆盖或读取 per-ID/data/gold。失败保留现场，不以其它成功步骤掩盖。

```bash
cd /media/lenovo/data2/promptwitness-delta-20260926/aris-incremental-20260927/code-final
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/media/lenovo/data2/promptwitness-delta-20260926/aris-incremental-20260927/venv/bin/python -m reproduce.check_incremental_pipeline /media/lenovo/data2/promptwitness-delta-20260926/aris-incremental-20260927/follow-doc-final /media/lenovo/data2/promptwitness-delta-20260926/aris-pipeline-20260927/code/reproduce/bfcl-runtime /media/lenovo/data2/promptwitness-delta-20260926/aris-incremental-20260927/code-final/reproduce/text-runtime
```

记录真实最终 exit、wall、stderr 和 aggregate qualification。`PASS_AUTHORED_ONLY` 不能表明
真实 online/GEPA/MIPRO/M1/Pilot 已通过。旧数据预览、失败、成本和默认 ARIS4轮保持不变。
