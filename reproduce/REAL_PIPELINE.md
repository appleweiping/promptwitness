# 真实完整参考向量与分阶段评分接入

`real_pipeline.py` 将已有限制推理进程、连续实际成本账本和阶段控制器连接起来。
完整 reference 在生成前冻结全部 input-only 请求及随机单元映射，全部实际响应成功后
才由独立 worker 对原始 annotation 评分。缺失或失败的响应不能补预测值或填零。
恢复只复用严格相同请求的已完成 receipt，不重放失败或未解决的原请求。

沿用现有 `ExecutionIdentity`：可信 controller 对 fit/search/selection 的 input/gold
文件哈希，并绑定已核验的 scorer source/resources 与 wrappers，不解释 gold 标签。
这不是新安全签名或 air gap；没有打开或哈希 final leaf。模型进程 benchmark 文件读授权
仍为零，只经 pipe 接收完整原始输入。scorer 只解释其授权阶段的真实 annotation。
更改冻结数据、代码、配置或 profile 不能以旧 receipt 恢复。

`selection()` 只能在原控制器结束 search 后，为冻结 survivor 生成完整独立 selection
向量。它不替换原生选择规则。`incremental()` 返回原 driver、同一账本和 `model.task`，
后者使用 driver 已预约的调用而非再次计费；调用者仍须建立实际执行假设及冻结 audit。
这里不提供原生 GEPA/MIPRO engine、不以 replicate 标签授予 `ONLINE_PINNED`。

## 完整 reference 组件

`run_reference.py` 要求原 proposed-v2 的 512 fit / 256 search / 256 selection，
ID 不重叠，使用原 seed，预检全部 search 输入，然后生成完整 256 条 reference。
不进行较小子采样或按结果挑选单位。它没有执行 search 优化、selection 或 final，
不训练 M1、不构成 Pilot，也不能测量方法收益。search 正确数仅为训练侧 seed 参考结果。
官方 Hotpot exact-match、BFCL offline AST/JSON-only 与 IFTrain all-constraint adaptation
保持原定义与披露，原模型 revision/BF16/sdpa/singleton/greedy/context/caps 不变。

每个实际调用及已知 token 都进入原连续 ledger；未知成本保留保守预约。
GPU allocation 从 cold load 到 scorer 等待、idle 和实际原 child 退出全部计费。
摘要携带原 history receipt：usage 中 unknown-cost 计数仅为 continued DB 本地行，
不能将其零值解释成全部历史未知成本为零。全局预算仍为用户明确的跨版本上限，
没有新增较小调用/GPU/token quota。

## 按文档执行的真实环境验收

本次首先执行完整 HotpotQA/Qwen 单元格，非小样本性能筛选。warm-reuse 已登记的
CPU@d41b2c6f 与 Torch@150b5834 环境，不安装、不改共享 CUDA/Python。
独占新 stage 的 public Python source 与完整 private training store 分开传输；
store、响应、per-ID scores、账本、日志和访问 trace 不上传 GitHub。
继续包含全部原失败成本的闭合历史 DB；不能使用较低历史或 authored 测试账本。

fresh agent 只获得本文、compute contract、private provider ledger 和 literal invocation，
逐行原样一次执行，不修复、重试、覆写或清理其他任务。实际调用前验证原 GPU UUID
memory<500MiB；busy 就停止，不改用其他 GPU。任何失败保留原 session 和实际 stderr。
观测超时不等于任务退出，也不允许启动替代 job。原 scorer runtime 已逐字节核验，
本次不下载 scorer/data 或重放旧 fit-wire 验收。

```bash
cd /media/lenovo/data2/promptwitness-delta-20260926/aris-real-reference-20260927/code
export PYTHONPATH="$PWD/src:$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
# 按 private ledger 的完整原样 run_reference invocation 执行一次。
```

private ledger 指定现有端点/私钥、fixed snapshot、实际历史 SHA、完整原 store/runtime、
独占输出和实际成本。源代码审查结论 same-family/provisional，不等于真实文档验收或
科学接受。不得把本地 authored mocks 当作真实模型或评分器通过；实际结果另记版本化报告。

首次完整 Hotpot/Qwen 调用已经实际失败（原 exit1、458.9171906s）：模型成本已闭账，
scorer 读取旧 sibling-runtime 清单时被 Landlock 拒绝，未发布 reference。
此前 authored 环境把 runtime 放在当前 `reproduce/` 下，这次新文档没有保持该布局。
必须将已核验 code/resource-only runtime 原字节复制到当前代码的
`reproduce/{bfcl-runtime,text-runtime}` 并比较全部 bytes；不授予旧目录 ancestor、
不开放更多数据叶、不修改 policy。原失败 source、logs、costs 不覆盖。

`recover_reference.py` 提供显式 CPU-only 恢复：验证原 data/scorer/tool 身份不变，
仅取完整冻结向量中各请求的原已完成 generation receipt，调用既有 controller 的
reference 评分。失败或未解决的 model 请求不能重放或补值；已知失败 scorer 的旧记录
保留，新的评分尝试单独记录。模型不重新加载、调用或分配 GPU；原调用成本继续披露。
这是评分环节的新尝试，不是把原失败运行改写成成功。具体原样验收见 REFERENCE_RECOVERY.md。

全阶段访问与 online/statistical audit、原生 engine early rejection、完整角色成本
forecast、真实 M1/Pilot 和原 144 pairs / 180 runs / 36 transfers 仍须各自实证。
Linux ABI1 的 network/stat/truncate/ioctl 局限及旧 IFBench 开发预览披露始终保留。
默认 ARIS MAX_ROUNDS=4、科学门限、其他四仓冻结和无定时任务的 active goal 不变。
