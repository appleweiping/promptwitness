# 固定模型请求与连续成本

本入口验证既有 Torch 后端能执行真实完整训练侧请求，并在持久进程中返回实际
token 数；不是 Pilot、模型效果、ONLINE_PINNED 或原生 GEPA/MIPRO 搜索验收。
推理进程不是数据 sandbox；此前 scorer/predictor 的 Landlock 路由独立保留。

## 前置条件

- 原物理 ResourceLedger 已关闭并独立核对原件与副本字节，不能导入 AUTHORED
  fixtures、只导入更早的 770-call 摘要，或将未结束分配计为零。
- 现有任务专属 Python3.10.15 / Torch2.7.1+cu118 / Transformers5.3.0，唯一已授权
  GPU，完整已验证的固定 Qwen/OLMo snapshot；不更改共享环境或终止其他作业。
- 已准备的三个任务 fit/inputs 各512条。HotpotQA/IFTrain 为真实规范化
  `{id,family,messages}` 格式：仅核对后替换第一条 seed，其后完整原始上下文和
  约束不删除。BFCL 保留原始全部 question/function。
- 完整原 `native-context-v1` 请求文件，固定摘要见准备脚本。原 typed proposer
  cost fixtures 包含已披露 fit 示例；不等于原生 bootstrap/proposal/reflection。

新阶段使用用户跨版本总额度，没有新增小阶段配额。真实旧成本、未知失败 token
 预留和原失败历史保留。cold/idle/退出时间在同一 GPU 分配窗口计费。

## 命令接口

从仓库根目录使用模块形式；私有服务器账本保存实际路径、hash 和原样可执行命令。

```text
python -m reproduce.prepare_model_transport FIT_STORE ORIGINAL_NATIVE_REQUESTS PREPARED_JSONL
python -m reproduce.run_model_transport --requests PREPARED_JSONL --request-sha256 VERIFIED_SHA --history VERIFIED_CLOSED_DB --history-sha256 VERIFIED_DB_SHA --ledger CONTINUED_DB --output NEW_PRIVATE_RUN_DIRECTORY --snapshot PINNED_SNAPSHOT --model-python REGISTERED_TORCH_PYTHON --model PINNED_MODEL --run-id UNIQUE_ORIGINAL_RUN_ID
```

预先固定两个模型共10请求：每模型三个 task fit 请求和两个原 typed proposer
fixtures。按固定 input-ID hash 选择，不根据输出或 gold 筛选。10是本次资格工作量，
不是研究总调用上限。两个模型顺序运行，使用同一连续物理 DB 和同一旧历史副本。
必须显式指定已登记的 Torch 环境绝对解释器路径：CPU controller/scorer 环境与
模型环境独立，不能默用 controller 的 Python、回退安装或更改共享环境。
不得更换 run ID 重刷同一失败；保留原进程日志、失败成本和实际退出。

## 与增量评测接入

模型加载后返回实测 backend/template/decoding profile；先冻结进实际 controller
ExecutionIdentity，再产生原 reference 或 candidate。scorer/data/tool environment
身份由科研调用方按真实已审计资源提供，transport-only 值不能直接用于科学准入。

`PersistentModel.metered_task` 对 reference、完整向量、selection 等实际请求先预留
再执行；`metered` 对 proposer/reflection 同样计费。`task` 是既有 IncrementalPipeline
回调，要求此同一账本中存在唯一匹配的预留，避免第二次收费或无账本执行。
单单元、左 padding、固定完整上下文和输出 caps，无 response cache/输出调度。
模型实际返回后，正式 correctness 仍须走官方 scorer，不能把生成文本当分数。

所有完整请求、原回复、profile freeze、失败、原子 ledger 和 GPU 窗口留在私有目录。
无自动重试；未知 tokens 保留 caps，已收到但验证失败的 actual tokens 仍计费。
GPU 只在原 child 经 wait 确认终止后释放；观察失败不会把活分配写成结束。
异常／预算中断立即 terminate 精确的原 child，再 wait/必要时 kill；只有成功退出
才允许先通过 EOF 等待正常清理。实际到终止的全部时间继续计费。
传输层在抛出物理调用异常前即执行中断；不依赖外层 context manager 收到异常，
因为增量 evaluator 会将 executor 异常转为 INCONCLUSIVE。失败 session 不可重用；
再次请求在新增预留或推理前拒绝，清理观察失败保留原开放 allocation 和已收到成本。
不读取 final bodies，不开启科学确认、PR、main merge、outreach 或定时任务。

## 资格边界

作者 CPU fixtures 不是真实模型成绩。实际按文档运行前，该环境/调用资格仍未建立；
每一次实际命令、失败和成本必须单独记录。成功只证明该源代码和环境上的真实
wire/token/原 child 生命周期，不证明固定潜在响应表、全部 native optimizer 路径、
完整角色 forecast、独立统计审核、M1 fit 或 Pilot GO。

2026-09-27 实际原样文档调用已完成：两个模型共10次，新增21,557输入token、
2,266输出token、0.12166419578923104 allocated GPU-hours，两个原child区间关闭。
当前窄验收及局限见 `research/ARIS_MODEL_TRANSPORT.json`，不能把初次env声明中的
NOT_RUN快照当当前调用状态，也不能把本次成功扩展为科研准入或最终cleanpeer审查通过。
