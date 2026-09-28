# ICMR 检索任务的原生 MIPRO 接口资格（自制数据）

记录：2026-09-28。状态仅为 `PASS_AUTHORED_NATIVE_RETRIEVAL_COMPILE`，**不是**
正式 CIRR/FashionIQ 结果、M1/Pilot、ONLINE_PINNED 或论文方法效果。

## 本轮真正接通的边界

`RetrievalMIPROGateEvaluator` 接受固定版 DSPy 的原始 `Example` 对象和
`task_input -> task_output` 单 predictor；reference 使用已实际评分的完整
search 向量。新 program 先由调用者冻结提示词、plan 与 ranker，然后经过
`RetrievalAuditBridge` 的固定随机前缀和 scorer-only 主指标；INELIGIBLE
以 `TrialPruned` 返回 Optuna，INCONCLUSIVE/执行失败不补 0。只有 ELIGIBLE
候选补齐全体原 search 查询的真实二元分数，才把原生请求的 batch 顺序与
完整原对象身份交还 MIPRO。`Prediction(primary_hit=...)` 是评分载荷，
**不是**生成描述或检索排名输出。适配器不能认证调用方的 reference 来源、
冻结时机、ranker 身份或实际物理成本，因此不自行授予在线资格。

独立于旧三文本任务的检查器 `reproduce/check_retrieval_mipro_search.py` 在
固定原 DSPy commit `da1736e21ffda8cc4b86379d4748b011764d507c` 的
隔离 CPU 环境中，核对 Git HEAD 及三个原生入口源码逐字节相等后，
运行真实原生 bootstrap、proposal、Optuna 和共享 strict cadence。
search `dspy.Example` 只有 input/ID、没有 target 或 `task_output`；
所有 ranking 和目标都是检查器自制，指标使用项目的 `score_ranking`。
scorer 在本资格中被替换为**进程内 authored 函数**，不是 Landlock 受限
worker；这与已单独验证的受限 CLIP/scorer 路径不可合并成一次真实在线实验。
`strict_mipro_search` 的异常和 due-full 调度是两臂共享的显式适配，
不是声称上游原样默认处理。`num_trials=12` 会随原生 full cadence 产生
13/17 个 Optuna trial；没有改 ARIS 默认 `MAX_ROUNDS=4`。

在私有新输出目录分别执行 full 和 minibatch，各得到：

| 路径 | COMPLETE | PRUNED | FAIL | 脚本化 winner | 候选外层 rank+score attempts | 逻辑 episodes |
|---|---:|---:|---:|---|---:|---:|
| full | 8 | 5 | 0 | `authored retain`, 100 | 560 | 344 |
| minibatch | 12 | 5 | 0 | `authored retain`, 100 | 560 | 344 |

每条路径的 64 个 reference 分数事先由自制排名计算，**未进入**上述外层
物理尝试账本；逻辑账本从 64 个 reference episodes 开始。每条路径另有
7 次 task、4 次 proposer 的**脚本化** LM 调用，不是 Qwen/OLMo 调用，
也没有 token/GPU 全角色计量。winner 的 100 分由自制 rank 函数按指令
刻意构造，不能用来推断 prompt 优化有效或节省真实计算。

本地可复跑命令（需已安装、与固定源码一致的 DSPy/Optuna 环境）：

```text
python -m reproduce.check_retrieval_mipro_search <fixed-DSPy-checkout> <new-private-output>
```

私有 `qualification.json` SHA-256：
`b8933de1609ac076c763c5a90098525e0b045ed7933cfcece89df0903e46634b`。
检查器源 SHA-256：
`5ac6dfe11e746f063e17c364c187a6edb64be86911eca034b101bd04adc5a5ef`。
未提交私有 SQLite、运行目录或旧失败/探针数据。固定 DSPy 真实
`Example` 经 `strict_mipro_search` 的额外无模型探针 exit 0。

## 验证与审查边界

新增定向测试 11 passed，覆盖真实指标函数的合法未命中、malformed
ranking 失败、混合 0/1 与乱序身份、补全失败/重放禁止、demo 差异。
Windows 全套 2093 passed、28 skipped，coverage 94.03%；Ruff check/format、
`mypy src` 61 文件、配置内 Bandit、两个发行包 build/Twine 均 exit 0。
全套测试启动后检查器只改了报告字段名及固定源码 HEAD 校验；最终
检查器本身已用上面私有完整原生 compile 再跑 exit 0，精确提交 CI 待核验。

ARIS `experiment-bridge` 的第一次同家族只读审查找出真实 DSPy
`Example` 不是 `Mapping` 的阻断项和两个测试缺口；均已修复。
第二次同家族静态复审对当前适配器、检查器与测试未发现新 BLOCKING，
但未独立重跑检查器或全套，因此仅 provisional。完整 prompt/response
在忽略的 `.aris/traces/experiment-bridge/2026-09-28_run01/007-*`、`008-*`。

仍缺官方图像/标注许可和独立封存、固定 captioner/融合权重、真实
Qwen/OLMo 描述、实际受限 scorer 与 optimizer 同次运行、GEPA 路径、
全角色资源预测、强 CIR/evaluator 对照、M1/Pilot 和 C1–C3 统计结果。
SSH 当前指纹与此前用户核验值冲突，未登录；无新增 GPU、付费 API
或正式 benchmark 访问。目标保持进行中，不能据此写 ICMR 论文结论。
