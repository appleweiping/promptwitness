# BFCL 真实 fit 标注与隔离评分

此闭环验证固定官方 AST matcher 对已有 fit-only 模型响应的真实评分。
不是 M1 拟合、优化效果、Pilot 或完整科学准入。分池仍为
`METADATA_PROPOSAL_NOT_FORMAL_FREEZE`，不覆盖历史失败与 final-card 预览披露。

## 数据与源码边界

官方版本固定为 `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`。
trusted controller 先验证 checkout revision/code diff，再从原版 Git blobs
只提取 `.py`。`stage_bfcl` 使用独占新目录，逐字节对照固定源码；stage
只有代码与文件名 inventory，不包含 Git objects、data、gold、`.env` 或响应。
inventory 不是密码学签名；这里假设合作操作者，不能抵御另一个无隔离本地进程。
传输后的 runtime 也要由 controller 与固定源码逐字节对照，不能以 receipt
文件存在代替验证。worker 不能访问 source checkout，不需要 Git。

data auditor 在读取任何标注/响应前，先读取已固定 proposed-v2 membership。
原版 dataset JSONL 第一字段必须为 ID；先解析该 token，再只 decode fit bodies。
历史 archive 的 ID 字段按其 sorted-key 记录格式识别，必须仅有一个不转义的
literal ID property；非 fit bodies 不 decode。遇到不符合已审计格式的记录、
跨池重复 ID/group、final-derived archive ID、重复或缺失记录时立即报错。
原版 fit question/function 必须与先前成本探针的输入语料完全相同；function
groups 必须与固定 membership 一致。没有按成绩挑选输入或响应。

原版五类 JSONL 的非 fit 字节仍经过 auditor 的流，非 fit ID token 被检查；
这不是物理空气隔离。非 fit input/gold body 未 decode，主开发者仅收到范围、
数量和错误。两个已有主模型的所有 base/task fit 响应均进入复核，不生成新响应。
irrelevance 按原版无可解码调用语义评分；其余四类使用真实 possible_answer。
所有配置继续使用同一 JSON name/arguments interface 适配。

store 的 11 个 leaf 独立创建，本次只填充 fit/inputs、fit/gold、fit/records；
其他 leaves 空置，不假装真实 search/selection/final store 已准备。数据不放入
runtime 或 scratch。`launch_role(fit_learner,fit,...)` 先施加 Landlock，再导入
应用及官方 checker。完成状态/错误不会偷偷转为零分。逐 ID scores 与原始
stderr/stdout 留在私有 scratch，公开结果只包含各模型 counts/correct、角色、
PID 和源码 provenance。参考 `PROCESS_ACCESS.md` 的 ABI1 限制；不是网络隔离。
worker 显式绑定已授予的 helper namespace，不依赖扫描未授权的父目录；
原生 sentinel 也在限制后执行同样的 `reproduce.process_access` 导入。
普通失败和超时均保留已捕获 stdout/stderr，超时不返回完成状态或分数。
SDK import 的 `Path.home()` 显式指向各 worker scratch；不继承真实 HOME，
不授予 `/etc/passwd` 或用户 home。首次原样运行因此失败的原始日志保留。

## 环境与准备

规格 `configs/research/bfcl-linux-env.json`：独立 Python 3.10.15 venv、CPU=1、
GPU=0、分阶段安装 SDK import dependencies，不实例化任何 inference handler。
不修改共享 GPU 环境。ledger 记录每个 pip phase、tier1 witness、fresh-doc 验收。

准备动作由有权限的 data auditor 执行，主开发者不打开真实文件内容：

```powershell
Set-Location 'D:/Company/nlp-original-projects/promptwitness'
& '.venv/Scripts/python.exe' -X utf8 -m reproduce.prepare_bfcl_fit 'D:/Company/research-artifacts/promptwitness-delta-20260926/gorilla-scorer' 'results/summary/g0/training-pools-proposed-v2.json' 'D:/Company/research-artifacts/promptwitness-delta-20260926' 'D:/Company/research-artifacts/promptwitness-delta-20260926/infra1/qwen-infra1.jsonl' 'D:/Company/research-artifacts/promptwitness-delta-20260926/infra1/olmo-infra1.jsonl' 'D:/Company/research-artifacts/promptwitness-delta-20260926/bfcl-fit-store-20260927'
```

新目录必须不存在；失败不得覆盖或清理原始 store。只 transfer fit store，不
transfer source checkout、raw archive、原版未过滤 data 或其他 final 数据。

## fresh-agent 按文档执行

controller 事先完成代码对照、真实 fit 准备与传输后，fresh agent 只获本文、
private provider ledger 和既定调用；只原样执行一次，不自己改代码/安装/修复。
不得打开 gold/input/response/per-ID scores，不能把失败的 output 文件当成功。

在已登记 Linux 主机执行以下两个命令：

```bash
cd /media/lenovo/data2/promptwitness-delta-20260926/aris-bfcl-fit-20260927/code
/media/lenovo/data2/promptwitness-delta-20260926/aris-bfcl-fit-20260927/venv/bin/python -m reproduce.check_bfcl_fit /media/lenovo/data2/promptwitness-delta-20260926/aris-bfcl-fit-20260927/store /media/lenovo/data2/promptwitness-delta-20260926/aris-bfcl-fit-20260927/code/reproduce/bfcl-runtime /media/lenovo/data2/promptwitness-delta-20260926/aris-bfcl-fit-20260927/follow-doc-home-work /media/lenovo/data2/promptwitness-delta-20260926/aris-bfcl-fit-20260927/follow-doc-home-result.json
```

900 秒是环境执行观察 deadline，不是新的实验额度。controller 与 worker PID
必须不同，实际输出 `COMPLETED_FIT_AUDIT_ONLY` 才证明该次运行完成。历史模型
已计费，本次评分新模型调用/GPU allocation 均为零；仍记录 CPU wall time。
这只关闭 BFCL real-fit rescore 小项，完整 S1 IPC/stage receipts、在线随机单元、
GEPA/MIPRO 原生 early rejection 与全角色成本仍须接通后才开展真实 M1/Pilot。
