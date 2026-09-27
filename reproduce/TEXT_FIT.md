# HotpotQA 与 IFTrain 的真实 fit 隔离评分

本闭环只验证官方 scorer 对两个已有主模型的 fit-only 成本探针响应评分。
不是 M1 拟合、Pilot、优化效果或完整 S1 准入；IFBench 仅用手写机械样例，
不读取 final inputs/gold。固定 proposed-v2 仍未正式冻结。

两项 training proposal 只有 fit/search/selection，不同于 BFCL 的 final_derived。
官方 Hotpot dev 与 IFBench final 数据独立，不为构造排除列表而读取它们的内容
或 metadata。首次准备错误引用 BFCL 字段的失败已保留，修复不改变 membership。

## 源码、资源与数据边界

`text_scorer_stage.py` 的 11 个 literal code paths 来自固定 Git revision，
HotpotQA 单文件与固定官方 HTTPS revision 的原文逐字节核对。仅这 11 个
文件、四类 NLTK resource trees 和 inventory 进入只读 runtime；没有 checkout、
Git objects、bundled IFBench test、标注、原始 archive 或模型响应。
NLTK 必须事先准备并核对资源字节，worker 缺资源即失败，不能偷偷下载。
官方 IFBench 导入会创建 `.nltk_data`，controller 预创建空目录，不授予代码写权。
传输后也需对照原始 code/resource bytes，inventory 不是签名或验证替代品。

独立 data auditor 先验证固定 metadata 的 ID/group 跨池不重叠，然后使用
Parquet 的 fit ID filter 和 explicit columns，只将选中 fit rows 转为 Python。
Hotpot 保留全部 distractor contexts；IFTrain 保留全部 messages/constraints，
不把训练 completion 放进输入、不重写约束。已有 base/task archive 先识别 ID
token 再 decode fit body，两个主模型均纳入，不按成绩挑选。对这些响应的输入
messages 逐项比较旧成本请求。非 fit ID tokens/bytes 被流式检查，Parquet
物理 row groups 可能包含非 fit 字节；这是逻辑选取，不是空气隔离。

store 的 11 个 leaf 只有 fit 三类填充，其余留空。`launch_role` 先施加 Linux
Landlock 再导入 helper/native code；worker 无原始源码/Git/data 目录权限。
真实逐 ID scores 留在独立 scratch，stdout 只有汇总/PID/版本信息。
失败、缺响应和未知约束不能当作零分；普通失败和超时日志均保留。
ABI1、stat/network/truncate 和其他本地进程限制见 `PROCESS_ACCESS.md`。

## 准备：controller 与 data auditor

Linux CPU 环境规格 `configs/research/text-linux-env.json`，Python 3.10.15，
一个 CPU worker，无 GPU、无推理服务、无新增模型调用。私有 ledger 记录
所有 pip phases、退出码与 wall time。不要修改共享 CUDA 或其他任务环境。

controller 只在已有官方源码/资源目录上执行代码 staging：

```powershell
Set-Location 'D:/Company/nlp-original-projects/promptwitness'
& '.venv/Scripts/python.exe' -X utf8 -B -m reproduce.text_scorer_stage 'D:/Company/research-artifacts/promptwitness-delta-20260926' 'D:/Company/research-artifacts/promptwitness-delta-20260926/text-scorer-stage-20260927'
```

data auditor 用已有 PyArrow 环境执行，主开发者只接收汇总，不能打开真实数据：

```powershell
Set-Location 'D:/Company/nlp-original-projects/promptwitness'
& 'D:/Company/research-artifacts/promptwitness-delta-20260926/scorer-venv-aris/Scripts/python.exe' -X utf8 -B -m reproduce.prepare_text_fit 'D:/Company/research-artifacts/promptwitness-delta-20260926' 'results/summary/g0/training-pools-proposed-v2.json' 'D:/Company/research-artifacts/promptwitness-delta-20260926/infra1/qwen-infra1.jsonl' 'D:/Company/research-artifacts/promptwitness-delta-20260926/infra1/olmo-infra1.jsonl' 'D:/Company/research-artifacts/promptwitness-delta-20260926/text-fit-store-20260927'
```

路径须为不存在的新输出；失败不得覆盖或清理。controller 只传输已过滤 fit
store、helper code、code/resource stage，并在传输后逐字节对照 stage；不传输
原始 archive、Parquet、源码 checkout、未过滤数据。敏感内容不得 commit。

## fresh-agent 原样执行一次

controller 已完成上述准备及 Linux build 后，fresh context 只接收本文、private
provider ledger 与调用。只执行一次，不改代码/安装/重试，不打开任何 gold、
inputs、responses 或 per-ID scores。在登记 Linux 主机执行：

Linux 本次采用独立 `text-runtime-packaged` stage：controller 将原版已核对的
code/resource stage 压缩传输，解到新的独占目录，并将实际解出的文件回读后
再次逐字节对照原始源码/资源。原始逐文件传输目录另行保留，不与解包目录
重叠，不以正在写入的半成品作评分输入；不是修改 official scorer 或扩大权限。

```bash
cd /media/lenovo/data2/promptwitness-delta-20260926/aris-text-fit-20260927/code
/media/lenovo/data2/promptwitness-delta-20260926/aris-text-fit-20260927/venv/bin/python -m reproduce.check_text_fit /media/lenovo/data2/promptwitness-delta-20260926/aris-text-fit-20260927/store /media/lenovo/data2/promptwitness-delta-20260926/aris-text-fit-20260927/code/reproduce/text-runtime-packaged /media/lenovo/data2/promptwitness-delta-20260926/aris-text-fit-20260927/follow-doc-work /media/lenovo/data2/promptwitness-delta-20260926/aris-text-fit-20260927/follow-doc-result.json
```

900 秒仅为环境观察 deadline，不是实验额度。实际 SSH/worker 退出码必须为 0，
controller/worker PID 不同、状态为 `COMPLETED_FIT_AUDIT_ONLY` 才计该次闭环完成。
正确率来自少量旧 fit responses，不能用于方法比较；历史推理成本保留，
本次 CPU rescore 新 calls/GPU-hours/tokens/paid API 为零，仍登记 CPU wall time。
search/selection stores、真实 IPC、stage-end/freeze receipts、在线采样、native
GEPA/MIPRO early rejection、全角色成本、M1/Pilot/确认实验继续按主目标推进。
