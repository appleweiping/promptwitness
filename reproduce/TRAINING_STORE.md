# 三任务完整 training store 准备

这是完整 S1 流水线的数据准备步骤，不是 reference 评分、候选/配置冻结、
M1、Pilot 或科学准入。既有 proposed-v2 不变：各任务 512 fit、256 search、
256 selection。真实 role IPC、stage-end receipts、online semantics 和 native
GEPA/MIPRO 必须继续接入，不能由目录存在或本次准备完成替代。

独立 auditor 使用固定 BFCL 原版 Git objects 和已有 Hotpot/IFTrain training
Parquet。先检查所有 ID/group 跨池隔离，再选 training ID，只 decode 选中正文。
BFCL final-derived 的 ID/group metadata 参与隔离检查，其正文/gold 不读取；
Hotpot dev、IFBench final 的文件和 metadata 均不打开。Parquet 的物理 row-group
字节可能包含非选中行，这不是空气隔离。历史 final preview 披露继续有效。

保留 Hotpot 全部 distractors、IFTrain 全部原约束和原 user/system messages。
annotations 与 inputs 分叶；BFCL 五类使用原生 ground_truth/irrelevance 语义。
重新准备的全部 fit inputs/gold 必须与上一已验收的过滤 fit store 完全相同，
只有其合法既有 fit 响应被复制。不能让 search/selection 响应进入 fit records。

只在不存在的新目录生成输出。所有语义检查通过后才创建 store；I/O 失败可能
留下 partial store，保留它，不覆盖、不清理；最终 receipt 最后写入，不是评分
许可。search/reference、search/parent、final/inputs、final/gold 必须保持空。
本次不复制源码/Git/Parquet/archive 到 runtime，不生成任何模型调用或分数。

## 新上下文原样执行一次

作者先完成代码审查。fresh context 只获本文、既有 CPU/PyArrow 环境 ledger 和
下列调用；原样执行一次，不改源码/安装/修复/重试，不另开输入、gold、响应
或逐 ID 文件。代码内部是已授权的训练数据 auditor，主开发者只接收汇总。

```powershell
Set-Location 'D:/Company/nlp-original-projects/promptwitness'
& 'D:/Company/research-artifacts/promptwitness-delta-20260926/scorer-venv-aris/Scripts/python.exe' -X utf8 -B -m reproduce.prepare_training_store 'D:/Company/research-artifacts/promptwitness-delta-20260926' 'D:/Company/research-artifacts/promptwitness-delta-20260926/gorilla-scorer' 'results/summary/g0/training-pools-proposed-v2.json' 'D:/Company/research-artifacts/promptwitness-delta-20260926/bfcl-fit-store-20260927' 'D:/Company/research-artifacts/promptwitness-delta-20260926/text-fit-store-20260927' 'D:/Company/research-artifacts/promptwitness-delta-20260926/training-store-20260927'
```

记录实际退出码、wall time、receipt 汇总和所有 stderr。仅 actual exit0 且
`PREPARED_NOT_SCIENTIFIC_FREEZE`、三任务实际 counts 与固定 metadata 相符，
才能称本次 preparation 完成。失败不转零分、目录/空输出不等于成功。
不改共享环境、没有 GPU 使用；不读取 final，也不宣称整个 S1 已通过。
