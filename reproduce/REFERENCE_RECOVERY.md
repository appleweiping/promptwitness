# 原完成响应的 CPU-only reference 评分恢复

只用于已经完成全部 model generations、但官方 scorer 有明确失败的原reference。
不重新生成，不重放失败或未决 model 请求，不预测/补填分数，不改变原seed/人口/随机单元。
`recover_reference.py` 恢复原controller，比较已冻结data/scorer/tool字节，检查全量原request
及唯一成功generation receipt的ID/replicate/provenance，再用原controller评分路径执行。
已知失败worker logs/成本完整保留；新score attempt追加原journal，未决worker不可重放。
已经发布的reference只复用原receipt，summary使用独占新文件；不授予final或科学准入。

本次根因是新文档把code/resource-only runtime放在旧siblingstage，Landlock正确拒绝清单读取。
不是模型/标签/官方matcher错误。新独占code-recovery从原publicPythonarchive提取，
将原BFCL/text runtime精确复制到本次 `reproduce/` 下并比较全部bytes；旧source/失败stage
不改，不新建readgrant、放宽policy或复制源checkout/data/Git。两环境原spec均warm-reuse。
原固定身份与已完成256条响应保持不变。此CPU路径不导入/加载Torch，不创建GPU allocation。

原实际ledger已闭账：1170calls/2136046input/195222output/3.8074371029887377allocatedGPUh。
本次原model component新增256calls/351583input/1778output/0.1254645915826162GPUh，
包括原失败时的完整cold/idle/scorerwait/childexit，不能删掉或算零。
CPU评分本身新model calls/GPU allocations为零，但不把历史费用归零。
historicalunknowncount未编码，localunknown0不是全历史证明；原preview/access披露不变。

fresh agent只获本文、compute contract、private provider ledger及literal invocation，
逐行原样一次，无安装/修复/重试/其他job清理。可信controller会读取原saved response log，
fresh executor与开发agent只看汇总/指定stderr，不打开rawresponse/gold/per-ID/final。
若失败先保留原exit/log，不能启动新模型或把失败改称方法负结果。原读授权只有本stage。

```bash
cd /media/lenovo/data2/promptwitness-delta-20260926/aris-real-reference-20260927/code-recovery
export PYTHONPATH="$PWD/src:$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
# 按 private provider ledger 的完整 recover_reference invocation 执行一次。
```

待实际originalvector/scorerPID/ABI/exit/aggregate验证后，只可声明本reference组件恢复成功。
source与fresh-doc assurance是same-family/provisional，不是科学或ONLINE_PINNED接受。
其他reference cell、真实selection、GEPA/MIPRO、完整角色forecast、M1/Pilot、144/180/36
仍须各自完成；默认ARIS4轮、无定时器的active goal和跨版本总预算保持不变。
