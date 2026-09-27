# 实际参考图像＋修改文本的直接融合基线

这是 ICMR v0.1 的 B3 无 LLM 原始图文融合机械基线，不是 PromptWitness 方法收益、
基准成绩、已冻结 captioner/fusion 或科学数据准入。固定第一方 CLIP ViT-L/14 和
`RETRIEVAL_CLIP_CPU.md` 的私有来源、checksum、55pins、CPU1/GPU0、batch1约束。
三个自制图像与两个修改文本均非 benchmark；没有数据标签或语义正确率阈值。

`direct_fusion_query` 对参考图像的原始 gallery 向量归一化一次，修改文本
沿 `ClipCPUEncoder.encode_text` 的一次归一化结果使用，不重复归一化；
按 image weight 加权，再对组合向量归一化一次。authored checker 用 0.5
只是预先写定的机械值，绝不由这两例调优、推广为论文权重或比较收益。
完整图库同一 CPU ranker 排名；此层不排除参考图，CIRR 的排除/子集与
FashionIQ 的保留须由各自 scorer 执行。不存在目标标签输入此组合器。

从源码 checkout 用已核验本地 checkpoint 和不存在的新输出目录：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m reproduce.check_retrieval_composed_cpu <verified-local-ViT-L-14.pt> new-output
```

实际入口只离线加载，不安装/下载/替换。三图 RGB/RGBA/L 各独立编码并反序复跑；
两个修改文本分别与红、蓝参考图组合，文本反序复跑，比较逐字节向量和完整
3-ID排名。期望单次 6 image＋4 text真实CLIP forwards；后续真实数据/全图库/
caption/两生成模型/强对照/训练分组/最终封存与全角色成本仍要单独准入。
