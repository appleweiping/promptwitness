# 实际 CLIP CPU 编码器运行验收

仅非部署研究软件资格：真实第一方 ViT-L/14 权重、自制三张 pattern images，
不下载/使用 benchmark 或论文/demo 图片，不产生 CIRR/FashionIQ 科学结果。
不调参、不冻结 fusion/captioner，不声称 image-search 产品可安全部署。

固定 source 为 OpenAI CLIP `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`；
其 code 为 MIT，版权与许可保留在私有 stage。模型卡列研究为 intended use；
checkpoint 的 MIT 再发布许可未确立，本项目不再发布 checkpoint 或把它说成 MIT。
下载只能由操作者明确调用原 loader 到专属 cache；验收命令只接已核验本地文件，
不下载、不接受条款、不改共享环境、不替换模型。正式数据/权重准入仍独立待审。

操作者先按 `configs/research/retrieval-clip-cpu-env.json` 构建专属软件层，核对
完整55pins、source文件与tokenizer bytes、官方原loader expected artifact identity；
复用既有Torch50pins/SciPy层，不修改其环境。CPU1/GPU0、thread1、offline，
`PYTHONPATH`包括checkout、指定OpenAI source及两个已核验专属software layers。

从本项目源码 checkout 一次执行，new-output 必须不存在：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m reproduce.check_retrieval_clip_cpu <verified-local-ViT-L-14.pt> new-output
```

每一图像/text 单独编码（float32/eval/no-grad），官方 preprocess 与77token不截断。
图像features验证后保持原值，交给已有ranker做唯一的gallery normalization；text
query仅在encoder做一次unit normalization，ranker不再归一化query。
三图像 RGB/非正方形、RGBA、L，两个自制英文查询；反序复跑核对 bitwise features
及完整3-ID排名。只作当前CPU runtime顺序检查，不证明任意benchmark/多GPU一致。
两畸形text须在模型前拒绝，保留 attempt 但不添 forward 或observed-zero成绩。
预期每次 image6forwards/text4forwards；text6attempts含2rejections。这些是实际
视觉/text encoder calls，不是 LLM generator calls；不能报告所有模型调用为0。
aggregate `qualification.json`含实际计数/有限CPU耗时/完整排名/失败。无语义准确率阈值，
无gold标签替代、科学风险证书或全角色成本准入；CPU下载/安装开销须另记。
