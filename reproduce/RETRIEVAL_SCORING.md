# ICMR M0：完整排名和二元门控的 CPU 诊断

本文件只说明 authored CPU 验证。手写向量不是图像/text encoder 输出；未加载
数据、模型或 GPU。没有方法效果、官方程序 parity、环境整体就绪或 Pilot 声明。
研究设计与下一步准入见 `research/ICMR_SCOPE.md`。

## 功能边界

`retrieval_scoring.py` 独立实现完整候选库 cosine 排名、严格单目标评分与聚合。
排名 ID 必须唯一且恰好覆盖 pool，真值/参考/subset 必须存在；失败不填零。
CIRR 删除 reference 后分别求 global/subset rank；FashionIQ 保留 reference。
输出0–1 fraction，不是0–100百分比。FIQ query-micro 与 category-macro 分开。
现有二元证书只作用于 primary query hit（CIRR@5 / FIQ@10），不是所有检索指标。
实际完整模型/encoder/native selector/数据权限接入尚待完成。

此 Python 标量 exact cosine 是诊断，不是可扩展的 ANN 或 SOTA 检索器。
有限数值先按最大绝对值缩放再归一化；空/零/NaN/Inf 向量明确失败。
ties 采用 cosine降序+ID字典序；不能声称与 Torch/FAISS 原 tie 规则一致。
传入普通 Python 数值序列；真实 tensor 转换与预处理精度需另行验证。

## 本机 authored 验证（项目已有环境，不安装、不访问服务器）

工作目录：`D:/Company/nlp-original-projects/promptwitness`。
下列 fresh-doc invocation 输出目录必须事先不存在；不要改目录或覆盖旧结果。

```powershell
& 'D:/Company/nlp-original-projects/promptwitness/.venv/Scripts/python.exe' -m reproduce.check_retrieval_scoring 'D:/Company/nlp-original-projects/promptwitness/.aris/compute/icmr-retrieval-fresh-doc'
```

预期退出0，stdout JSON `status=PASS_AUTHORED_CPU_ONLY`，三case依次
ELIGIBLE / INELIGIBLE / INCONCLUSIVE。eligible 的 survivor 最终有64实际手写向量
排名/评分、无重复查询；failed 无观察分数且消耗1失败attempt。三个 journal
仅逻辑 authored episodes，不写实际cross-version账本。
输出 `qualification.json` 保留实测 CPU wall 和资格边界；不把短prefix说成
模型/GPU节省。fresh-doc executor 仅按命令执行一次、收集aggregate，不修复或重跑。

单元验证：

```powershell
& 'D:/Company/nlp-original-projects/promptwitness/.venv/Scripts/python.exe' -m pytest -o addopts='' --no-cov tests/reproduce/test_retrieval_scoring.py tests/incremental
```

## 来源与未完成门禁

约定核对来自 SEARLE `validate.py` 固定a9c314b（CC BY-NC4.0）。本模块未复制
其代码/权重；数学评分独立实现。尚未在原程序和真实图像数据上执行parity。
训练数据许可/分组，sealed final，视觉encoder/固定captioner原程序，独立统计/
online语义、M1、native GEPA/MIPRO全角色执行与完整forecast，均不能由此命令准入。
