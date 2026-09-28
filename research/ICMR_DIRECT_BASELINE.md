# ICMR 输入侧 CLIP 直融检索闭环，2026-09-28 01:27:19 UTC

状态：`AUTHORED_SEARCH_WIRING_NOT_M1_ADMITTED`。这是 B3 无 LLM 直融基线的工程接线，
不是 CIRR/FashionIQ 的正式结果，也没有修改 ARIS 默认四轮、历史成本或科学门禁。

`reproduce/retrieval_direct_clip_ranker.py` 读取与受限 scorer 对齐的 input-only
query JSONL 和有序完整图库 JSON；拒绝 target/subset 等额外字段、重复 query/图库 ID、
缺失类别及类别外参考图。调用者须明确传入每张图库图像的路径、已核验 checkpoint、
训练侧选定的融合权重及持久操作账本。一个共享图像索引按图像 ID 只编码一次，
FashionIQ 的 dress/shirt/toptee 仍各自按原图库顺序排名。每查询把参考图原始向量
与修改文本向量送入原 `rank_direct_composed`，返回完整图库；CIRR 参考图排除只由
原 scorer 做。加载、图像打开/编码、文本编码和查询排名均有独立预留/结算收据；
失败不补零、不静默重放。`RetrievalAuditBridge` 再负责审计随机前缀、restricted
scorer 与 survivor 真实向量补全。新类仅支持 search，不冒充 fit/selection 通用运行器。

回归使用操作者自写元数据、空文件路径及假图像解码器/编码器，检验 input-only schema、全库/类别顺序、
重复调用拒绝、图像打开失败的费用收据；Linux-only 集成回归将同一 ranker 的排名
送进真实 Landlock scorer 和 AuditJournal/gate。假编码器不能证明 CLIP 数值、合法
原图来源或强检索 baseline。旧真实 CLIP 自制图资格与这次新运行器是不同源码/证据，
不能合并成新运行器已执行真实权重的声称。

本机新文件定向 `5 passed, 1 skipped`，跳过 Linux Landlock；`tests/reproduce`
全组 exit 0（计数以精确 CI 为准）；Ruff 和格式检查通过。全仓套件此记录时仍在运行，
不能提前记为通过。fresh-context gpt-6-astra/xhigh 审查为同家族 provisional，
未发现 BLOCKING；指出 FashionIQ 三类别接线覆盖不足后已补回归。审查者本人
修改前六文件定向为 `33 passed, 5 skipped`，不能代替补测试后的 CI。

仍缺：合法 CIRR/FashionIQ 原图和标签、训练分组/validation 封存、权重与融合权重
科学冻结、captioner 和原两生成模型链、输入/模型进程级权限、真实 CLIP 新路径运行、
全角色 CPU/GPU/token 成本和保守 forecast、official parity、native 在线优化、M1/Pilot/
确认实验与论文效果。当前 ranker 代码不打开 gold，但运行它的控制进程尚未证明
不能打开 gold；不能由 input-only 函数推断进程级隔离。图像路径、checkpoint 和
全进程 CPU allocation 仍需外部来源与费用审计。未新增真实模型调用/GPU/付费 API，
本机测试 CPU 工作量非零且没有完整计量。
