# ICMR M1 训练侧元数据与权限增量，2026-09-27 22:58:57 UTC

状态：`PARTIAL_NOT_ADMITTED`。这是同一 ARIS run 的 experiment-bridge 增量，
不是正式 CIRR/FashionIQ 数据实验、科学效果、Pilot 或许可意见。

## 新核对的第一方数据障碍

- [FashionIQ 固定 README](https://github.com/XiaoxiaoGuo/fashion-iq/blob/f71b16c3b09ef03b02c0fed160fa072fb3b8aa16/README.md)
  只写通用 CDLA，没有明确具体版本；[CDLA 官方站](https://cdla.dev/)列出
  Permissive 2.0、Permissive 1.0、Sharing 1.0 等不同文本。
  [原仓库 issue #22](https://github.com/XiaoxiaoGuo/fashion-iq/issues/22) 仍公开询问
  Permissive 或 Sharing，没有可见的维护者澄清。故不能替作者选定条款。
- [原仓库 issue #10](https://github.com/XiaoxiaoGuo/fashion-iq/issues/10) 的报告者
  在 2021 年称约 905 个 dress URL 不可用；这不是本研究测出的当前缺失数，
  但提示完整图库分母/可复现性需实测。[issue #18](https://github.com/XiaoxiaoGuo/fashion-iq/issues/18)
  的第三方再分享链接不是原图许可或官方完整性证明。本轮没有读取其归档。

## 具体代码改进与边界

`reproduce/retrieval_split_audit.py` 只接受 query/reference ID、dataset/category、
预声明 fit/search/selection 分配以及外部给定的参考图近重复组。它拒绝遗漏、
重复、额外 query，和同组跨训练池；不接受 target/subset/caption/final 成员。
近重复组的产生和质量尚未验收，不能用每图一个组冒充去重完成。

旧文本角色确实给 optimizer/predictor `fit/gold` 读取权，违反本次 CIR 的
scorer-only target 设计。因此 `process_access.py` 新增七个 `retrieval_*` 角色：
三个非 scorer 角色均无任何 `*/gold` 授权，四个 scorer 只取各自阶段 gold；
旧六个文本角色保持原权限。独立 sentinel 期望矩阵扩为 13 角色×11 leaves，
应为 143 次 read 与 143 次 write 尝试。新增本地授权矩阵与分组回归通过；
Windows 没有实际 Landlock，不能宣称真实 Linux 边界或工作流接线通过。

审查为 fresh same-family/provisional，未发现这两项窄实现的 blocker，但明确
指出 CIR controller 仍可误用旧角色；真正 M1 必须建立合法完整原图/annotation
来源、资格化近重复检测、训练数据与 gold 分离存储、CIR controller 强制新角色、
实际 Linux sentinel、sealed final 和原生全图库评分/数值 parity。现有状态不改变
`M1_NOT_ADMITTED`，也不解开 FashionIQ/CIRR 原图访问门槛。

本轮无 benchmark image/annotation、最终标签、外部归档或 test 服务器访问，
无新模型/GPU/paid API；CPU 开发测试耗时非零且未完整计量。原历史账本不重写。
