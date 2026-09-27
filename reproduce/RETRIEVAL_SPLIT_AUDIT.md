# CIR 训练侧分组审计

`RetrievalInputIdentity` 仅含 query ID、reference ID、dataset 与 FashionIQ 类别。
`audit_training_groups(inputs, assignments, reference_groups)` 在准备任何 gold 前
验证 fit/search/selection 恰好覆盖每个 query，且同一 reference/外部近重复组
不跨池。它没有选择分组算法或比例；调用方须事先固定并提供来源可核验的组。

返回 `INPUT_METADATA_DISJOINT_ONLY` 只证明输入元数据不重叠，**不证明**近重复
检测质量、官方 train/validation 交叉重叠、sealed final、进程隔离或科学准入。
target/subset 不可加到此对象或优化器输入中。原图、annotation、真实分组还未
获本方向 M1 准入；请勿用自制 fixture 的通过代替真实证据。

检索专用的进程角色定义在 `process_access.py`。非 scorer 的
`retrieval_fit_learner`、`retrieval_optimizer`、`retrieval_predictor` 均不得读
任何 `*/gold`；fit/search/selection/final scorer 仅能读对应阶段 gold。旧文本
角色有不同的历史权限，不可直接重用于 CIR。`check_process_access.py` 的
authored sentinel 需要在 Linux 上真实执行；即便它通过，仍须检查 CIR
controller 实际只使用这些专用角色与分离的 store。
