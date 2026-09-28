# ICMR 检索 GEPA：真实受限评分器接入（仅自制资格）

记录：2026-09-28 12:39:15 UTC。结论仅为
`PASS_AUTHORED_NATIVE_GEPA_RESTRICTED_SCORER`：固定原 GEPA 搜索在自写 CIRR 形状
query ID 上，reference、原生 search/selection 与审计均使用实际 Landlock
评分子进程的返回值。不是正式 CIRR/FashionIQ 成绩、真实 CLIP/生成器优化、
M1/Pilot、C1–C3 证据或 ICMR 论文结果。旧自制 GEPA 检查器的进程内
scorer 替身和按规则预置的 reference，**不能**再作为本次受限资格的证据；
本次是独立新运行。

## 接线与进程边界

固定原 GEPA commit `d771eb21b5dd3228bc3f567293d2ccfc423fc900`；
检查器核对 checkout HEAD 及所需六个原生源码文件逐字节一致，继续使用
reflective engine、`AllImprovements`、`use_merge=False`、fresh run。
原生示例只含 query ID。脚本化提案和自制排名函数产生候选排名，但
reference、native minibatch、完整 selection 和审计的 hit 分数均从
`score_rankings_restricted` 返回的 `primary_hit` 取得；预期参考分数仅用于
比较，不能替代计算。审计桥的包装调用原受限入口，未将失败补成 0。
搜索计划先于 child 原生评分冻结；只有通过原生完整性、改善和审计后，
才补齐 64 条实际 search survivor 并进入 selection。

子进程由项目 Landlock launcher 限制，返回角色、阶段、ABI、不同 PID 和
完整观察人口，检查器逐项验证。**仅 scorer 受限**：可信优化器父进程仍
持有自制 store 路径，能够读取 gold；没有 optimizer-versus-gold 进程隔离，
也没有在本次 GEPA 运行中做越权读取 sentinel。Landlock 不是网络隔离。
固定 scorer 是自制 12 项图库 ID 上的确定性指标，不是 CLIP forward。

## 物理自制运行与失败收据

私有 WSL ext4 Python 3.12 环境执行最终 `qualification-gepa-restricted-20260928-04`
exit 0；源码从 Windows 工作树机械拷贝到 ext4 后逐文件 SHA-256 相等。
初次 DrvFs 源码被 Landlock 拒绝、第二次原生 GEPA 参数组合不合法的失败
目录均保留；第三次成功是修复报告前版本，最终以第四次为准。

| 自制场景 | 实际 reference query | 原生 search + selection query | 审计 score query | 审计外层 rank+score | 成功受限 worker 收据 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| 有益候选 | 64 | 16 + 32 = 48 | 64 | 128 | 69（不同 PID 69） | 完整 search survivor 64，进入 selection |
| 退化候选 | 64 | 16 + 16 = 32 | 4 | 8 | 8（不同 PID 8） | INELIGIBLE，survivor 0 |

正例 native 账本 5/5 completed、审计 128/128；反例 native 4/4、
审计 8/8；两例 failed/unresolved 均 0。上述 native query 不是 worker
次数，更不是生成模型调用数；69 = 1 个 reference 批次 + 4 个 native
批次 + 64 个单 query 审计 worker，8 = 1 + 3 + 4。完整检查器 wall
38.472 秒、正例 35.374 秒仅是该自制 CPU 执行观测，不是全角色成本。
外层账本仍把未知 forward 数记为 `forward_count_unmeasured_attempts`，
不得把它解释为零成本。

另用私有故障注入核对真实接线：原生引擎在实际受限 reference 64 条完成后
故意抛错，`scenario.json` 保留 `FAILED_RETAINED`、64 条 reference、
1 个成功 worker 收据与 native ledger completed=1；启动前故意抛错，
顶层 `qualification.json` 保留 `FAILED_RETAINED` 和
`score_worker_is_real_landlock_child=false`。中途失败时 `scorer_worker_launches`
仅统计**成功返回并验证的 worker 收据**，不包含失败启动；失败、未决和
未测量工作必须看各账本，不能从成功计数推断为零。

最终私有 qualification SHA-256：
`b6aa67b290409246c83cc7131c8ec0a253d208992401068200a81ba901f4c5a6`；
两份故障报告 SHA-256：
`cbc427cb20a6f8fdae6bbe05265dee9b99c2c04950bbb650d4c59154cb4a959c`、
`cf94d2d08c8027336c2fa95ccd1af394ee7a7ecbc030d2cc8d7795a3583c7b7e`。
公开 checker/helper/test 源 SHA-256 分别为
`e9b80d69c0ebe90bc6902fc8322882c4696a7926391ca47b6726a1501b3910dc`、
`2b1f6df3c69e6feff03a10221980ee43e224d96ede996cffadcc1a964a7a1e20`、
`5f4bc2787307491297323d02804060729e0ccdfac262f725dd90799a85fac138`。
私有 stores、SQLite、模型环境和 trace 不提交。

## 验证、审查和未准入项

新增 5 条辅助层测试；Windows 全套 `2110 passed, 28 skipped`、
coverage 94.03%。Ruff check/format、mypy src 61 文件、配置 Bandit、
sdist/wheel build 和两个包 Twine 检查均 exit 0。最终源码精确 SHA CI
须在提交推送后另行核验；仓库 CI 不运行私有 GEPA/WSL qualification。

同家族 GPT-6-Astra/xhigh 初审 0 BLOCKING、2 NON-BLOCKING：失败前过早
声明 worker 已验证、中途失败丢失已完成计数。已修并在 WSL 故障探针
执行。复审 0 BLOCKING、1 NON-BLOCKING：5 条新增测试覆盖辅助层，尚未
把完整 `check/_run_scenario` 故障报告接线固化为仓库自动回归；复审本身
做了 7 组只读内存探针，没有复跑私有 WSL 原生运行。两次审查的请求、
回复和来源元数据在忽略的 `.aris/traces/experiment-bridge/
2026-09-28_run01/011-*`、`012-*`，结论 `same-family/provisional`，
不是跨家族独立科研验收。

正式 CIRR/FashionIQ 图像与标注、来源许可和 sealed final、真实 CLIP/
Qwen/OLMo、优化器对 gold 的同次进程隔离、全角色真实成本与预算预测、
M1/Pilot/C1–C3 均 `NOT_ADMITTED`。本次新增真实生成模型调用、GPU 小时、
付费 API 均为 0；完整角色 CPU 成本 `UNMEASURED_NOT_ZERO`。历史账本
1170 calls、2,136,046 input、195,222 output、3.8074371029887377
allocated GPUh 不清零。先前公开 CIRR 页面样例的接触披露不撤销。
SSH 端口可达，但当前指纹与先前用户核验值冲突，未登录、未改
`known_hosts`。ARIS 默认 4 轮、无定时器、其他四个 NLP 仓库冻结；
研究目标继续 active。
