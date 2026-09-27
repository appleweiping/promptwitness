# PromptWitness 的 ARIS 接入记录（当前）

时间：2026-09-27 05:10:00 UTC。
本记录替代最新接入副本中的旧工程授权阻塞，不覆盖历史 timestamp 记录。

## 来源和默认值

采用 [ARIS](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep)，
固定源码 `341f914024d270dc5c8fa51337d1ad38829273aa`（MIT）。
83 个 Codex 镜像 skills 已本项目本地安装；原文、许可证和共享参考保留。
使用前完整阅读所选 skill 及必要引用，不等于已执行所有 skills。

主入口为 `research-pipeline`；接续固定方向，跳过重新选题。
保持上游默认：`AUTO_PROCEED=true`、`HUMAN_CHECKPOINT=false`、
`CODE_REVIEW=true`、`REVIEWER_DIFFICULTY=medium`、`AUTO_WRITE=false`；
`auto-review-loop MAX_ROUNDS=4` 不修改，不叠加旧的两轮上限。
未指定 venue，不自动投稿；同家族 review 为 `same-family/provisional`。

## 最新用户授权和调度

用户要求保留 ARIS 默认轮数、撤销额外小额度阻挡，随后要求取消定时任务、
准确设目标。`promptwitness-aris` 心跳已由应用工具实际删除，
返回 `deleteStatus=deleted`；没有替代定时器。
本聊天 active goal 已实际建立，未添加自定 token budget。
目标、验收和预算见 `ARIS_GOAL.md`；历史账本与 2/2 工程记录不清零，
但旧 2/2 不再阻挡新核心工作。不能用目标 active 冒充科研完成。

## 当前事实与接续

- 源码起点 `bcc0e37cd9dffab59ab5442f245eaf2595c90321` 的
  [CI 36293992956](https://github.com/appleweiping/promptwitness/actions/runs/36293992956)
  completed/success，15/15 jobs。新变更尚无自己的 CI 结果。
- 科学状态仍为 `INTEGRITY_BLOCKED`：G1 partial、真实 M1 UNFITTED、
  Pilot not_run、confirmatory not_started；不把工程授权写成科学准入。
- 跨版本账本快照 898 calls、3.4812718904634763 allocated GPU-hours、
  1,758,607 输入、190,976 输出。此工作流修改未启动推理。
- 新核心实现已授权。下一步 `experiment-bridge`：官方严格评分器，
  进程级 split 访问，在线执行/采样语义，原生 GEPA/MIPRO 搜索路径和全角色成本。
- batch 4/8 已观察到不同输出；重复相同不是普遍 batch invariance 证明。
  原生接口 smoke 不是完整 NLP 搜索，冻结表证书不是在线科学保证。
- 真实 Pilot 和全矩阵实验须通过原有科学、访问和成本门禁；没有在运行的
  科学实验可报告，不因取消阶段小额度而绕过这些门禁。

按 ARIS 使用已有可恢复状态、单一 orchestrator、审查记忆与真实迭代记录。
只在已完成步骤后审查实际新证据，不重刷未变内容；默认停止条件必须遵守。
实际执行按总预算和真实可用配额计费，不新增自定小额度。
