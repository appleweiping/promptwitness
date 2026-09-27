# PromptWitness research instructions

## Pipeline Status

language: zh

研究仅限 PromptWitness-Delta。其他四个 NLP 原创项目保持冻结。
以 `research/PROTOCOL.md`、`research/scope.lock.json`、
`research/PROTOCOL_v1_1.md` 和实际账本为科学与资源约束来源。
续推授权以用户最新指令及 `research/ARIS_GOAL.md` 为准；它覆盖下述历史
文件中的旧工程轮数停止点，不覆盖科学门限、访问披露和已发生的成本。
读取 `research/FINAL_STATUS_v1_1.json`、`research/G1_STATUS_v1_1.json`、
`research/PILOT_STATUS_v1_1.json` 和 `research/ACCESS_AUDIT.md` 后再行动。
历史终态、失败记录、预算消耗和数据预览披露不得覆盖或归零。

## ARIS workflow

用户要求采用 `wanshuiyin/Auto-claude-code-research-in-sleep` 的完整 skill 工作流。
项目本地 Codex 版位于 `.agents/skills/`，源码版本固定为
`341f914024d270dc5c8fa51337d1ad38829273aa`。完整安装有 83 个 skills；
不是要求每轮执行全部 skills。使用前完整读取所选 `SKILL.md` 和必需引用。
共享参考位于 `.agents/skills/shared-references/`。
辅助脚本通过 `.aris/installed-skills-codex.txt` 的 `repo_root` 定位。
不要运行全局安装脚本、改写上游 skill、安装付费后端或覆盖其他项目配置。

采用 `research-pipeline` 的可恢复状态，接续已有固定研究方案；
不重新选题，不把旧方案标成新审查通过。用户已取消本项目定时任务，
改用本聊天的 active goal 持续推进；不得重建心跳、cron 或外部定时器。
保留可恢复状态、真实进展和审查记忆，不用计时器判定论文质量。
Codex 同模型家族的独立上下文评审是 `same-family/provisional`，
不能写成跨模型接受。机械验证只接受其真实覆盖的机械事实。

## Current goal and unchanged ARIS defaults

用户明确要求“不要动aris默认轮数，取消你的吝啬额度限制”，随后要求
“取消定时任务……直接准确的设置目标”。已建立本聊天 active goal，
已删除 `promptwitness-aris` 心跳。继续核心实现已授权，不再请求追加工程轮次。
历史 v1.1 的 2/2 保留为已发生记录，不再作为新续推的阻挡，且不清零账本。
按固定源码的 ARIS 默认执行：`auto-review-loop MAX_ROUNDS=4`，
`AUTO_PROCEED=true`、`HUMAN_CHECKPOINT=false`、`CODE_REVIEW=true`、
`REVIEWER_DIFFICULTY=medium`、`AUTO_WRITE=false`，不修改安装的技能原文。
不增设低于用户总预算的自定调用/GPU/token 小额度或另加轮数上限。
旧阶段额度仍保留在历史文件，但不作为新续推授权的停止点。
默认轮数用尽后照技能记录结果和未解决项，不通过新 run ID 循环重刷审查。

当前接续 `experiment-bridge`，优先官方严格评分、进程级数据访问隔离、
在线执行/采样语义、原生 GEPA/MIPRO 搜索路径、完整角色成本。
工程授权通过不等于科学准入通过；真实 Pilot 与确认实验仍需各自证据。
不得自动开启 v1.2、放宽门限、删基线、减种子或改变模型来通过门禁。

总上限跨版本累计：800,000 次真实调用、1,000 allocated GPU-hours、
20 亿输入 token、2 亿输出 token、付费 API 为零。
v1.1 解阻已消耗 128 次/0.2274557214975357 GPU-hours/309,728 输入/12,462 输出；
跨版本快照为 898 次/3.4812718904634763 GPU-hours/1,758,607 输入/190,976 输出。
这些是快照，每次执行须核对实际账本。按真实工作量为下一步测量和登记成本，
不把资源余量当已获服务器配额。G2 的评分、隔离、执行语义、原生接入和
成本准入仍未通过。正式确认实验仍需 Pilot GO 和
`consumed + 1.20 * remaining_confirmatory_forecast <= ceiling`。
不将吞吐基准、合成测试或通过 CI 当成 Pilot 或科学效果。

2026-09-27 13:35 UTC 新快照：实际 Qwen/OLMo transport 各5次，总10次，
增加0.12166419578923104 allocated GPU-hours/21,557输入/2,266输出；
跨版本实际908次/3.6029360862527073 GPU-hours/1,780,164输入/193,242输出。
原898次账本保留；新10次COMPLETED、两个GPU区间关闭，无新增reserved/unknown成本。
详见 `research/ARIS_MODEL_TRANSPORT.json`。只证明完整wire/token/ownedchild成本路径，
未评分、未验证ONLINE_PINNED或inference数据sandbox、原生fullworkflow/M1/Pilot。
最终吞异常中断修复有本地机械回归，但没有最终cleanpeer CODE_REVIEW判定；
科学门禁与缺失验证不得据此改为通过。环境清单补正不是环境重建或模型重跑。

目标完成须交付可复现代码、有效真实实验、统计分析、局限和研究叙述。
正结果不是必需条件；缺失实验、INTEGRITY_BLOCKED、技能安装或 CI 不能冒充完成。
保持 active goal，只有真正完成可核验交付后才结束；ARIS 停止条件本身不是科研成功。

不得读取最终测试内容来开发，不对已预览数据宣称盲态。
不得访问或占用其他任务的 GPU，不更改共享 CUDA/Python。
敏感响应、私有端点、模型、数据与大账本、审查 trace 和 `.aris/` 状态不提交。

## Git and delivery

在 `research/promptwitness-delta-v1` 上保留用户改动。
只对已审查、脱敏、任务范围内的文件进行显式暂存、commit、push。
不使用 reset/clean，不合并 main，不自行创建 PR、联系导师或投稿。
每次交付区分实际 SHA 的 CI、局部测试、研究效果和未完成事项。
