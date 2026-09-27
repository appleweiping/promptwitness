# PromptWitness-Delta 接续 tracker

记录：2026-09-27T13:35:00Z。ACTIVE、无定时任务、原默认4轮与确认矩阵不改；其他四仓冻结。

| ID | 状态 | 实际覆盖/边界 |
|---|---|---|
| S0/BFCL/text-fit | 历史窄fit完成 | 40BFCL+18text旧响应，不是全池性能/M1 |
| S1-store | ACTUAL_PREPARED | 原3tasks512fit256search256selection，非科学freeze |
| S1-full-IPC/incremental | AUTHORED_NATIVE_QUALIFIED | 前78workers264officialchecks，固定前缀/完整survivor/未知成本保留 |
| S1-model-transport | REAL_NARROW_QUALIFIED | Qwen+Olmo10实际wire，21557input2266output，0.1216642GPUh |
| S1-resource | CONTINUED_ACTUAL | 旧898→908，原件副本字节核对、2GPU区间关闭，无新未知成本 |
| S1-abort | LOCAL_MECHANICAL_FIX | 单次复审捕获gate吞异常路径；transport内部stop，实际gatefixture通过，无最终cleanpeerverdict |
| S1-env | METADATA_DOC_QUALIFIED | corrected50pins/spec150b5834，freshmetadata-only一次，不重跑模型 |
| S1-real-scoring/sandbox | NOT_COMPLETE | 新10wire未做正确率，inference进程非data sandbox |
| S1-ref/selection/config | REAL_NOT_GENERATED | 固定真实完整reference/parent/selection与科学identity仍缺 |
| S1-online | NOT_VALIDATED | 原batch差异保留，重复或greedy不证明ONLINE_PINNED |
| S2-native | ENGINE_NOT_IMPLEMENTED | 原版GEPA/MIPROengineearlyreject与fullrole成本未验收 |
| G2 | NOT_RUN | M1UNFITTED/PilotNOT_RUN，独立统计准入未通过 |
| G3 | NOT_ADMITTED | 0/144pairs、0/180runs、0/36transfers |
| G4 | NOT_READY | 统计/局限/NARRATIVE、默认有记忆review未完成 |

当前422passed1skip44.19s/Ruff297/mypy61src/configuredBanditsrcPASS；
新helperBandit2LOW1MEDIUM与旧WMI异常保留。当前SHA CI待commit/push，不与旧15/15混淆。
跨版本actual908calls/3.6029360862527073GPUh/1780164input/193242output、paid0。
完整trace/DB/wires及端点保持私有。未给任何未运行scientific gate补零或标完成。
