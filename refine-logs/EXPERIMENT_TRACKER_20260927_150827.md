# PromptWitness-Delta 接续 tracker

记录：2026-09-27T15:08:27Z。ACTIVE、无定时任务；默认4轮/科学0轮/原确认矩阵不改，其他四仓冻结。

| ID | 状态 | 实际覆盖与边界 |
|---|---|---|
| S0/BFCL/text-fit | 历史窄fit完成 | 40BFCL+18text旧响应，不是全池性能或M1 |
| S1-store | ACTUAL_PREPARED | 原3tasks各512fit256search256selection，非科学freeze |
| S1-full-IPC/incremental | AUTHORED_NATIVE_QUALIFIED | 前78workers264officialchecks；非真实reference或在线统计准入 |
| S1-model-transport | HISTORICAL_REAL_NARROW | 原Qwen+OLMo10wire/0.1216642GPUh，旧source保留 |
| S1-model-access | ACTUAL_REPAIRED_NARROW | ABI1/零benchmark读/先限制后导入；2CPU probes+两模型6实际input-only fit wires |
| S1-access-repair | ACTUAL_FAILURES_RETAINED | 初模型失败与两kernel失败计费；第二kernel通过，后续新modeldoc一次通过 |
| S1-resource | CONTINUED_ACTUAL | 898→908→914；所有failedstartup/diagnostic/cold/idle/exit计费，closed_history重算 |
| S1-abort | CURRENT_SOURCE_REVIEWED | 新边界code031/032及actual修复036/037；旧P2/followup与机械回归保留 |
| S1-real-scoring/stages | NOT_COMPLETE | 此6新wire未评分；fullpipeline身份/逐阶段科学访问审查仍缺 |
| S1-ref/selection/config | REAL_NOT_GENERATED | 固定真实完整reference/parent/selection与科学identity仍缺 |
| S1-online | NOT_VALIDATED | greedy/singleton/匹配response不能证明ONLINE_PINNED |
| S2-native | ENGINE_NOT_IMPLEMENTED | 原版GEPA/MIPROengineearlyreject与全角色成本未验收 |
| G2 | NOT_RUN | M1UNFITTED/PilotNOT_RUN，独立统计准入未过 |
| G3 | NOT_ADMITTED | 0/144pairs、0/180runs、0/36transfers |
| G4 | NOT_READY | 统计/局限/NARRATIVE、默认有记忆review未完成 |

当前443passed3skip59.84s/最终Ruff311/mypy61src/configuredBanditsrcPASS；
新helperBandit4LOW保留。no-isolation build失败保留，isolated build+Twine两包PASS。
全本机package及安装wheel smoke本轮未跑；前b67de274 exactCI15/15成功，仅属于前source。
当前commit/push/exactSHA CI需另核验，不混淆CI和研究验收。

跨版本actual914calls/3.6819725114061215GPUh/1784463input/193444output/paid0。
新六调用4299input/202output/0.07292376862631889GPUh；失败与diagnostic成本从未reset。
历史unknowncount未编码而非已证明0；完整trace/DB/wires私有。见ARIS_MODEL_ACCESS.json。
