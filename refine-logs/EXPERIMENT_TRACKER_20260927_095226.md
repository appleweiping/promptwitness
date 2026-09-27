# PromptWitness-Delta 接续 tracker

日期：2026-09-27 09:52:26 UTC。完整目标 active，无timer，ARIS科学默认4轮不改。

| ID | 状态 | 证据与边界 |
|---|---|---|
| S0-cpu | PARTIAL_VERIFIED | 官方authored checks，仅其实际覆盖 |
| S0-BFCL-fit | REAL_FIT_RESCORE_VERIFIED | 旧simple_python40条，12/20与2/20，不是全512/五类成绩 |
| S1-files | AUTHORED_NATIVE_VERIFIED | 历史六进程132权限checks，不代表真实pipeline全部IPC |
| S1-BFCL-fit-app | ACTUAL_NATIVE_VERIFIED | 固定fit/gold分叶，ABI1独立worker已完成 |
| S1-text-fit-app | ACTUAL_NATIVE_VERIFIED | 各512fit/gold、18旧响应+13authoredchecks；SSH/scorer0，不同PID，transport警告保留 |
| S1-pipeline | NOT_COMPLETE | search/selection分叶、allowed IPC、stage-end与config freeze receipts待实现 |
| S1-online | NOT_VALIDATED | 既有batch差异不变，在线随机单元/采样须验证 |
| S2 | NOT_IMPLEMENTED | nativeGEPA/MIPROearly rejection、actualsurvivors/all-role costs未完成 |
| G2 | NOT_RUN | M1 UNFITTED/Pilot NOT_RUN |
| G3 | NOT_ADMITTED | 0/144、0/180、0/36，final开发禁令不变 |
| G4 | NOT_READY | 科学统计/局限/叙述及默认有记忆review待完成 |

上一0286bdc exactCI15/15、1776passed94.03%；本次newdoc/qualification SHA待CI。
新cost0，旧账本/首审漏检/真实失败和Windows transport警告全部保留。
