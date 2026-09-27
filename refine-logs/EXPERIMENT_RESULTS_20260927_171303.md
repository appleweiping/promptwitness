# 当前实验记录：MIPRO component，不是科学研究结果

记录：2026-09-27T17:13:03Z。细节及machine-readable汇总：research/ARIS_MIPRO_SEARCH。

原Qwen/Hotpot256 reference仍是此前官方GT评分190correct的训练侧seed，不重跑。
新实际运行是CPU authored原MIPRO full/minibatch compile：原67910 exit0，
COMPLETE8/10、PRUNED各5、拒绝valueNone，完整非seed实际vector进入selector，
75seed到100winner。trial7拒绝跳过周期full，selection/cadence科学等价仍不成立。
对应strict native control compile未资格化，fresh doc因thread limit未执行。

主执行者诊断≠fresh独立验收；scripted LM/data/scorer≠方法效果/Pilot。
全角色科学forecast仍未测量。新真实calls/GPU/modeltokens/paid0，原DB26a7e04b...
未改/重算1170calls/3.8074371029887377GPUh，不抹掉先前失败和未知历史成本。

68targetedpassed；本机1917passed6skipped/695.56s/94.04%src覆盖，Ruff329/mypy61/
configured src Bandit/isolated build/Twine通过。newhelper保留1LOW B404，不冒充clean。
原依赖缺失、quote诊断失败、NumPy lazy import cycle及首次lint失败保留于私有观察。
当前source同家族provisional043/044；新exactSHA CI等待提交后独立核验。

experiment-bridge仍running，M1UNFITTED、PilotNOT_RUN、confirmationNOT_STARTED，
0/144pairs、0/180runs、0/36transfers。Goalactive无timer/额外小quota/default4不变。
