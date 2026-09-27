# 真实模型 wire 与连续成本：审查和窄验收

记录：2026-09-27T13:35:00Z。Goal ACTIVE、无定时任务，默认4轮/原144pairs180runs36transfers不改。

Fresh CODE_REVIEW 初审发现异常先等30s EOF才终止child的P2，63tests21.21s通过不能消除该问题。
旧回归3failed1.78s，修复异常退出立即terminate；同时CPU/scorer与Torch环境实际分离，
改为必填绝对model_python，固定module argv，无install/fallback/sharedenv变更。
唯一followup69passed22.97s，却明确同一P2仍在gate捕获executor异常转INCONCLUSIVE的路径。
main按该具体意见在transport内部先stop再抛错、标记failedsession不能再执行/新增reserve；
旧回归1failed2.05s，修复后70targeted23.60s。实际gate回归初fixture漏start TypeError保留，
纠正后1passed1.70s；最终全helper+incremental422passed1WindowsLinuxskip44.19s。
最终局部修复只有mechanical证据，没有新的cleanpeer CODE_REVIEW verdict，不重开第三review。

Fresh doc按原样一次实际Qwen+Olmo顺序执行，原SSHsession11030退出0，观察443s。
三任务每模型1fit input、原typed-proposer每模型2fixture，共10calls，21557input/2266output，
完整allocation增加0.12166419578923104GPUh，历史898→908且未归零。两个区间均关闭，
实际10COMPLETED，无新reserved/openallocation。模型/模板/解码profile真实冻结；无cache、
batching、drop字段、自动重试、final bodies或scoring。Proposer fixtures含已披露fit示例，
不是完整nativebootstrap/GEPAreflection/MIPROproposal；scorer/tool identity仍transport-only。
GPU最初其他任务忙时未启动，空闲后才运行；原其他service保留。coldload权重145s比旧慢，
如实计费不重启，不据此宣称算法吞吐或模型优劣。

环境WHAT初48pins而actual--all50，漏pip25.2/setuptools65.5.0；原freeze47packagepins而非
作者ledger误称49。补正spec150b5834后另一fresh metadata-only doc一次exit0/4.8427259s，
50pins exactmatch，无新模型/GPU/环境变更。spec NOT_RUN为最初声明快照，当前实证见
ARIS_MODEL_TRANSPORT.json；不是改写原声明或重新推理。完整private traces027–030不上传。

Ruff/checkformat297/mypy61src/configuredBanditsrc PASS；新helpers Bandit2LOW1MEDIUM保留：
固定无shell子进程B404/B603、local-only固定snapshot B615；无新增ignore。
较早414passed1skip80.25s另有WindowsWMI0x8007000e日志；最终changed-source422次无该日志，
不抹除旧异常。中间59case handle观察丢失未报告passed。完整本机package/build本轮未跑，
历史timeout1800s/wheelWinError127等不抹掉。

本次只证明实际训练侧wire/token/ownedchild分配生命周期，非ONLINE_PINNED、数据sandbox、
原生引擎、M1/Pilot/科学效果或独立统计接受。完整角色forecast/真实reference/selection仍缺。
当前commit/push前snapshot CI待核验，前SHA135df75的15/15成功不冒充新SHA。
