# 实际完整训练侧参考：首次评分失败、CPU恢复成功

记录：2026-09-27T16:17:21Z。直接goal ACTIVE，PromptWitness timer不存在；ARIS默认4轮、科学0轮不变。

原完整HotpotQA/Qwen seed11 reference256已真实生成。首次fresh-doc原37849 exit1、
458.9171906s：已有Landlock边界拒绝旧sibling code-only runtime清单，未发布reference。
所有原响应、日志、失败和cold/idle/scorerwait/childexit费用保留，不能将首次运行改写成成功。

修复仅将同一核验code/resource bytes放在独占code-recovery的既有reproduce读授权下；
不放宽policy、数据grant、环境或模型配置。唯一sourcefollowup041无BLOCKING。
新fresh-doc仅四文档原样执行CPU恢复一次，原69725 exit0、8.3443399s。
官方scorer PID591355/ABI1实际评分全部原256响应，190严格正确。下载汇总与原输出一致。
新增modelcalls0/GPUallocationfalse，原physicalDB SHA不变；新score attempt追加、旧失败保留。

这只是训练侧seed参考，不是方法效果、held-out accuracy、模型比较、Pilot或ONLINE_PINNED。
其他reference/selection、原生GEPA/MIPROengine、全角色forecast、独立统计审查仍未完成。
M1UNFITTED、PilotNOT_RUN、确认0/144pairs/0/180runs/0/36transfers。

实际更高闭账1170calls/2136046input/195222output/3.8074371029887377allocatedGPUh；
新原256calls为351583input/1778output/0.1254645915826162GPUh。closed_history重算、
本地/remote SHA26a7e04b...一致，原914/base898不归零；CPU恢复后DB bytes不变。
localunknown0不能解释成全历史unknown0，历史未编码/保守token费用原样保留。paid0。
原模型terminal后GPU466MiB0%，其他job不动；没有新模型或GPU重跑。

前本机full1897passed6skip478.69s/94.04%是在CPU恢复helper加入前，不冒充最终fullsource。
最终helpers+incremental467passed3skip59.50s、Ruff321check/format/new3helpersBandit通过；
unchanged src的mypy61/configuredBandit与isolatedbuild/Twine通过。installedwheel smoke未跑。
旧Mockfixture5失败、E501、staging错误、首次scorer失败、错误trace文件路径读取保留。

前50b709e exactCI36328917670 success15/15，Ubuntu3.12实际1883passed161.92s94.06%，
只证明前source。本报告新提交CI尚待commit/push后独立核验，不伪报为已通过。
source039/041同家族provisional；doc040/042继承设置，canonical metadata的model=inherited、
family_relation=unknown/independence=unverified原样保留，不声称已验证跨家族独立性。
原15:55:49运行中版本不覆盖；完整raw/trace/provider/DB/per-ID数据始终私有。

详见research/ARIS_REAL_REFERENCE.json与ACCESS_AUDIT.md。全桥和科学门禁仍未通过；
目标继续保持active，默认轮数和原科学范围不变，不新建定时器。
