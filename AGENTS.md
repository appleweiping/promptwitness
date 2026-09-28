# PromptWitness research instructions

## Pipeline Status

language: zh

研究仅限 PromptWitness-Delta。其他四个 NLP 原创项目保持冻结。
2026-09-27最新方向授权：用户已确认ACM ICMR，并明确要求改进尚未建立的主题匹配。
当前主研究设计以 `research/ICMR_SCOPE.md` 为准：真实组合图像检索，CIRR/FashionIQ，
版本v0.1设计而非科学冻结；新实验须通过各自许可、评分、隔离、在线与成本准入。
这明确覆盖下文旧“不改三文本任务/旧矩阵作为未来必做前置”的限制，不改历史记录。
旧scope.lock/PROTOCOL仍保存数学、总资源与历史事实；不把旧144/180/36转写为已完成。
同一ARIS run和审查记忆继续，不创建新run重置default4或已发生费用。
最终交付包含符合ICMR当届要求的英文稿/PDF；不得把旧AUTO_WRITE=false作为省略理由。
2026-09-28 03:56 UTC 增量：固定 SEARLE 原指标函数体与独立核在
11 自写无并列查询、55候选上17项匹配；`research/ICMR_SEARLE_METRIC_PARITY.md`
仅是 M1 前置窄工程资格。已有 tie 反例、官方 server/完整库、原图许可、
分组封存、双生成模型和全角色预算仍未准入，不得把它写成 ICMR 结果。
精确源码 `040313894f74a70d15c6ff8e66eba8718625087a` CI run36376271977
已success15/15；这只关闭该代码的CI待验，不替代私有WSL正向函数体运行，
更不关闭正式数据/官方server/完整库/tie/M1/Pilot。见
`research/ICMR_SEARLE_METRIC_PARITY_CI.md`。
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

2026-09-27 15:08 UTC 新快照：修复后的 inference 文件边界已由 fresh doc 原样一次
完成两模型各3实际 input-only fit wires，原 exit0/268.0048149s、ABI1/零 benchmark 读授权。
新6calls/4299input/202output/0.07292376862631889 allocatedGPUh；跨版本实际
914calls/1784463input/193444output/3.6819725114061215 GPUh，原模型启动与诊断失败均保留计费。
新连续DB字节核对/closed_history重算，无开放allocation；historicalunknowncount未编码非0证明。
own-task 文件metadata写是整个child自身task子树，不是仅comm名匹配；ABI1限制明确披露。
source/rescue/doc031–038均same-family/provisional，仅模型窄资格，不是全角色/science准入。
当前helpers+incremental443passed3skip59.84s、isolated build/Twine两包通过；no-isolation失败保留。
详见 `research/ARIS_MODEL_ACCESS.json`；M1/Pilot/ONLINE_PINNED/nativeengine/原确认矩阵尚未完成。

2026-09-27 16:17 UTC 新快照：原完整Hotpot/Qwen256 reference先实际生成后scorer布局失败
（exit1/458.9171906s），所有失败/响应/费用保留。仅同bytes的code/resource runtime
colocate到既有reproduce grant，唯一sourcefollowup041无BLOCKING；显式CPU恢复fresh-doc
原69725 exit0/8.3443399s，官方评分256/190correct、新modelcalls0/GPUallocationfalse、DB未改。
latestactualDB26a7e04b... byte核对/closed_history重算1170calls/2136046input/195222output/
3.8074371029887377allocatedGPUh；继续此更高历史而非914/908/898。新增原256calls成本
351583input/1778output/0.1254645915826162GPUh含cold/idle/scorerfailurewait/actualexit。
这是训练侧seed参考，不是方法效果、Pilot或ONLINE_PINNED。其他reference/selection/native
engine/全角色forecast/独立统计/M1/Pilot/原确认矩阵仍未完成；bridge保持running。
source039/041 same-family/provisional；doc040/042继承设置canonicalfamilyunknown/
independenceunverified原样保留，不声称跨家族接受。详见research/ARIS_REAL_REFERENCE.json。
最终helpers+incremental467passed3skip59.50s/Ruff321/new3helpersBandit；本机1897全套94.04%
在recoveryhelper之前，不冒充新final全套。无timer/额外小quota，default4科学0不变。

目标完成须交付可复现代码、有效真实实验、统计分析、局限和研究叙述。
正结果不是必需条件；缺失实验、INTEGRITY_BLOCKED、技能安装或 CI 不能冒充完成。
保持 active goal，只有真正完成可核验交付后才结束；ARIS 停止条件本身不是科研成功。

不得读取最终测试内容来开发，不对已预览数据宣称盲态。

2026-09-27 17:13 UTC：strict MIPRO helper/component 已接原 incremental driver，主作者原67910
实际CPU authored full/minibatch compile exit0；真实OptunaPRUNED各5/valueNone，complete8/10，
75seed到100完整非seed。source043/唯一followup044同家族provisional，允许限定诊断。
但到期TrialPruned跳过周期full/study.add_trial可改变winner，原source反例保留；这项
cadence/selector缺陷未修，strict native控制也未资格化，不能开展科学native/Pilot部署。
fresh doc spawn失败thread limit，未执行；主作者不替代独立环境验收。见research/ARIS_MIPRO_SEARCH。
本机1917passed6skip/695.56s/94.04%，68targeted/Ruff329/mypy61/srcBandit/build/Twine通过；
helper保留1LOWB404仅构造authored CompletedProcess无执行、不加ignore。原环境缺依赖/
NumPy lazy import失败保留，新task-ownedCPUenv0279efce不改服务器/原interfaceenv。
物理DB26a7e04b... bytes未变/closed_history重算1170calls/2136046input/195222output/
3.8074371029887377GPUh；本轮新realcalls/GPU/tokens/paid0，未知历史count未编码保留。
default4科学0/no timer/extra quota不变；完整native/online/statistical/fullroles/M1/Pilot/原矩阵仍缺。
不得访问或占用其他任务的 GPU，不更改共享 CUDA/Python。
敏感响应、私有端点、模型、数据与大账本、审查 trace 和 `.aris/` 状态不提交。

## Git and delivery

2026-09-27 18:29 UTC：共享MIPRO objective适配完成限定CPU机械验收：原73744和
fresh-doc91776各一次四路径compile exit0；minibatch objective6/native trial7拒绝
仍实际full promotion，空池/已full池不伪造trial。source唯一followup045同家族
provisional；初次完整请求未落盘、旧verdict原样重附，保留trace缺口。doc047继承
模型canonicalfamilyunknown/independenceunverified，不声称跨家族接受。
最新22214f6 CI14/15暴露constructor preflight并发错误，本机确定性复现后最小
in-process RLock修复；76passed1skip，不作跨进程保证。fresh journal review
thread limit未创建，trace046保存完整失败，local-only而非独立审查通过。
最终1924passed6skip1116.10s/94.03%，WMI fatal0x8007000e日志保留，终态exit0；
Ruff343/mypy61/configuredsrcBandit/build/Twine通过，helper1LOW B404不隐藏。
实际DB26a7e04b... bytes未变，续1170calls/2136046input/195222output/
3.8074371029887377GPUh，historicalunknown未编码保留，新real/GPU/tokens/paid0。
详见research/ARIS_MIPRO_SEARCH；接真实nativebackend/allroles与GEPA原engine，
M1/Pilot/online/statistics/原矩阵/ICMR主题匹配与稿件仍未完成，goalACTIVE。
不以旧a77b622成功重试替代交付新exactSHA CI；无timer/额外小quota/default4科学0不变。

2026-09-27 17:36 UTC 用户新增：论文目标 ICMR，遵守最新届规格/字数/页面要求。
按 `research/ICMR_REQUIREMENTS.md` 的官方核验与明确假设准备英文稿；当前按
ACM ICMR 2027 regular long paper 登记，不默认换成 ICML、不照抄 generic ACM 页数。
写作现已明确要求；旧默认 AUTO_WRITE=false 不得成为遗漏后续论文交付的理由。
上游 skill 默认轮数和原研究冻结不变；先完成证据冻结/研究准入及主题匹配审查，
再执行 paper-writing，不把格式登记写成研究或投稿完成。投稿前重新核验官方要求。
现有文本/工具任务尚未建立多媒体检索主题匹配，不私自添加实验或改题。

在 `research/promptwitness-delta-v1` 上保留用户改动。
只对已审查、脱敏、任务范围内的文件进行显式暂存、commit、push。
不使用 reset/clean，不合并 main，不自行创建 PR、联系导师或投稿。
每次交付区分实际 SHA 的 CI、局部测试、研究效果和未完成事项。

2026-09-27 19:18:39 UTC：最新用户已授权实质ICMR主题改进，主设计以ICMR_SCOPE
v0.1为准而非继续把旧三个texttasks作为整套未来必做。候选CIRR/FashionIQ图像+
修改文本→共享固定caption→原两文本模型→固定视觉/文本编码排名；96/120/24
为NOT_FROZEN/NOT_RUN新设计，旧144/180/36保留未运行历史。无科学准入或稿件。
新独立CPU评分核/CLI接原gate，33new+135targeted通过；source048/唯一049限定
same-family/provisional；fresh-doc050原93654一次exit0/25.0974021s，familyunknown/
independenceunverified。只有authored vectors，不是实际images/encoder/模型节省。
whole39297与build40858仍运行，不称通过；WMI诊断保留。见ARIS_ICMR_RETRIEVAL。
CIRR官网许可核查自动展示testannotation示例，ACCESS_AUDIT披露；不转入开发，
不声称CIRRtest开发者完全盲态。CIRR原图NLVR2条款由用户完成访问，FIQexactCDLA/
image-source与weights审计未完成；不代签/联系/下载镜像/测试上传绕过。
物理DB26a7e04b...未改，1170calls/2136046input/195222output/3.8074371029887377GPUh；
unknown未编码保留、新real/GPU/tokens/paid0。default4科学0/no timer/goalACTIVE。

2026-09-27 19:32:23 UTC更新：原39297已terminalexit0/1957passed6skip2013.77s/
94.03%；WMI日志保留。finalRuff/format366通过；实现27f90c5实际push并exact
CI36344286371success。来源审计文档e95bd09已push/CIqueued，本机原build40858
仍未终态，不假称localbuild过。ICMR_DATA_AUDIT记录第一方CLIPexactsource与
上游expectedweightidentity，但未下载/编码；原图/许可/native/image/cost/M1/Pilot/
science/manuscript仍pending。待续原build40858；新latestSHA CI须另外核验。

2026-09-27T20:17:00Z最新组件：published原metric核两次actualCPU（48422/54921）各一次exit0，
11query/15比较/5native函数；source051无具体blocker但same-family/provisional，
doc052familyunknown/independenceunverified。原all-tie global0%vslexical100%反例
保留，strict-order authored PASS不授official/fullnumeric parity。50pins metadata
6687ae36实匹配、CPU1GPU0无install/rebuild；原source/许可仅私有不改MIT。
新14util+33scoring局部47passed，whole80877运行中。Ruff374/mypy61/srcBandit过；
新helper保留B102 MEDIUM/HIGH-confidence可信exec非sandbox，不屏蔽。
原build40858实际失败虽最后sdistTwine使terminal0；新39828实际两包/Twine过，
uv.CMD空77 owned artifact已私有归档。99876b8 exactCI36345101338 success15/15，
不是新未提交helperCI；newreal/GPU/token/paid0、原1170历史/DB不变。
M1原图/许可/封存/fullpool tie、实际encoder/caption/native/online/statistics/fullroles/
Pilot/确认矩阵/英文稿仍pending，goalACTIVE无timer/default4科学0不改。见新
research/ARIS_ICMR_NATIVE_METRICS；真实访问门槛不得因CPU通过就跳过。

2026-09-27T21:21:00Z：ICMR单queryfloat32 CPU排名前置实际验收：主23828/doc41132
各一次exit0，四authoredcase/8原metriccalls/8全72-ID排名/36比较；8malformed拒绝，
原gate survivor64/rejection4/failed1attempt0score。旧lexical反例保留；querybatch1
gallery顺序/1-dot/Torchargsort固定，不自授images/encoder/GPU/batch/fullbenchmark。
新owned SciPy1.15.3层，原50pins只读，新51spec87663299实匹配；71stagefiles
逐字节往返核对。source053同家族provisional；doc054身份unknown/independence
unverified，aggregate未单独展示CUDA值/各72-ID列表限制保留。原42147全套终态
1978passed6skip/1014.27s/94.03%src（非reproducecoverage），Ruff387/mypy61/
src+新3helpersBandit/build15144两包Twine过；旧exec B102 MEDIUM不屏蔽。
原80877已1971/6/1159.27s仅bc8304e3；bc exactCI36347625475 success15/15，
新代码记录时未push，不借旧CI充数。DB26a7e04b...本轮hash未改，1170历史费用/
unknown未编码保留；新增model/token/GPU/paid0，CPU总开销未完整计量非0。
ARIS novelty-check初审001/路径澄清002/同记忆新HbBoPs续查003完整私有trace，
5/10 PROCEED_WITH_CAUTION仅早期过、same-family/provisional，不科学接受。
HbBoPs已含结构vs整段text、stage配对/superset/真实最高fidelity，C1必须检验
typed父子edit×query相对强模块控制的边际作用；不能写首次结构/paired/风险。
六configs映射/数据访问/final封存/视觉与native在线/全角色forecast/M1/Pilot/
统计/英文稿待准入，goalACTIVE/default4科学0/无timer/其他四仓冻结不变。
详见research/ARIS_ICMR_CPU_RANKING和idea-stage/ICMR_NOVELTY_CHECK。

2026-09-27T22:00:00Z：实际CLIP CPU组件已实现并两次验收。第一方ViT-L/14真实
932768134B权重actualSHA b8cca3fd...，224px/77token/float32evalno-gradbatch1，
仅3自制RGB/RGBA/L图+2英文query；主73988/doc10832各一次exit0/40.6711s/40.0724s，
3image/2text反序bitwise、2完整3-ID排名一致；每次image6/text4forwards，两输入模型前拒绝。
55pins fbea8bf4实际匹配，15原stagefiles往返bytes一致；base50+SciPy只读/ownedvisual层。
codeMIT/模型卡research用途已审，weightsMIT再发布未确立，私有研究非formal准入。
source055实际double-normalization blocker→旧测试fail→rawimage最小修复；唯一同记忆
056无blocker，同家族provisional；doc057身份unknown/independenceunverified，不科学接受。
59targeted44.38s/Ruff409/mypy61/configuredsrc+新两helpersBandit通过；未配置src11旧LOW
与旧execB102保留。0.2.0两包build79353/Twine94088实际过；首次旧0.1.0误查保留。
whole9530仍pre-fix collection运行/WMI保留，不final源码PASS；旧27c2139 exactCI
36351707078success15/15不代新未提交源码。新20 CPUencoderforward/24attempt另记，
generator/tokens/GPU/paid新增0、CPU全开销未量非0；原1170/26a7e04b...DB重核不变、
unknown未编码非0证明、已知保守和1190不完整allrole。benchmark/M1/Pilot/封存/
captionfusion/nativeonline/强对照/forecast/统计/英文稿仍pending，goalACTIVE/default4科学0、
同一run/无timer/其他四仓冻结。见research/ARIS_ICMR_CLIP_CPU，不重复此fixture救准入。

2026-09-27T22:36:18Z：ICMR B3真实图文融合机制已在authored-only资格跑通：参考图像
raw向量＋修改文本unit向量→预写定0.5组合→原完整图库CPUrank；0.5非科学冻结/
优化结果。主40503/doc一次各exit0/checker38.7071/38.5269s，3图/2修改反序
bitwise、2完整3-ID排名一致，各6image＋4text实际CLIPforward。source058
gpt-6-astra/xhigh无具体blocker同家族provisional；doc059按文档一次无retry、
identity inherited/familyunknown/independenceunverified，不科学通过。ownedstage
新两源码hash与本地匹配、原55pin fbea8bf4/权重/base环境不变。
局部67项66passed1skipped51.23s（本地无Torch数值）、Ruff420/mypy61/两
helper Bandit通过。首df53792 exactCI36354407932确为15/15 success但不代
此未提交源码；原whole9530终态1981passed6skip1failed2830.81s/94.03%src，
旧closed-pipe Windows wait15s超时，单例6.29s复测通过不能覆盖整套失败，WMI保留。
新组合额外20CPUencoderforwards，上一组件20，历史generator1170；已知保守和
1210/unknown未编码，不写allcall0或CPUtotal0；新generator/token/GPU/paid0、
原物理DB26a7e04b...未改。无benchmark数据/许可/fullpool、caption/融合训练
冻结、M1/Pilot、强CIR、全角色forecast、统计/英文稿；goalACTIVE/default4
科学0/同一run无timer其他四仓冻结。见research/ARIS_ICMR_COMPOSED_CPU。

2026-09-27T22:58:57Z：M1输入元数据分组审计及七个CIR专用Landlock角色已实现，
旧文本role未变。FashionIQ固定README的泛CDLA在原issue#22仍有版本歧义，
原issue#10仅有旧URL缺失报告；不用#18第三方镜像。authored本机测试及
fresh same-family/provisional review不授真实数据/进程/科学准入；Windows无法
实际执行Landlock，CIR controller必须强制新角色。见research/ICMR_M1_ACCESS；
M1/Pilot仍NOT_ADMITTED，default4/scientific0/原账本/无timer不变。

2026-09-27T23:10:53Z：上一增量源提交5ba033187a22a3175eae552067c2cf1b301a83ac
已推送且精确SHA CI run36357426043终态success15/15。Ubuntu3.14为2018passed/
1skipped/127warnings；Linux实际test_real_linux_process_sentinels在通过job中执行，
覆盖13角色×11leaves=143读/143写检查。本机reproduce475pass4skip。见
research/ICMR_M1_CI；这不证明CIR controller接线/真实数据/许可证/M1/Pilot。

2026-09-27T23:34:12Z：CIRR/FashionIQ fit/search/selection 新增可调用的受限评分
路径：控制端只发排名，检索专用scorer进程从本阶段gold读标签，只回指标；search
可部分评分，fit/selection须完整，final未开放。fresh same-family/provisional
审查发现并复审关闭相对store路径blocker。本机最终reproduce481pass7skip，
含最后fit增量；真实Linux执行需等新提交精确CI。仅自制fixture，无正式
benchmark数据/新模型/GPU/paid；M1/Pilot/论文效果仍NOT_ADMITTED。详见
research/ICMR_RESTRICTED_SCORING.md；原账本/default4/无timer不变。

2026-09-27T23:44:46Z：该源SHA fda4464dc4186338e8ed1c06777f369785ebea08
CI run36359379065终态success15/15；Ubuntu3.14 2027pass1skip127warnings，
新Linux实际worker测试在通过增量中。受限scorer仅不直接返回原始target/subset；
逐查询指标可被反复探测推断标签，控制器预算/journal/selector尚缺，不能宣布
完全盲评、M1或Pilot。见research/ICMR_RESTRICTED_SCORING.md。

2026-09-28T00:08:54Z：新检索审计桥已将原冻结AuditPlan/AuditJournal/gate
与检索专用scorer连接；失败不补零/不重放、ELIGIBLE survivor补全、显式
selector人口顺序和混合0/1对齐。本机最终reproduce486pass8skip/119.85s，
定向5pass1skip；fresh审查same-family/provisional无本增量blocker。
WSL六项3pass3fail，缺SciPy及既有worker包PermissionError，不算Linux通过。
待新SHA CI；物理模型成本、完整controller、ONLINE_PINNED、selection/final、
正式数据/M1/Pilot/论文仍未准入。见research/ICMR_AUDIT_BRIDGE.md。

2026-09-28T00:22:04Z：检索桥提交b1609b75e0276fd93275cf6c9f49e62d35e30b82
已推送，精确SHA CI run36361517661终态success15/15；Ubuntu3.10
2033pass1skip，Ubuntu3.14 2033pass1skip128warnings，包含Linux authored
实worker桥接测试。只关闭工程CI待验；WSL失败、物理成本/完整controller、
ONLINE_PINNED/selection/final及正式数据/M1/Pilot/ICMR效果仍未准入。
见research/ICMR_AUDIT_BRIDGE_CI.md。

2026-09-28T00:54:13Z：检索侧物理操作收据新增 SQLite reserve/settle、未知费用
保留与禁止重放；审计桥强制经此记录 rank/scorer，authored CLIP checker 接
encoder/load/rank。定向18pass2skip，最终reproduce492pass8skip/81.68s；
全套2029pass11skip始于末两处小调整前，不当最终源码证明，exactSHA CI待验。
fresh reviewer发现嵌套耗时误称lower bound的BLOCKING，确定性修复并唯一
follow-up复审关闭，同家族provisional。没有新真实模型/benchmark/GPU/paid；
全角色成本、正式数据/许可、ONLINE_PINNED、M1/Pilot/ICMR效果仍未准入。
见research/ICMR_WORK_LEDGER.md；历史账本/default4/无timer不变。

2026-09-28T01:03:59Z：检索操作账本精确源码83e574a9f615aa24ae5a8851830903a941ec5846
CI run36364121379终态success15/15；Ubuntu3.10 2039pass1skip、Ubuntu3.14
2039pass1skip127warnings，authored Linux worker在矩阵内。仅工程CI通过，
不授真实CLIP新账本资格/M1/Pilot/ICMR效果。EXPERIMENT_PLAN claim map已按
ICMR_SCOPE将paired审计列C2、等预算优化列C3；未改矩阵或科学冻结。
见research/ICMR_WORK_LEDGER_CI.md。无timer/default4/其他四仓冻结不变。

2026-09-28T01:27:19Z：search-only direct CLIP ranker 读取input-only元数据和
有序全图库、逐图共享索引、逐查询修改文本+参考图融合，可接原固定审计桥。
自写fixture新文件5pass1skip；reproduce组exit0，Ruff/format通过；全套
与精确SHA CI待终态。fresh审查同家族provisional无BLOCKING，三类别建议
已补回归。未运行新源码真实CLIP/benchmark，进程级ranker隔离、合法数据、
权重/融合科学冻结、captioner、两原模型、全角色forecast/M1/Pilot/效果/稿件
仍未准入。历史账本/default4/无timer/其他四仓冻结。见research/ICMR_DIRECT_BASELINE.md。

2026-09-28T01:43:04Z：direct baseline源码c08f5c654dd8600ad1ff5ee2cc1afb15380a0183
已推送，精确SHA CI36366650505 success15/15；Ubuntu3.10/3.14各2045pass1skip，
后者127warnings。Linux authored 实受限 scorer/gate 集成执行；CLIP/融合仍替身，
未测正式数据/真实权重/生成模型。只授 ENGINEERING_PASS_AUTHORED，不授
M1/Pilot/进程级ranker隔离/全角色forecast/ICMR效果/论文。见
research/ICMR_DIRECT_BASELINE_CI.md；历史费用/default4/无timer/其他四仓不变。

2026-09-28T02:15:19Z：新增 Linux-only search ranker 持久 Landlock 进程，
只读 search/inputs，模型/图库索引留子进程；独立 scorer 读 gold。
fresh 审查复现并关闭整帧 timeout BLOCKING；Windows 定向40pass6skip，
WSL纯管道3pass但完整受限 worker 在 drvfs 路径失败，精确 SHA CI 待验。
只自制替身，不是新源码真实CLIP或官方数据。M1/Pilot、全角色成本、
selection/final、ICMR效果仍NOT_ADMITTED。见research/ICMR_RANK_ISOLATION.md；
历史账本/default4/无timer/其他四仓冻结不变。

2026-09-28T02:34:43Z：ranker隔离源码769b05a736a9c7d8dc7e97188a81cd316efda440
精确CI run36369807411 success15/15；Ubuntu3.10/3.14各2052pass2skip，
Linux authored受限ranker/scorer/gate与失败收据已实际执行。本增量只授
ENGINEERING_PASS_AUTHORED；真实CLIP在该进程、正式数据/许可、全角色成本、
M1/Pilot/ICMR效果仍NOT_ADMITTED。见research/ICMR_RANK_ISOLATION_CI.md；
WSL旧失败、历史费用/default4/无timer/四仓冻结不变。

2026-09-28T03:22:04Z：真实 first-party CLIP ViT-L/14 已在 WSL 原生
Landlock search-only ranker 中执行，3自制图/2自制查询，独立受限
scorer返回自制hit；inner 8/8、已知encoder forward5，outer3/3，
58.455116s checker wall。两次前置失败及仅ranker `/dev/urandom`、
`/proc/cpuinfo` 单文件只读修复均留痕。Linux focused45pass1skip；
Windows full/精确SHA CI另记。fresh same-family/provisional审查发现的
失败结果保存blocker经回归与唯一复审关闭。只授 authored 工程资格，
正式CIR数据/许可、官方parity、captioner、双模型、全角色成本、
M1/Pilot/论文效果继续NOT_ADMITTED。见research/ICMR_REAL_RESTRICTED_CLIP.md；
default4/历史账本/无timer/其他四仓冻结不变。

2026-09-28T03:35:38Z：上述受限真实CLIP源码精确提交
`bf627a6b1be27b0ae06219cf0e0be1a5f30fc876` 已推送；CI run36373923799
15/15 success，Ubuntu3.10/3.14各2057pass2skip，coverage94.04/94.05%。
CI不运行932MB真实权重；真实CPU自制输入另由private WSL运行证明。
官方数据/许可、full-gallery parity、captioner/双LLM、全角色费用、
M1/Pilot/论文效果仍NOT_ADMITTED。见research/ICMR_REAL_RESTRICTED_CLIP_CI.md。

2026-09-28T04:37:25Z：ranker 新增 caller-supplied target description 数据
路径，真实 WSL CPU 第一方 CLIP/Landlock 在三张自制图上跑通直检和描述检索，
独立 scorer 与重放/身份失败测试通过；详见
research/ICMR_DESCRIPTION_TRANSPORT.md。描述是作者自写，绝不可称为
Qwen/OLMo 生成或正式 CIRR/FashionIQ 结果；本地全套2046pass20skip、
coverage94.03%，精确SHA CI待记。
M1/Pilot/C1–C3、正式数据许可、full-gallery/tie parity、captioner、
双模型与全角色物理成本仍 NOT_ADMITTED；ARIS 默认4轮、历史预算、无timer、
另外四仓冻结的覆盖指令不变。

2026-09-28T05:11:08Z：caller-supplied description 增量精确 SHA
`42ba46f4563a78dbf944fd9c0c6282074bb53b76` 的 CI run36380294671
completed/success15/15；Ubuntu3.10/3.14各2064pass2skip、coverage
94.04/94.05%。见research/ICMR_DESCRIPTION_TRANSPORT_CI.md。
CI 不运行私有 CLIP 或正式数据；描述仍自写，不是Qwen/OLMo结果。
M1/Pilot/C1–C3以及官方许可/评分/全角色成本继续NOT_ADMITTED；目标
仍 active。默认4轮、历史预算、无timer、四仓冻结不变。

2026-09-28T05:45:24Z：新增 input-only 固定 caption+修改文本→原双模型
`PersistentModel.metered`→受限 CLIP 完整排名的 authored 接线，
`cir_description` task cap 256，旧三个 `TASK_CAPS` 不变；新增 cap 改变新
backend profile digest，旧历史账本不清零。Windows全套2059pass19skip、
coverage94.04%，Ruff/format/mypy/configuredBandit/build/Twine通过。
fresh reviewer 因 agent thread limit 未创建，审查仅`[local-only]`；
SSH 现行 ED25519 指纹与先前用户核验值不一致，未连接服务器或跑真实模型。
正式captioner、双模型、图像/数据许可、全角色预算、M1/Pilot/C1–C3/
英文稿仍待准入；见research/ICMR_GENERATED_DESCRIPTION_BRIDGE.md。
goal仍active，ARIS默认4轮、无timer、其他四仓冻结不变。

2026-09-28T05:56:28Z：生成描述桥源码精确 SHA
`da9a4ba4b10ad9cca701e7db48d68755bd841231` 已推送；
CI run36383516624终态success15/15，Ubuntu3.10/3.14各2076pass2skip、
coverage94.04/94.05%；详见research/ICMR_GENERATED_DESCRIPTION_BRIDGE_CI.md。
CI只覆盖authored代码/构建，不运行私有Qwen/OLMo、CLIP或正式CIR数据。
SSH指纹未重新核验，fresh审查因agent thread limit未完成；M1/Pilot/
C1–C3/全角色预算/稿件仍未准入，goal仍active，default4/无timer不变。

2026-09-28T06:14:38Z：新增单一 Linux authored 测试，将模拟模型的
`PersistentModel.metered` 账本→Landlock ranker→受限 scorer→冻结 gate/
survivor 连成 64 查询的一次实际进程链；Windows全套2059pass20skip/
coverage94.04%，Linux新用例与精确SHA CI待验。fresh审查仍因
agent thread limit未创建，仅`[local-only]`。模拟 `_execute`、虚拟图库
不是 Qwen/OLMo/CLIP/正式数据结果；M1/Pilot/C1–C3不变。
详见research/ICMR_GENERATED_CHAIN_AUTHORED.md；服务器SSH指纹未确认，
未登录，default4/历史账本/无timer/其他四仓冻结不变。

2026-09-28T06:26:24Z：生成描述同计划authored链源码SHA
`a9707c441694e63646f62feaca1b4fc862d3c279` 已推送；
CI run36385678817 completed/success15/15，Ubuntu3.10/3.14各
2077pass2skip，较上个SHA增加1pass而skip不变，Linux用例实际执行。
只授`PASS_AUTHORED_LINUX_END_TO_END_CHAIN`，模型 `_execute` 与ranker
均替身；不证明Qwen/OLMo/CLIP/正式CIR数据或科学收益。
fresh审查与SSH身份未解决；M1/Pilot/C1–C3/完整预算/稿件仍未准入。
见research/ICMR_GENERATED_CHAIN_CI.md，goalactive/default4/无timer。

2026-09-28T07:03:20Z：固定 SEARLE 函数体对照新增自制 CPU 55-ID
全并列样例，CIRR5/FashionIQ4查询、17项指标及4次完整排列比较，
连同原严格排序17项均在私有固定源码上实际通过。WSL Python3.12.3/
Torch2.7.1+cpu最终run exit0；Windows定向43pass、Ruff/隔离mypy过。
报告v2保留`ties_qualified=false`与官方/full-gallery parity=false；
fresh审查因thread limit不可用仅local-only，精确SHA CI待验。
无benchmark/真实模型/GPU新费用；正式数据许可、全图库/GPU数值、
M1/Pilot/C1–C3仍NOT_ADMITTED。见research/ICMR_SEARLE_TIE_PARITY.md；
同一run/default4/历史账本/无timer/四仓冻结不变。

2026-09-28T07:11:20Z：上述本机原配置完整套件 exit0，
2058pass21skip/372.73s/coverage94.03%。相较前一次2059/20
有一项由pass变skip，原因未在本次原输出中归因；不把它算通过。
精确新SHA CI仍待提交；正式M1/Pilot科学状态不变。

2026-09-28T07:30:48Z：`b6faa72` 已推送，精确CI run36390612426
success15/15。新上下文只读同系列review 0 BLOCKING、2 NON-BLOCKING；
后续本地修正自制CIRR subset cutoff 区分力，R@1/2/3现为20/40/60%，
新增可选私有固定源码正向/词典序故障回归，WSL 4pass、Windows定向
43pass1skip。reviewer未复审后续diff；完整本机套件及新SHA CI待验。
此仍仅自制CPU函数体对照，正式图库/GPU/许可/模型、M1/Pilot/C1–C3
全部NOT_ADMITTED；SSH指纹未重核验，未连接。见
research/ICMR_SEARLE_TIE_PARITY.md、refine-logs/EXPERIMENT_CODE_REVIEW.md。

2026-09-28T07:37:37Z：审查后diff本机完整套件exit0，2058pass22skip/
490.24s/coverage94.03%；新增可选私有源码测试Windows按设计多1skip，
WSL有固定私有源码时4pass。Ruff/format/mypy src/配置Bandit/build/
Twine通过，但单文件AST exec B102未消失。新SHA CI待提交后核验；
正式M1/Pilot科学状态不变，见research/ICMR_SEARLE_TIE_PARITY.md。
