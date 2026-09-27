# ICMR 方向早期新颖性核查

记录：2026-09-27T21:21:00Z；ARIS novelty-check，同一研究记忆/default4/科学round0。
检索：3核心claims各至少3种query；补查Scholar/Semantic Scholar域名、第一方arXiv、
ICLR2025/2026、NeurIPS2025、ICML2025/2026。包含2026年4月至9月的近期来源。
这是定向检索与相关章节核查，不是穷尽数据库、逐篇完整阅读/证明审计或基线复现。

## 结论和可核验定位

初次fresh-context gpt-6-astra/xhigh评审：5/10、PROCEED WITH CAUTION。
通过此次早期筛查，可保留方案；5分锚点是有明确近邻但有值得Pilot验证的差异。
没有找到一篇已经包含三项精确结果的论文，不据“领域拥挤”ABANDON。
same-family/provisional、独立性false；不是科学接受，novelty仍UNESTABLISHED。
完整请求/原响应/路径澄清私有trace001/002保留。新HbBoPs来源续查与原意见一并保留，
不是重启四轮或择优重评。trace003原记忆续查仍5/10/CAUTION，无新精确结果碰撞。
HbBoPs已含结构对整段文本消融和真实配对筛选，因此仅typed胜flat仍不足够。

一句话定位：在冻结backbone的组合图像检索监督prompt搜索中，检验显式父子编辑
表示的经验边际作用，能否在合法随机的固定池paired审计和原生优化器里，转成
相同完整资源预算下的未见检索改善。没有真实数据支持前，只能写“检验是否”。

## 三项待测内容与具体近邻

| 内容 | 最近的已知机制 | 尚可检验的具体差异 |
|---|---|---|
| C1 显式typed parent→child edit×query | SOPL模板/role/约束特征，HbBoPs模块结构kernel，MASPOB工作流图，ProEval学习prior | 对完整serialization/相同父信息/拟合数据、容量、调参与计算，且超过无显式编辑关系的模块化结构控制；不是新增信息 |
| C2 fixed finite-pool paired Delta/R | aLTT有界风险/e-process，RG-PT先验＋可靠性检验，Cer-Eval分区评测，已有无放回推断 | candidate输出揭示前冻结分层/计划/SRSWOR顺序；真实二元净改善与无条件退化联合判断。既有推断工具的设计，不是新风险原理 |
| C3 native optimizer CIR后果 | POES评测调度、CAPO racing、HbBoPs多fidelity选择，GEPA/MIPRO原生minibatch，CIReVL检索链路 | typed收益在计入fit/proposal/reflection/caption/encoder/index/cache/full-survivor/failure/cold/idle后仍存在，并提升未见CIR |

不同estimand或保证范围只说明证书不能互换，不能借此豁免效率/错误判定/选择效果
比较。固定池证书仅原query-micro二元端点，不覆盖FIQ category macro、其他cutoff、
多真值AP、未来安全或campaign-wide5%。完整文本可能已经编码结构，故主张归纳偏置。

## 优先相关工作

| 第一方来源及版本 | 核查到的实质重合 | 阅读与比较边界 |
|---|---|---|
| [SOPL 2501.03508v1](https://arxiv.org/html/2501.03508v1) | 多类型prompt特征＋Bayesian regression/KG关联评测 | 主/peer读§3.1–3.2；C1主要近邻，不再abstract-only |
| [HbBoPs ICML2025](https://proceedings.mlr.press/v267/schneider25b.html) | structural-aware deep-kernel GP＋Hyperband；instruction/exemplar模块 | PMLR出版元数据＋[作者v2](https://arxiv.org/html/2412.07820v2)§3/4.3/5.3/E.4；原出版PDF获取失败，不称逐字一致 |
| [RG-PT NeurIPS2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/38853020527b0f84187870114ff7a686-Abstract-Conference.html) | prior/留出学可靠性DAG，DAGGER控制；包含prompt实验 | 相关方法/先验错误/应用章节；不依赖先验准确性保证，否定首次结构＋可靠性 |
| [POES 2604.11328v1](https://arxiv.org/html/2604.11328v1) | IRT区分度、覆盖与warm-start调度进优化循环 | abstract/related/§3–4.1；其子模目标不是当前证书，仍须强经验对照 |
| [aLTT 2409.15844v2](https://arxiv.org/html/2409.15844v2) | 自适应测试/提前停、FWER/FDR；明确prompt实验 | 用v2的§5.2，不将缺少该应用的v1混引 |
| [CAPO AutoML2025](https://proceedings.mlr.press/v293/zehle25a.html) | prompt population racing、paired t-test、长度惩罚 | 出版元数据＋[arXiv v1](https://arxiv.org/html/2504.16005v1)方法；出版PDF获取失败，不称全文；其检验非当前证书 |
| [ProEval 2604.23099v1](https://arxiv.org/html/2604.23099v1) | 历史/表示GP prior、BQ选点 | related/§2；学习先验不是首创，预测均值不替代真实认证 |
| [Cer-Eval 2505.03814v1](https://arxiv.org/html/2505.03814v1) | 主动分区、组内统计和区间减少认证成本 | peer§3–5，主定义段落；不能只以population risk不同排除效率对照 |
| [NLLP2025 Proxy Prompt Evaluator](https://aclanthology.org/2025.nllp-1.18.pdf) | prompt/input correctness logistic/MLP进MCTS；含构建调用的break-even | §4.3/5.3，已有fit成本先例；它还用gold label输入，原协议不能当本项目input-onlybaseline |
| [CIReVL 2310.09291v2](https://arxiv.org/html/2310.09291v2) | caption→LLM目标描述→CLIP检索 | §3.2；该链路不是新发明。我们的训练标签prompt搜索不能冒称training-free |
| [MASPOB 2603.02630v2](https://arxiv.org/html/2603.02630v2) | topology-aware GAT、LinUCB、coordinate ascent | §3；结构是agent topology，非query级typed edit，但实质表示先例 |
| [Contrastive Reflection 2606.30840v1](https://arxiv.org/html/2606.30840v1) | 成功/失败slice、定向编辑与回归约束 | peer相关方法；不称首次结构化反馈＋回归检查 |
| [CoRR ACL Findings2026](https://aclanthology.org/2026.findings-acl.1120.pdf) | CIR候选反馈、caption优化、历史query融合 | 主相关方法/任务定义，强系统近邻；本文CPU核来自SEARLE，不是复现CoRR |
| [OSCAR 2602.08603v2](https://arxiv.org/html/2602.08603v2)、[CoCo-IR 2608.05149v1](https://arxiv.org/html/2608.05149v1) | 检索动作MIP/示范与多轮交互TIE | 任务/方法定位；非当前单轮prompt-change机制，不把arXiv venue自述当已核验出版状态 |
| [Stop Guessing 2607.08522v1](https://arxiv.org/html/2607.08522v1)、[optstop 2608.14425v1](https://arxiv.org/html/2608.14425v1) | group-sequential与Bayesian稳定性停止 | 各相关方法；原假设/终点不同，不声称首次知道何时停止 |
| [WoR confidence sequences](https://arxiv.org/pdf/2006.04347)、[2026 WoR区间](https://arxiv.org/html/2603.14423v1) | 有限总体随机次序/同时区间/超几何工具 | peer理论相关段；非项目发明，未完成全部证明审计 |

未将所有源码/超参数/许可适配成可执行对照，也没有任何我们实际方法成绩。
Google Scholar/Semantic Scholar使用域名定向web检索，不声称其完整数据库导出。
部分长页面输出截断，报告只认实际可见相关段落；不会因此声称完整阅读。
追加[Coin Flip作者v2](https://arxiv.org/html/2604.14585v2)的研究/局限/附录C已核查，
属CTB@ICML2026 workshop，不称主会或已复现。其PROSE有五组件定向变异和
风险调整fitness，没有当前精确paired证书。原OpenReview PDF browser challenge保留。
HbBoPs还共用同stage随机实例、递增superset/cache，并从真实full/highest-fidelity
候选选incumbent，不能称它non-paired或完全proxy最终score。其配对效果在保留
superset的最后预算比较中未显著，不夸称普遍配对收益；预算是evaluation calls，
不是已核验all-role。结构对block文本消融已有，必须把无编辑关系的模块控制入C1。
这些新增定位事实来自原方法，不是我们实际结果。

## 决定性检验与下一设计要求

真实训练侧完整父子表只给scorer，各审计臂只见同一合法随机前缀；完整text
serialization与所有允许父信息均对齐，fit样本、容量、调参/计算预算匹配。
原certifier保持epsilon=.01/r_max=.05/alpha=.05、32slots/8strata/6looks/
tail allowance。结构置乱是训练侧机制诊断，不拿final选择适配。

测转移预测、错误判定/INCONCLUSIVE、clear-eligible保留、真实cost（含fit）。
先确认同certifier下typed优于完整text及强模块结构控制；只胜flat或uniform不足。
现六configs与未实现模块控制的识别差距须在训练侧设计审查记录；不自行添第七配置。
对近邻做
原算法与当前协议适配的明确区分；主动/确定性选点不能直接塞入SRSWOR证书。
机制表replay不等于在线native GEPA/MIPRO最终收益，后者需真实搜索并补完
survivor全评分/费用，最终效果证书端点与主效果指标不可混用。

六configs/原5seeds/两模型、门限和跨版本总资源不改；新增对照/诊断映射、
成本/配对统计和许可需训练侧冻结及独立科学审查，不是本次执行新矩阵。
这次未进行M1、图像/模型或Pilot，不主动追加小quota，也不重置科研round。

## 诚实的写作边界

不写首次结构先验、首次可靠prompt优化、首次correctness proxy、首次拟合费用
核算、首次CIR改写或新风险控制理论。英文稿要说具体归纳偏置/机制的未知价值，
包括失败条件和break-even；若真实增量不成立，保留有效负结果而不包装为成功。
选中CIR只是设计主题对齐；empirical venue fit/scientific admission尚未建立。
本轮software1978test/CPU排名资格不证明新颖性。公开论文例子/表格接触增量已
入ACCESS_AUDIT，不复述IDs或用作开发，不新增完全盲态声明。
