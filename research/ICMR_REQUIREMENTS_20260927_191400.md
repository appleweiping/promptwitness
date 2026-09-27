# ICMR 投稿写作约束

官方格式核验：2026-09-27 17:36:45 UTC；方向更新：2026-09-27 19:14 UTC。
本文是写作约束，不是论文、研究完成或投稿就绪证明。原版本单独保留。

## 目标与官方规则

按用户明确指定的 ICMR，登记 **ACM International Conference on Multimedia Retrieval 2027**。
暂按 regular long research paper 规划；长文类别是本项目的规划选择，不是用户已明确选择的 track。
不自动替换成 ICML，也不混用 ICMR 同名会议或往届规则。

当前官方 [Call for Contributions](https://www.icmr2027.org/cfc) 要求：

- 长文正文至多 8 页，参考文献至多另加 2 页，额外页面只能用于参考文献。
- 短文为正文至多 4 页、参考文献至多另加 2 页；不自行把本项目降为短文。
- 两类均不允许附录或 supplementary materials。
- 使用 ACM 双栏 proceedings 模板；LaTeX 文档类为 `\documentclass[sigconf,review,anonymous]{acmart}`。
- Regular paper 为双盲；去除可识别作者的署名、致谢与网站链接。
- 该 CFP 未列独立的总字数或摘要字数上限；不得编造硬性字数标准。最终以编译后的正文及参考文献页数验收。

官方 [Important Dates](https://www.icmr2027.org/date) 当前列 regular full/short 截止为
**2027-01-31 23:59:59 AoE**（UTC 为 2027-02-01 11:59:59）。这是当前发布值，不是永久冻结的日期。
CFP 目前尚未给出完整 submission 操作细节；不猜测投稿系统或 camera-ready 格式。
写作开始和实际提交前须再次核验官方 CFP、日期、模板与最新作者政策，记录变更后采用届时规则。
CFP 的 ACM 模板链接本次返回 HTTP 403；文档类由 CFP 本身确认，未声称下载或编译了模板。

## 对本项目写作的落实

英文稿正文必须自洽：核心方法、保证的假设与适用边界、实验设置、主要比较和关键局限
在正文中完整交代。不能把必要证明、采样条件、访问披露或失败结果挪到被禁止的附录、
补充文件或网页来规避篇幅。研究复现文件仍保留，但不当作额外送审材料。

内部暂分配正文 8 页：问题与动机 1 页、相关工作 0.75 页、问题定义 0.75 页、
方法 1.75 页、实验 2.25 页、分析 1 页、局限与结论 0.5 页。这是可随证据调整的
编辑预算，不是会议规定；不靠缩字号、改页边距或删实验条件凑页数。

写作交付需包括英文 LaTeX 源与可编译 PDF、实际页数核验、图表可读性检查、匿名检查，
以及按 ARIS paper-writing 的 claim/proof/citation 等审查与真实 assurance 判定。
每个结果数字绑定实际证据；负结果照实呈现，CI 和 authored fixture 不转写成方法收益。
本次已明确提出写作要求，不能以后仅以旧的默认 `AUTO_WRITE=false` 遗漏论文交付；
但这不更改上游 skill 默认设置，不跳过研究证据冻结及写作准入。

## 主题匹配与当前状态

ICMR 的官方主要范围是多媒体与多模态检索。现有固定 IFBench、HotpotQA、BFCL
任务并未直接验证该范围，不能仅换标题就声称多媒体贡献。
用户随后明确确认本会议并要求改进主题匹配，已授权实质方向调整。
当前主设计见 `ICMR_SCOPE.md`：CIRR/FashionIQ 组合图像检索，以参考图像和修改文本
检索目标图像，检验结构变化感知评测的同风险调用节省与同预算检索质量。
当前 `venue_fit=DESIGN_ALIGNED_EMPIRICAL_PENDING`、`manuscript=NOT_STARTED`。
CPU 手写向量排名/评分/gate 机械验收不是图像实验、方法收益或主题匹配实证。

登记会议不授权暗中新增数据集、模型、模态、基线、预算或改变冻结研究问题。
此次方向授权已取得；新 v0.1 增量方案登记实际资源影响与未测量 forecast，仍不是科学冻结。
旧数据访问、失败与已发生费用保留；CIRR 官网意外测试样例接触详见 `ACCESS_AUDIT.md`。
新方向必须完成许可、评分原生 parity、访问隔离、在线/全角色成本及 M1/Pilot 准入。
不伪造 Pilot GO、确认实验、创新性通过或投稿就绪；默认 ARIS 四轮不变。
实际提交和可能的出版费用/机构覆盖需另外核验；本轮不提交、不支付、不联系会务。
