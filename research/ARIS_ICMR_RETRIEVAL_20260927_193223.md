# ICMR 组合图像检索方向与 M0 实际验收

记录：2026-09-27 19:18:39 UTC。同一 active goal/ARIS run 接续；不是研究完成。

## 实质改变

用户已确认 ACM ICMR 并要求改进主题匹配。新 `ICMR_SCOPE` v0.1 将主问题从
三个纯文本/工具任务改为 CIRR/FashionIQ 组合图像检索：参考图像+修改文本→
同一固定 reference caption→两个原文本生成模型→目标描述→固定 text/image
encoder 与 fusion→全图库排名→目标 ID 评分。两个生成模型共享信息，不把
Qwen 额外视觉能力作为优势；exact encoder/captioner/融合与数据尚未科学冻结。

检验 C1 同风险更少实际调用、C2 同全角色预算更好 held-out 检索。已有 LLM
改写/规划 CIR 相关工作不作新颖性声明。拟新矩阵96pairs/120runs/24transfers
全为 NOT_RUN/NOT_FROZEN；旧144/180/36保留为未运行历史，不作先决实验或完成。
论文仍须真实证据、独立统计及 ICMR 当届8正文+至多2参考页英文稿，无附录。

## 已实现与实际验证

- `reproduce/retrieval_scoring.py`：独立 CPU full-pool cosine 排名/固定融合、
  单目标 gold 严格检查；CIRR 去 reference 与 subset R@1/2/3，FashionIQ 保留
  reference 的 R@10/50，以及 category macro 和 query micro 分开。
- 二元 gate 只验证 CIRR query-micro hit@5 / FIQ query-micro hit@10；不授予
  macro、其他 cutoff、multi-positive AP 或未来用户安全保证。
- `check_retrieval_scoring.py` 接原 paired gate/journal/survivor：参考与候选
  都来自实际手写向量排名+gold ID，不用 predictor 补全 survivor，不将缺失
  排名记为零。CLI 只接受新目录，失败/成功都关闭 journal。
- 主作者 original15705：新增33+原incremental，共135passed/134.50s，exit0。
  初次33passed后 Ruff 的2zip-strict+2line findings保留，已修后再验收。
- reviewer 实际另测33passed/104.64s；source048/唯一followup049无已证实
  阻止 M0 的 BLOCKING/NON-BLOCKING；same-family/provisional，不跨家族接受。
- 主作者一次 authored CLI exit0/32.2339146000013s；fresh doc 原93654一次
  exit0/25.09740210000018s。trace050继承设置/familyunknown/independence
  unverified；只复验允许的 aggregate，不称独立 journal/uniqueness 检查。

| authored case | early attempts | total attempts | observed scores | logical episodes |
|---|---:|---:|---:|---:|
| eligible | 4 | 64 | 64 | 128 |
| rejected | 4 | 4 | 4 | 68 |
| failed | 1 | 1 | 0 | 65 |

这些是64单元/手写向量的 CPU 机械事实，不是实际图片/encoder/模型输出、
调用节省、Pilot 或科学形式保证。没有新模型/GPU/token/paid 消耗。

Ruff whole tree通过/format355files已格式化，mypy61src通过，configured src
Bandit及新2helper Bandit通过。原完整pytest39297及isolated build/Twine40858
在本记录时仍运行，不能称通过；WMI 0x8007000e诊断保留，不重启抹除。
HEAD c9f85da CI15/15成功是旧树，不替代这些新文件的 exact-SHA CI。

## 成本、访问与未完成项

实际成本 DB `26a7e04bbe7f256df6796877cdfb24c5e86cd57495268491804687d44d88bf93`
字节未改：1170calls/2136046input/195222output/3.8074371029887377allocatedGPUh。
historicalunknown count仍未编码，不据此推断为0；新矩阵 forecast 未测量非0。
总上限与20% reserve公式不变；不加小quota、不减default4，科学rounds0，无timer。

官方 CIRR 许可页自动返回 public test annotation 示例的意外访问已记入
ACCESS_AUDIT：不转入模型/优化器/fixture、不声称全部盲态，独立污染/访问
审查待做。CIRR 原图需用户完成 NLVR2 条款访问流程；FashionIQ 精确 CDLA
edition/原图来源与视觉权重许可未定。无下载/接受条款/联系/上传/付费。

下一顺序：数据/权重许可与 source→训练分组+sealed final→native评分 parity→
实际 image/text encoding/index/caption/cache及全角色成本→M1/online/native优化器→
Pilot→准入后新矩阵/统计→英文论文。其他四仓冻结；旧历史全部保留。
`venue_fit=DESIGN_ALIGNED_EMPIRICAL_PENDING`、`novelty=UNESTABLISHED`、
`scientific_admission=NOT_PASSED`、`manuscript=NOT_STARTED`。goal ACTIVE。

## 2026-09-27 19:32:23 UTC：原完整测试终态与实际推送

上述19:18:39运行中快照保持历史；原完整pytest39297现在终态exit0：
1957passed/6skipped/2013.77s，configured src覆盖94.03%，达到原90%门槛。
WMI异常日志保留。最终Ruff pass/format366files已格式化；与前355快照均为
当时真实输出。未重启完整测试或修改门槛/skip/timeout取通过。

实现提交 `27f90c530d53d2796804b00b35b334180e213247` 已实际push并核对remote，
对应[CI36344286371](https://github.com/appleweiping/promptwitness/actions/runs/36344286371)
实际completed/success。包含distribution build、三平台wheel smoke及测试/quality。
后续仅来源审计文档提交 `e95bd093f75357f7d43e887981dc30d788825b4a` 也已实际push，
其CI36344686524在此时间queued，不冒充新exactSHA已过。full代码未变化。
本机原build40858仍无终态，remote distribution通过不改写本机结果。
只提交显式35个public文件及4个来源审计文件；无data/weights/private ledger/trace。

第一方CLIP exactsource与上游预期权重identity见ICMR_DATA_AUDIT新版本；未下载/
实际编码。数据权限/污染审查/native评分/真实费用/M1/Pilot/新矩阵/稿件仍缺，
新科学效果没有产生。default4科学0/同一run/no timer/历史账本/goalACTIVE不变。

原191839机器报告错误地对schema标识与Ruff格式结果重复使用`format`键；
本版本将后者改为`ruff_format_check`，原文件保留。此为报告元数据修正，
不是实验代码/结果/评分改变；最终JSON另做严格duplicate-key检查。
