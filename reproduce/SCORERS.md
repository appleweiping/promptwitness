# 官方严格评分的 CPU 验证

这份文档只验证 scorer 集成，不宣称 Pilot、模型效果或完整 GPU 环境 ready。
软件规格为 `configs/research/scorer-env.json`。只使用官方源码和已经冻结的
fit 侧历史成本响应；不打开 final 文件、上游 `eval/` 或数据卡样本。

## 评分语义与来源

- HotpotQA：直接调用 [官方 answer EM](https://github.com/hotpotqa/hotpot/blob/3635853403a8735609ee997664e1528f4480762a/hotpot_evaluate_v1.py)，
  不移除 reasoning 或抽取一个更有利答案，不用 F1/joint EM 替代二元正确性。
- IFTrain：使用 open-instruct 的 pinned `IFEvalG` classes，版本
  `99b1ee970490a2d0d5664663eb0b1acc410b944c`。明确适配为 raw-text、
  **all constraints**，不沿用其移除 thinking/fractional reward 的训练奖励。
- IFBench：直接调用 pinned `test_instruction_following_strict`，版本
  `1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d`；当前只运行自主编写机械 fixture，
  包含原生 OOD checker，未加载真实 final prompt。
- BFCL：固定 `name/arguments` JSON interface，转成官方 AST checker 的 decoded
  call 结构，精确保留名称。禁止 upstream Python-expression decoder 的 `eval`。
  这是明确、所有控制组共享的 JSON-only 适配；native irrelevance 为 absence of
  decodable call，不等于遵循 JSON 输出格式。独立环境的原版 runtime 验证见
  `BFCL_SCORER.md`；本命令仍只执行 Hotpot/IF 子集，不能证明 BFCL real-GT 门禁。

空的已完成文本可以按实际规则评分；缺失/failed/cancelled 响应、未知约束、
错位 annotation、scorer 异常和缺失的 native flags 都是错误/unsupported，不填零分。
原始输入/输出不因评分而被修改。随机 checker 构造 seed=11、语言检测 seed=11，
均须在最终 scorer 身份中记录；在线有限总体语义仍需独立验证。

## 环境和原文资源

CPU-only task-owned venv，不改主项目或共享 Python/CUDA。
用 `uv venv --python .venv/Scripts/python.exe <task-owned-scorer-venv>` 创建环境；
按 spec 的第一 pip phase 安装明确版本。IFBench import 内置了 NLTK 下载，
所以先显式通过 NLTK 官方 loader 准备 spec 中四项语言资源；运行时先核验资源存在。
Hotpot 的单个 scorer 文件从上述 pinned URL 下载到 `source/hotpot-scorer/`；
不要克隆或读取任何 Hotpot dev/final 样本。其他两个 pinned source 已在 source 下。
源码保持原文及其许可证：IFTrain/IFBench Apache-2.0；Hotpot 数据 CC BY-SA 4.0。
不复制第三方数据或代码进公共发布包。

## Windows 本任务实际调用

在 `D:/Company/nlp-original-projects/promptwitness` 运行（所列目录均为此任务已有
或刚建立的受控目录；不是 GPU 端点）：

```powershell
$env:PYTHONPATH = 'D:/Company/nlp-original-projects/promptwitness'
& 'D:/Company/research-artifacts/promptwitness-delta-20260926/scorer-venv-aris/Scripts/python.exe' -X utf8 -m reproduce.check_strict_scorers 'D:/Company/research-artifacts/promptwitness-delta-20260926' 'D:/Company/nlp-original-projects/promptwitness/results/summary/g0/training-pools-proposed-v2.json' 'D:/Company/research-artifacts/promptwitness-delta-20260926/strict-scorers-aris-follow-doc.json' --prepare-nltk
```

fresh agent 应原样执行命令并报告不一致，不自行修复。输出使用 exclusive create，
已有同名输出是保留证据，不覆盖；重试经审查的更改应给新输出名。
输出仅保存在受控外部目录，含 fit unit 的 checks，不推送原始样本/响应。
公共交付只发布汇总与 source/profile/环境说明，不把它说成模型准确率基准。
这一步不训练 M1、不拟合 sampling bins、不更改任务或候选；不清隔离/在线/原生搜索门禁。
