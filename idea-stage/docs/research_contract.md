# 研究合同：PromptWitness-Delta

此合同从已有冻结方案提取，不重新选题、不代表新颖性接受。
来源：`research/PROTOCOL.md`、`research/scope.lock.json`、
`research/PROTOCOL_v1_1.md`、最新用户授权 `research/ARIS_GOAL.md`。

## Selected Idea

检验 prompt 编辑结构是否在同样的文本、parent score 和允许的 parent trace
之外提供额外的正确性转移信号；该信号是否在同等决策质量下减少评测调用，
并在同等全角色资源预算下帮助原生 GEPA/MIPRO 找到更好的 prompt。
选择源自用户已固定方向，不存在新 `IDEA_REPORT` 排名或已成功 Pilot。

## Core Claims

以下均为待验证假设，不是现有结果：

1. 结构特征具有超出 text-only 的转移预测/抽样价值。
2. 固定分层、无放回随机审计、精确超几何计数区间可以在其已核验假设内
   作有限总体的风险受控判定；不声称未来用户安全或全 180-run 的总体 5% 保证。
3. 在相同 proposer、模型、约束、评分器与总预算下，获得真实成本和优化效果收益。

## Method Summary

M1 使用可追溯的编辑结构、固定文本特征、输入特征、缺失标记和允许的
parent 信息，拟合轻量 logistic 转移头及成本模型。当前只有合成拟合，真实 M1 UNFITTED。
M2 在读取候选输出前冻结同质 reference 层、随机排列及单调分配，使用同时
超几何区间计算 Delta/R；M3 持久化候选 slot、观测计划、恢复、预算和判定。
当前已有 offline recorded-table 核心，在线语义和原生 early rejection 仍未完成。

## Experiment Design

- 数据：BFCL-derived 五个 Python offline 类别；HotpotQA distractor；IFTrain-to-IFBench。
- 模型/规模：沿用两主模型家族、第三模型固定迁移、种子 11/23/37/53/71；
  144 mechanism pairs、180 main runs、36 transfers，完成计数均仍为零。
- 指标：二元机器可验证正确性、paired Delta、unconditional R、物理/逻辑资源成本，
  决策一致性及 run 级优化效果。HotpotQA 使用 answer EM，不用 F1/joint EM 偷换。
- 数学门限：epsilon=0.01、r_max=0.05、alpha=0.05，32 slots、8 strata、6 looks，
  每方向 tail allowance=1/61440；不根据成绩放宽。
- 资源：跨版本 800,000 calls、1,000 allocated GPU-hours、20 亿输入、2 亿输出；
  零付费。完整角色的剩余成本仍 UNMEASURED，不写虚构 GPU 小时估计。

## Baselines

原六配置及机制基线全部保留；原生 minibatch screening、ProEval、uniform、
metadata、同 certifier 的 text-only 和结构方案都需按实际协议执行。
当前没有可填写的真实优化基线成绩。

## Current Results

已有机械核心及成本探针，不是方法效果。真实 Pilot not_run、confirmation
not_started；具体历史见 `research/G1_STATUS_v1_1.json` 和真实账本。

## Key Decisions

评分器使用真实 dataset ground truth 和官方逻辑，不用模型答案充当 gold。
IFTrain 的 raw-text/all-constraints 和 BFCL 固定 JSON 接口适配必须明确披露，
且每个控制组使用同样的适配。最终样本不能用于开发；保留此前 card 预览披露。
按 ARIS 默认轮数续推、不复位既往成本；只完成有证据的工程步骤后才清相应门禁。

## Status

- [x] Idea selected（已有固定方向，非新 review acceptance）
- [ ] Baseline reproduced
- [ ] Main method implemented（offline core 已有，research integration 未齐）
- [ ] Representative dataset results
- [ ] Full dataset results
- [ ] Ablation studies
- [ ] Paper draft（AUTO_WRITE=false）
