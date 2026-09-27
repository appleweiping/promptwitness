# ICMR 单查询 CPU 排名：限定实际验收

记录：2026-09-27T21:11:00Z。同一 ARIS run/default4/scientific0；goal ACTIVE，不是科学结果。

## 改进与实际执行

旧 lexical/native all-tie 反例保留，不改旧函数。新 `rank_float32_cpu` 使用已 unit 的
CPU float32 query、原图库行顺序、gallery normalize、`1-dot` 和原 Torch argsort；
不再次 normalize query，不改为 `-dot` 或 lexical tie。single query batch1。
原 [SEARLE validate.py](https://github.com/miccunifi/SEARLE/blob/a9c314ba4b6e6e14be5a9c3fdf6b66e6a0c23e37/src/validate.py)
两 metric bodies/decorators 未改，只用 witness 观察实际 argsort 返回，不改返回值。
原 source/LICENSE 保留私有 CC BY-NC4.0；AST 执行是可信代码，不是 sandbox。

新增可复现说明：`reproduce/RETRIEVAL_CPU_RANKING.md`，实际规格51pins。
原50pin Torch环境只读；实际缺 SciPy 的 gate 依赖，在独占 task-owned CPU layer
安装 SciPy1.15.3（官方PyPI/no-deps），不是改共享Python/CUDA或重跑模型。
actual Python3.10.15/Torch2.7.1+cu118/NumPy1.26.4，完整 metadata87663299匹配。
CPU1/GPU0，offline/thread1。explicit71代码/许可/env文件实际上传并逐文件往返byte核对。

| 实际调用 | 原session/终态 | checker wall | 每次覆盖 |
|---|---|---|---|
| 作者一次 | 23828/exit0 | 1.1561418620403856s | 8原metric calls/8全72-ID排名/36cutoff比较 |
| fresh-document一次 | 41132/exit0 | 1.1156352909747511s | 相同四case、8 malformed拒绝、3gate路径 |

strict/all-tie/near-tie/reversed-gallery-tie全部case匹配；不同float32 dot确实
坍缩为相等 `1-dot`。这修复本路径具体算子与 tie 语义，不证明任意near-tie、
GPU/batched/FAISS或真实encoder相同。旧全局native0%vslexical100%反例仍有效。
原gate真实authored survivor补全64score、rejection4、failed1attempt/0score；
logical episodes128/68/65不是模型calls，也不写真实费用DB。

## 新上下文与质量证据

source053无具体BLOCKING/NONBLOCKING，gpt-6-astra/xhigh same-family/provisional。
54本地no-cov tests/21.64s；首个继承coverage门槛的局部调用exit1保留，非测试失败，
不充作全套通过。fresh-doc054只读允许文档和launcher，原样一次，无代码检查或修复；
模型继承、canonical family unknown/independence unverified，不称跨家族接受。
stdout aggregate没有单独打印CUDA空值和各72-ID列表；launcher/checker机械执行这些
约束，不能声称doc执行者独立观察了未公开列表或stderr归属。

最终作者54targeted/13.02s；原42147全套terminalexit0，1978passed6skipped/
1014.27s；configured **src** coverage94.03%（不冒充reproduce模块覆盖率）。
Ruff/format387、mypy61src、configured src Bandit通过；新3helpers339LOC/0issue/
nosec0。旧external loader B102 MEDIUM/HIGH-confidence原样保留，不屏蔽。

原15144真实uv.EXE+官方PyPI isolated build，sdist+wheel及两包Twine实际通过，
两阶段setuptools84.0.0。reproduce模块走source checkout，不承诺wheel-only CLI。
旧40858实际build失败/最后Twine terminal0与owned空77 artifact仍保留，不擦除。
上一原80877已终态1971passed6skip/1159.27s/94.03%，仅属于bc8304e3。
该提交exact [CI36347625475](https://github.com/appleweiping/promptwitness/actions/runs/36347625475)
已actual success15/15；本报告新代码在记录时未commit，不借旧CI声称新源码CI通过。

## 成本、访问与未完成

真实DB SHA26a7e04b...本轮再核对字节未变，沿用1170calls/2136046input/
195222output/3.8074371029887377allocatedGPUh；historicalunknown未编码非0证明。
本次新增模型/token/GPU/paid0；CPU/SCP/PyPI/QA开销未完整计量，不说总成本0。
新矩阵仍NOT_FROZEN/NOT_RUN，未预支 early-rejection收益。

没有新benchmark annotation/image或weights下载、encoder、在线embedding、M1/
Pilot/确认实验。原图合法访问、FashionIQ exact许可链、final封存、强检索系统/
原生优化器、全角色forecast、统计与英文稿仍待准入。公开论文及CIRR官网示例接触
见ACCESS_AUDIT，不声称developer完全盲态。CPU parity和CI不建立ICMR研究效果。
