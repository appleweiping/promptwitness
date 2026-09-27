# ICMR 已发表评分核实际验收：严格排序通过，并列反例保留

记录：2026-09-27T20:17:00Z。同一 ARIS run/default4/scientific0，goalACTIVE，无定时器。
这是 ICMR v0.1 M1 前置组件的 authored CPU 验证，不是 M1 全部通过或真实图像结果。

## 已实际执行

新增 `reproduce/check_retrieval_native_metrics.py`，由合作操作者提供固定 SEARLE
`a9c314ba4b6e6e14be5a9c3fdf6b66e6a0c23e37` 的原 validate.py。仅选择并编译
两个原 metric 函数 AST，保留原函数体和 decorator，prediction generation 用
手写 CPU 特征替代；未导入原数据/模型/main。这是可信外部代码执行，不是 sandbox。
原源码及 CC BY-NC4.0 许可只在私有研究 stage，不 vendoring 或改为我们 MIT。

- 新 CPU 声明 canonical `6687ae36`；实际 Python3.10.15、Torch2.7.1+cu118、
  NumPy1.26.4，与全部50 pins相符。既有 task-owned 环境复用，无安装/重建/权重访问。
- CPU1/GPU0，CUDA visibility 空、线程1、offline；CPU2x2 matmul 前检43585 exit0。
- 原 SCP 因库存错误包含不存在的 namespace `reproduce/__init__.py` 而 exit1。
  改正库存后的原32848 exit0；23610 roundtrip逐字节比较6个源码/许可/配置文件全相同。
- source051：gpt-6-astra/xhigh，same-family/provisional，无具体 BLOCKING/NON-BLOCKING。
  reviewer 既有 Python3.14 实测47passed/18.16s、四文件 Ruff通过；未执行远端资格。
- 主作者原48422一次terminalexit0；checker wall0.4262390269432217s。
- fresh-doc052原54921按文档原样一次terminalexit0；checker wall0.09774209395982325s。
  model/effort inherited、canonicalfamilyunknown/independenceunverified；不声称跨家族。
  仅 aggregate 输出，combined stream无另列stderr；没有独立读取真实成本账本。
- 每次11 strict-order查询、15数值比较、5次原native CPU metric函数调用。
  验证 CIRR 全库R@1/5/10/50、subset1/2/3，FIQ三类别R@10/50及类别宏平均。
  unequal-category query-micro与category-macro明确不同，binarygate不保护宏平均。

## 必须保留的数值反例

同一手写 all-tie 几何上，原 Torch argsort 得到 CIRR global R@1/5/10/50
全部0%，独立标量核的 lexical tie policy 全部100%。两次实际调用均观察到此差异。
`PASS_STRICT_ORDER_AUTHORED_ONLY` 不掩盖它：tie_policy_qualified=false，
official_benchmark_parity_established=false、scientific_admission=false。
后续实际检索必须明确固定真实排名器与 scorer 数值/tie语义，并验证 near-tie、
完整库和图像向量；不得把标量诊断的字典序偷偷当 native 排名，或把反例删掉。

## 实际软件质量与保留失败

主作者新util/scorer局部47passed/33.26s，其中14新util tests不执行Torch原函数。
该复合命令之后读不存在的__init__.py导致整体exit1；pytest子步骤通过不等于整命令通过。
初始Ruff I001已按源码改正；最终Ruff/format374、mypy61src、configuredsrc Bandit通过。
新两个helper的Bandit实际exit1：保留1 MEDIUM/HIGH-confidence B102(exec_used)，
用于执行明确指定的可信原函数，非可接受任意不可信文件的安全界面；不加ignore/nosec。

上一轮whole39297 actual1957passed6skip/2013.77s/94.03%是旧源码证据。
本次新whole80877正在运行，未用预计1971数字宣称通过。

原isolatedbuild40858依赖安装长期停滞，ProtocolError/OSError access violation
writing0x48日志保留；仅结束已确认owned安装child，build实际失败。
随后旧sdist Twine通过使复合命令terminalexit0，不能冒充wheel/build成功。
本机无setuptools metadata，不执行无依赖校验的no-isolation捷径。
改用既有uv installer+官方PyPI，原39828实际sdist+wheel成功、两包Twine通过。
实际两阶段setuptools84.0.0满足>=77；uv.CMD的%*产生空重定向文件77，
已确认是本次owned artifact并移入私有归档，无删除或共享脚本修改；此gotcha保留。
新build是不同安装路径的实际修复，不擦除原失败或把工具最后exit0等同每步成功。

上一实际交付99876b8 exact CI36345101338现completed/success/15of15；
27f90c5和e95bd09旧CI成功仍只属于它们。新增helper尚未提交/推送，
不把旧SHA CI或纯src覆盖率叫新helper原生parity/科学证据。

## 成本与科学边界

物理历史DB原字节26a7e04b...未变：1170calls/2136046input/195222output/
3.8074371029887377allocatedGPUh，historicalunknowncount未编码，不推断0。
本次新realcalls/input/output/allocatedGPU/paidAPI均0；CPU checker wall已列，
没有把它当全pipeline的完整CPU分摊账本或未来forecast。

无benchmark record、图像/encoder、模型推理、final读或测试上传。
CIRR NLVR2原图访问仍需用户按第一方流程确认；FashionIQ exactCDLA/image
source链未澄清。当前只补查固定仓库根目录路径：FashionIQ没有LICENSE文件，
CIRR有LICENSE；路径元数据不授权图像。未读取caption/split/test/demo内容。
公开CIRR例子接触仍在ACCESS_AUDIT，不改完全盲态声明。
强CIR baseline/captioner/weights/真实index/final封存/online/native/统计/全角色
forecast/M1/Pilot/0of120确认runs/ICMR英文稿均未完成；不能提交论文效果声明。

## 2026-09-27T21:11:00Z：真实 CPU 数值前置终态及查新决定

详见 `research/ARIS_ICMR_CPU_RANKING.md/json` 与
`idea-stage/ICMR_NOVELTY_CHECK.md/json`（均相对仓库root）。主23828与fresh-doc
41132各原样一次exit0，四authoredcase/8原metriccalls/8完整72-ID排名/36比较；
8malformed拒绝，原gate64/4/failed1attempt0score。51pin ownedCPUlayer87663299，
71stagefiles逐字节往返一致。source053同家族provisional；doc054身份unknown/
independenceunverified，aggregate CUDA值/各rank列表未另公开限制保留。
原42147全套1978passed6skip/1014.27s/94.03%src；Ruff387/mypy61/srcBandit/
新3helpersBandit过，旧exec B102 MEDIUM不隐藏。原15144两包build/Twine过。
原80877终态1971/6/1159.27s仅bc源码；bc8304e3 exactCI36347625475 success15/15。
新代码记录时未提交，不借旧CI取通过。成本DB26a7e04b...本轮bytes核对未改；
1170历史/unknown未编码保留，新model/token/GPU/paid0，CPU全开销未完整计量。

typed-edit增量需在完整预声明文本/父信息和匹配拟合/调参/容量/计算后检验；
结构置乱仅训练侧诊断。现成统计工具不是新风险定理；不同estimand不豁免
强对照；CIR标签搜索不是整套zero-shot。强近邻适配和额外机制检验尚未冻结，
不自动替换原六configs/5seeds或新增已运行矩阵。真实数据准入/M1/Pilot/方法效果/
全角色forecast/原生在线引擎/统计/英文稿仍缺，同一goalACTIVE/default4科学0。
