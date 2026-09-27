# ICMR 实际视觉编码器 CPU 前置验收

记录：2026-09-27T22:00:00Z。只通过 authored-images 机械运行，goal ACTIVE；原default4/
科学round0、同一ARIS run、无timer、其他四仓冻结。详见同名 JSON 的原receipt。

## 实质实现及实际运行

新增离线 `ClipCPUEncoder`：第一方 ViT-L/14，224px/77token；官方 preprocessing/
tokenizer，float32/eval/no-grad、每item batch1。image输出保持raw给既有ranker
做唯一gallery normalization；textquery仅归一化一次。原rank算子与tie约定未改。
超长text不截断、不记录forward或zero成绩；失败输出/attempt保留、旧目录拒写。

主73988与fresh-doc10832原样各一次terminalexit0；三张自制 RGB/RGBA/L pattern
images、两自制英文查询。每次image6/6/6 attempt-forward-complete，text6/4/4；
三image与两text反序bitwise相等、两个完整3-ID排名相同。无semanticaccuracy阈值
或tuning，不把此小图库结果转为CIRR/FashionIQ成绩或任意online/GPU一致性证明。
主checker40.6710608550s；doc40.0723752792s。没有captioner或生成LLM调用。

## 来源、许可与环境

OpenAI source `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`，8指定code/tokenizer/
notice文件只在私有stage。code为[MIT](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/LICENSE)；
[模型卡](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/model-card.md)列研究intended use，
并警示未经in-domain testing的image-search/部署。未确立weights也为MIT；此次仅
私有非部署研究运行，不再发布checkpoint、不接受terms，不授formal许可准入。
原loader实际下载932768134B，SHA256 b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836，
不是expected冒充actual。下载+验证51.0110s；software粗粒度18s，CPU开销非0。

原50pins/SciPy层只读，新增owned TorchVision/Pillow/ftfy/wcwidth层；完整55pins
实匹配fbea8bf4。CPU1/GPU0、emptyCUDA/线程1/offline，seeded8x8 dispatch通过。
15原stage文件逐字节往返一致，源码保留版权与许可、不vendoring为本项目MIT。
原build47723terminal0；tool进度中段截断、stdout/stderr合流的观察限制保留。

## 审查与软件验证

新上下文source055发现真实double-normalization blocker，旧针对测试1failed/
14.74s后最小修复；同记忆唯一followup056无blocker。same-family/provisional。
freshdoc057仅doc/provider/launcher原样复跑，无repair/retry/install；canonical模型
身份unknown/independenceunverified，仅执行验收，不独立审计源码或scientific acceptance。

最终59targetedpassed44.38s，Ruff/format409/mypy61/configuredsrcBandit/新两helper
Bandit通过。未配置srcBandit的11旧LOW原样保留，不冒充configured结果。package
79353两0.2.0包build成功，首次Twine误查旧0.1.0；94088实际检查两0.2.0包通过。
9530完整pytest仍运行、collection在normalization修复前，WMI日志保留，不能称
final-source全套通过。旧27c2139 exactCI36351707078实际15/15成功，不代新encoderCI。

## 成本与科研状态

两次共12image+8text=20真实CLIP CPU forwards，24inputattempts含4拒绝。
新增generatorcall/token/GPU allocation/paid0，不写“所有modelcalls0”。CLIP tokenization
未完整计量；build/download/SCP/QA/metadata/CPU总开销未完整计量，非totalcost0。
历史generatorDB26a7e04b...本轮重核未改，1170/2136046input/195222output/
3.8074371029887377GPUh；unknowncount未编码非0。已知generator+encoder保守
调用和1190另列，不重写历史DB。尚不授all-role forecast或120runs成本准入。

没有benchmark annotation/image下载、CIR研究参考/M1/Pilot、科学效果、通用online
资格或英文稿；fusion/captioner/强对照/真实全库仍待冻结。CIRR原图需用户完成
NLVR2第一方访问，FashionIQ exactCDLA/source仍pending，不用mirror/换模型绕过。
下一步是数据权限/分组封存/强controls与原native all-role真实链路，不重复此fixture。
