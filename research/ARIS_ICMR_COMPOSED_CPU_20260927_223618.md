# ICMR 参考图像＋修改文本直接融合：真实 CPU 限定验收

记录：2026-09-27T22:36:18Z。这是同一ARIS run的B3无LLM图文融合机械基线，不是论文方法
效果、正式CIRR/FashionIQ成绩、M1或Pilot。goal ACTIVE/default4科学0、无timer、
其他四仓冻结。详见同名JSON与私有原receipt。

## 实际多媒体路径

第一方ViT-L/14对自制参考/候选图像及修改文本分别编码；参考图raw图库向量
归一化一次，文本沿原编码器已unit结果不重复归一化，再以预写定0.5加权并
归一化组合query，通过既有CPU核给完整候选图库排名。该权重只为机械验收，
未用成绩调优，也未在训练数据冻结；不声称检索效果。CIRR参考图排除属于后续
scorer，不在此层悄悄排除；FashionIQ需另保留。无gold/目标标签进入组合器。

主40503一次terminalexit0/checker38.707057795953006s；fresh-doc按文档一次
exit0/checker38.526855575852096s。各用三张自制RGB/RGBA/L图、两条英文修改，
6image＋4text真实forwards；3image/2text反序bitwise一致，两完整3-ID排名一致。
两次完整rank均为[red,blue,grey]与[blue,red,grey]；这是自制输入的可复跑
算子观察，不是以自制名称为CIRR/FIQ正确率或语义鲁棒性阈值。

55pin fbea8bf4/CPU1/GPU0/emptyCUDA/float32/batch1与上轮相同；无新依赖、
权重下载或其他模型。新两源码本地与owned服务器SHA逐项一致。source058
same-family/provisional无具体blocker；fresh-doc059只按公开doc/provider/launcher
运行，无repair/retry/install，model/effort inherited、familyunknown、independence
unverified，未独立检查源码、权重/个别pins或数据；合流流不证明stderr为空。

## 软件与成本证据

新模块局部67项：66passed、1skipped（本地无Torch数值项；服务器真实模型已跑）/
51.23s；reviewer另18passed1skipped/13.96s，原全局Python3.14尝试停在Torch
import且被终止，不冒充通过。Ruff/format420、mypy61、两新helper配置Bandit
通过。前一df53792提交的exactCI36354407932是15/15 success，仅覆盖前一编码器
提交，不代此次未提交融合代码。旧本地whole9530终态1981passed6skip1failed/
2830.81s/94.03%src；原Windows closed-pipe子进程wait15秒超时，单独重跑
1passed6.29s不能抹去全套失败；WMI异常日志保留。此whole在先前修复前收集，
也不作为本次最终源码全套通过。

两次融合验收新增20真实CPU encoder forwards（12image＋8text）、20inputattempts；
先前编码器验收也有20forwards。生成LLM调用/token、新GPU分配和付费新增0，
但下载/安装/CPU/QA总开销未完整计量，不能写totalcost0。原真实generator DB
SHA26a7e04b...重新核对不变：1170calls/2136046input/195222output/
3.8074371029887377GPUh，unknown历史数未编码不是0；已知generator＋两组件
保守forward和1210，不是完整全角色费用或120 runs预测。

[ICMR 2027官方CFP](https://www.icmr2027.org/cfc)确实以多媒体/多模态检索、
查询表示和评测为范围，8页主文＋最多2页仅参考文献且无附录。这里获得的
仅是问题链路机械可执行，不等于新意或venue实证匹配已经成立。CIRR原图仍需
用户第一方NLVR2访问，FIQ exact CDLA/image-source未明确；无benchmark原图/
annotation下载、无captioner/强CIR对照、正式全库native parity、封存final、
真实M1/Pilot或统计。权重私有研究使用不确立其MIT再发布许可。
下一步是合法数据/分组封存、训练侧冻结caption/fusion/强controls、真实
全角色native执行与保守forecast；不能重复此fixture充当论文结果。
