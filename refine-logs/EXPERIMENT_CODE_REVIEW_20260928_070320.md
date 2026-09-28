# Experiment bridge 代码审查：SEARLE 自制 CPU tie 差分

记录：2026-09-28 07:03:20 UTC。状态 **`[local-only]`**，不等于
fresh 独立审查、更不等于科学接受。按 ARIS `CODE_REVIEW=true` 请求
gpt-6-astra/xhigh fresh-context reviewer，协作工具返回
`agent thread limit reached`，未创建 reviewer，也没有编造其 verdict。

主作者检查当前 diff `reproduce/check_retrieval_searle_metric_parity.py`
及负向测试：固定 SEARLE `validate.py` SHA 在 AST 编译前核对；原函数体
不进入 MIT 源码；自制无并列样例未删；新全并列 55-ID 样例的单查询
排序与该固定函数体使用的批量 Torch 距离/argsort 表达式完整排列一致，
且原函数体实际指标对照全部通过。CIRR 删除 reference、FashionIQ 保留
reference；二者不互相套用。原始失败保留 `FAILED_RETAINED`，新增
`authored_cpu_tie_fixture_checked` 在检查完成后才置 true；
`source_metric_function_bodies_executed` 在原两类函数无并列对照完成后
置 true，不误称 tied fixture 也成功。`ties_qualified`、完整图库和
官方服务器 parity 仍 false。

本地未发现新的 BLOCKING 逻辑错误；该判断仅限当前自制 CPU fixture。
尚缺 fresh review、真实图库/GPU tie parity、正式数据与许可、captioner、
双模型、在线原生优化器、全角色成本。单文件 Bandit 对既有固定 AST
`exec` 报 B102 Medium/High；SHA 匹配不构成 sandbox，未屏蔽。
详情与实际运行证据见 `research/ICMR_SEARLE_TIE_PARITY.md`。
