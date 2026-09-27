# 完整 training-store 审查与真实文档验收

日期：2026-09-27 10:26:22 UTC。CODE_REVIEW fresh gpt-6-astra/xhigh，
no blocking/no nonblocking findings，same-family/provisional，trace018。
Reviewer只读代码/授权metadata，不打开raw corpora/GT/archive/final。

实际metadata核验三任务512/256/256、BFCL额外214final-derived与五类分布，
未重复旧文本 final_derived 错误。源码保留完整上下文/约束；先校验membership，
再读取training正文；全部fit数据与旧store相同，复制合法已有fit子集响应，
不按正确率选择、不伪造缺失标注。exclusive输出、partial I/O失败保留，receipt
最后写入，不能作科学评分许可。未建议新hash/compat/feature flag等无据机制。

Reviewer首次局部pytest13tests执行通过但默认全包coverage失败exit1；随后
--no-cov局部验证exit0。全程未修源码、未执行真实准备，原始trace保留。
Authored BFCL fixture覆盖simple_python/irrelevance；另外三类由metadata分布和
代码路径核对，不声称新测试对全部五类逐项native评分。

fresh /root/aris_training_store_doc只获doc/ledger/invocation，PowerShell两行
原样一次exit0/toolwall7.93127秒、无stderr或doc divergence；trace019同家族/
继承Codex/provisional。实际counts全3072inputs/gold、fit相等、58旧fit响应；
metadata-only stat25files9241430B，四个应空leaves确空。无final内容读取。

主作者13tests/wholehelpers249passed1skip、lint/format/mypy61src/Bandit过。
新helperBandit0issues；早期F401和reviewcoverage失败保留。未跑新full local
package/build，历史失败不改。当前新源码exactSHA CI待交付，不继承旧CI。
这不是角色IPC/controller/freeze验收、M1/Pilot或科研接受。
