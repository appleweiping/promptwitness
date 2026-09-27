# 真实参考向量接入：已实现，原 native 进程运行中

记录：2026-09-27T15:55:49Z。直接goal ACTIVE，无PromptWitness定时任务；默认ARIS4轮、科学0轮不变。

新增 `RealPipeline` 连接既有限制推理进程、实际连续账本与官方GT阶段worker。
完整reference/selection请求在响应前冻结；失败或未决请求不重放，不补分数，不双重预约。
selection保留原survivor/调用者规则；incremental复用原driver的单次预约。现有科学identity
绑定实际training输入/annotation/scorer字节，可信controller不解释标签、不读取final。
模型benchmark文件读授权0，完整input-only请求经pipe进入；ABI1合作边界局限不变。

fresh CODE_REVIEW039无BLOCKING，53authored测试通过51.96s，same-family/provisional。
唯一nonblocking历史未知成本披露已补入摘要，localunknown0不能冒充全部历史0。
旧5个Mock fixture失败、E501和错误metadata路径观测均保留，未被后续成功覆盖。

原完整HotpotQA/Qwen reference256组件，fresh文档上下文原样一次启动session37849，
当前仍运行；没有真实terminal或accuracy汇总。本轮不重放旧6fitwire，也没有按结果缩小人口。
只读GPU/事件元数据确认实际生成活动，不打开响应/标签/per-ID分数；事件数不是完成样本数。
所有cold/idle/scorerwait/exit沿实际ledger计费；当前使用更高历史账本，而非重置历史。
最新**先前已闭账**快照914calls/1784463input/193444output/3.6819725114061215GPUh，
不代表进行中job新增成本为零或当前总消耗仍只有该数。新job闭账后再报实际更高总量。
historicalunknowncount未编码、保守token成本及旧IFBench预览披露均保留。

独占source/private training store传输、SHA匹配提取、现有CPU CLI --help真实exit0；
3tasks各512fit256search256selection不变，25files；不改共享环境或其他任务。
最终source archive829533c12339a0335311830fb7fb0fd7db3f259c10332fc4d8999e5f2fc63b21，
只含publicPython；完整store/响应/DB/trace始终私有。runtime沿用原核验code/resource-only stage。

本机完整pytest原87326 exit0：1897passed6Windows限制skip478.69s，覆盖率94.04%。
helpers+incremental463passed3skip53.52s；Ruff315、mypy61src、configuredBanditsrc及新2helpers
Bandit通过；isolatedbuild实际sdist+wheel、Twine两包通过。安装wheel smoke本轮未跑。
前50b709e exactCI36328917670 success15/15，Ubuntu3.12实际1883passed161.92s94.06%，
不是当前未提交source的CI；新提交exactSHA CI随后独立核验。

真实fullreference本次尚未终态，selection/nativeGEPA/MIPRO/ONLINE_PINNED/fullroleforecast
未完成；M1UNFITTED、PilotNOT_RUN、确认0/144pairs0/180runs0/36transfers，方法效果未测。
这次source实施与native运行不是实验桥全部完成，更不是科研接受或有效负结果。
