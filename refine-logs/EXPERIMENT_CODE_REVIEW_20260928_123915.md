# Selection 真实 CLIP 资格脚本代码审查

记录：2026-09-28 09:10:03 UTC。新上下文、只读 `codex exec`、
`gpt-6-astra`/xhigh 两次；因代理入口 `agent thread limit reached`，
改用 CLI 新会话。请求/回复/元数据保存在忽略的
`.aris/traces/experiment-bridge/2026-09-28_run01/004-*` 与 `005-*`。
同属 OpenAI 系列，`same-family/provisional`、非独立科学验收；审查没
运行测试、模型或联网。

初审：1 BLOCKING——selection 描述评分仅提交单查询，真实 scorer
要求完整人口；替身测试错误容忍。1 NON-BLOCKING——scorer 失败
stderr 未留存。修复为两条 description 排名及一次完整 selection 评分，
测试约束同真实 scorer；正常非零退出的 stderr 写到私有 scratch。
唯一 follow-up：BLOCKING 无；确认 search 单查询诊断仍合法、selection
完整人口及 10/12 内层操作、6/7 forward 计数一致。

Follow-up 另报两项非阻断：第二条描述排名失败时第一条成功结果原先
没有写入报告；已在审查后补逐次写入并加失败回归，**未再复审**。
`launch_role` 超时或零退出后 JSON/身份验证失败的 stderr 不受新增
留存分支覆盖；不得宣称所有失败都持有 stderr。真实 WSL/CLIP 资格
与本地测试另见 `research/ICMR_SELECTION_REAL_CLIP.md`，不是 reviewer
亲自执行的证据。正式数据、许可、M1/Pilot 和科学主张仍未准入。

## 2026-09-28T09:55:11Z：fit input-only ranker 审查

因 `spawn_agent` 报 `agent thread limit reached`，使用新会话只读
`codex exec`/`gpt-6-astra`/xhigh；原请求、回复和元数据保存在私有
`.aris/traces/experiment-bridge/2026-09-28_run01/006-*`。同系列
`same-family/provisional`，不作为独立科研准入。审查 0 BLOCKING，
1 NON-BLOCKING：新 fit Linux 集成用例只覆盖完整人口成功，未覆盖少查询
失败与外层收据。主作者随后在该用例补真实受限 scorer 的少查询拒绝、
外层 `failed=1` 和 stderr 留存；Linux 最终 2 passed/10 deselected。
这一后续纯测试补丁未再由 reviewer 复审。审查未亲自执行测试或真实 CLIP；
作者的私有 WSL 运行及完整状态见 `research/ICMR_FIT_REAL_CLIP.md/json`。

## 2026-09-28：检索 MIPRO 原生接口两次只读审查

`spawn_agent` 仍报 `agent thread limit reached`；改用隔离的
`codex exec`/`gpt-6-astra`/xhigh，只读请求和回复保留在私有
`.aris/traces/experiment-bridge/2026-09-28_run01/007-*`、`008-*`。
首轮 1 BLOCKING：固定 DSPy `Example` 不继承 `Mapping`，reference
也会被拒；2 NON-BLOCKING：伪造的剪枝排名不是合法 CIRR miss，
以及混合/乱序、补全失败和 demo 差异覆盖不足。主作者改为原对象下标
读取 unit ID，用12项完整图库与真实指标函数，补上述用例；固定环境
真实 Example 的 strict helper 探针 exit0。第二次只读静态复审未发现
新 BLOCKING，确认前三项在源码中关闭；审查者没有独立复跑11测试、
完整原生编译或查看其私有输出，仍是 `same-family/provisional`。
作者另执行两条自制 full/minibatch 原生 compile exit0，详情见
`research/ICMR_NATIVE_MIPRO_RETRIEVAL.md/json`；不能合成科学准入。

## 2026-09-28T11:48:28Z：检索 GEPA 原生搜索两次只读审查

因现有 agent slots 已满，使用独立只读 `codex exec` / GPT-6-Astra
xhigh 会话；请求、回复及模型来源元数据保存在忽略的
`.aris/traces/experiment-bridge/2026-09-28_run01/009-*`、`010-*`。
同系列 `same-family/provisional`，不是科学独立接受。

初审发现 1 BLOCKING：原生 `subsample_indices` 声明 8 项却只给 1 项
score，或 outputs 全 `None`，旧适配仍可接受。修复为 indices、两侧
score/eval-score 和 output 一一等长、非空、二元及非 `None`，并补两项
仓库回归。复审从当前源码抽取原方法做 41 个纯内存畸形样例，均在
improvement 与 audit 之前拒绝；固定上游原 proposal 使用 list 容器。
复审判定上轮阻断关闭、0 新 BLOCKING、1 NON-BLOCKING：before 输出、
单项缺失、长度和 eval scores 不一致、不改善却不完整等已在内存证伪，
建议后续固化为仓库回归。复审未自行执行完整仓库 pytest 或原生 checker。

审查限定为 fresh run、原生 reflective、`AllImprovements` 与禁止 merge；
控制器不强制这些全局配置。脚本化 proposer、进程内 authored scorer、
预置 reference 不构成真实在线或全角色成本资格。实际作者测试与私有
qualification 的记录见 `research/ICMR_NATIVE_GEPA_RETRIEVAL.md/json`。
正式图像、许可、Qwen/OLMo、同次受限 scorer、M1/Pilot 继续未准入。

## 2026-09-28T12:39:15Z：检索 GEPA 真实受限 scorer 接入复审

因 agent slots 已满，沿用独立只读 `codex exec` / GPT-6-Astra xhigh；
请求、回复和来源元数据在忽略的 `.aris/traces/experiment-bridge/
2026-09-28_run01/011-*`、`012-*`。初审 0 BLOCKING、2 NON-BLOCKING：
失败前报告即声称 worker 已验证；中途失败未写已完成计数。已修为顶层
失败默认 `false`、两场景成功后才 `true`，并在 `finally` 快照 reference/
native/audit 成功评分数、成功 worker 收据及失败/未决账本。新增两条辅助层
回归；作者另在私有 WSL 做实际接线故障注入，启动前失败及受限 reference
64 条完成后失败都保留准确报告。

复审用 7 组只读内存故障探针确认上述两缺口关闭、0 BLOCKING、
1 NON-BLOCKING：自动测试尚未直接覆盖完整 `check/_run_scenario` 报告接线，
未来改动仍可能绕过辅助函数；私有故障脚本不在 CI。复审未亲自运行
WSL 原生 GEPA/Landlock、真实模型或完整 pytest。`scorer_worker_launches`
在失败报告中只表示成功返回且验证的 worker 收据数，不包括失败启动。
同家族 `same-family/provisional`，非独立科学验收；实际作者运行与限制见
`research/ICMR_NATIVE_GEPA_RESTRICTED.md/json`。优化器仍是可读 gold 的
可信父进程，正式图像许可、Qwen/OLMo、全角色成本、M1/Pilot/C1–C3
均未准入。
