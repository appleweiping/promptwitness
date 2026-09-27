# 研究角色的实际进程读取边界

本模块落实 experiment-bridge S1 所需的文件读取机制，不是完整 S1 验收。
Linux Landlock 在 worker 导入应用代码前生效；不需要 root，不改共享环境。
不支持的系统直接报错，绝不退回无限制进程。Windows 的单元测试不证明
Linux 的限制已经执行。系统调用定义依据
[Linux 官方 Landlock 文档](https://docs.kernel.org/userspace-api/landlock.html)。

## 固定授权

数据根下必须建立下表叶目录；输入、gold 和历史记录不可混存。
运行时与可写 scratch 不能覆盖数据根；所有数据授权都只读。

| 角色 | 阶段 | 可读取的数据叶目录 |
|---|---|---|
| fit_learner | fit | fit/inputs, fit/gold, fit/records |
| optimizer | search | fit/inputs, fit/gold, search/inputs, search/reference |
| predictor | search | fit/inputs, fit/gold, fit/records, search/inputs, search/reference, search/parent |
| search_scorer | search | search/inputs, search/gold |
| selection_scorer | selection | selection/inputs, selection/gold |
| final_scorer | final | final/inputs, final/gold |

fit gold 是已许可的训练监督和示例；search/reference 只能放已完成的固定
reference，search/parent 只能放允许的已查询 parent 信息。当前候选的未查询
结果、selection/final gold 不得放进这些目录。路径授权不检查文件语义，
因此真实数据准备及线上消息接口还须独立验收。

可信 controller 必须依据科学准入、候选/配置 freeze 和阶段结束记录推进
selection/final；不能把任意命令行 stage 当作科学准入。本模块验证 role/stage
组合，不判断 controller 是否已经履行这些科学门禁。

`launch_role` 用结构化 argv、干净环境、`close_fds=True` 和 Python `-I`
启动 worker。只保留 stdin/stdout/stderr 管道；应用非零退出和异常照实返回，
不是错误答案零分。模块仅允许解释器安装、系统库、reproduce 源码和当前
角色数据；scratch 是唯一写入目录。runtime 目录不得包含语料、响应或 gold。
Linux ABI1 不限制网络、stat 和 truncate 系统调用；这不是 air gap，不能
限制同用户的未沙箱进程，也不防恶意本地操作员。不要扩大这个保证范围。

## 本次环境与测试命令

规格：`configs/research/process-access-env.json`。复用已登记的任务拥有 venv，
仅使用标准库；无需 pip 安装、不加载 torch/权重、GPU=0。
已查明服务器 Python 3.10.15、x86_64、Linux 5.15，Landlock ABI1。
这是能力探针，不是以下实际访问验证的通过记录。

部署只复制 `reproduce/process_access.py` 和 `reproduce/check_process_access.py`
到下述 staging 的 `reproduce/`，不复制结果、benchmark 数据或其他项目。
staging 源码与该命令创建的工作目录必须分离。下述命令必须首次运行，
已经存在的输出/工作目录会失败以保留历史，不覆盖旧结果。

```bash
cd /media/lenovo/data2/promptwitness-delta-20260926/aris-process-access-20260927
/media/lenovo/data2/promptwitness-delta-20260926/venv-torch271/bin/python -m reproduce.check_process_access /media/lenovo/data2/promptwitness-delta-20260926/aris-process-access-20260927/follow-doc-work /media/lenovo/data2/promptwitness-delta-20260926/aris-process-access-20260927/follow-doc-witness.json
```

期望 exit 0：六个实际 worker PID，各测试 11 个自编写哨兵文件，共 66 次读取、
66 次写入；允许读取仅限表中角色叶目录，全部数据写入必须被拒绝，角色
scratch 必须可写。探针不使用任何 benchmark 样本/标签或模型响应，
不能解释为模型分数、Pilot、真实数据隔离或完整科学准入。
完整记录保存在 exclusive JSON，异常则保留空/不完整输出及 stderr。

本机机械测试：

```powershell
.venv/Scripts/python.exe -m pytest --no-cov tests/reproduce/test_process_access.py
```

Linux native 测试在不支持 Landlock 的 CI 上会明确 skip；独立服务器
qualification 不允许 skip。先做新上下文代码审查，再按上面的原命令
进行 fresh-agent 文档验收，记录实际 kernel/ABI/进程输出。

## 尚未完成

真实分池 store、各 scorer 与 optimizer 的实际调用路径、只返回允许的
消息接口、selection/final 的阶段 receipt 尚未接好。BFCL fit-only real-GT
评分和在线执行仍待这些路径验收；此机制不能清除全部 S1/S0 门禁。
此前 final card 预览和 bundled final 下载的访问披露保持不变。
