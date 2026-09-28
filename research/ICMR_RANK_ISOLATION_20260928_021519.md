# ICMR search ranker 进程隔离，2026-09-28 02:15:19 UTC

状态：`AUTHORED_PROCESS_WIRING_PENDING_EXACT_SHA_CI`；不是 M1/Pilot 或论文结果。
此增量专门缩小上一版 input-only 直融基线的进程边界缺口。

`retrieval_search_ranker` 只获 `search/inputs` 数据 leaf 读取权限、独立 scratch
写权限及必要运行时读取/执行权限。Linux Landlock worker 在导入 ranker 应用前
实施限制；清理继承环境和文件描述符，并将该 CPU 基线的 CUDA 可见设备设为空。
图像和 checkpoint 的相对路径必须留在 input leaf 内。可信控制端建立持久会话，
受限子进程装载一次模型/共享索引，只接收 query ID、返回完整排序；scorer 是
另一个只持对应 search gold 权限的进程。这里的角色与榜单属于 search，
不声称 fit、selection 或 final 的同等接线已经完成。

初版实现经 fresh-context 审查发现一个 BLOCKING：只对响应首字节做 `select`
超时，半帧 `readline` 和大初始化帧的阻塞写可越过期限。审查者以真实管道
复现；已改为非阻塞 `os.read/os.write`、同一个单调 deadline 覆盖每次完整
请求与响应，并在异常时终止/回收子进程。复审未发现新的可复现 BLOCKING，
但仍是 same-family/provisional，不等于独立科学审查。新增半帧、背压、
超时回收和真实受限 authored worker 的回归。停顿回归要求 outer 物理尝试
记为失败而非零分，子进程已预留的 inner 尝试保留 unresolved/forward unknown。

本地 Windows 新会话与相关定向测试 `40 passed, 6 skipped`（复审时源码）；
本地完整 pytest/coverage `exit 0, 94.03%` 始于最后一轮超时修复之前，
不能充当最终字节验证。独立 WSL Linux 三项管道超时回归 `3 passed`，
但完整 Landlock worker 在该 WSL 的 `/mnt/d` 路径上发生旧有
`promptwitness/__init__.py` PermissionError，且环境缺 SciPy/Torch，
因此不报本机 Linux 受限会话通过。精确 SHA Linux CI 在提交后核对。
最终修改后的全仓 `pytest --no-cov -q` exit 0；最后六文件 Ruff lint/format、
ranker/process helper 定向 mypy 和 `git diff --check` 通过。Bandit 报四项
subprocess 相关 LOW（B404/B603）、无 MEDIUM/HIGH；保留警示而非压制。

全部受限端到端测试只用操作者自写 query/gold/虚拟图片与 checkpoint，
worker 由 `AuthoredRanker` 替代，不是 CLIP 数值验证。没有读取/下载官方
CIRR/FashionIQ 图像或 annotation、没有运行新进程内真实 CLIP、生成模型、
GPU、付费 API 或正式检索评测。CPU 测试开销非零，未作为全角色费用预测。
旧公开样例接触、合法图像/标签权限、分组审计、正式强 baseline、完整
controller/selection/final/ONLINE_PINNED、保守全角色成本和科学准入仍未完成。
ARIS 默认四轮、历史费用账本、无定时任务和其他四仓冻结状态不变。
