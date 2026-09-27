# 推理进程文件访问边界

`PersistentModel` 现在强制使用 Python `-I` 和标准库 bootstrap；不通过普通 module
入口先导入应用。启动消息只有 model/snapshot/data_root/scratch。worker 重建固定 policy，
验证单线程、先施加 Landlock，再绑定已知 source 叶目录、导入 Torch 并加载模型。
所有 benchmark 文件读取授权为空；原始输入只能从已有 pipe 的完整请求进入。
无内核支持、错误 policy/receipt 或加载失败均终止原 child，不回退为无限制进程。

沿用已有 `process_access` 实现，不改变原 scorer/predictor/optimizer 的数据叶授权。
模型 weights snapshot 和精确 symlink blob FILE 可读，不授予整个 HF cache ancestor；
解释器、source 和系统库只读可执行，写入限本次 scratch、固定 CUDA/null 字符设备，
以及本推理进程自己的 `/proc/self/task` 文件元数据子树。后者允许 CUDA 给新驱动线程命名，
只授予文件写入（支持时含 truncate），不授予创建、删除、重定位或执行；不会开放其他 PID。
这是整个 own-task 子树的文件写授权，不是按 `comm` 文件名匹配的授权，必须如实披露。
系统元数据只读授权另含精确文件 `/proc/cpuinfo` 和 `/proc/sys/vm/mmap_min_addr`，不是整个 `/proc`。
进程不继承凭据/PYTHONPATH/数据 descriptors；HOME、TMPDIR、CUDA/HF cache 指向 scratch。
固定 GPU UUID、bfloat16/sdpa、singleton、context32768 和输出 caps 保持不变。
仅设备节点 `/dev/nvidia0`（已登记 UUID）、ctl/uvm/uvm-tools；不使用其他 GPU。

这仍是合作操作者的研究文件边界，不是恶意本地用户 sandbox 或 air gap。
本机 Linux5.15 的 Landlock ABI1 不封闭 network/stat/truncate/ioctl；不得据此声明完整隔离。
机制依据 [Linux 官方 Landlock 文档](https://docs.kernel.org/userspace-api/landlock.html)。
controller 是否发送允许的 input、是否包含合法 fit demonstrations、是否完成阶段科学门禁，
仍由研究 controller、配置 freeze 和独立访问审查负责。

## 文档实际验收

warm-reuse 原 Torch@150b5834/Python3.10.15 环境；不重建、安装或改变共享 CUDA/Python。
新独占 code stage 只含已核对 public src/reproduce.py；无 Git/raw data/records/traces/credentials。
fresh context 只获得本文、compute contract、private provider ledger 和 literal invocation。
先执行 CPU access probe：11 个 authored benchmark 叶，各读/写被拒绝、sentinel 不变；
scratch 可写、原始模型 config symlink 可读、应用在限制后导入；不导入 Torch或分配 GPU。
该 probe 单独通过不能证明真实 CUDA 加载和生成也可运行。

随后顺序 Qwen/OLMo：仅使用之前按固定 input-only hash 预选的一条 fit 输入/任务/模型，
不重放旧 proposer fixtures。六次调用用于新限制下的实际 CUDA/task-wire 验收，不是阶段额度。
所有调用沿用 context/caps、在含已发生失败与诊断成本的连续实际账本上预留和计费，
原 908 次调用不归零，cold/idle/child exit 全部计入。
完整 wire/response、ledger、per-ID 记录留在 private outputs，只输出汇总。
不打开 gold store/scoring/final body，不把 matching/greedy/singleton 视为 ONLINE_PINNED。
可信 controller 仍解析原十行 prepared 文件，其中未执行的 proposer fixtures 含此前已披露
的 fit-gold examples；不能声称 controller 完全没加载标签。模型进程只收到三个 input-only
fit rows，不收到这些 unused fixtures。这不是新的 gold-store 或 final 数据访问。

```bash
cd /media/lenovo/data2/promptwitness-delta-20260926/aris-model-access-20260927/code-repair-2
export PYTHONPATH="$PWD/src:$PWD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
# 按 private provider ledger 的完整原样调用，先 CPU probe，再顺序两个模型。
```

private invocation 指明固定请求、snapshot、已关闭历史 DB SHA、continued DB、独占输出和
当时 GPU idle 预检查；本公开文档不包含用户端点或私钥。fresh executor 必须逐行原样一次，
保留原 process/session 和所有失败，不能修改、重试、覆写旧输出或停止其他任务。
若失败先读 actual stderr 定位具体层，再按 ARIS 的一次修复复审与环境流程推进。

首次 fresh-doc 运行的两次 CPU probe 通过，但 Qwen CUDA 启动失败，OLMo 未开始；零新生成，
启动分配时间已实际计费。针对性 seeded kernel/file trace 复现错误；仅补两个系统元数据读取后，
读取成功但 CUDA 仍失败，剩余拒绝包括 own CUDA thread 的 `comm` 写入。这不是成功验收。
原始 source、两次诊断失败、退出码和连续成本均保留；第二处 own-task 修复的实际 seeded
kernel 已通过：shape8x8/finite/RTX A6000、正常退出、完整 allocation 关闭。NVIDIA 尝试
访问另一 GPU 和部分非必要系统路径仍被拒绝；没有为它们添加授权。之后新的模型 fresh-doc
按四份文档原样一次完成，原 SSH exit0、实测268.0048149s，两模型各三个实际 input-only fit
wires；完整分配增加0.07292376862631889 GPUh/4299input/202output，连续账本实际914calls。
两次 CPU probe 各11read/11append denied，真实模型也在限制后加载生成。汇总捕获的合并
stdout/stderr 没有 warning/traceback，但未单独 instrument stderr。完整证据见
`research/ARIS_MODEL_ACCESS.json`；same-family/provisional，仅是当次机械资格，不是科学准入。
真实 M1、Pilot、online/statistical audit、原生 optimizer engine 和 144/180/36 均未完成。
已有 MODEL_TRANSPORT.md 的历史十次调用对应其当时已部署 source，不是此新 boundary 的证据；
当前 run_model_transport 入口新增必需 `--data-root`，旧 source/archive/results 不改写。
