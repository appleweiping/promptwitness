# 已发表检索 metric 核一致性检查（authored CPU）

此独立 qualifier 运行 SEARLE 固定 revision 的原 `cirr_compute_val_metrics`
和 `fiq_compute_val_metrics` 函数，核对本项目独立评分公式。不是 dataset 官方
测试服务器/原 whole CLI/完整检索模型、图像或科学实验资格。

第一方源码：[SEARLE validate.py](https://github.com/miccunifi/SEARLE/blob/a9c314ba4b6e6e14be5a9c3fdf6b66e6a0c23e37/src/validate.py)，
项目为[CC BY-NC4.0](https://github.com/miccunifi/SEARLE/blob/a9c314ba4b6e6e14be5a9c3fdf6b66e6a0c23e37/LICENSE)。
原文件和许可只能放入受相应许可约束的私有研究 stage，不当成我们 MIT 源码。
该 wrapper 不 vendoring upstream code；由合作操作者明确提供上述可信源码文件。
AST 仅选两个原 metric 函数，保留函数体与 decorator；future annotations 避免
为 type names 导入 CLIP/Dataset。未执行原 top-level imports/main/data loader。
仅 prediction generation 替换为手写 CPU feature tensors；不是安全 sandbox。

所需软件声明为 `configs/research/retrieval-metric-cpu-env.json`，仅 CPU1/GPU0、
CUDA visibility 为空、单线程。既有已登记 task-owned 50pins 软件可按 metadata
确认 warm-reuse；不要求在本机未装 Torch 的 package venv 安装新依赖或改共享环境。
无 weights、image/text encoder、benchmark records、最终测试访问或 inference。

从有 Torch/NumPy 的已核验代码 stage，设置以下**进程级**环境变量，再原样执行
一次（`pinned-native/validate.py` 为上面指定原文件，`new-output` 必须不存在）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m reproduce.check_retrieval_native_metrics pinned-native/validate.py new-output
```

checker 验证11个 strict-order authored queries、CIRR cutoff/reference/subset 与
FIQ3category宏/微差别，native metric百分比和独立fraction*100逐项核对，float32
舍入允许绝对1e-5。另真实观察一个 all-tie source kernel；不因恰巧相等自授tie
策略资格。只在无错误时写 `PASS_STRICT_ORDER_AUTHORED_ONLY` aggregate。
资格预计15个数值比较、5次 native CPU metric 函数调用；仍须实际 receipt，
不得把这份文档或本机util测试当实际native执行通过。

输出仅新目录 `qualification.json`；失败保留原目录/错误，不原样重跑或覆盖。
它不改旧成本账本，不证明官方评分/近tie数值/GPU排名/强baseline/真实M1/Pilot。
实际访问/许可、fullpool image encoding、sealed final和全角色成本门禁仍独立。
