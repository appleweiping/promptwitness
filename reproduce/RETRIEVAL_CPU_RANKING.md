# 单查询 float32 CPU 排名与原 gate 验收

此路径针对已复现的 lexical/native 并列差异，不改变旧诊断或原 SEARLE metric body。
引用原 [validate.py](https://github.com/miccunifi/SEARLE/blob/a9c314ba4b6e6e14be5a9c3fdf6b66e6a0c23e37/src/validate.py)；
原源码与CC BY-NC4.0许可只在私有stage，不vendoring/改为本项目MIT。
复用 `check_retrieval_native_metrics.load_metric_kernels` 仅执行两原metric AST；
这是明确可信外部代码执行，不是安全sandbox。

`retrieval_cpu_ranking.rank_float32_cpu` 接收float32 CPU、已unit查询、完整图库及
其不可变行ID顺序；不再次归一化query。与两原metric对float32库相同，归一化
gallery，计算`1-dot`再Torch默认argsort；不换成`-dot`或lexical tie规则。
一query固定matmul batch1，全量/补全survivor也不能改Qbatch；embedding生成
的online语义另需验证。没有GPU/multiquery batch/FAISS/真实图像parity声明。
畸形、nonfinite、zero/degenerate features是失败，不是observed zero。

源码checkout stage须包含本项目完整src运行文件及5个reproduce模块；不是仅wheel
安装。原metric source为操作者提供的指定可信文件。软件WHAT为
`configs/research/retrieval-ranking-cpu-env.json`：CPU1/GPU0、固定线程1/offline、
既有50pins保持只读，新增SciPy1.15.3在任务专属CPU依赖层。按完整51pins和
actualPython3.10.15验证，不把50pin旧metadata结果当新环境通过。
PYTHONPATH必须指向此checkout的src及已核验SciPy层；不改共享CUDA/Python。

按下式一次执行，source为指定原文件，new-output须不存在：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="src:<verified-scipy-layer>" \
  python -m reproduce.check_retrieval_cpu_ranking pinned-native/validate.py new-output
```

四个手写case：strict/all-tie/near-tie/reversed-gallery-tie。只观察原真实argsort
返回值，不修改结果或原函数体；比较完整72-ID排名，随后各用原CIRR/FIQ函数
与独立gold scorer核对cutoffs。预期8原metriccalls/8完整排名比较/36数值比较；
near-tie展示float32 `1-dot`会合并不同dot。旧lexical反例仍保留，不能从新的
同算子特定case吻合推出旧核等价或任意近tie/GPU数值保证。

另8个畸形feature路径实际应拒绝。actualranker接原paired gate三authored路径，
eligible必须补全64实际score、rejected4、failed1attempt0score；SQL只logical
authored journal，不写旧真实费用DB。新输出aggregate写qualification.json，
失败传播且保留新目录，不原样重复/覆盖。没有model/data/image/encoder调用、
最终评测读取或科学M1/Pilot。四case与计数待actualreceipt，文档不自授通过。
