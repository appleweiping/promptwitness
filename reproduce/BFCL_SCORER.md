# BFCL 官方 AST checker 的 CPU 验证

本命令只验证原版 scorer 的运行时接入。所有 fixture 均为自主编写，未读取
BFCL 数据文件、历史模型响应、final prompt 或 gold；不是模型准确率或 Pilot。
`BFCL_real_GT_end_to_end` 仍为 pending；完成 fit-only auditor 访问控制和真实
annotation 接入后才可判断完整 scorer 门禁。

## 原版代码与明确适配

使用 [BFCL 原版 AST matcher](https://github.com/ShishirPatil/gorilla/blob/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py)，
`Language.PYTHON`，五个冻结类别：`simple_python`、`multiple`、`parallel`、
`parallel_multiple`、`irrelevance`。不改其类型、字符串、可选参数、多个函数
与无序并行匹配规则，不以自写等值匹配器代替。

接口为严格 JSON list，每个 call 只有 `name` 和 `arguments`。名称原样传递；
不执行 upstream Python-expression decoder 或 `eval`。这是所有控制组共享的
明确适配，不是复现官方 leaderboard 的 vendor function-calling generation。
irrelevance 沿用“没有可解码调用”的语义，因此 refusal/无效 JSON 可通过该类别，
不代表它满足 JSON 格式。缺失/failed/cancelled 响应从不充当观测零分。

固定源码的注册表没有本研究的两个主模型。接入时只对不存在的冻结任务模型
增加 `scorer-only` 元数据，`underscore_to_dot=false`，handler 为不可调用的字符串；
不启动推理 SDK、不声称 upstream 正式支持这些模型。已有条目若名称重写配置
冲突则报错，不覆盖。官方 matcher 仍直接读取原版注册表并运行原版函数。

## 独立环境

规格 `configs/research/bfcl-scorer-env.json`，独立 Python 3.12 CPU venv，GPU=0。
按 `pip_phases` 顺序安装，不修改主项目、共享 Python/CUDA 或 upstream checkout。
SDK 依赖是原版 model registry 的传递导入要求；安装不等于使用收费 API。
原版 eval_config 在 import 时会创建目录，因此先把 `BFCL_PROJECT_ROOT` 指向
任务专用外部 scratch。OMP/MKL/OpenBLAS 均为 1；没有 foundation-model 权重。

## 本任务原样运行

在 `D:/Company/nlp-original-projects/promptwitness` 执行：

```powershell
$env:PYTHONPATH = 'D:/Company/nlp-original-projects/promptwitness'
& 'D:/Company/research-artifacts/promptwitness-delta-20260926/bfcl-scorer-venv-aris/Scripts/python.exe' -X utf8 -m reproduce.check_bfcl_scorer 'D:/Company/research-artifacts/promptwitness-delta-20260926/gorilla-scorer' 'D:/Company/research-artifacts/promptwitness-delta-20260926/bfcl-scorer-work-aris' 'D:/Company/research-artifacts/promptwitness-delta-20260926/bfcl-scorer-aris-follow-doc.json'
```

fresh agent 只获得本文、provider ledger 和此 invocation；原样执行，不修复或
重试。输出 exclusive create，不覆盖旧证据。失败的空输出和 traceback 也保留，
经修改、审查后用新的输出名。成功只代表三种固定模型配置下的 authored native
checks，不清 S1/S2、完整科学准入或确认成本门禁，不读取 final 来调 checker。
GPU kernel witness 对本 CPU 环境不适用；实际 GPU 推理环境另行验收。
