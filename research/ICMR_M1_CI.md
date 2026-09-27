# ICMR M1 权限增量精确提交验证，2026-09-27 23:10:53 UTC

源提交 `5ba033187a22a3175eae552067c2cf1b301a83ac` 已推送
`research/promptwitness-delta-v1`。其[GitHub CI run 36357426043](https://github.com/appleweiping/promptwitness/actions/runs/36357426043)
终态 `success`，15/15 jobs success；包括 Linux/Windows/macOS 测试、quality、
build、incremental coverage 与三平台 wheel smoke。Ubuntu Python 3.14 job
日志为 `2018 passed, 1 skipped, 127 warnings`；`test_real_linux_process_sentinels`
出现在运行记录/警告中，且该测试包含 13 角色×11 leaves 的143 read/143 write
与13个不同 worker PID 断言，job成功。警告保留，不写零警告。

本机最终 `tests/reproduce` 为475 passed/4 skipped（Windows Linux-only项未运行）；
Ruff、format、mypy显式package-bases、`git diff --check`通过。Bandit新文件检查
无medium/high，旧`process_access.py`的subprocess两项low发现保留，未降安全门槛。
新上下文审查同家族/provisional，非跨模型接受。

**只证明**自制数据权限机械检查在CI Linux实际通过与代码质量；没有合法完整
CIRR/FashionIQ原图/annotation、真实近重复组、CIR数据store及controller强制
retrieval角色、sealed final或全库official parity。因此 M1/Pilot、科学效果及
ICMR稿件仍未准入。原历史真实模型费用和访问披露不变。
