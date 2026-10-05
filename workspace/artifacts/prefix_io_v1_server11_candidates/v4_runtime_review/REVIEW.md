# Server11 runtime_v4 + C4 receipt 只读审查

日期：2026-10-03。审查者：server11_continue_protocol。

结论：发现的“实际运行 reactor 与共同 C4 校准来源未完全连接”问题，已由 hotpath 在 **runtime_v4 内**修复。独立本地复核和 **59 / 59 CPU 测试**通过。当前读取的代码中未发现剩余必修问题；本报告不声称正在运行的 v6 GPU 校准已经完成或通过。

## 必修问题与修复

原 runtime 将实际 native owner 绑定到锁中的某份源码。由于完整锁同时保留 legacy 和 C4 源码，仅检查计划的 C4 引用及 sys.path 优先级不足以证明实际 owner 来自 C4。

修复后的 `native_source_binding` 检查实际 `py_kvcache.reactor` 模块路径、真实原始 `_run` 的代码来源、严格整数 parent cap = 8、实际 bridge 对象身份和模式，以及全部已加载 native 包的 C4 路径和 SHA。独立验证器要求实际引用等于 receipt 的共同 C4 校准源码，并要求实际 identity helper 的 source refs 确实包含该引用。

新增 11 项反例使用真实 `original_method` 源码/代码对象验证器和合成的 Python owner。已验证旧、新源码均在锁中但实际加载旧模块、旧代码文件名、实例方法覆写、错误 parent/bridge、旧包越界、源码漂移及独立验证器错误绑定等情形会被拒绝。没有加载 CUDA 引擎。

## 动态成本、资格及预算

- C4 receipt 固定 v6 serializer/verifier 与原 estimator 的字节哈希；从真实六窗口原始证据重新序列化、重新验证。存储的 PASS 本身不构成资格。
- 校准及运行都必须使用共同 C4 reactor 和共同源码。成本上界来自校准集；工程预算只来自校准 A 样本。不将 B 或 heldout 纳入预算，也不借 heldout 重新拟合。
- off、shadow、on 逐个显式准备和启动，前序资格会从原始证据重放。相同模型、完整输出、单文件工作量及执行器配置的门控保留。
- 每 arm 仍为 300 秒执行 + 20 秒收尾预留；累计上限仍为 28,800 秒；活动 reservation、既有运行/launch intent、2 GiB 预留及 8 GiB 空间下限均保留。不得复用作业名称盲目重试。
- 审查任务提供的历史余额为 7,441.368204549421 秒。仅按 v6 最坏预留 1,220 秒及三个 arm 共 960 秒算术估计，余 5,261.368204549421 秒。此值不是当前实测余额；下一次启动必须使用 guard 完成后的真实账本。
- on 的生命周期/调度资格与成本迁移结果分别记录；成本迁移失败不会被包装成性能提升。production 和 strategy effect 保持 false。
- 若共同 idle 修复使 v6 成本上界不再高于 A-only budget，shadow 无合格延期提案是应保留的停止条件。不得为强制实验动作而抬高阈值或伪造释放信用。共同修复收益不归给研究策略。

## 实际本地 CPU 执行

命令均通过 PowerShell 调用同一 Python 3.12，使用 `-B -I -S`，未进行 RPC/GPU：

- `& 'C:/Users/mamengkui/AppData/Roaming/uv/python/cpython-3.12.14-windows-x86_64-none/python.exe' -B -I -S 'artifacts/prefix_io_v1_server11_candidates/p4_single_file_runtime_v4/test_native_source_binding.py'` → exit 0；11 tests；unittest 0.456 s。
- `& 'C:/Users/mamengkui/AppData/Roaming/uv/python/cpython-3.12.14-windows-x86_64-none/python.exe' -B -I -S 'artifacts/prefix_io_v1_server11_candidates/p4_single_file_runtime_v4/test_p4_single_file_harness.py'` → exit 0；10 tests；unittest 0.290 s。
- `& 'C:/Users/mamengkui/AppData/Roaming/uv/python/cpython-3.12.14-windows-x86_64-none/python.exe' -B -I -S 'artifacts/prefix_io_v1_server11_candidates/p4_single_file_runtime_v4/test_control_p4_single_file.py'` → exit 0；9 tests；unittest 0.408 s。
- `& 'C:/Users/mamengkui/AppData/Roaming/uv/python/cpython-3.12.14-windows-x86_64-none/python.exe' -B -I -S 'artifacts/prefix_io_v1_server11_candidates/p4_single_file_runtime_v4/test_verify_p4_single_file.py'` → exit 0；14 tests；unittest 0.028 s。
- `& 'C:/Users/mamengkui/AppData/Roaming/uv/python/cpython-3.12.14-windows-x86_64-none/python.exe' -B -I -S 'artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v4/test_single_file_receipt.py'` → exit 0；15 tests；unittest 0.161 s。

总计 59 项通过。完整真实 stdout/stderr 保存在同目录 `REVIEW.json`。

## 实际读取的文件哈希

- `artifacts/prefix_io_v1_server11_candidates/p4_single_file_runtime_v4/control_p4_single_file.py`：14296 bytes；SHA-256 `48c1d04ea4b2154e4ad699f14673a676433add4741ca00e9c74e7fd009d532d0`。
- `artifacts/prefix_io_v1_server11_candidates/p4_single_file_runtime_v4/run_p4_single_file_experiment.py`：44493 bytes；SHA-256 `cca77444fa821938c3506c41616f49a454991dd35aa25448abecac59ad90f8b4`。
- `artifacts/prefix_io_v1_server11_candidates/p4_single_file_runtime_v4/verify_p4_single_file.py`：30099 bytes；SHA-256 `31d23f3a19a9437ae056be82dc4a2b26a6ea9c587b6f45ceb65a4eacfa553056`。
- `artifacts/prefix_io_v1_server11_candidates/p4_single_file_runtime_v4/test_native_source_binding.py`：7558 bytes；SHA-256 `bdc325836656ae64bdccc03516caee265d528005a719a5b10f9abe4c419a418a`。
- `artifacts/prefix_io_v1_server11_candidates/p4_single_file_runtime_v4/test_p4_single_file_harness.py`：10623 bytes；SHA-256 `3f30931f018a97b01841edf3bc645c574e975eb209b9e5499e1cc7d60d1bb7e1`。
- `artifacts/prefix_io_v1_server11_candidates/p4_single_file_runtime_v4/test_control_p4_single_file.py`：7021 bytes；SHA-256 `c12ada340645ecfa27b5a0a38c4796dceba87b7800b552a44c6fed7c06c5d62a`。
- `artifacts/prefix_io_v1_server11_candidates/p4_single_file_runtime_v4/test_verify_p4_single_file.py`：7320 bytes；SHA-256 `432b9d670b0457dc288bba3262d3b646f82c85c8ebbd4f9d7ea1f567fe062f10`。
- `artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v4/test_single_file_receipt.py`：10389 bytes；SHA-256 `102f45a1363ae7266fad3f1f72f3c09833b10e839dbfb71797f709ba8cf4735c`。
- `artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v4/single_file_runtime_binding.py`：10790 bytes；SHA-256 `8b66b3aba1c936b85f4d82fe73c568650b712acf250aae454823cd35da060d60`。
- `artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v4/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_single_file_receipt.py`：14761 bytes；SHA-256 `11312ab7a6f5c7823e77b22cf17c93bd255438af7f886568e693f76fc5b9825d`。
- `artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v4/source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py`：178979 bytes；SHA-256 `96bfd88dcee7f9c7996518ffe762b87be5ad81ef11d598329d41145e2b01fd87`。
- `artifacts/prefix_io_v1_server11_candidates/native_cost_v6/native_conditional_cost.py`：36869 bytes；SHA-256 `e20944ea470a4fbdbd10cb67e516cfc2b9f1de3f53d61d0249e0fa6d0e59a291`。
- `artifacts/prefix_io_v1_server11_candidates/native_cost_v6/prepare_and_verify_native_cost.py`：22006 bytes；SHA-256 `6cb02c8d2ff148a1c25f221a2350f71ae71daff39c1cdcd191954ae8e6c9570f`。
- `artifacts/prefix_io_v1_server09_candidates/g3_calibration_launcher_cpu_v1/g3_calibration_runtime_metrics_v2.py`：29978 bytes；SHA-256 `0c04287325fd84c31aaea35dc6ff7431229511a04f3ab71bf9aa96a2ce83f1cc`。

## 限制与交付范围

本次审查只读取本地文件及运行 CPU 测试；新增写入仅为本目录两份报告。没有修改冻结的 C4、native_cost_v6，也没有启动任何 GPU/RPC。GPU 在 root 控制下运行，其成功/失败、原始结果、实际累计时间及服务端空间仍须另行验收。服务端交付应与本报告 SHA 一致并通过相应 CPU 测试。

审查者此前参与实现 C4 共同 idle 修复；本次是独立进行的 runtime/receipt 只读交互审核，不声称对 C4 完全没有作者参与。有限的原生生命周期与条件调度资格不等于 P4 全面通过、统计性能提升、正式 SLO 或 P5 证据。
