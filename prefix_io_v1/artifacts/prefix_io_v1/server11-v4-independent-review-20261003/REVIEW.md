# C4 与 native_cost_v6 只读复核记录

结论：本次审查未发现必须修复的正确性或源码绑定问题。未运行 GPU，未连接服务器，未修改 C4/native_cost_v6 或旧版文件。

审查范围与实际源码哈希、完整命令（含临时反例脚本）见同目录 REVIEW.json。哈希记录实际读取的字节，不代表 GPU 已执行这些代码。

## C4 idle 与派生快照缓存交互

- Idle waits only when all seven original polling-work collections are empty; retained cache does not imply a runnable operation.
- Original _has_work, STOP intake, native pump, drain cleanup and Queue.get(timeout=0.5) remain unchanged.
- Mandatory, STOP, snapshots and P4 controls arrive through the existing Queue and are processed before the next pump; STOP prevents new blocking.
- A deferred ready preload remains polling work. The idle fix does not hide or fix active retry cost.
- Bridge cache requires the same immutable input snapshot object, complete live-step tuple and runtime signature prefix.
- Every preview still checks original freshness/max_wait and the second live state; record_single_file_deferral checks live state again.
- Original reactor retry-key covers owner state, event sequence, capacity, ready identity, epoch, progress and time bounds; changed facts force a fresh observation.
- Cached values are immutable snapshots/scalars, not runner/capture/job/Future/slot owners; no action or budget is cached.

## native_cost_v6 实际源绑定

- Child package search paths select C4; actual reactor module path must match C4.
- The existing original_method validates bound method, actual globals, co_filename, frozen-source compiled code and exact class identity.
- All loaded py_kvcache/prefix_io_control modules must be inside C4 and match actual frozen bytes/SHA.
- The collector historical lookup key maps only in a local view to the actual C4 reactor reference; original source-lock entries are unchanged.
- Before/after scalar-adapter source SHA must match C4; the adapter object is unchanged and all 128 exported frames equal its actual frame dataclasses.
- Each calibration child and serializer require bridge=None and max_accepted_parents=8.
- Final verification retains before/after complete-source receipts, original guard validation, complete outputs, actual I/O and original shutdown/drain.
- Legacy source/hash, changed parent bound, enabled policy and raw child/parent disagreement are rejected by CPU counterexamples.

## 已实际执行的本地 CPU 验证

| 命令末尾 | 通过数 | 测试报告耗时 |
|---|---:|---:|
| `test_retained_idle_wait.py` | 15 | 0.188 s |
| `test_enriched_snapshot_reuse.py` | 13 | 0.06 s |
| `test_common_source_binding.py` | 7 | 0.031 s |
| `test_prepare_and_verify_native_cost.py` | 14 | 1.914 s |
| `test_native_conditional_cost.py` | 17 | 0.066 s |

这些命令统一使用本机 Python 3.12：`python -B -I -S <test.py>`；完整解释器路径、工作目录和精确 argv 记录于 JSON。各组有重叠，不相加充当唯一系统测试数。

另外，临时 CPU 反例调用真实既有 `original_method`：合法绑定通过；相同路径不同代码、旧 co_filename、实例方法覆盖分别被拒绝。该反例确认 Torch/vLLM 未导入。

## 限制

- CPU tests and source checks do not establish real CUDA execution, DMA completion, runtime performance or valid cost migration.
- The reviewer implemented the common idle correction earlier; bridge interaction and native-v6 source-binding checks were subsequently performed as a separate read-only review. This is not a claim of independent authorship for the idle change.
- The thread/Queue test uses actual source-extracted control methods and explicit fake backend/pump work; backend lifecycle qualification still requires server tests.
- The in-process identity checks prevent accidental stale/mixed source use; they are not a cryptographic security boundary against malicious process code.
- All off/shadow/on arms must share the common fix. New-on versus historical-off is not a strategy comparison.
- Historical v5 calibration does not automatically qualify C4; v6 needs real guarded runs, source-before/after checks, and heldout verification.
- Test groups overlap and their counts must not be summed as unique system tests.
- No additional tests were run to create these records; the recorded commands/results were executed during the prior review.

## 关键源码身份

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v4/source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py` | 178979 | `96bfd88dcee7f9c7996518ffe762b87be5ad81ef11d598329d41145e2b01fd87` |
| `artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v4/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_bridge.py` | 25553 | `6bec5dd57e25cfc2c0e62540742a781d656ba528cb6c8057cd2402197db93e1a` |
| `artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v4/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_policy.py` | 24976 | `bce19c96c7651f69ebb91a515ea3e9e3d6855636b4bd4aa84e60fb2fbc10d673` |
| `artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v4/native_full_step_collector.py` | 17073 | `b8249e9a59a65aa13d452ba24f8fa255ca90ec290f3f1924db883b1fa550a1ce` |
| `artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v4/single_file_runtime_binding.py` | 10790 | `8b66b3aba1c936b85f4d82fe73c568650b712acf250aae454823cd35da060d60` |
| `artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v4/test_retained_idle_wait.py` | 15943 | `aa7c372d70cb5d52bb1a259b9f211d17fc213bcbb382826787915f2eb5773462` |
| `artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v4/test_enriched_snapshot_reuse.py` | 10728 | `68f3c1143bd6d36b7b6b8bfca2792a75b57af2da270b8dc0a33c073806214201` |
| `artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v4/test_single_file_retry_observation.py` | 12888 | `a24eb1c3d9aadb6e491ad0bfa3f6b123d349ec58a8fb3952fa06f4806e306c71` |
| `artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v4/IDLE_CHANGE_NOTES.md` | 3485 | `6a1e212c2d600115dc940e3cca0b3452f441562492540128212dfb4294bc24e2` |
| `artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v4/ENRICHED_CHANGE_NOTES.md` | 1832 | `2034d706f2e0abc8f0c6d1a80bdc84f1f553f060e628c4fd437e77b3da8b6910` |
| `artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v3/source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py` | 178317 | `fb69ddde5d3d28ac3860b531f3ed24edbcbfa0a0c64519512a64063bd840d1c6` |
| `artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v3/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_bridge.py` | 24422 | `c9901e8e5c0f0cd46cb0cf08a5763f27be19d0e121f2d6261ded1be8c506aae5` |
| `artifacts/prefix_io_v1_server11_candidates/native_cost_v6/run_native_cost_experiment.py` | 43112 | `666d778d8299ea81276925b111b9d9953c249629598a82f6d59d3cc6755764b7` |
| `artifacts/prefix_io_v1_server11_candidates/native_cost_v6/control_native_cost_job.py` | 12349 | `9728b29101c63aae97bc1b7253df8001ead634497e1d23529bb9527389b5f974` |
| `artifacts/prefix_io_v1_server11_candidates/native_cost_v6/prepare_and_verify_native_cost.py` | 22006 | `6cb02c8d2ff148a1c25f221a2350f71ae71daff39c1cdcd191954ae8e6c9570f` |
| `artifacts/prefix_io_v1_server11_candidates/native_cost_v6/native_conditional_cost.py` | 36869 | `e20944ea470a4fbdbd10cb67e516cfc2b9f1de3f53d61d0249e0fa6d0e59a291` |
| `artifacts/prefix_io_v1_server11_candidates/native_cost_v6/test_common_source_binding.py` | 5135 | `28aa75904fca313abfd836e68a37664bddf84779bb39631d1e787dd6ea1c69d4` |
| `artifacts/prefix_io_v1_server11_candidates/native_cost_v6/test_prepare_and_verify_native_cost.py` | 9050 | `30b7df4cbbe7b077cc18ad98b79c9e377ae0df308b06278eeed05d93a1d599e4` |
| `artifacts/prefix_io_v1_server11_candidates/native_cost_v6/test_native_conditional_cost.py` | 13143 | `56304f4f1fcb82188a66450b1bcb1927701fd681fed4d5eeb4de231dc332ab76` |
| `artifacts/prefix_io_v1_server09_candidates/g3_calibration_launcher_cpu_v1/g3_calibration_runtime_metrics_v2.py` | 29978 | `0c04287325fd84c31aaea35dc6ff7431229511a04f3ab71bf9aa96a2ce83f1cc` |
