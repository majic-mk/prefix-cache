# P1 CPU baseline report

P0 source lock/audit complete. P1 CPU scope complete; full P1 acceptance BLOCKED.

Untouched py-kvcache tests initially: 258 passed, 2 failed, 9 skipped. The two failures occurred when missing vLLM erased an otherwise installed Torch import and blocked CPU alignment tests. Original logs are preserved.

Common patch 0001 separates CPU Torch availability from the real vLLM copy backend. Canonical GPU-layout and IoReactor guards still require TORCH_COPY_AVAILABLE. No stub is promoted to runtime evidence.

Common patch 0002 accounts for the retained 4095 alignment bytes in the original staging backing allocation. It rejects a minimum slot floor that exceeds the nominal approved staging budget and uses the same corrected slot count in reactor and the original planner. Real allocation storage nbytes is recorded and checked. This is tensor-backing accounting, not a measured total process/pinned-allocator overhead budget; auxiliary arrays, ring mappings and allocator overhead still require P1 deployment measurement.

A pre-existing planner capacity assertion changes from 112 to 111 because the extra alignment backing must fit the same budget. The original failed intermediate run is retained in p1/upstream-common.txt. Separate CPU tests verify actual Torch storage nbytes and that rejecting the floor occurs before any allocation or ring creation.

SciPy was installed inside the isolated CPU venv to execute the previously skipped pure CPU test; simple-profiler was installed from its fixed author checkout. The first profiler install attempt lacked hatchling and failed; normal isolated build then succeeded. No system package or driver was changed.

Final: 328 passed, 8 skipped, zero failed. This comprises 261 upstream tests and 67 project tests. Skips: 5 GPU E2E, 3 io_uring EPERM. Evidence: artifacts/prefix_io_v1/p1/cpu-verified.txt/xml. Earlier attempts are not overwritten.

Source-invariance checks confirm unchanged LoadPlanner, adapter, file mapping, fs executor, native reactor dispatch, continuation, fusion and parent completion methods. See native-path-preservation.json. The source observer/dependency modules are preparatory CPU components; they are not connected to a live engine.

GPU hours: 0. Model downloads: 0 GiB. No throughput, TTFT, ITL, overlap, model accuracy or real KV round-trip result is claimed.

## New server 02 — P1 partial GPU qualification

The author source build, base CUDA, guarded native copy, import/API and approved-UUID platform checks now pass. A common vLLM UUID compatibility patch is shared by all arms. Full P1 remains blocked: no model weights or model/GPU Prefix execution, native LiburingRing(2) EPERM, and real staging/SSD/handler/production-KV lifecycle unverified. Current GPU budget consumed is 45.273048002272844 seconds, model download is 0 bytes. Earlier zero-GPU totals describe the original P1 CPU snapshot only. See NEW_SERVER_02_REPORT.md and GPU_STAGE_REPORT.md for current evidence.

## New server 03 — native model and GPU Prefix verified

The native GPU-only diagnostic passed: identical128-token input gives cached0→112 and equal16-token outputs. Actual28-layer CUDA BF16 KV backing is66,977,792B under64MiB configuration. All11 official ModelScope model files are pinned/verified. Complete P1 still BLOCKED by io_uring EPERM, real staging/SSD/handler/production-KV lifecycle and local cost-curve/driver prerequisites. Current budget, failures and scope: NEW_SERVER_03_REPORT.md. Earlier no-model/no-GPU statements are historical snapshots.


## 2026-09-29 新服务器 07 GPU 验证

新指定 RTX5090 上作者真实拷贝、原 handler/reactor + AIO SSD、shared preload、pinned staging、真实模型 KV 字节往返及正常 wait/shutdown 通过。最终证据为 runs/server07-native-aio-kv-02，成功后不重写 CPU 备份，Prefix 0/112 与输出 token 一致。当前模型 scheduler 未接入外部 connector，AIO 成本曲线与 GPU 源块释放见证尚待验证；P1 仍 partial，不进入策略验收。完整范围、预算、命令见 NEW_SERVER_07_GPU_REPORT.md。


## 2026-09-29 server07 原生标定与规划器验证

详见 SERVER07_CALIBRATION_REPORT.md 和 REPRODUCE_SERVER07_CALIBRATION.md。本轮完成有界 P1 AIO 原生路径、实测成本、原 LoadPlanner 调用与自然 GPU 复用。短前缀 SSD 成本不利；未改变原准入或缓存执行器，未启用研究策略。P2 mandatory/release witness 和观察开销仍待验证。
