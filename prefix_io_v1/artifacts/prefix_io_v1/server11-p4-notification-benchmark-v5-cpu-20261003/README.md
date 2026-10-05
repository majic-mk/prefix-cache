# Full native-control CPU comparison harness

This directory implements the already frozen `notification_v5_cpu_gate/CPU_COMPARISON_PROTOCOL.json` (SHA-256 `6c00cf687f4903988a363e18a178ccca413076ced81920271bf9c962abde1218`). It does not change the candidate, deadline, repetitions, thresholds or GPU permissions. This is a synthetic CPU engineering gate. It cannot qualify GPU benefit, real I/O bottleneck controllability, D/J release credit, P4 completion or paper readiness.

`cpu_full_pump_compare.py` loads every original reactor class by AST without importing torch/vLLM. An Owner constructed from the CPU fixture binds the native class methods, including complete `_run`, `_pump_once`, scheduling, incoming intake/drain, ring polling, CUDA-copy drain/flush, ready drains, finalization, accepted-parent retirement, `submit_job`, `request_mandatory` and shutdown. The pump is not replaced. Source bytes, method ASTs/line anchors, actual code filenames, collector, loaded pure modules and helper identities are recorded. The original control methods must match between arms except the documented candidate changes in `_run`, `_intake` and `_prefix_single_file_retry_key` (the last also validates the post-release scalar capacity used for waiting).

The explicit physical fixtures are a scalar 16-slot pool, empty ring CQE polling, a synthetic read submitted once and completed after 1 ms, and synthetic event record/query. Completion calls original preload bookkeeping and `_file_terminal`; mandatory uses an original `_ReactorJob`, Future, waiter and accepted-parent retirement. No actual disk/AIO/H2D/GPU/pinned-memory payload or exceptional native DMA drain is exercised. A constant CPU payload digest checks fixture completion only. `native_gpu_recording` is solely an existing collector branch selector, not native qualification. The runtime identity has a synthetic runner weakref; the bridge performs its original borrow checks.

Both arms include symmetric transparent counting wrappers which call the original functions. Their CPU cost remains in measurements. These are instrumented CPU costs, not completely uninstrumented production numbers. `SystemSnapshot.fresh` and retry-key calls are counted; individual inline age predicates are not separately counted. Capacity/decision/live/preview counts are also recorded.

Each trial is a fresh Python interpreter, followed by an equal fixed 200 ms setup settling period. A single startup barrier publishes shared state. The independent producer executes `after_prepare` before that barrier inside its measured thread CPU span, including C5's per-step weakref/lock registration. Its later end-record, notification, mandatory and STOP costs are also measured. One-time attach/construction is setup. The original ready arrival and absolute external event deadlines use real `monotonic_ns`; they do not wait for candidate parking or reset the original 100 ms maximum wait. Actual producer lateness is retained. Worker CPU, producer CPU, their verified sum, wall times, queue timestamps, original submission/completion and final ownership are recorded.

Mandatory is submitted through original `submit_job` followed by original `request_mandatory(Future)`. Native work can start between these operations. The raw signed first-submission minus `_MandatoryWait` enqueue delta and `already_in_progress_at_control_enqueue` are preserved. A negative delta is **not** scored as faster notification; its latency is ineligible and prevents upgrade (`CPU_LATENCY_MEASUREMENT_INELIGIBLE_NO_UPGRADE`), while successful native completion still passes semantics. No worker lock, artificial producer delay, sample discard or retry is used to hide this case.

All cgroup throttle observations remain in paired statistics. Insufficient unthrottled pairs or unknown Linux quota counters prevent latency qualification, as preregistered. A half-core/2 GiB host may therefore produce resource-limited evidence. Every warmup and scored subprocess command, exit, stdout/stderr and raw SHA is saved. A failed trial stops the suite and preserves its prefix; exceptions before barrier completion can leave only the command failure evidence. Output directories/files must be new and are never overwritten. Formal score runs are not adaptively repeated.

## Commands

Use an existing Python interpreter. `ROOT` below is the server project containing the complete original source set; `C4` and `C5` are the two candidate bundle directories. No environment installation is needed. All arguments accept absolute paths.

```text
python -B -I -S cpu_full_pump_compare.py --mode smoke --baseline-root C4 --candidate-root C5 --author-source-root ROOT --protocol CPU_COMPARISON_PROTOCOL.json --output-dir NEW_SMOKE_DIR
python -B -I -S analyze_cpu_comparison.py --directory NEW_SMOKE_DIR --output NEW_SMOKE_ANALYSIS.json
python -B -I -S test_analyzer.py
python -B -I -S cpu_full_pump_compare.py --mode score --baseline-root C4 --candidate-root C5 --author-source-root ROOT --protocol CPU_COMPARISON_PROTOCOL.json --output-dir NEW_SCORE_DIR
python -B -I -S analyze_cpu_comparison.py --directory NEW_SCORE_DIR --output NEW_SCORE_ANALYSIS.json --race-receipt SERVER_SOURCE_BOUND_RACE_RECEIPT.json
```

For the validator tests on the server, set `CPU_COMPARE_SMOKE_DIR` to its completed smoke directory before invoking `test_analyzer.py`; otherwise it uses local `smoke_dev/suite05`. The score plan is 48 warmups plus 84 scored trials (132 total), exactly from the protocol. The analyzer's zero exit means functional replay succeeded; read its explicit CPU/latency/resource/measurement/race gates for qualification. It always grants zero GPU runs.

The optional independent race receipt must bind both `candidate_reactor_sha256` and `candidate_collector_sha256` to arm B, with `status="PASS"`, `gpu_workloads_run=0`, `failed=0`, and a positive integer `passed`. Its source/result evidence is owned by the separate race suite; this analyzer records receipt SHA and does not execute that suite.

## Local fixture validation, not scored performance

On Windows Python 3.12.14, the final harness completed `smoke_dev/suite05` with all 12 scenarios/arms passing. `smoke_dev/SUITE05_ANALYSIS.json` is the first replay of that smoke; smoke is never scored. The initial local development failure `smoke_dev/01-A.json` remains: fabricated +2 ns start completion timestamps could be future timestamps under the Windows clock. The harness now uses actual shared `t0` with a strictly preceding synthetic start bracket. Earlier smoke revisions are retained; no development run is a performance sample.

The analyzer's in-memory positive/negative test cases cover source/hash/order errors, missing native stages, producer cost transfer and sum/raw mismatch, negative/bool CPU, omitted preparation, early deadlines, unretired parents, throttle evidence, pre-enqueue progress, incomplete runs and both race source hashes. Artificial timing values exist only in memory during those validator tests; they are not benchmark results. Windows thread CPU clock quantization may record zero for a short preparation callback; no local speedup is claimed. No formal score run or GPU operation was executed by this harness author.
