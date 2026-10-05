# P4 paired-cost CPU final follow-through

Final status: CPU_SEMANTIC_VERIFIER_RAW_BUILDER_AND_PREPARED_LOADER_COMPLETE;
GPU_QUALIFICATION_AND_PRODUCTION_CONTROL_BLOCKED.

Final source: two new modules in prefix-io-p4-02-cpu, and the dedicated
tests/prefix_io_v1_p4_measurements suite/CPU guard. This subtask did not edit the
old CostTable/candidate parser, policy/options/bridge/native files. Root integrates
diagnostics separately. See source-lock-final.json for final exact bytes/SHA.

Final dedicated qualification: run-08-wrapper-final, 75 passed, 0 failed,
0 skipped, 1.58 seconds, process exit0. Backend imports are blocked, visible GPU
devices empty. GPU import/init/probe/run/reservation/download/install/system
modification count is zero. No CPU synthetic duration is a GPU experiment.

Semantics now verify actual pinned raw-byte refs, A/B context and workload match,
balanced sequential AB/BA, positive nonoverlapping raw windows, fixed single
physical stage/quantum/operation geometry, completed accepted-I/O bytes,
warmup exclusion, counts and recomputed cost. Global plan isolation covers
trace/prefix-family/workload SHA across calibration and margin-validation,
including different cells. Per cell, workload-content hashes are unique; new
trace/family/seed labels cannot turn the same content into independent data.
active_decode must be positive and fit the actual execution batch. Plan and
bound-candidate objects are reloaded from pinned bytes and compared, preventing
manually changed typed objects from detaching metadata from raw files.

Only paired_residual_margin is implemented. Point estimates use calibration
independent run means rounded upward; the margin uses maximum positive measured
action-window residual over calibration and margin-validation. A long window
cannot disappear in its run mean. Validation therefore participates in margin
construction; it is NOT untuned independent effect evaluation. This finite
empirical envelope is not a confidence interval, SLO guarantee, coverage claim
or causal-effect experiment. Paired interval estimation remains unimplemented
and is explicitly rejected rather than silently approximated.

The raw builder reuses the sole calculator. Given an independently pinned plan,
strict one-stage geometry and four complete raw refs, it writes an append-only
analysis candidate and returns the recomputed candidate cell mapping. It does
not accept caller timing/count numbers. The full loader later revalidates the
assembled five-role candidate and qualification binding chain. The demonstrated
builder-roundtrip-01 produced identical cost119ns only as a CPU synthetic fixture,
and production lookup remained null. Full commands and actual receipt refs are
in builder-roundtrip-01/result.json and executed-commands-final.json.

The public PreparedCostTable constructor now rejects a fake lookup object, a
conditional scope, altered source digest and different computed cells. Production
lookup directly returns None without delegating. Reported PASS, hashes, or a
native_gpu_recording origin label never open the gate. The source uses exact9
active-decode signatures and does not alias root scheduled-work-v1 planned
request/task counts to actual GPU decode.

Preserved sequence, not cumulative test counts:
- run-01: fixture self.load dictionary shadowed load() method; 8 failed/42 apparent
  passes, none accepted as trustworthy negative-test evidence. Failure XML,
  guard and failed test source retained.
- run-02: 50 passed; run-03: 52 passed; run-04: 53 passed.
- Independent review exposed content/split and active-batch metadata gaps.
- run-05-red-split reproduced seven actual unmet rejection contracts on the
  earlier source: 7 failed, 53 passed. Full stdout, XML, source/test copies retained.
- run-06-green-split: 60 passed after the actual repair.
- Eta's three original saved counterexamples were loaded directly and rejected;
  no fixtures were altered. See independent-counterexample-replay-v2.json.
- run-07-builder: 70 passed after ten raw-builder/roundtrip negatives.
- Eta also reproduced a manual FakeLookup constructor boundary. The real loader
  and native activation were already gated; the public wrapper was tightened.
- run-08-wrapper-final: final 75 passed. Earlier positive runs are corroborating
  revisions, not extra unique tests.

Final interfaces are documented in SCHEMA_FINAL.md. Evidence:
source-lock-final.json, executed-commands-final.json,
run-08-wrapper-final/{tests.xml,cpu-guard.json,process-result.json},
builder-roundtrip-01/result.json, independent-counterexample-replay-v2.json,
and all original negative/positive fixtures and logs.

Next stage: root CPU integration and final independent review. When actual
authorized GPU capacity returns, collect authentic native step/load/I/O events
through a source-locked runner; qualify real backend correctness, dependency
prediction, interference control, liveness and effect. Internal file consistency
does not prove GPU origin. Full P4 and positive performance effect remain open.
No P5/P6/P7 run or system/driver changes were performed.
