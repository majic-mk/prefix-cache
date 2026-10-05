# C5 CPU-only single-file notification candidate

This is an isolated incremental candidate copied from frozen C4. It has **no GPU qualification, GPU launcher, calibration receipt for the changed reactor, or performance claim**. C4, native-cost-v6 and runtime-v4 have not been modified. Their negative GPU result remains the last actual strategy result.

Two runtime source files change: the original reactor and its existing thin full-step observation wrapper. The policy, bridge decision logic, prefix cache, admission/capacity owners, native I/O/copy implementation and model executor remain inherited. Historical notes and tests copied from C4 are provenance, not C5 results; the C4 197/547 counts and native-v6 receipt are not transferred to C5.

## Installation boundary

The new API is `capture.attach_single_file_wait(reactor)`. It is explicit and accepts only an active interference bridge with a single-file receipt and matching run. Call it only after the existing bridge's capture attachment has set `shadow=False`. `install()` does not enable waiting. Off, shadow and uninstalled active modes use the original pump path. This directory contains no changes to the frozen GPU runtime to call the API.

The collector weakly references the reactor and bridge; the reactor weakly references the collector. End-event proxies weakly reference their collector. Queue messages contain only `(run_id, capture_nonce, step_ordinal, generation)` in a frozen metadata class. At most one wake message is pending from this mechanism. No message contains a tensor, Future, FD, staging slot, completion, action or resource credit.

## Narrow waiting protocol

1. Original `reserve -> collect/preview -> record deferral -> release` must already have produced the existing legal retry cache. C5 does not compute or retain a final policy action.
2. Before the next original pump, `_prefix_wait_single_file_retry()` rechecks the original retry-key conditions. There must be exactly one ready one-file preload, zero accepted parents and stage operations/bytes, no pending open/read/copy/mandatory support or other poll work, a valid journal/capacity sample, unchanged epoch/identity/age and no scheduled-load publication.
3. The reserved-capacity key is compared with current released capacity by subtracting exactly one from free-slot and reclaimable-slot metadata. Every other capacity component must match. No actual slot is reserved while waiting and no capacity credit is published.
4. The reactor arms a scalar token under its submit lock, releases that lock, then arms the collector under its separate short lock and checks the live condition again. The deadline is the earliest of original ready arrival plus `max_wait_ns`, original snapshot/native sample freshness and original completed-start-query freshness. Neither stale messages nor new attempts reset any original age.
5. The only blocking operation is a deadline-bounded `get()` on the original `_incoming` Queue. Every message, including old/duplicate wakes, returns to intake and the original pump. Wakes are not filtered by waiting for another message inside the wait. Native intake exceptions reach the original fatal/drain handler.
6. After waking, the original reserve, capacity/native checks, policy preview, second live-state check and deferral record all execute again. Pending physical I/O and copy events always keep their original polling path.

End recording closes the live condition under the collector lock by setting `record_before_ns`; notification is emitted **after** the original `raw.record()` and its `record_after_ns` timestamp, including a finally path for an original exception. It denotes host-condition invalidation, not GPU completion. The new code adds no event query, wait, synchronize, sleep tuning or GIL switch-interval change.

Collector fail/detach, bridge fail, existing observer invalidate/detach, and scalar observer fail/detach also wake through thin, weak-reference wrappers that call the original method exactly once. Observer `disable/observe/_original_failed` already converge on those failure methods. Detach restores a wrapper only if it is still the installed wrapper. Notification failures are diagnostic-only, disable new parking and leave the current wait bounded by its original deadline; they must not suppress original event recording, failure handling or native intake.

`publish_p4_scheduled_load()` now wakes under its existing submit lock on new publication and on conflicting same-sequence metadata invalidation. Job/preload/mandatory/STOP producers already write the same original Queue. Collector locks are never held while taking the reactor submit lock. Capacity and journal mutations are native-owner operations in this scoped path; an external arbitrary write into `free_count` or private containers is not a supported producer API. Legal cross-thread job admission/mandatory work wakes through the Queue. CPU tests reject changed capacity/journal/work facts before parking and again after wake.

## CPU verification

Use the explicit source roots; in particular the unchanged enriched-cache test derives a wrong baseline if a C5 name is used without `SERVER11_V3_CANDIDATE_ROOT`.

```text
python -B -I -S run_cpu_candidate_qualification.py \
  --author-root PROJECT_SOURCE --previous-root C4_ROOT --v3-root C3_ROOT \
  --output SERVER_CPU_QUALIFICATION.json
python -B -I -S run_original_cpu_regression.py --project-source PROJECT_SOURCE
```

The first command selects 106 CPU tests: 26 new wait semantics, 16 original retry, 14 policy, 15 receipt verifier, 5 runtime binding, 10 reactor, 13 derived-snapshot equivalence and 7 empty-window functional tests. The second runs the unchanged 101 original policy/ABI tests against the C5 overlay. All passed locally; `LOCAL_CPU_QUALIFICATION.json` records the first command and actual source hashes. Server execution is a separate root-owned result.

The copied `test_retained_idle_wait.py` contains a V3-to-C4 AST assertion and a source extraction list without new C5 names; run it against frozen C4, where its 15 cases passed locally. The copied empty-window whole-file AST allowlist describes earlier revisions and is intentionally excluded from the C5 selector; its seven functional cases still run. A new strict C4-to-C5 AST test checks exactly the four changed existing reactor methods and verifies original pump, scheduling, native polling, copy, intake tail, ready-defer/release and borrower semantics. The initial unconfigured enriched test (self-baseline, 2 instead of 156) and old empty-window source-allowlist failure were diagnostic selection errors, not passed C5 evidence; neither historical test was weakened.

All raw events, I/O completions and time-control fixtures in these CPU tests are explicitly synthetic. The fixture's `native_gpu_recording` accessor branch only selects the existing scalar accessor; it does not call the native installer/source qualification or produce a valid GPU receipt. CPU semantic success does not establish GPU improvement or explain the prior 64.9 ms deferral interval. Full original-pump dual-thread CPU measurements and independent adversarial review are separate artifacts.
