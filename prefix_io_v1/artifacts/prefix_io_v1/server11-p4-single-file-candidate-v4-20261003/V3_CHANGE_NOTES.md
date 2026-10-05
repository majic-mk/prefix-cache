# One-entry repeat-observation reuse

V1 and V2 remain immutable. V3 changes only the native reactor observation path
and adds a bridge `observation_reuses` counter. It does not change the policy,
cost receipt, cost threshold, step budget, workload, original queues, polling,
sleep, slot reservation/release, native completion or model execution.

The first eligible actual deferral still performs the original full owner
collection. One tuple may then retain its immutable Snapshot and WorkDescriptor,
with their ORIGINAL snapshot and stage-sample times. It contains no ready/job,
Future, stream, event, tensor, model or pool owner. Every retry still executes
the original policy preview and actual deferral recorder; no decision is cached
unconditionally.

Reuse requires the same original ready sequence/FD/hash/open time/request values,
same real live-step tuple, unchanged journal sequence and bridge epoch, and
freshly checked identical capacity components. There must be one preload ready,
zero existing stage I/O and accepted parents, no other active/open/ready/copy or
pending work, no preload waiter, and no scheduler-load publication. The sampler
and original max-wait bounds are rechecked without refreshing cached timestamps.
The original caller checks mandatory/downstream/shutdown and reserves the real
slot before this gate. A changed or unknown condition falls through to the
original observation/policy path; mandatory, continuation and context fallback
remain. Oversized observation windows retain their original rejection.

The source-extracted CPU replay invokes the actual collect, capacity, reserve,
ready-queue and preview implementations. With `_preload_inflight_total=1` for an
opened descriptor, 155 same-step CPU deferrals cause one collection, 155 previews,
154 reuses, and 155 original reserve/release pairs. The descriptor remains in its
original queue and its open timestamp is unchanged. These counts are CPU evidence,
not a latency measurement or a GPU performance result.

Validation: the previous 151 tests plus 16 targeted retry tests pass (167 total).
The prior empty-window audit is updated only to allow the documented V3 method
diff. Counterexamples cover mandatory and ordinary waiters, changed/ended GPU
context, new parent work, changed journal/capacity/epoch/scheduler state, new I/O,
faults, original max-wait and sample expiry, wrong stage/geometry, off/shadow,
and recursively immutable cache values. The original policy is unchanged.

Server integration tests and one bounded off/shadow/on requalification are the
parent's next checks. The previous negative GPU result is not overwritten. No
GPU operation, RPC, source promotion or performance claim occurs in this local
candidate delivery.
