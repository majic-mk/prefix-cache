# Independent P4-02 semantic-cost and raw-recording review

Review scope: read-only new semantic verifier, prepared cost loader, raw schema,
fixtures and existing GPU drivers. Independent counterexamples executed on CPU.
No source edits in this review. GPU workloads: 0. Production bypass found: none.

## Concrete reproduced findings

The original verifier (SHA-256
4d67324baf83f912a7df7ddfdddceadb747436009f578256f9290daf773df2a4)
accepted these internally bound fixtures with
PASS_CPU_SEMANTICS_NOT_GPU_QUALIFICATION:

1. Calibration and validation share the same workload_sha256 content while using
   different trace/family labels. Per-cell trace/family isolation alone did not
   establish disjoint workload content.
2. Cell 2 validation reuses cell 1 calibration trace, prefix-family and workload
   content. Isolation was only checked within each individual cell.
3. active_decode=3 with batch=2. Exact executed-batch load metadata was not
   constrained to its physically possible relationship.

All three remained production_qualified=false, gpu_verified=false and production
lookup=None. The fabricated native_gpu_recording origin also cannot open that
gate. Raw native source and GPU identity drift were correctly rejected.

Actual full fixture bytes, pinned refs and source snapshots are retained under
counterexamples-01/. The independent executed script is
review-counterexamples-executed.py. Its backend-import blocker recorded zero
attempts. The cost implementer has been notified to repair these boundaries and
retain the original failing evidence.

Validation windows currently contribute to the explicitly defined conservative
maximum-positive-residual envelope. This is margin validation, not an independent
effect-evaluation cohort or empirical confidence-interval coverage. P5 has not
started; no statistical generalization or performance claim is made.

## Additional scalar-observation review

Four malformed SchedulerLoadObservation constructor combinations were accepted:
nonzero prefill requests with zero prefill tokens, nonzero decode requests with
zero decode tokens, scheduled requests not all represented by any phase, and
fewer decode tokens than decode requests. The real producer does not emit these,
and the public values remain unqualified scheduled work; however strict intake
should reject them. Evidence is load-constructor-01/. Suggested invariants:
per-stage request/token zero equivalence, tokens >= nonempty requests, and
requests <= prefill_requests+decode_requests <= 2*requests. A mixed prefill/decode
request legitimately contributes to both stage counts.

## Existing GPU drivers do not directly emit the new raw schema

acquire_native_aio_costs_capacity_p316.py acquires TTFT and defaults to exactly one
output token. New paired semantics require sustained decode with >=2 output
tokens, so renaming its results cannot create a valid input.

run_concurrent_capacity_p316.py records host engine.step float timestamps,
outstanding frontend requests, token timelines and cohort/tail accumulated I/O.
It does not supply exact per-window active GPU decode/batch/context, start-time
existing I/O, original-owner accepted physical action operations attributed to
that step, or frozen per-cell AB/BA calibration/margin-validation pair identities.

The new SchedulerLoadObservation signature is scheduler-work-v1; it is expressly
not the old exact9 active GPU state. No adapter may equate those values, convert
frontend queue counts to active decode, distribute cohort I/O among steps, fill
missing counters with zero, or convert P3 durations to P4 interference effects.

Safe CPU adapter scope: strict intake/export requiring actual native step
boundaries, actual executed GPU load, original-owner accepted/completed operation
identities and bytes, full run drain, per-window token work, exact frozen
trace/workload/family/pair/order/split metadata, and source/hardware/model/layout
refs. Missing fields must be BLOCKED_MISSING_NATIVE_PAIRED_RECORDER. Fake timelines
can exercise validation and export but cannot confer real GPU measurement
qualification. Actual recording and source/context qualification remain GPU work.

## Root load/batch delta

Read-only review saw correct closed boundaries: batch activation requires a
production-qualified advice, qualified I/J table and production_gpu_load_state;
scheduled observations supply none of those capabilities. Continuations return
before optional batch cutting; source release, native fusion and allowance
settlement retain their original owners. Off remains before clock/scan.

This report's findings are not GPU outcomes. Repair verification is recorded
separately after the implementer freezes the corrected sources.
