# P4-02 CPU causal closure ETA delivery

Status: CPU_CAUSAL_CLOSURE_HISTORY_AND_OWNER_HOOK_COMPLETE_PRODUCTION_ETA_BLOCKED.

## Actual change

The new p4_eta.py records an original-owner complete-ready closure observation,
then measures elapsed time only when the same native parent successfully drains
all physical children. It stores scalar parent identities and immutable geometry/
context values. It does not retain job, Future, tensor, slot or resource owners.

Bounds: 32 pending scalar observations, 32 exact geometry/context cells, 16
successful measurements per cell, 64 recent diagnostic records. Parent identity
reuse is blocked by a monotonic accepted-parent high-water mark. Excess or
out-of-order observations are omitted without consuming or truncating native work.

All forecast inputs satisfy completed_ns < forecast now_ns. Forecast point is
median_high of the exact cell's fresh completed intervals. The empirical margin
is the maximum of absolute sample deviations and previously completed causal
forecast residuals. Neither the point nor margin is a GPU qualification, confidence
coverage guarantee, or allocator release proof. History freshness is bounded by
the existing P4 max_wait_ns; the minimum is four successful past samples.

NativeP4Bridge owns this optional bounded observer. Native reactor integrates
only complete-ready observation, successful final-child physical drain, and
failed final-parent discard. Collection uses actual SSD-read/H2D ready geometry,
physical byte quanta, and native queue metadata. Those queue values are explicitly
native closure context, not decode load or an interference-cost signal.

A later geometry/context change never resets the first observation. Successful
finish is recorded against that initial geometry/context and represents the
observed initial-state-to-whole-drain trajectory, including later native changes.
It does not claim that geometry, queue state or compute load remained constant.
Independent GPU qualification must inspect drift in the underlying raw timeline;
an initial context match alone cannot establish a stable conditional cost cell.

No per-file stage durations are summed. Original fused H2D, native queues,
accepted-parent protection, physical accounting, callbacks, admission, budgets
and resource release remain authoritative. Off returns before time capture,
imports, job inspection or closure scanning.

Production Parent.completion_estimate_ns and ReleaseWitness.estimated_unblock_ns
remain unknown. NativeClosureHistory.estimate(execution="production") always
returns None. Shadow estimates are diagnostic only and cannot publish production
ETA values or authorize a GPU release. Existing actual-owner publication equality
continues rejecting externally forged forecasts.

## CPU execution

- scalar-02: 31 passed, no failures/errors, backend-import blocker attempted zero
  Torch/vLLM/py_kvcache/CUDA/CuPy/NumPy imports.
- native-04: 122 passed, 0 failed, 0 skipped: 43 new ETA tests plus 79 existing
  P4-02 bridge tests. The 31 scalar tests are included in these 122 and must not
  be added again.
- CUDA initialization remained false; GPU workloads 0; ledger bytes unchanged.
- Seven ETA source/test inputs were SHA-256 checked before and after native-04.

Evidence: scalar-02/result.json, process.log, cpu.xml and command.json;
native-04/result.json, process.log, cpu.xml, cuda-guard.json, command.json,
source-inputs-before.json and executed.py.

Negative and initial failed evidence is retained:
- entry-preflight-01: import path failed before tests started; corrected isolated
  metadata driver to use the frozen P4-02 source and existing project venv paths.
- native-01: 120 passed, 1 failed. The fake native clock was held at zero, correctly
  rejecting a zero elapsed timing. Test now explicitly advances the fake clock
  before publishing completion; production code and zero-interval rejection were
  not weakened.
- native-03: 121 passed, 1 failed. Existing test waited for the fake backend entry
  then indexed the native pending list before the owner appended it. Only the
  P4-02 fixture condition now also waits for that native pending record; all held
  source/fence assertions remain. Original P3/P4-01 tests were not edited.

## Handoff interfaces and next permitted stage

bridge.observe_native_closure(parent_id, ClosureGeometry, scalar_context,
now_ns=...) returns no production forecast. bridge.complete_native_closure(
parent_id, now_ns=..., successful=..., drain_known=...) emits a successful scalar
measurement or discards failed/unknown intervals. The context is an exact tuple
of at most 16 scalar strings/integers; future actual scheduler load state can be
bound through a canonical scalar fingerprint. Parent counts cannot stand in for
decode load.

On an authorized GPU stage, live shadow can collect native completed intervals
and causal forecast-to-completion residuals. A real source/context/residual
qualification gate and production ETA loader still need GPU qualification.
CPU mocks, declared PASS text and old P3 timings cannot activate this gate.
No effectiveness or acceleration claim has been established.
