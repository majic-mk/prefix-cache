# Bounded restore-parent D_R candidate

This is a private candidate for the explicitly authorized, exploratory **U vs
F8+D_R combination**. It is not a pure D_R ablation or a production qualification.
The frozen original files have not been edited. There is no interference table,
allocator release authority, new executor, or new backend here.

## Original core and minimal wiring

The C5 `_prefix_p4_order` and original `P4Policy.choose` already reorder bounded
native SSD-ready entries. Original owner-thread closure observations already
measure first complete-ready observation to successful whole-parent drain.
Previously all parent/witness estimates were `None` and history was diagnostic.

The private bridge setup is:

```python
bridge.configure_restore_development(
    calibration_or_None,
    history_max_age_ns=declared_age_ns,  # positive, at most 600 seconds
    order_journal=common_journal,
)
```

It must run before original `bind`, only for `dependency_only`. C5 still requires
the original fixed controller; the authorized method uses F8. Only restore
targets/witnesses receive forecast input. CPU/GPU release witnesses retain their
old unknown values. The original generation, AND protectors, freshness,
mandatory/progress, bounded-ready and original whole-window H2D continuation
checks remain. Original order, parent terminal, file terminal and copy/release
functions are unchanged. No code here frees an owner or retires a native fence.

## Actual calibration input

```python
calibration = load_restore_calibration(
    root, current_refs, actual_P316_result_ref, completed_original_guard_ref,
    expected_gpu_uuid=actual_current_uuid,
    expected_common_runtime_domain_sha256=frozen_common_domain,
    expected_runtime_refs=frozen_common_source_leaf_refs,
    now_ns=time.monotonic_ns(), max_age_ns=declared_age_ns,
)
```

The thin original P316 adapter supplies `result.probe` metadata read from its
actual process/device/frozen source and saves original
`current_native_snapshots[*].p4.eta_history`. Required probe fields are `run_id`,
`gpu_uuid`, `common_runtime_domain_sha256`, `runtime_refs`, `subprocess_sid`,
`subprocess_pid`, `original_engine_shutdown_returned`, `native_tail_drained`,
and `current_native_snapshots`. Result, guard and source leaf references must be
in the independently frozen current reference map and match actual file bytes.
Original guard exit/child exit must be zero, timeout/error absent, session
drained and before/after members empty. Source/device/raw timing errors are
ordinary errors and stop setup.

Samples are the original actual successful whole-parent history, not a sum of
SSD/H2D unit times. They keep the exact original geometry and all thirteen
original context values. Forecasts require four fresh samples in a single exact
pure-SSD complete-ready geometry/context with `copy_ready_count == 0`. Other
cells remain uncovered. The point is the original `median_high` arithmetic;
error is empirical dispersion. Absolute ETA uses this method parent's preserved
first observation. An exceeded forecast, changed context or expired calibration
falls back. This is **not a deterministic earliest-completion guarantee**.

Only fully byte/source/device/guard-validated calibration lacking an actionable
cell raises `RestoreCalibrationCoverageMissing` with actual sample/cell counts.
The caller may configure `None` and report safe fallback, as authorized. It must
not catch generic `ValueError` and reinterpret invalid calibration as coverage.
The candidate then retains the F8+D_R configuration and reports zero D_R action;
combination timing cannot be attributed to D_R if no order changes occurred.

## Common scalar journal and actual submit boundary

Both arms instantiate `RestoreOrderJournal(run_id)` before the original reactor
initialization and assign `_prefix_restore_order_journal`. It records actual
SSD-ready parent/file identities around the original order call, plus accepted
`queue_read` work IDs and original `user_data`, FD and slot. It exports through
the original owner snapshot, including U with bridge disabled.

D_R additionally records every actual changed before/after permutation and
dependency parent IDs, first parent/reason/coverage windows, complete-ready and
known-ETA parent IDs, and full reason/window counts. Queue acceptance is labelled
as queue acceptance. The thin runner separately taps original successful kernel
submit `batch[:count]` and joins by `user_data` to prove actual commit order.
Neither nominal choices nor these queued rows alone prove kernel submission.

Submission rows are bounded at 16,384; order changes and deduplicated diagnostic
windows each at 1,024. Any overflow or identity error invalidates order evidence
while leaving native operations unchanged. No raw pointers or owner objects are
retained. Save one complete final journal; avoid unnecessary duplicate exports.

## Local CPU evidence

Command:

```text
python -B -I -S test_restore_development_cpu.py
```

Nine meaningful tests cover source/device/guard/timing rejection, typed coverage
only, exact-context/overrun fallback, real original policy value contracts and
generation/H2D protection, native parent/file identity, journal overflow,
success/failure branches of the actual copied original read-submit function,
and an AST audit of preserved native release/order functions. All inputs in
these tests are explicitly CPU fixtures. They are not actual GPU receipts,
calibration, performance evidence or qualification. GPU/RPC operations: zero.
