# Server11 native conditional cost validator

This module performs no GPU operation, I/O submission, model execution, cache
eviction, system change, or author-source mutation. It is a finite evidence
bridge around the existing native observer and original paired estimator.

## Interfaces

`analyze_paired(plan, windows, journal=..., drain=..., original_source_path=...)`
validates the complete raw collection and computes a **candidate**. Its native
execution and conditional qualification booleans remain false. Tests fabricate
raw numbers solely to exercise rejection paths, never to claim GPU execution.

`verify_native_cell(project, plan_ref=..., measurements_ref=..., guard_ref=...,
expected_plan_ref=..., expected_guard_ref=..., original_source_path=...)` reads
the expected frozen plan, raw measurements and actual original guard result.
The guarded parent records the expected plan before launch and the expected
guard after natural session closure. The final six windows use six fresh original
processes, handlers and private storage directories within one original guard.
The final entry validates source-closure membership, independent before/after
receipts, collector/runtime pins, real GPU identity and successful non-forced
OS drain. Origin strings or caller PASS booleans do not substitute for this.

`test_native_conditional_cost.fixture()` gives a complete schema example with
CPU-fabricated values. The actual launcher must serialize **its real values**.
The measurement envelope has scope `server11_native_paired_measurements_v1`,
origin `native_gpu_recording`, `plan_ref`, and `windows`, each carrying its real
`native_journal`, original `native_post_shutdown` and directly read
`native_tail_assertions`, plus `source_verification_refs.before/after`.
Source receipts have `phase`, `source_lock_ref`, `files_verified`, and `failed`.
The exact frozen source-lock file count is required; none may fail.

## Original arithmetic, independent held-out gate

The source-locked original `p4_paired_measurement_verifier.py` is exactly 48,136
bytes with SHA-256
`3cd840c6dd388e3dcf9023e54dd5eb172ab59777c66f731a5d1e3f9afecd99ac`.
The numerical AST and original CUDA-duration helpers are compiled without
altering their mathematical expressions. No duplicate estimator is maintained.

Two calibration pairs execute AB and BA; the independent held-out pair is AB.
The calibration-only application of the original formula determines T0, delta,
and margin. Its upper bound is frozen for the held-out comparison. Held-out
values never update this bound or margin. The ordinary all-pair original result
is also retained, clearly marked **audit only**, to show that using its residual
would fold the held-out error back into the estimate.

The new finite engineering gate is
`zero_observed_holdout_underprediction_no_refit_v1`: every selected held-out
action duration must be no greater than the calibration-only bound. This is not
an existing SLO, a population probability guarantee, or an arbitrary percentage
error tolerance. Any observed underestimate leaves conditional qualification
false. One held-out sample is reported as one sample, never expanded into a
confidence claim. A missing externally frozen step budget leaves interference
allowance `null`; no operational budget is invented.

The initial cell is single-request decode at pre-context 143; measured offset
16 after a 128-token prompt whose 112 tokens are already GPU-cached. Each A/B
request produces all 128 original tokens. The SSD-only preload is independent
of the foreground request, with 1/2/4/8 storage quanta and explicit operation
count. Full prompt, first-block prefix family, workload and trace hashes are
recomputed, ensuring calibration/held-out separation is substantive.

Each fresh process first performs a separate 128-output original warmup and a
one-token original completion flush. The in-request `warmup_offsets=[1]` is a
different, excluded observation window. `prepare_and_verify_native_cost.py`
provides CPU-only `--prepare`, `--verify` and `--window-check` commands. The last
can reject a bad closed child while the original guard is still running, but
cannot grant final native or conditional qualification. External request IDs
and the original `add_request` returned native IDs remain distinct; the native
ID is matched against original frames without renaming their contents.

## Important fail-closed boundaries

All 128 original ClosedFrames and CUDA-event witnesses must match real output
IDs and contiguous native ordinals. The original scalar frame's GPU fields stay
null; actual CUDA elapsed durations stay in separate witnesses. No absolute
GPU/host clock mapping is invented. SSD acceptance must occur in both the
original host step and the strict causal bracket from completed start-event
query to the end-event record. Accepted/completed native events, every owner
frame counter, byte results, no-I/O baseline, initial/final empty owners and
actual shutdown drain are independently reconciled. There is no early release
credit. Extra synchronization, lost events, partial outputs, short reads, stale
source/identity, output mismatch, or independent-prefix aliasing reject.

Successful finite evidence may set only `conditional_cost_cell_qualified` for
the exact returned condition. Other conditions reject. `production_qualified`,
`strategy_effect_verified`, and `resource_release_credit` remain false.

## CPU verification

Run Python 3.12 with `-B -I -S test_native_conditional_cost.py -v`.
On the server set `NATIVE_ORIGINAL_ESTIMATOR` to the frozen original verifier
file before running. The 14 validator tests and 8 serializer tests cover unchanged arithmetic, held-out
underprediction without refit, missing output/event, altered token, source
drift, short CQE, counter mismatch, ambiguous causality, nonempty drain, aliasing
payload, added synchronization, and insufficient origin-only guard evidence.
