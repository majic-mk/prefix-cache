# Observer preparation

The isolated `third_party/work/py-kvcache-observer` worktree contains CPU
pre-integration of the mandatory wait bridge and bounded snapshot hook. Apply
`common/0001`, `common/0002`, `observer/0001`, then `observer/0002` to the locked
upstream source; keep the common baseline separate.

Both optional constructor arguments remain disabled by default:
`progress_run_id=None` and `observation_sink=None`. The normal vLLM adapter does
not enable these options. When explicitly supplied, `SnapshotPublisher` captures
bounded owner-thread observations after the native reactor pump and publishes
only the latest immutable snapshot. Collector or owner-validation failure makes
previous snapshots unavailable; the hook disables a failed observer while native
transfers continue.

CPU and mocked CUDA/io_uring tests are preparation evidence only. Real backend
qualification, the live scheduler ownership/generation adapter, mandatory progress
under real transfers, and end-to-end observation overhead still need the applicable
phase acceptance. No production quota or research scheduling policy is enabled.
