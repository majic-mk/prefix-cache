# P4-02 CPU observation seam delivery

This is CPU seam preparation, not an installed real GPU collector. Every module
keeps `production_qualified=False`; no native/GPU origin string supplies
qualification. The actual vLLM worker was not imported, attached, or executed.

## Actual additions

- `p4_gpu_step_observation.py`: optional wrappers around original
  `_prepare_inputs` and `_model_forward`. Original calls, arguments, results,
  and exceptions are preserved. Disabled observation leaves original lookup.
  It snapshots actual prepared input batch after original preparation, restricts
  candidates to known same-context pure decode with one token per request,
  and rejects mixed, speculative, async, distributed, unknown or changed work.
  The native ordinal is actual `_profile_step - 1`, zero based.
- Injected Event factory records only original model-forward scope. Resolution
  queries event readiness at the tail without synchronization or waiting.
  `gpu_elapsed_ns` is a real explicit dataclass field; CUDA-event elapsed,
  host call timestamps, and optionally mapped wall timestamps are separate.
  Missing reference, fallback, event failure, or observation overflow cannot
  become a measured GPU cost. Additional cross-clock qualification is required
  even for diagnostic native-window attribution.
- `p4_native_window_journal.py` subclasses the existing `StageAccounting`
  API, first calling the original acceptance/completion implementation. Only
  bounded scalar event identities and immutable owner frames are retained.
  Existing native owner observation-sink injection is supported. Worker
  `published()` reads copies of frames/events and never reads native stats.
  Missing owner coverage, loss, clock ambiguity, wrong geometry, partial
  completion, or extra stages remain unknown. Only SSD-read/H2D single-stage
  diagnostics are representable here. D2H/store-write isolation is unproved.
- Neither module infers sample output token IDs, post-context, whole decode
  step, full output count, full-run I/O totals, or final accepted-I/O drain.
  Those need separate evidence from original sampling and final native owner
  accounting. V2 measurement selection and cost verification remain in the
  existing verifier; no statistics were copied into these modules.

## Actual CPU commands and results

The exact CPU runner is saved as `cpu-test-command-02.py`. It invokes:

```
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv-prefix/bin/python [saved import-guard runner]
pytest -q --import-mode=importlib tests/prefix_io_v1_p4_observation \
  --junitxml=artifacts/prefix_io_v1/server08-p4-02-cpu/observation-cpu/run-02/tests.xml
```

Actual final result: **42 passed** (0 failed, 0 skipped). Fixtures explicitly
use fake events and CPU source identities. Test coverage includes original
API call/result/exception preservation, observer-off path, exact prepared
batch, nonblocking tail resolution, failed reference/fallback, event failure,
bounded overflow, resource-owner nonretention, original StageAccounting
injection, owner-only frame publication and unknown window attribution.
The earlier 35-test run remains as run-01; run-02 is the final added counterexample
set. No claim of CUDA timings is derived from either fixture run.

Evidence: `run-02/tests.log`, `run-02/tests.xml`,
`run-02/CPU_GUARD.json`, `source-lock-v1.json`.
The guarded runner denies imports of torch/vllm/py_kvcache/cupy/pycuda/cuda.
Actual imported backend modules: none. Actual GPU runs: **0**.
The budget ledger SHA stayed
`31199998369e35fcd40daddd7af340353b0feaf134bbe37d2dd8f1f35391efc1`.

The earlier thin launcher/calibration/raw-preparation delivery remains **116
passed** at `gpu-next-day/cpu-tests-08.xml`, with exact command,
owned-source-lock-v5, CPU-first launch denial 78 and unchanged source/ledger
receipts. The two observer modules did not change those launcher files.

## Actual source and binary boundary

The new author overlay received precisely 92 missing pure Python files from
the same author HEAD after path/bytes/SHA/no-overwrite verification. Copy and
static receipts are `gpu-next-day/AUTHOR_PYTHON_OVERLAY_COPY_92.json` and
`AUTHOR_OVERLAY_CPU_STATIC_RESULT.json`. This fixed known missing Python
paths without rebuilding an image or proving ABI.

A read-only inventory found **seven** .so files in the old author build.
Current G1 allows exactly the existing top-level `_C` and
`_C_stable_libtorch` locks, as recorded in `source-lock-v1.json`.
Full model attention may require nested FlashAttention FA2 (191,414,528 bytes)
and FA3 (44,999,560 bytes), plus other paths depending on selected runtime.
Those extra paths are not authorized by G1's two-binary fallback. They need
separate exact source/binary eligibility; the new overlay is not a rebuilt
standalone author installation. No .so was copied, downloaded or rebuilt.

## Next permitted GPU phase

Today's explicit CPU-only scope still blocks real `--launch` with exit 78
before old GPU guard/preflight/budget consumption. A fresh explicit GPU-scope
receipt, final source lock, exact context/path/storage eligibility and existing
GPU budget guard are required before G1 off/shadow primitive qualification.
That phase must validate actual Python module origins and the locked binary ABI.

Real model measurement collection additionally needs runtime installation of
the optional seam, original sampling/full-output trace, final native drain,
clock-reference qualification and a pre-registered V2 ordinal/exact-load
selection. These have not run today. G1 success alone would not establish model
collection, cost-table activation, I/J policy benefit, full P4 or P5 SLO.
