This append-only delivery fixes future finite development/effect startup's
collector selection. It does not change the running calibration source lock,
its old runtime/driver files, cache, executor, native stages or cost estimator.

native_runtime_v2.preflight_collector_binding(root,gates,driver=...) selects the
actual descriptor.collector_ref (collector_source_ref is accepted only when
it agrees with collector_ref). For development/effect it requires the exact
V2 source13016B/SHAa710f3f55ecbc6d8fbce8f3cf85ce4587daf0e1a863783df25a4ba8bc37e2aa0,
its original25070B/SHA9915ca0c18147eb26f44e2292f541da95e33d0321c6236b224d19d4446e09e9d
delegate, actual calibration plan bytes, same device/common domain and exact
calibration-source ancestry in current runtime_refs. Appending new host source
is allowed; changing any inherited row is rejected. Whole-lock hashes need not
match. The original activation parser/issuer retain their full gates.

The helper checks copied CPU metadata/source only. It cannot issue a table,
grant a GPU capability, supply a deadline/SLO or authorize strategy I. It runs
before output creation, framework imports or LLM initialization. The selected
source is then used by the original install/export/drain path. Qualification
and no-table shadow continue selecting the original collector unchanged.

strong_trace_runner_v2 changes only verify_configuration's final selection:
finite development/effect require native_runtime_v2 and invoke the same pure
source gate during CPU preflight. Phase/workload validators, permissions,
original guard/budget and all other methods remain unchanged. Formal U/I input
and missing natural data/deadline are not bypassed by this append-only fix.

Six CPU tests use actual old/new source bytes and execute actual source gate,
early execute rejection and the new original configuration-selection AST.
Source-only positive fixtures are explicitly nonissuable and have no table,
deadline or GPU permission. Missing/old/drifted sources and wrong ancestry are
rejected. No torch/vLLM/native backend is imported by the tests.

Server commands from the project root, D=artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004:

    CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S $D/raw_runtime_collector_binding/test_runtime_collector_binding_cpu.py
    CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S $D/runner/strong_trace_runner_v2.py --help

No GPU/RPC/model/CUDA event or private qualification is created. Current
calibration evidence remains bound to its original frozen source lock.
