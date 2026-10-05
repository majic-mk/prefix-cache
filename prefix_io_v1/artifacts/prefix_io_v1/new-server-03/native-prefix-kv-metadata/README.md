# Native Prefix bounded KV storage observation

Status: CPU preparation passed; this work ran no GPU, downloaded no weights, loaded no model, and did not change author engine source or the GPU ledger.

The smoke calls the author's existing LLM.collective_rpc exactly once after LLM initialization and before the cold request. LLM.collective_rpc (entrypoints/llm.py:570–601) delegates through LLMEngine and EngineCore to UniProcExecutor (v1/executor/uniproc_executor.py:85–108). WorkerWrapperBase forwards model_runner access. The fixed Qwen configuration must expose 28 initialized CUDA BF16 Tensor entries and one layer group with 28 distinct layer names. Only tensor shape, device, dtype, untyped_storage().nbytes() and storage.data_ptr() metadata are read. The loop never reads tensor values, copies tensors, synchronizes CUDA, scans KV blocks, or installs hooks.

Storage base pointers, rather than Tensor view pointers or Python object identities, deduplicate shared backing allocations. Pointers remain inside the worker callback and are not returned. The unique storage sum must be positive, <= frozen 64 MiB, and equal the sum of initialized native KVCacheConfig.kv_cache_tensors sizes. Native block rounding may produce a sum below the configured budget. Zero/missing/invalid metadata and all exceptions fail the smoke rather than fabricating a result. Only one worker response is accepted.

New real-run evidence will be kv-tensor-storage-metadata.json and matching smoke-result.json fields. The measured quantity is unique Tensor backing storage nbytes. It is not CUDA allocator reserved bytes, not process GPU memory, not weights/workspaces, not a free-memory measure, and not a physical release witness. The latter two fields remain null.

## Author callable serialization

Pinned v1/serial_utils.py:221–231 rejects callable serialization unless VLLM_ALLOW_INSECURE_SERIALIZATION=1; ext_hook at 477–481 checks the same flag. The script explicitly sets this official opt-in in configure_runtime_environment before importing vLLM, records it in frozen-config.json, and passes the inherited process environment to the native workers. This is the trusted local callback defined in this same project script, on the offline single-user LLM/uni path. No bytes encoding workaround is used, and no system or engine configuration is modified.

The script does not launch the OpenAI HTTP/API server or any request listener. On this single local DP engine path, v1/engine/utils.py:979–1007 selects ipc:// addresses for client/engine data, and :1100–1131 selects the local IPC handshake. multiprocessing fork/spawn workers inherit the parent environment. This is a source-level statement about the selected native offline path, not an observed GPU socket scan and not a blanket assertion that native CUDA/distributed initialization never creates any internal transport sockets. Do not reuse this serialization flag for an exposed untrusted service.

The public RPC timeout=10.0 is requested, but the pinned UniProcExecutor synchronous implementation does not enforce it. The outer authorized budget runner remains the wall-time/process-session guard. The callback itself is bounded to 28 tensor entries and small native configuration metadata.

## Patch chain and CPU evidence

0003-native-prefix-kv-storage-metadata.patch is the final patch against the post-0002 cwd script (SHA256 37eebb38bfa2d79af8219172c55838e2f4ad01f6d8a6cbc0e938ae1600e9d569). Apply after the provider patch and cwd patch, exactly once. It includes the official serializer environment flag. The earlier unready version is preserved as native_gpu_prefix_smoke.before-env.py, 0003-native-prefix-kv-storage-metadata.before-env.patch and verification.before-env.json; do not apply the before-env patch as well.

Final script SHA256: ae8ffb8320785d582d6991672b8132918f1c616032c04e7aabc40bd605f8dfc2
Final patch SHA256: 657efb3d30e7ab02df30a9f4e4b4fe89897904ac8f2b95ce7462cabc7913db46

Commands, run from /root/autodl-tmp/prefix-io-v1-handoff/project:

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/prefix_io_v1_native_prefix -q --junitxml=artifacts/prefix_io_v1/new-server-03/native-prefix-kv-metadata/cpu-tests.xml
CUDA_VISIBLE_DEVICES='' .venv/bin/python artifacts/prefix_io_v1/new-server-03/native-prefix-kv-metadata/verify_cpu.py
```

49 CPU tests passed in 0.13 s. The 25 new cases use explicitly synthetic test-only Torch/Tensor metadata classes and do not import torch/vLLM or emulate a successful GPU result. They cover 28 separate storages, distinct wrappers/views sharing one storage, invalid/missing/zero/budget/shape/device/dtype/config metadata, exceptions, and process environment/frozen flag placement. Compilation, CLI help, unchanged engine/prompt/sampling and single metadata-only RPC AST checks, retained cwd ordering, and forward/reverse patch checks passed. See cpu-tests.log/xml, verification.json/log and cli-help.log.

The next permitted step is only the parent-scheduled, fully budgeted real native GPU Prefix smoke after verified model download. Native staging/SSD integration and P2 remain outside this smoke.
