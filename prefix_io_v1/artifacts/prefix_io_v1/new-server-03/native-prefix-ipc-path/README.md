# Native Prefix startup compatibility: short IPC path and native prefill

CPU preparation complete. No GPU operation, model load, weight download, engine-source modification, or GPU ledger change was performed by this repair. The actual native-prefix-01 failure remains at experiments/prefix_io_v1/runs/native-prefix-01/process.log and its original frozen configuration/results. The first run failed in the native client ZMQ bind before model weights loaded; do not turn this attempt into a model qualification.

## Minimal changes

1. configure_runtime_environment now sets both TMPDIR and the author's explicit VLLM_RPC_BASE_PATH to the real project directory /root/autodl-tmp/prefix-io-v1-handoff/project/.p1tmp. The path must not be a symlink and must resolve under the project. Both environment values are frozen in frozen-config.json via RUNTIME_PATH_ENV_KEYS. This changes only the smoke process and inherited native workers. Other cache locations remain in experiments/prefix_io_v1/runtime-cache.
2. ENGINE.enable_chunked_prefill changes False to True. Every other ENGINE entry, the 128 input IDs, greedy 16-token sampling, 64 MiB KV budget, expected cold/repeat cached tokens 0/112, exact output comparison, cwd isolation, and bounded KV metadata RPC remain unchanged.

The author's get_open_zmq_ipc_path (vllm/utils/network_utils.py:141–143) concatenates VLLM_RPC_BASE_PATH, a slash, and a 36-byte UUID. vllm/envs.py:672–674 defaults that variable to tempfile.gettempdir(). An explicit project-local RPC base avoids any inherited override or previously cached tempfile choice. The original path plus UUID is 119 bytes; the corrected path is 89 bytes; installed zmq.IPC_PATH_MAX_LEN is 107 bytes.

## Author prefill support evidence

vllm/engine/arg_utils.py:2365–2387 obtains the default from ModelConfig.is_chunked_prefill_supported; a generative model with that True emits the observed warning when False is forced. vllm/config/model.py:1813–1822 returns True for generative, non-encoder-decoder models such as this Qwen2 configuration. This is a native startup configuration compatibility fix, not a new scheduling strategy.

Native scheduler defaults max_num_partial_prefills=1 and long_prefill_token_threshold=0 (vllm/config/scheduler.py:68–84). vllm/v1/core/sched/scheduler.py:654–669 only caps a pending prompt by a positive long threshold and the remaining token budget. With the fixed single 128-token request, 256-token budget, no positive long threshold, and no other concurrent request, enabling the supported feature does not split this prompt because of these caps. This is a source-level conditional conclusion; actual cold/repeat GPU results still require the newly labeled run.

## CPU verification

Run from /root/autodl-tmp/prefix-io-v1-handoff/project:

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/prefix_io_v1_native_prefix -q --junitxml=artifacts/prefix_io_v1/new-server-03/native-prefix-ipc-path/cpu-tests.xml
CUDA_VISIBLE_DEVICES='' .venv/bin/python artifacts/prefix_io_v1/new-server-03/native-prefix-ipc-path/verify_cpu.py
```

51 tests passed in 0.16 s, comprising the prior 49 tests with the expected temporary directory assertion updated, plus two IPC cases. The real installed pyzmq ROUTER socket successfully bound and closed a unique UUID endpoint in the actual project's .p1tmp; its exact test-created socket file was removed afterward. The second case rejects a symlink temporary directory. No torch/vLLM import or GPU access is needed. verify_cpu.py independently checks compile/CLI, the exact one-entry ENGINE difference, unchanged sampling/prompt/RPC and output assertions, runtime/frozen environment keys, real local IPC bind/close, and positive/negative patch applicability.

Evidence: cpu-tests.log/xml, verification.json/log, cli-help.log, native_gpu_prefix_smoke.before.py, test_model_provenance.before.py. The failed 01 log SHA256 is e941b12b8324f54e84c376e9b19cf24436faf42bef962fa6ce69d5e361222170.

The final 0004-native-prefix-short-ipc-and-prefill.patch applies exactly once after final 0003. Before-script SHA256: ae8ffb8320785d582d6991672b8132918f1c616032c04e7aabc40bd605f8dfc2.
Final script SHA256: 6c2c679c672c41401c2ed82ecbf5fd2804777aabf695a280f84b83bf7d86f042.
Patch SHA256: 6025c22f783d8e389c2f5eb038d42de04e68a5ac25e5fd71abb0ce0852c427a2.

Next permitted action: parent-scheduled native-prefix-02 through the authorized budget runner, preserving 01 as the actual failed attempt. These CPU checks do not qualify the model, GPU Prefix result, staging/SSD, or P2.
