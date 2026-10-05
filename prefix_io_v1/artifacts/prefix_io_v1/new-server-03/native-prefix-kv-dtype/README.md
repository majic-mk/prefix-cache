# Native Prefix auto KV dtype with strict BF16 verification

Status: CPU repair passed; no GPU run, model load, download, kernel/attention edit, sampler edit, or GPU ledger modification was performed in this repair. The actual native-prefix-03 failure remains unchanged. This script is ready for a separately labeled, parent-scheduled budgeted native-prefix-04 attempt.

## Actual failure and original author mapping

native-prefix-03 completed the native sampler JIT and reported GPU KV cache size 1,168 tokens, then failed during native warmup at vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:353–359. That kernel requires the string cache dtype to be auto or a supported quantized format; the prior explicit bfloat16 string was rejected. The failure log SHA256 is b49cde4fdecf86fea2306de9ee9d0f075b0b6154f519ff62b31bbc2cd640f84b. JIT proof is maintained separately by the parent.

The pinned author's GPUModelRunner calls kv_cache_dtype_str_to_dtype(cache_config.cache_dtype, self.model_config) at vllm/v1/worker/gpu_model_runner.py:448–450. The original implementation at vllm/utils/torch_utils.py:396–402 returns model_config.dtype for auto. The model remains fixed unquantized Qwen BF16; therefore this uses the original BF16 tensor allocation path while passing the auto string supported by the original Triton cache writer. No conversion implementation or backend was replaced.

## Exact changes

ENGINE changes only kv_cache_dtype from bfloat16 to auto. Weights dtype remains bfloat16 and quantization remains None. The existing effective configuration checks are grouped in validate_effective_config and extended to require actual config.model_config.dtype == torch.bfloat16 and config.model_config.quantization is None. Effective cache_config.cache_dtype must equal the frozen ENGINE string auto; Prefix/block size/offload/64 MiB checks remain strict.

The report records effective_model_dtype, effective_model_quantization, effective_kv_cache_dtype, and effective_kv_cache_memory_bytes. The existing worker metadata RPC still strictly verifies every one of 28 initialized CUDA KV Tensors has torch.bfloat16 dtype, unique backing storage sizes agree with the native configuration, and bytes stay within 64 MiB. Its implementation is unchanged. actual_kv_tensor_dtype is also copied from that verified native worker metadata into the result. CPU tests provide no actual tensor evidence.

Prompt IDs, 16 greedy output tokens, exact output comparison, cold/repeat cached-token requirements 0/112, single native RPC, all runtime environment/cache/IPC/cwd behavior and other ENGINE values are unchanged. The parent's existing launch environment supplies the private FlashInfer cache, no-download policy, private CUDA compiler and .venv/bin PATH.

## Verification and patch

Commands from /root/autodl-tmp/prefix-io-v1-handoff/project:

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/prefix_io_v1_native_prefix -q --junitxml=artifacts/prefix_io_v1/new-server-03/native-prefix-kv-dtype/cpu-tests.xml
CUDA_VISIBLE_DEVICES='' .venv/bin/python artifacts/prefix_io_v1/new-server-03/native-prefix-kv-dtype/verify_cpu.py
```

61 CPU tests passed in 0.14 s: the previous 51 plus 10 configuration guard cases. New tests use explicit synthetic metadata to check acceptance of auto/BF16 and refusal of changed effective model dtype, quantization, cache string, pool size, block size, Prefix setting or offload connector. They import neither torch nor vLLM. verify_cpu.py also evaluates an AST-extracted copy of the actual pinned dtype mapping with CPU-only metadata, checks the exact one-entry ENGINE change and unchanged worker function, compiles the script, runs CLI help, and checks the patch forward and reverse. All passed.

0005-native-prefix-auto-kv-bf16.patch applies exactly once after 0004.
Before script SHA256: 6c2c679c672c41401c2ed82ecbf5fd2804777aabf695a280f84b83bf7d86f042
Final script SHA256: 7d31cce851ca2334fedbe6b66c4e2d9bfff19e8768bafaa52f68e414518d77e2
Patch SHA256: 2ada5b01178e8427eafb7e3b956ed9c3459c87cfe04551ce78d9db1b3010018c

Evidence: native_gpu_prefix_smoke.before.py, cpu-tests.log/xml, verification.json/log, cli-help.log. Actual Prefix and tensor dtype qualification still require the real budgeted GPU run. Staging/SSD and P2 are not qualified by this change.
