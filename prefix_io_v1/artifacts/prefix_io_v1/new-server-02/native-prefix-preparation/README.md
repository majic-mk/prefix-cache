# P1 native GPU Prefix smoke preparation

Status: **PREPARED / GPU MODEL RUN BLOCKED, NOT EXECUTED**.

The parent task reported that official Hugging Face connectivity failed and no model weights are present. The model revision has not been obtained. `frozen-pending-plan.json` therefore records `model_revision: null`, `actual_model_manifest: null`, and `gpu_execution: UNEXECUTED`. It is a frozen diagnostic plan, not an actual model manifest or a result.

## Change

Added only `experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py` plus this preparation evidence. No production cache or model executor changes, no permissions/ledger changes, no model downloads, no GPU execution.

The diagnostic fixes Qwen/Qwen2.5-7B-Instruct, BF16 weights and KV, native TRITON_ATTN, 16-token blocks, a 64 MiB configured KV pool, max model length 256, one sequence, and two sequential requests with the exact same 128 synthetic token IDs (1000 through 1127). Both requests greedily generate 16 tokens with ignore_eos enabled. No reset occurs between requests. Native cached-token counts must be exactly 0 then 112, and both output token ID lists must be exactly equal. A difference fails; no numerical tolerance or fallback is introduced.

The model is a local project directory only. `--model-plan` must be the genuine official fixed-revision plan produced by `prepare_qwen_download.py`, once official metadata is available. The smoke checks every planned file against its exact size and SHA-256 or Git blob SHA-1 before any vLLM import. It requires all safetensors shards from the verified index, rejects extra local safetensors files, uses load_format=safetensors, and checks the fixed Qwen architecture. The pending blocked model artifact must never be used as this input.

The script requires the budget runner's full GPU UUID plus HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1. Offline is not an authorization substitute: run exclusively through the budget runner so permissions and remaining budget are checked. All hash validation and initialization time count toward that runner's wall-clock budget.

Runtime frozen-config.json is created only after genuine local model bytes and the author source commit are validated. It contains model revision, verified file hashes, source commit, prompt IDs, engine/sampling settings, and the bound offline environment. Cold/repeat outputs are saved separately. The original native EngineCoreClient.shutdown(timeout=15) is invoked in finally; the outer runner remains responsible for timeout/session cleanup.

## Pinned source API review

All paths below are relative to third_party/work/vllm-author-build, HEAD 817a7e3124f817cd6e549581d3e5483207a753a4.

- vllm/entrypoints/llm.py:180 and :422: native LLM constructor and generate token-prompt API.
- vllm/engine/arg_utils.py EngineArgs: explicit load_format, cache budget, prefix caching, chunked-prefill, executor, no-offload and generation-config arguments.
- vllm/inputs/llm.py:106: TokensPrompt.prompt_token_ids.
- vllm/config/attention.py:91: string backend conversion; vllm/v1/attention/backends/registry.py:48: native TRITON_ATTN implementation.
- vllm/v1/attention/backends/triton_attn.py:272–299: BF16 compute/KV supported, block size multiple of 16 supported. The older CacheConfig prose about BF16 is narrower than this concrete backend's supported list.
- vllm/v1/request.py:170: PrefillStats initialized independently of log_stats.
- vllm/v1/core/sched/scheduler.py:627–632: local/external cached-token counts populate PrefillStats.
- vllm/v1/engine/output_processor.py:630–632 and :352: counts forwarded to RequestOutput.
- vllm/outputs.py:121: native RequestOutput.num_cached_tokens.
- vllm/v1/core/kv_cache_manager.py:217–229: final prompt token must be recomputed, and hit length is block-aligned. Thus floor((128-1)/16)*16 = 112.
- vllm/v1/engine/core_client.py:134, :304, :645: native shutdown API and in-process/multiprocess implementations.

No py-kvcache modules are imported by this script. cpu_offload_gb=0, offload_group_size=0, kv_offloading_size=None and kv_transfer_config=None keep this isolated to the native GPU cache path. The effective connector, prefix/block/BF16 settings and 64 MiB budget are checked after LLM initialization.

## CPU verification actually performed

Command (exit 0):

```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES='' PYTORCH_NVML_BASED_CUDA_CHECK=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python artifacts/prefix_io_v1/new-server-02/native-prefix-preparation/cpu_review.py
```

This compiles the script, imports only its stdlib definitions, matches every engine/sampling option against the pinned source AST, checks the frozen dimensions, runs --help (exit 0), and verifies that a launch with CUDA hidden fails before creating any run directory (expected exit 1). Neither torch nor vllm is imported. The generated verification JSON, both CLI logs, pending plan, and patch are in this directory. There are no GPU model outputs or a runtime frozen-config.json.

Script SHA-256: 6747fd0e0a7e5f454139e7254d0cf77b03be6340f80267b2df127fde3a622c7f.
Patch SHA-256: dae0dd56a52e3484fe5741cc281390507243d4c5ba357d66c3f116bae44d8d8e.

## Next allowed execution, after real model preparation

The following command has **not** been executed. It does not download weights. First obtain a real official revision/metadata plan and the complete verified weights through the separately authorized download accounting process. Set both variables to those real project-local paths. The GPU runner will recheck permission and remaining cumulative budget; a fresh label is required.

```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
: "${PREFIX_MODEL_DIR:?Set the real project-local verified Qwen model directory}"
: "${PREFIX_MODEL_PLAN:?Set the real official fixed-revision download plan JSON}"
.venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py \
  --label native-prefix-01 --seconds 900 -- \
  .venv/bin/python experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py \
  --model-dir "$PREFIX_MODEL_DIR" --model-plan "$PREFIX_MODEL_PLAN" \
  --output-dir experiments/prefix_io_v1/runs/native-prefix-01/details
```

## Limits

The 64 MiB figure is an explicit KV pool budget. It excludes model weights, workspace and other allocations; actual KV tensor allocation is transparently recorded as null. Native execution/backend compatibility, the observed prefix count, and exact generated-token equality remain untested until the authorized GPU model run occurs. BF16 shape-dependent differences may cause the exact output comparison to fail; that must be reported as a failure, not weakened.

This smoke does not qualify SSD, shared staging, preloading, stores, I/O bytes, production-KV byte identity, performance, research benefit, or full P1. It is a synthetic short-input native GPU path diagnostic only. No online model lookup, cache reset, approximate identity, quantization, or alternate model/executor is introduced.
