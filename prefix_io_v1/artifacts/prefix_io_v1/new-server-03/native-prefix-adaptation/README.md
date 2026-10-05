# Native Prefix smoke: official provider provenance adaptation

Stage: P1 preparation only. No GPU operation, model load, weight download, or core source change was performed by this subtask.

## Changes

- Updated experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py.
- Added tests/prefix_io_v1_native_prefix/test_model_provenance.py.
- Accepted source origins remain an exact set: https://huggingface.co and https://modelscope.cn.
- Existing HF plans without an explicit provider normalize to huggingface; ModelScope requires provider=modelscope. ModelScope revisions are recorded under modelscope_git_commit, never treated as HF commits.
- ModelScope requires Qwen publisher/channel/GitHub identity and all six saved official provenance roles: official_release, provider_model, pinned_files, config, safetensors_index, tokenizer_config. Each referenced project-local evidence file is checked against its saved size and SHA-256; endpoint identities and fixed-revision query values are checked.
- ModelScope model files require content SHA-256. Existing actual local model size/hash, architecture/BF16, shard-index completeness and extra-safetensors rejection remain.
- Runtime frozen-config.json now records source, provider, provider-specific revision namespace, actual model revision, manifest_sha256 (and retained plan_sha256), and verified provenance references.
- On script startup only its own process environment is set to VLLM_USE_MODELSCOPE=0, VLLM_NO_USAGE_STATS=1, VLLM_DO_NOT_TRACK=1; these values are included in the runtime frozen environment. HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 and one complete approved GPU UUID still must come from the budget runner.

Runtime cache isolation is also fixed before vLLM import: VLLM_CACHE_ROOT/TRITON_CACHE_DIR/TORCHINDUCTOR_CACHE_DIR/XDG_CACHE_HOME/HF_HOME point respectively to the project experiments/prefix_io_v1/runtime-cache/{vllm,triton,inductor,xdg,huggingface}; TMPDIR points to experiments/prefix_io_v1/runtime-tmp. These directories are created after verifying their resolved paths remain in the project. All six paths are frozen in runtime evidence. The author envs.py:654–658 directly honors VLLM_CACHE_ROOT. The existing CPU environment test now asserts every directory is created and remains project-local.

No ModelScope SDK or provider-specific model executor is introduced. The same author vLLM 817a7e3124f817cd6e549581d3e5483207a753a4 receives a fully validated local HF-format model directory. ENGINE, SAMPLING, prompt IDs and expected cache counts were compared against the saved pre-adaptation script and are unchanged: BF16 weights/KV, 64 MiB configured KV pool, native TRITON_ATTN, 128 input tokens, 16 output tokens, native cached-token counts 0 then 112, exact output-token equality. No intermediate cache reset.

## Verification actually performed

All commands ran on the server with CUDA hidden and no torch/vllm/modelscope import:

```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/prefix_io_v1_native_prefix \
  --junitxml=artifacts/prefix_io_v1/new-server-03/native-prefix-adaptation/cpu-tests.xml
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  artifacts/prefix_io_v1/new-server-03/native-prefix-adaptation/verify_cpu.py
```

24 synthetic CPU tests passed in 0.09 seconds. They cover both providers, provider mismatch, branch/null revision rejection, unapproved origins, required publisher/provenance, hash tampering, wrong revision URLs, invalid evidence paths/roles, hash algorithm restrictions, modified weights, extra safetensors and local environment settings. Synthetic bytes are never loaded by a model.

Source compilation and pinned-author constructor field AST checks passed. --help returned 0; a launch with CUDA hidden correctly returned 1 before making a run directory. Reverse application check of 0001-provider-provenance.patch passed.

The real saved manifest artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json passed only the provider/provenance validation, including the six real saved evidence files. It identifies ModelScope revision 16c174980d8a1492910551634b4969e69cdc2444. Manifest SHA-256 is 9b1c79986aa1fd62eb9064ecb8b4e2a03fc235a34ac2d69c33a4fcb531a91017. No weight bytes or complete local model were validated by this subtask.

Script SHA-256: e2badf5a54ed9240dd3a103fc0b05e8814f28923454ed3c9570b2f8d074b5e0d.
Patch SHA-256: e8bec3b4c71ebfbbab4cd5f2d5ca0093bbcce5a8d04ce062f658b72e3ae0b238.
The patch applies on top of the existing native smoke preparation, not instead of that original new-file patch.

## Next allowed operation

The parent task must complete the separately authorized, budget-accounted download and verify the exact manifest before choosing the real model directory. Only then may it run the smoke through run_gpu_stage.py with its fixed GPU UUID and remaining budget. Example, not executed here:

```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
: "${PREFIX_MODEL_DIR:?Set the real verified project-local model directory}"
.venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py \
  --label native-prefix-01 --seconds 900 -- \
  .venv/bin/python experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py \
  --model-dir "$PREFIX_MODEL_DIR" \
  --model-plan artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json \
  --output-dir experiments/prefix_io_v1/runs/native-prefix-01/details
```

This adaptation does not establish GPU model execution, actual KV tensor allocation, staging, SSD, production KV byte identity, performance, research benefit, or full P1 qualification. Unknown HF equivalence is not inferred from the ModelScope model name or commit.
