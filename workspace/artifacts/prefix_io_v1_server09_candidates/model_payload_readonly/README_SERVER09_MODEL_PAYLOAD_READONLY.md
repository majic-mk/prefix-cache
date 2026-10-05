# Server09 model asset CPU verification

Actual server execution returned exit 0 with empty stderr. All 11 existing files match the independently pinned ModelScope plan and the SHA-256 values in its saved official provider listing. Four weight shards were read completely, in 8 MiB chunks, without parsing or loading tensors. Total model file bytes: 15,242,788,168. Weight shard file bytes: 15,231,271,888, including safetensors file headers. The index describes 339 tensor entries and 15,231,233,024 tensor payload bytes. Remote verification took 12.720283459872007 seconds.

The script reuses exact ASTs of five locked functions from native_gpu_prefix_smoke.py: require, under_project, digest_file, validate_provider and validate_local_model. It never executes that script's main or runtime environment/cache setup. The original offline model validation therefore also checked architecture, BF16 configuration, exact sharded index membership and saved provider evidence. Before/after device, inode, size, mtime and ctime remained stable for every model file. All six small provider evidence files were byte/SHA rechecked after payload reads.

Execution was purely CPU file reading: CUDA_VISIBLE_DEVICES was empty; Python used -B -I -S; framework imports, tensor/tokenizer/model loads, GPU runs, downloads and remote application file writes were all zero. Existing source, data, permissions and GPU ledger were not changed. This is an asset integrity result, not a normal model execution, CUDA kernel compatibility check, GPU qualification, method-effect result or P4 completion.

The fixed plan is 7,310 bytes, SHA-256 9b1c79986aa1fd62eb9064ecb8b4e2a03fc235a34ac2d69c33a4fcb531a91017. The saved official pinned-file response is 6,065 bytes, SHA-256 e84a0c966b2e295e3182e47e13c678765381503ba4819e2176c3589f7352d6e4. Every expected model hash was available and matched that response. Its body leaves LatestCommitter.Id empty; revision binding uses the already frozen API URL Revision and response SHA, and is not claimed as a fresh network verification. No expected hash was generated from the local weights.

The first preflight run was rejected before any weight reads because an added check incorrectly expected the API body's empty commit-ID field to contain the revision. Preserve SERVER09_MODEL_PAYLOAD_READONLY_RESULT.json as failed preflight evidence. The corrected run preserves this source limitation explicitly and is recorded separately in SERVER09_MODEL_PAYLOAD_READONLY_RESULT_V2.json.

Frozen successful script: verify_server09_model_payload_readonly.py, 11,990 bytes, SHA-256 405f3a16f301434f9115e7a942b43baccf4bc4457878e9fc10652a09a2970991. Successful receipt: SERVER09_MODEL_PAYLOAD_READONLY_RESULT_V2.json, 97,302 bytes, SHA-256 6eed7b029b9d8c6f0e01e3d1ea2e5e9ea7aaee004d0266fab50c0a2a727c1cbb. It includes the exact remote command, source AST hashes, stream progress, expected and actual hashes, stable metadata, source references and final qualification limits.

Actual remote command was the source supplied on stdin, with no server script written:

```sh
cd /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -I -S - <<'SERVER09_READONLY_MODEL_PY'
# Exact script bytes are embedded in the successful receipt's command field.
SERVER09_READONLY_MODEL_PY
```

The next allowed stage is CPU source/worker-provenance and normal-model entry preparation. A real G2 execution still requires its own frozen authorization, reservation in the unchanged original GPU ledger, a real full-step CUDA event timer and complete I/O/drain evidence. G1's two authorized GPU runs remain exhausted.
