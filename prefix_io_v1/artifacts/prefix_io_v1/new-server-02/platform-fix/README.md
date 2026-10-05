# Common author-vLLM full GPU UUID compatibility fix

Source: author vLLM commit 817a7e3124f817cd6e549581d3e5483207a753a4.
Read applicable AGENTS.md before editing. Project .venv pre-commit4.6.2 and
generated hook were already installed. No PR, model, GPU call, environment
change, main lock/state/doc change or GPU ledger modification was performed.

The original real failure is copied verbatim from
experiments/prefix_io_v1/runs/author-platform-01/process.log into
gpu-before-process.log. Full CUDA_VISIBLE_DEVICES UUID reached Platform's
int(token) conversion through NvmlCudaPlatform.get_device_uuid.

Only third_party/work/vllm-author-build/vllm/platforms/cuda.py changed: 17 added
lines override the conversion in NvmlCudaPlatform. GPU- tokens resolve through
NVML GetHandleByUUID then GetIndex; the helper uses the existing
with_nvml_context wrapper. Numeric, unset and empty paths delegate to the
original super method without additional NVML calls. No environment rewrite,
index guess, unknown-UUID fallback or research-policy condition was added.
Nested NVML lifetimes remain balanced. This is a common compatibility fix for
all experimental arms. The original upstream checkout stays clean.

Patch:
patches/prefix_io_v1/common/vllm-author/0001-full-gpu-uuid.patch
This is deliberately outside the py-kvcache common patch glob.

CPU tests use exact AST-extracted source methods and the original lifecycle
decorator with explicit NVML doubles. They do not import the GPU-probing cuda
module, Torch, or NVML. These are CPU unit tests, not runtime CUDA qualification.
They cover full UUIDs, mixed numeric/UUID lists, multiple UUIDs/logical indices,
unchanged environment/numeric/empty/unset branches, unknown UUIDs, failed
initialization/index queries, invalid/out-of-range tokens, and nested caller
cleanup on success/failure.

Results:
- cpu-before.txt/xml: 8 expected UUID-related failures, 8 baseline cases passed.
- cpu-after.txt/xml: 16 passed, zero failed, 0.17 seconds.
- ruff-isolated.txt: pinned upstream ruff-check and ruff-format both passed.
- verification.json: forward apply check against clean upstream, reverse check
  against modified build worktree, and git diff --check all passed; protected
  permissions and GPU ledger SHA256 values unchanged.

Lint note: running the full upstream pre-commit config initialized unrelated
repositories and exceeded 80 seconds. The exact owned pre-commit descendant
processes were stopped and recorded in lint-initialization-stop.json. The same
upstream-pinned ruff hooks (v0.14.0) were then run from ruff-only.yaml and passed.
No other hook was claimed as run.

CPU command:
```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -ra tests/prefix_io_v1_vllm_platform --junitxml=artifacts/prefix_io_v1/new-server-02/platform-fix/cpu-after.xml
```

The main agent must rerun real platform metadata and import qualifications under
the existing authorized GPU budget. This Python-only fix does not rebuild or
alter the author's compiled copy kernel. No post-fix GPU result is invented here.
