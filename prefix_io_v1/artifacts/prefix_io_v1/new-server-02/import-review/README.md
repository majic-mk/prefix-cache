# Author import qualification preparation

Added experiments/prefix_io_v1/scripts/qualify_author_imports.py.
Only AST parsing and compile-to-code-object were executed. No runtime imports,
NVML/CUDA operations, package install, model, ring, or handler construction was
performed by this preparation task.

The script must be executed inside run_gpu_stage.py after the author source
build/install succeeds. vLLM platform imports may probe NVML. It does not change
sys.path or environment, uses no fake objects or fallbacks, and requires actual
VLLM_AVAILABLE, PLAN_API_AVAILABLE and TORCH_COPY_AVAILABLE to all be True.

It requires the pinned Git HEAD of author vLLM, common py-kvcache and
simple-profiler, verifies imported module paths, records their SHA256 and source
tracked status, checks exact Plan fields/enums/type identities, handler/base
inheritance, required preload/transfer methods, actual profiler object identity,
compiled vllm._C origin and registered CPU-dispatch swap_blocks_batch schema.
It never calls the copy operation and explicitly reports no end-to-end cache,
actual copy, io_uring, or model execution qualification.

Default expected source roots:
- third_party/work/vllm-author-build
- third_party/work/py-kvcache
- third_party/upstream/simple-profiler

The exact profiler editable installation root is
third_party/upstream/simple-profiler/python (hatchling, package simple-profiler
0.1.0). Root-level simple-profiler is not a Python build project.

Preparation command for the parent, NOT executed here:
```bash
uv pip install --python .venv/bin/python --no-deps -e third_party/upstream/simple-profiler/python
```

Planned runtime command (not executed by this review):
```bash
.venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py \
  --label author-imports-01 --seconds 120 -- \
  .venv/bin/python experiments/prefix_io_v1/scripts/qualify_author_imports.py
```

GPU usage by this review: zero.
