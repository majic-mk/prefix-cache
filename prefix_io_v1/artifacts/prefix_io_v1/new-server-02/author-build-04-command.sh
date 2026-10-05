#!/usr/bin/env bash
set -u
PROJECT=/root/autodl-tmp/prefix-io-v1-handoff/project
OUT="$PROJECT/artifacts/prefix_io_v1/new-server-02"
cd "$PROJECT/third_party/work/vllm-author-build"
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export PYTHONDONTWRITEBYTECODE=1
export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_PRECOMPILED=0
export VLLM_USE_PRECOMPILED_RUST=0
export VLLM_REQUIRE_RUST_FRONTEND=0
export VLLM_USE_RUST_FRONTEND=0
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=8
export NVCC_THREADS=1
export CMAKE_BUILD_TYPE=Release
export CUDA_HOME="$PROJECT/experiments/prefix_io_v1/cuda-toolkit"
export PATH="$CUDA_HOME/bin:$PROJECT/.venv/bin:$PATH"
export FETCHCONTENT_BASE_DIR="$PROJECT/experiments/prefix_io_v1/build-cache"
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=http.version
export GIT_CONFIG_VALUE_0=HTTP/1.1
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
timeout --signal=TERM --kill-after=30s 5400s "$PROJECT/experiments/prefix_io_v1/tools/uv-bootstrap/bin/uv" pip install --python "$PROJECT/.venv/bin/python" --no-build-isolation --no-deps -e . > "$OUT/author-build-04.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$OUT/author-build-04.exit"
exit "$rc"
