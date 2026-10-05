#!/usr/bin/env bash
set -u
PROJECT=/root/autodl-tmp/prefix-io-v1-handoff/project
OUT="$PROJECT/artifacts/prefix_io_v1/new-server-02"
LABEL="${1:?usage: build_author_vllm.sh unique-label}"
case "$LABEL" in *[!a-zA-Z0-9_-]*|"") exit 2;; esac
if [[ -e "$OUT/$LABEL.log" || -e "$OUT/$LABEL.exit" ]]; then
    printf 'Refusing to overwrite build evidence: %s\n' "$LABEL" >&2
    exit 2
fi
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
export UV_CACHE_DIR="$PROJECT/experiments/prefix_io_v1/uv-cache"
export TMPDIR="$PROJECT/experiments/prefix_io_v1/build-tmp"
export UV_LINK_MODE=copy
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
timeout --signal=TERM --kill-after=30s 5400s "$PROJECT/experiments/prefix_io_v1/tools/uv-bootstrap/bin/uv" --verbose pip install --python "$PROJECT/.venv/bin/python" --no-build-isolation --no-deps -e . > "$OUT/$LABEL.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$OUT/$LABEL.exit"
exit "$rc"
