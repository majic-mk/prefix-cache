#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES=""
export RUN_E2E_TESTS=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="src:third_party/work/py-kvcache"
mkdir -p artifacts/prefix_io_v1
out="$(mktemp -d artifacts/prefix_io_v1/cpu-replay.XXXXXX)"
.venv-prefix/bin/python -m pytest -q -ra tests/prefix_io_v1 third_party/work/py-kvcache/tests   --junitxml="$out/results.xml" 2>&1 | tee "$out/results.txt"
