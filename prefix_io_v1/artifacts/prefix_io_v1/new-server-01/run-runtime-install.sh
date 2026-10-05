#!/usr/bin/env bash
set -uo pipefail
cd /root/autodl-tmp/prefix-io-v1-handoff/project
export CUDA_VISIBLE_DEVICES=""
export RUN_E2E_TESTS=""
export PYTHONDONTWRITEBYTECODE=1
timeout --kill-after=10s 300s experiments/prefix_io_v1/tools/uv-bootstrap/bin/uv pip sync \
  artifacts/prefix_io_v1/new-server-01/runtime-candidate-hashed.txt \
  --python .venv/bin/python --require-hashes --no-cache --only-binary :all:
code=$?
printf '%s\n' "$code" > artifacts/prefix_io_v1/new-server-01/runtime-install.exit
exit "$code"
