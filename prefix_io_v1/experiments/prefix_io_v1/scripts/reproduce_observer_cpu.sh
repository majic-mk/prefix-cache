#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES=""
export RUN_E2E_TESTS=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="src:third_party/work/py-kvcache-observer"
test -f third_party/work/py-kvcache-observer/py_kvcache/reactor.py
mkdir -p artifacts/prefix_io_v1
out="$(mktemp -d artifacts/prefix_io_v1/observer-cpu-replay.XXXXXX)"
.venv-prefix/bin/python -m pytest -q -ra --import-mode=importlib \
  -W error::pytest.PytestUnhandledThreadExceptionWarning \
  tests/prefix_io_v1 tests/prefix_io_v1_progress tests/prefix_io_v1_observer \
  third_party/work/py-kvcache-observer/tests \
  --junitxml="$out/results.xml" 2>&1 | tee "$out/results.txt"
.venv-prefix/bin/python experiments/prefix_io_v1/scripts/measure_observation_cpu.py \
  --output "$out/observation-overhead-cpu.json" --iterations 20000 --repeats 7
