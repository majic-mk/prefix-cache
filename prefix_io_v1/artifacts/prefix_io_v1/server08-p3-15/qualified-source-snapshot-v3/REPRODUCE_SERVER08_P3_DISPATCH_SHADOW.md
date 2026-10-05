# P313 actual command record

These are historical executed commands, not a request to rerun completed labels. Before any new GPU repetition freeze a new label/plan/source lock, fresh UUID+idle+budget+storage checks, and the unchanged wrapper. Model performance remains unqualified on new cost provenance.

## server08-p3-13-order-migration-01

GPU wrapper bounded by 90 seconds. Actual environment delta from artifacts/prefix_io_v1/server08-p3-13/gpu-environment.json. Completed labels are append-only; do not rerun.

```sh
cd /root/autodl-tmp/prefix-io-v1-handoff/project
env CUDA_HOME=/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/cuda-toolkit TORCH_CUDA_ARCH_LIST=12.0 NVCC_THREADS=1 MAX_JOBS=4 PATH=/root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin:/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/cuda-toolkit/bin:/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin FLASHINFER_WORKSPACE_BASE=/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/runtime-cache/flashinfer FLASHINFER_NVCC=/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/cuda-toolkit/bin/nvcc FLASHINFER_NO_DOWNLOAD=1 PYTHONPATH=/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-order-cpu:/root/autodl-tmp/prefix-io-v1-handoff/project/src:/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/scripts PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py --label server08-p3-13-order-migration-01 --seconds 90 -- .venv/bin/python experiments/prefix_io_v1/scripts/qualify_store_order_gpu.py --output /root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/runs/server08-p3-13-order-migration-01/details
```

## server08-p3-13-native-fullref-01

GPU wrapper bounded by 240 seconds. Actual environment delta from artifacts/prefix_io_v1/server08-p3-13/gpu-environment.json. Completed labels are append-only; do not rerun.

```sh
cd /root/autodl-tmp/prefix-io-v1-handoff/project
env CUDA_HOME=/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/cuda-toolkit TORCH_CUDA_ARCH_LIST=12.0 NVCC_THREADS=1 MAX_JOBS=4 PATH=/root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin:/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/cuda-toolkit/bin:/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin FLASHINFER_WORKSPACE_BASE=/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/runtime-cache/flashinfer FLASHINFER_NVCC=/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/cuda-toolkit/bin/nvcc FLASHINFER_NO_DOWNLOAD=1 PYTHONPATH=/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-order-cpu:/root/autodl-tmp/prefix-io-v1-handoff/project/src:/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/scripts PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py --label server08-p3-13-native-fullref-01 --seconds 240 -- .venv/bin/python experiments/prefix_io_v1/scripts/run_concurrent_native_reference.py --output /root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/runs/server08-p3-13-native-fullref-01/details --manifest artifacts/prefix_io_v1/server08-p3-13/native-reference-manifest.json --native-config artifacts/prefix_io_v1/server08-p3-13/native-reference-config.json --model-dir models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444 --model-plan artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json
```

## server08-p3-13-dispatch-shadow-01

GPU wrapper bounded by 90 seconds. Actual environment delta from artifacts/prefix_io_v1/server08-p3-13/gpu-shadow-environment.json. Completed labels are append-only; do not rerun.

```sh
cd /root/autodl-tmp/prefix-io-v1-handoff/project
env CUDA_HOME=/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/cuda-toolkit TORCH_CUDA_ARCH_LIST=12.0 NVCC_THREADS=1 MAX_JOBS=4 PATH=/root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin:/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/cuda-toolkit/bin:/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin FLASHINFER_WORKSPACE_BASE=/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/runtime-cache/flashinfer FLASHINFER_NVCC=/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/cuda-toolkit/bin/nvcc FLASHINFER_NO_DOWNLOAD=1 PYTHONPATH=/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/py-kvcache-p3-shadow-cpu:/root/autodl-tmp/prefix-io-v1-handoff/project/src:/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/scripts PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py --label server08-p3-13-dispatch-shadow-01 --seconds 90 -- .venv/bin/python experiments/prefix_io_v1/scripts/qualify_dispatch_shadow_gpu.py --output /root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/runs/server08-p3-13-dispatch-shadow-01/details
```

## CPU final two-process qualification

```sh
cd /root/autodl-tmp/prefix-io-v1-handoff/project
env PYTHONPATH=src:experiments/prefix_io_v1/scripts PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python experiments/prefix_io_v1/scripts/run_shadow_cpu_qualification.py --label server08-p3-13-shadow-matrix-02
```

Exact underlying pytest commands/environment/deselected legacy IDs are in experiments/prefix_io_v1/runs/server08-p3-13-shadow-matrix-02/cpu-evidence/plan.json. Both historical failures remain packaged.

## Evidence packaging command

The packager requires final locks and an idle ledger, uses a fixed append-only ZIP and a 256 MiB primary reservation. Its actual command receipt is supplied separately because it is written after the ZIP has been finalized.

```sh
cd /root/autodl-tmp/prefix-io-v1-handoff/project
env PYTHONPATH=src:experiments/prefix_io_v1/scripts PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/prefix_io_v1/scripts/package_dispatch_shadow_delivery.py
```
