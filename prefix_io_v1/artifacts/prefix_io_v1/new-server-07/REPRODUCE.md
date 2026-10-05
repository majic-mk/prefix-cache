# 新服务器 07 复现入口

工作目录：/root/autodl-tmp/prefix-io-v1-handoff/project。
使用已有 .venv、已验证的本地模型与固定作者源码。必须先核对 permissions.yaml 当前指定 UUID、剩余预算、active_reservation，以及运行 label 尚未存在。下列 replay label 仅用于未来复现；不要覆盖现有证据。

```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
export PYTHONPATH="$PWD/third_party/work/py-kvcache-aio-cpu:$PWD/src:$PWD/experiments/prefix_io_v1/scripts"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME="$PWD/experiments/prefix_io_v1/cuda-toolkit"
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
export NVCC_THREADS=1
export MAX_JOBS=4
export FLASHINFER_WORKSPACE_BASE="$PWD/experiments/prefix_io_v1/runtime-cache/flashinfer"
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export FLASHINFER_NO_DOWNLOAD=1

.venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py \
  --label server07-reactor-replay-01 --seconds 150 -- \
  .venv/bin/python experiments/prefix_io_v1/scripts/qualify_aio_gpu_reactor.py \
  --output experiments/prefix_io_v1/runs/server07-reactor-replay-01/details

.venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py \
  --label server07-kv-replay-01 --seconds 240 -- \
  .venv/bin/python experiments/prefix_io_v1/scripts/qualify_native_aio_kv.py \
  --output-dir experiments/prefix_io_v1/runs/server07-kv-replay-01/details \
  --model-dir models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444 \
  --model-plan artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json
```

必须逐项检查上一命令 exit/status，失败时先诊断，不连续消耗下一项预算。GPU 运行通过 runner 设置 UUID、离线开关及预算，不手动绕过。

CPU 回归（独立新目录）：
```bash
run_dir=$(mktemp -d "$PWD/artifacts/prefix_io_v1/server07-cpu-rerun.XXXXXX")
CUDA_VISIBLE_DEVICES='' AIO_CPU_GPU_GUARD_PATH="$run_dir/guard.json" \
  .venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py \
  -q --import-mode=importlib tests/prefix_io_v1_aio \
  third_party/work/py-kvcache-aio-cpu/tests \
  --ignore=third_party/work/py-kvcache-aio-cpu/tests/test_e2e_kvcache.py \
  --basetemp "$run_dir/pytest" --junitxml "$run_dir/tests.xml"
```

本轮真实执行 argv 与返回值保存在 new-server-07/*.json 和各 runs/*/result.json。最终模型证据是 server07-native-aio-kv-02，初版 01 的备份回写限制详见报告。
这些命令是功能诊断，没有导出成本曲线、启动正式研究实验或启动外部 HTTP 服务。
