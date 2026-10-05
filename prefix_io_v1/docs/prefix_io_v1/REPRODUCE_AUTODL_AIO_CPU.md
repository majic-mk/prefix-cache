# 重现 AutoDL AIO CPU 验证

所有命令在服务器项目根目录执行：
/root/autodl-tmp/prefix-io-v1-handoff/project

使用已有锁定的 .venv，不下载模型，不更改驱动。可选后端工作树为 third_party/work/py-kvcache-aio-cpu。
若在新 checkout 重建：先取得作者提交 3abba7a502d553f6e7e2e58b92086487e3395d7e，应用本包 common-before.patch，再应用 compatibility.patch；不要对已有活动目录重复应用。

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/third_party/work/py-kvcache-aio-cpu:$PWD/src"
run_dir=$(mktemp -d "$PWD/artifacts/prefix_io_v1/aio-cpu-rerun.XXXXXX")
export AIO_CPU_GPU_GUARD_PATH="$run_dir/gpu-guard.json"

.venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py \
  -q --import-mode=importlib tests/prefix_io_v1_aio \
  third_party/work/py-kvcache-aio-cpu/tests \
  --ignore=third_party/work/py-kvcache-aio-cpu/tests/test_e2e_kvcache.py \
  --basetemp "$run_dir/pytest" --junitxml "$run_dir/tests.xml"

.venv/bin/python experiments/prefix_io_v1/scripts/aio_cpu_microbench.py \
  --output "$run_dir/microbench"
```

第一条预期 310 passed、16 skipped；跳过原因见 skip-reasons.json。路径采用新目录，避免覆盖旧证据。
第二条写入专用测试文件，约 640MiB 逻辑读写及 22.4MiB 预写，保留文件；它不是模型性能实验。
脚本会检查 CUDA 未初始化；CPU 微基准的导入可能打印 CUDA 扩展警告，不能据此宣称 GPU 路径已验证。
results.json 为完整原始数据，性能筛查采用 microbench-preconditioned/results.json。其余轮次仅用于开发诊断。

实际配置新增字段为 kv_connector_extra_config 中的 io_backend="linux_aio" 和 aio_metadata_workers=2。
这些字段当前只在可选工作树存在；未切换活动环境。缺省 io_backend="io_uring"，不支持自动回退。
