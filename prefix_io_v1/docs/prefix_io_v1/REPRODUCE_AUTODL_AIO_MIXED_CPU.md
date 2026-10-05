# 复现 AutoDL AIO CPU 混合续验

在服务器项目目录 /root/autodl-tmp/prefix-io-v1-handoff/project 执行，使用已有 .venv 与可选工作区。以下命令不下载模型、不初始化 CUDA、不改系统配置。

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/third_party/work/py-kvcache-aio-cpu:$PWD/src"
run_dir=$(mktemp -d "$PWD/artifacts/prefix_io_v1/aio-mixed-rerun.XXXXXX")
export AIO_CPU_GPU_GUARD_PATH="$run_dir/gpu-guard.json"

.venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py \
  -q --import-mode=importlib tests/prefix_io_v1_aio \
  third_party/work/py-kvcache-aio-cpu/tests \
  --ignore=third_party/work/py-kvcache-aio-cpu/tests/test_e2e_kvcache.py \
  --basetemp "$run_dir/pytest" --junitxml "$run_dir/tests.xml"

.venv/bin/python experiments/prefix_io_v1/scripts/aio_cpu_mixed_screen.py \
  --output "$run_dir/mixed"
```

最终联合测试预期 320 passed、16 skipped，环境性跳过原因见 skip-reasons.json。每个运行用新目录，避免覆盖旧证据。
微基准一轮产生 18 行，两次重复；计时内逻辑 I/O 465,043,456 字节，另有预写 192,847,872 字节与文件校验读取 227,016,704 字节。测试文件仅写入新建实验目录。吞吐包含 Python 与字节校验成本，不用于宣称 SSD 带宽或模型加速。

作者提交 3abba7a502d553f6e7e2e58b92086487e3395d7e。全量兼容补丁须应用在已有共同修复的基础上，不要向已修改工作区重复应用。empty-submit-fix.patch 仅适用于上一轮冻结的 AIO 适配实现；完整版本使用 compatibility.patch。

本轮真实命令与输出：
- empty-wake-before.json：修复前故障复现，预期失败。
- mixed-tests.json：10 项新增测试。
- release-qualification.json/.xml：最终联合测试。
- mixed-before.json、mixed-after.json：修复前后微基准。
- patch-forward-check.json、patch-reverse-check.json：完整兼容补丁适用性。
GPU 运行命令不在本次 CPU 交付范围；先满足报告中的设备/授权/预算核验。
