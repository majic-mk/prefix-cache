# 复现命令与证据索引

所有命令在服务器 /root/autodl-tmp/prefix-io-v1-handoff/project 中运行。
实际执行的完整 argv、退出码、GPU UUID、耗时与 session 清理记录位于 artifacts/prefix_io_v1/server07-calibration-01/gpu-job-ledger-slice.json；各任务原始日志在 experiments/prefix_io_v1/runs/<label>/process.log。CPU 命令和输出在同 evidence 目录的 *tests*.json、export-paired-v2.json。

## CPU 命令
```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=third_party/work/py-kvcache-aio-cpu:src \
.venv/bin/python -m pytest -q --import-mode=importlib tests/prefix_io_v1_calibration

CUDA_VISIBLE_DEVICES='' PYTHONPATH=third_party/work/py-kvcache-aio-cpu:src \
.venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py -q \
--import-mode=importlib tests/prefix_io_v1_native_costs
```

CPU 导出：
```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=third_party/work/py-kvcache-aio-cpu:src \
.venv/bin/python experiments/prefix_io_v1/scripts/export_native_aio_costs.py \
--cold experiments/prefix_io_v1/runs/server07-cal-cold-01/details/result.json \
--populate experiments/prefix_io_v1/runs/server07-cal-populate-02/details/result.json \
--restore experiments/prefix_io_v1/runs/server07-cal-paired-01/details/result.json \
--out artifacts/prefix_io_v1/server07-calibration-reexport
```
输出目录必须是新的。独立的旧 restore-02 是功能证据，最终曲线导出必须使用 paired-01。

## GPU 复现入口
固定环境覆盖来自 artifacts/prefix_io_v1/new-server-03/native-prefix-04-launch.json 的 environment_overrides。以下 Python 片段为可审阅的完整复现入口；必须从项目根运行。每次生成新的私有标签，所有 GPU 任务均通过原预算 runner，不重置累计账本。此片段在文档中提供，本轮没有再次执行它。

```python
from datetime import datetime
from pathlib import Path
import json, os, subprocess
root = Path.cwd().resolve()
assert str(root) == "/root/autodl-tmp/prefix-io-v1-handoff/project"
env = os.environ.copy()
env.update(json.loads((root/"artifacts/prefix_io_v1/new-server-03/native-prefix-04-launch.json").read_text())["environment_overrides"])
env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=":".join(str(root/p) for p in [
    "third_party/work/py-kvcache-aio-cpu", "src", "experiments/prefix_io_v1/scripts"]))
python = str(root/".venv/bin/python")
tag = "server07-repro-" + datetime.now().strftime("%Y%m%d-%H%M%S")
storage = str(root/f"experiments/prefix_io_v1/runs/{tag}-populate/storage")
details = {}
def run(mode, sizes="16,64,128,256,512,1024", reps=6, curve=None):
    label = tag + "-" + mode
    output = str(root/f"experiments/prefix_io_v1/runs/{label}/details")
    args = [python, "experiments/prefix_io_v1/scripts/acquire_native_aio_costs.py",
        "--output-dir", output, "--storage", storage, "--mode", mode,
        "--sizes", sizes, "--reps", str(reps),
        "--model-dir", "models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444",
        "--model-plan", "artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json"]
    if curve:
        args += ["--curves", str(curve)]
    subprocess.run([python, "experiments/prefix_io_v1/scripts/run_gpu_stage.py",
        "--label", label, "--seconds", "300", "--", *args], env=env, check=True)
    details[mode] = str(Path(output)/"result.json")
run("populate")
run("cold")
run("paired")
curves = root/f"artifacts/prefix_io_v1/{tag}-curves"
cpu_env = dict(env, CUDA_VISIBLE_DEVICES="")
subprocess.run([python, "experiments/prefix_io_v1/scripts/export_native_aio_costs.py",
    "--cold", details["cold"], "--populate", details["populate"],
    "--restore", details["paired"], "--out", str(curves)], env=cpu_env, check=True)
run("planned", sizes="16,128,512", reps=2, curve=curves/"curves-v2.json")
```

如源数据拒绝检查失败，脚本停止；不得复用旧曲线掩盖失败。读取本轮 source-lock.json 与最终 frozen-config.json 核验 SHA、模型实际目录、哈希 seed、GPU、预算与采样域。固定版本之外的环境需要重新审计。

## 证据
- 最终配对原始记录：runs/server07-cal-paired-01/details/result.json 与 72 份 trace。
- 冷重算：runs/server07-cal-cold-01/details/result.json。
- 原规划器：runs/server07-cal-planned-03/details/result.json 与 12 份多请求 trace。
- 精确 KV 字节回环：上一交付 runs/server07-native-aio-kv-02/details/result.json。
- 本轮曲线、汇总、负结果、CPU XML 和 GPU 预算切片均在 artifacts/prefix_io_v1/server07-calibration-01。
- ZIP 只含代码、配置和证据，不复制模型权重或 KV 数据文件；真实 KV 文件保留在服务器私有运行目录，不删除。
