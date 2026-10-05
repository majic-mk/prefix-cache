# Server07 P3 复测复现

工作目录：`/root/autodl-tmp/prefix-io-v1-handoff/project`。本增量包依赖已完成的 P1/P2/P3 工作区、锁定作者版本、模型文件和原私有缓存；包内不分发模型权重或大体积 KV 文件。

实际执行 argv/退出码位于 `artifacts/prefix_io_v1/server07-p3-02/commands.json`。每个 GPU 作业对应 `runs/<label>/result.json`（预算包装器）、`process.log`、`details/frozen-config.json`、`details/driver-source.py`、`details/result.json` 和原始 profiler trace。所有已存在的输出保持只读；重跑必须使用新的 label/output-dir。

环境在 `artifacts/prefix_io_v1/server07-p3-02/environment-overrides.json`，由既有 CUDA 工具链环境加上 P2 工作树的 PYTHONPATH 组成。GPU 作业必须通过现有 `run_gpu_stage.py` 核验 permissions.yaml、UUID 和累计预算，不直接调用采集脚本绕过保护器。

数值诊断的实际命令：
```text
.venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py --label server07-p3-native-logprobs-01 --seconds 240 -- .venv/bin/python experiments/prefix_io_v1/scripts/acquire_native_aio_costs.py --output-dir experiments/prefix_io_v1/runs/server07-p3-native-logprobs-01/details --storage experiments/prefix_io_v1/runs/server07-p3-long-storage-01 --model-dir models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444 --model-plan artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json --mode cold --sizes 2048,4096 --reps 3 --domain 4096 --native-hot-diagnostic --diagnostic-logprobs
```

16K 采集沿用同一脚本，参数为 `--sizes 16384 --reps 3 --domain 16384`，storage 为 `experiments/prefix_io_v1/runs/server07-p3-16k-storage-01`。精确顺序：cold-02、paired-02、paired-03、cold-03、cold-04、paired-04；完整 label 均以 `server07-p3-16k-` 开头。各自受 240 秒 GPU 作业预算保护。未重跑 populate，比较使用以前 `server07-p3-16k-populate-01` 的原始 store 结果。

CPU 数据分析：
```text
.venv/bin/python experiments/prefix_io_v1/scripts/analyze_repeated_native_costs.py --plan artifacts/prefix_io_v1/server07-p3-02/repeat-plan.json --output artifacts/prefix_io_v1/server07-p3-02/repeated-point.json
```

CPU 测试（GPU 作业全部完成后才执行）：
```text
.venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py -q --import-mode=importlib tests/prefix_io_v1_pilot tests/prefix_io_v1_native_costs --junitxml artifacts/prefix_io_v1/server07-p3-02/cpu-tests.xml
```

CPU 命令的 PYTHONPATH 指向 `third_party/work/py-kvcache-p2-aio:src`，CUDA_VISIBLE_DEVICES 设为空；GPU 命令的 PYTHONPATH 另含采集脚本目录。分析脚本是离线证据检查，不修改作者成本表、不安装新策略。重跑不能对同名已有输出覆盖；应重新冻结计划并另开输出目录，保留既有负结果。
