# Server07 P3 统一预算与长请求复现

在服务器 `/root/autodl-tmp/prefix-io-v1-handoff/project` 执行。本包是现有作者环境/P1/P2 的增量交付，不含模型权重、CUDA 二进制或大体积 KV 缓存。

实际命令、退出码：`artifacts/prefix_io_v1/server07-p3-03/commands.json`。GPU 环境覆盖值在同目录 `environment-overrides.json`。每条 GPU 命令均经 `run_gpu_stage.py --label <唯一label> --seconds <上限> -- .venv/bin/python <脚本> ...`。该包装器核验 permissions.yaml、已授权 UUID、累计 8 小时预算、离线模式和作业排空。不得绕过包装器或重置账本。

所有同名输出已存在；重跑必须另取 label 和输出目录，重新冻结计划。可读旧结果进行离线分析，不能覆盖旧证据。

## 已执行作业

- 数值参考：`server07-p3-uniform-native-ref-01`；以及 `server07-p3-uniform-{long,8k,16k}-ref-01`。均使用 domain=16384、reps=3，2K/4K/8K/16K。原生组使用 `--mode cold --native-hot-diagnostic --diagnostic-logprobs`；外部组使用 `--mode paired --cached-reference-logprobs`。
- 成本采集：`server07-p3-uniform-{cold,long,8k,16k}-cost-{01,02}`。诊断开关关闭，参数与顺序由 `uniform-cost-plan.json` 冻结。模型目录与 storage 来源均在 commands.json 中，无数据重建或删除。
- 连续请求：`server07-p3-long-replay-01` 和 `server07-p3-long-timeline-01`，调用 `run_long_calibration_replay.py`，参数为候选表、`long-calibration-manifest.json` 和已发布的 `server07-p3-16k-storage-01`。后一作业额外使用 `--cuda-timeline`。每作业上限 300 秒。

连续请求源码分别在 run 的 driver-source.py、probe-source.py 固定。仓库最终版本仅把未来 profiler 的磁盘预留从 512 MiB 提高到 1 GiB（非 CUDA profiler 预留 128 MiB），GPU 执行结果对应各自源快照。

## 离线命令

```text
.venv/bin/python experiments/prefix_io_v1/scripts/analyze_cached_references.py --plan artifacts/prefix_io_v1/server07-p3-03/cached-reference-plan.json --output artifacts/prefix_io_v1/server07-p3-03/cached-reference-result.json

.venv/bin/python experiments/prefix_io_v1/scripts/export_uniform_calibration_candidate.py --plan artifacts/prefix_io_v1/server07-p3-03/uniform-cost-plan.json --out artifacts/prefix_io_v1/server07-p3-03/calibration-candidate

.venv/bin/python experiments/prefix_io_v1/scripts/analyze_long_replay.py --folders experiments/prefix_io_v1/runs/server07-p3-long-replay-01/details experiments/prefix_io_v1/runs/server07-p3-long-timeline-01/details --output artifacts/prefix_io_v1/server07-p3-03/long-replay-summary-verified.json

.venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py -q --import-mode=importlib tests/prefix_io_v1_pilot tests/prefix_io_v1_native_costs --junitxml artifacts/prefix_io_v1/server07-p3-03/final-cpu-verified.xml
```

CPU 命令 CUDA_VISIBLE_DEVICES 为空，PYTHONPATH 为 P2 py-kvcache 工作树和 src。导出器会重新核验全部参考/成本数据，任何不一致都会拒绝。新候选仅供明确限定的标定集成诊断；旧严格导出器不变，P1 曲线没有被替换。

## CUDA 活动分析

最终结果为 `long-replay-summary-verified.json`。初版按单层页面过滤的结果保留但已被取代；实际 driver 批量复制以 storage-block 的整数倍合并记录。最终分析用方向、块大小、全部传输字节精确闭合筛选缓存复制，再以实际 CUPTI 设备区间求并集/交集，避免重叠 kernel 被重复计数。不会用 host API 的提交/返回时间推测 GPU 重叠。

原始大型 trace：
`experiments/prefix_io_v1/runs/server07-p3-long-timeline-01/details/native-cohort.trace.json.cuda.json`。
它包含 profiler 扰动，只用于设备机制证据，不参与 goodput、策略收益或 observation overhead 结论。
