本修订提供 C5 公共六窗口真实 GPU 入口；当前无 GPU，不创建配置、有效授权、绑定、计划、预算 reservation 或 native receipt。

新目录固定为 `artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004`，作业名称为 `server11-c5-native-common-cost-gpu03`。它继承 Stage C 的 `common_candidate` 64 个 Python 文件，63 个文件字节完全相同；唯一修改的公共文件是规范 typed receipt 的新 serializer/verifier 路径、来源检查和 SHA pins。`NATIVE_SOURCE_INHERITANCE.json` 固定这些来源以及 32 个未修改的数值、模型输入、执行观测和原生命周期辅助函数 AST。

`run_native_cost_experiment.py` 的 CLI 和直接调用都在接触原模型/torch/文件缓存引擎之前验证新的现场配置、完整来源闭包、fresh UUID、明确的人类执行授权及原 guard 的 active reservation。目标 UUID 来自已验证配置；旧机器 UUID、旧冻结授权和旧实验名称无法复用。`gpu_entry_binding.py` 和 `control_native_cost_job.py` 提供这些现场绑定以及唯一原 GPU guard 的控制；此文件不改原预算 guard、服务器系统或驱动。

公共标定保持原 `AB / BA / AB` 顺序、前两对拟合与第三对 holdout、六个新 OS 模型进程、129 个 prompt tokens、128 个 cached tokens、每窗口完整 128 个输出、offset 16、917504 字节单文件、最多 8 个 accepted parents、`bridge=None`、原始 preload、共享 staging、异步流水线和 shutdown。配置、实际模型函数、原公式及 holdout 禁止回拟合均保持原行为。

`prepare_and_verify_native_cost.py` 从新现场闭包创建 preregistered plan，复核真实 parent/child 原始记录、完整 Event 来源及 128 步、native IO 因果 enclosure、24 份已有生产 KV 的来源和关闭 drain。`native_conditional_cost.py` 验证新的现场绑定及原 guard 自然结束与唯一 ledger event；计算沿用原 numerical AST。`p4_single_file_receipt.py` 重做原始记录序列化与数值验证，校验新源码固定 SHA、实际 identity helper、runtime/common 来源，满足有限 holdout 门槛后才可能发行原 policy 所要求的 exact typed receipt。数据中写一个 `origin=native_gpu_recording` 或 `PASS=true` 不会获得资格。

Stage C 的 synthetic CPU plan/raw、CPU fixture、旧 C4/v6 记录、CPU-only preparation、空 binding、错误 UUID、漂移源码和未完成 guard 均保持拒绝。这里只提供真实分支源码，尚无 real native receipt、公共成本资格、完整入口开销或策略性能证据。`test_native_entry.py` 的 23 项检查是拒绝反例和来源保留检查，没有 native 成功 fixture。原候选测试文件原样继承，不能把其历史结果当成本轮结果。

本轮 CPU 执行命令为：

```
.venv/bin/python -B -I -S artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/run_cpu_native_entry.py --project-root /root/autodl-tmp/prefix-io-v1-handoff/project --source-lock <本轮CPU源码锁> --output-dir <新CPU证据目录> --location server_cpu
```

真实 GPU 阶段由 controller 的 `context`、`bind`、`scope`、`launch`、`after` 及 serializer 的 `--prepare`、`--verify` 处理。`context` / `bind` 要求新鲜现场资源观察、对当前来源与该作业的明确人类执行授权、原 permissions 根目录和累计 8 小时预算；绑定 revision + site 两层来源并生成 before proof。serializer `--prepare` 创建真实 prelaunch plan，然后 `scope` 冻结该计划。`launch` 重新现场核验并生成完整 launch proof 与独立 plan pin，然后仅使用原 `run_gpu_stage.py` 的 1200 秒执行 + 20 秒收尾 reservation。第一次失败停止，无自动换配置、标签或重试。`after` 在原 session 排空后生成完整 after proof，再由 serializer `--verify` 做真实原始记录/成本校验；失败不会发行 receipt。

有 GPU 并完成授权/现场绑定后先运行这一个公共标定作业。真实公共 receipt 成功后，下一阶段才接入 off → shadow → on 的完整入口生命周期及观测成本，最后在门槛通过后进行 P4 公平比较。此处未证明 GPU 提升、完整系统可行或论文效果。
