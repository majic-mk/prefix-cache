# C5 成本绑定准备：仅 CPU，尚未实现原生凭据颁发

本目录只新增严格源码/契约准备、CPU 正反例及运行器。未修改冻结 C5/C4、前轮运行准备、作者源码、GPU 账本、阈值和旧结果。`C5CostBindingPreparation` 是私有 CPU 文档载体，不是原 `ExactSingleFileReceipt`，没有签名、有效成本上界或步预算，不能传入原策略消费。`load_verified_single_file`、`launch_gpu` 及 CLI 无条件阻断；CLI exit 2，在参数读取、模型导入及预算变动之前返回。

`prepare_cost_binding` 要求调用者独立传入 binding、plan、factory 与新 source lock 引用，然后读取实际字节校验。固定 C5 reactor/collector SHA；完整前轮准备锁的每个 scoped ref 都必须纳入新闭包，并核对实际 C5 Python 文件全集。校准共源 reactor 留在 `runtime_common_refs`，collector 唯一位于 `runtime_overlay_refs` 且严格等于 `plan.collector_source_ref`；其他 C5/preparation/factory 源码保留 overlay。同源 model/Event/runner、原 math/serializer/verifier 也必须锁定。源码 SHA 用于漂移检测，不能证明授权，也不能证明这些源码在真实模型执行中被运行。

固定契约仍为 3 对 AB/BA/AB、前 2 对校准/最后 1 对 holdout、6 个新原进程及 handler、每次 129 prompt/128 cached/128 output、offset 16、warmup offset 1、1 个 917504 字节 SSD read、parent 上限 8、校准 bridge=None。CPU checker 实际调用冻结 `native_conditional_cost.analyze_paired`，继承全部 128 帧/输出/事件因果及原数值 AST；不调用 `verify_native_cell` 原生颁发入口。原公式为 A_cal 均值上取整加校准平均正增量，再加最大正残差；holdout 不参与拟合，A-only budget 仅使用校准 A。测试会真正计算合成 100/110/120/118 ns 的反例，合成 holdout 130 ns 保持不覆盖。这些数值只检验代码，不能用作系统成本或性能结果。

所有输入显式由外层 `origin=synthetic_cpu_contract` 标明。内层 fixture 沿用旧 shape validator 要求的 `native_gpu_recording` 字段，这些文字不产生任何原生可信性。审查输出不提供可消费成本，`native_cost_qualified`、`native_execution_verified`、`conditional_cost_cell_qualified`、`full_runtime_cost_qualified`、`on_observation_cost_measured`、`gpu_launch_allowed` 始终 false。

特别是原六窗口校准 bridge=None，**不会安装 on-only 通知/Queue 观测**。完整源码纳入闭包只完成绑定，不能说明其新增开销已测量。不能用旧 C4/v6 真实成本、旧 73.6% CPU 降幅或原六窗口数字冒充完整 on 路径资格，也不能把 on 延期后的 B 窗口套入原 overlap-I 校准因果规则。真正新原生凭据颁发、完整入口成本资格和 GPU 对照均未在本目录实现，须由后续独立有限协议及明确授权完成。

运行器使用调用者冻结的新闭包，显式传递全部依赖路径；输出目录必须是本目录内全新子目录。服务器闭包须纳入实际前轮 80 个 scoped 引用及其 lock，不能用本机 32 文件锁替代。依赖 closure 由 root 冻结，runner 在测试前后核验，不建立 GPU source lock 或授权 scope。

```sh
CUDA_VISIBLE_DEVICES='' "$ROOT/.venv/bin/python" -B -I -S "$D/run_cpu_cost_binding.py" \
  --candidate-root "$C5" --preparation-root "$PREP" --native-root "$V6" \
  --original-estimator "$ROOT/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_paired_measurement_verifier.py" \
  --project-root "$ROOT" --source-lock "$D/CPU_SOURCE_LOCK.json" \
  --output-dir "$D/SERVER_CPU_RESULT_01" --location server_cpu
```

测试临时文件只在新输出目录中创建和回收。没有 GPU、模型、物理 I/O、真实 `collector.install`、正式 CPU 计分或新增策略实验；本地测试与服务器测试分别报告。
