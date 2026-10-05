# 有限精确 cell 的最小 CPU 接口审阅

本笔记是对真实保留源码的只读接口映射，不是新 GPU 资格回执。旧表和旧诊断反例均保持不变。

## 构造链

- 原 `p4_cost_table.CostTable` 只接受 mock_only/conditional，production_qualified 永为 false，production lookup 永返回 None。不能翻 mock flag。
- 原 `p4_policy.P4Policy.__init__` 还要求 `type(table) is CostTable`。单独增加 duck type 或子类不能被消费。应在新的私有 overlay 里增加明确、准确类型的私有 issuer 接口，原 mock 仍保持原资格。
- 原 `p4_options.build_p4_kwargs` 可作为最小构造位置：读取新精确成本请求、实际证据全链回放成功后，将有限资格表传给新增桥构造口；在 reactor 工作前完成。不要从其他线程任意修改已工作的 policy.table。
- 只允许 1–8 个预先冻结的 exact cells，复用原 CostCell/CostEstimate/key/lookup 格式和成本域。私有 issuer 防意外字典/PASS 升级，不声称抵御任意同进程 Python。

## 原计算口与证据边界

- 原 `p4_paired_measurement_verifier` 的核算函数返回 CPU semantic 资格 false；`_verify_cell` 的 residual 集合包含 validation 动作值。直接把这一 residual 当独立 heldout 未拟合上界不正确。
- 旧 `native_conditional_cost.original_estimator` 提取原数值 AST，`analyze_paired` 演示正确分隔：full 集合仅审计；成本估计只使用 calibration 子集；独立 heldout 比较该 frozen upper，失败即 stop，不重新拟合。
- 旧 `analyze_paired`、`ExactSingleFileReceipt` 与 `single_file_runtime_binding._geometry` 全部硬限定 planner=off、maxseq1、1040 上下文预算与 selected context144。它们可作为接口范例，不能原样资格化 planner-on 强请求流。
- 新资格必须实际回放每个 cell 的原 raw pair、完整输出与实际 CUDA Event 见证、native I/O journal 与完整收尾、原 guard、源/模型/硬件/KV layout/共同配置及 calibration/heldout隔离。不接受只读一份报告中的 PASS、bool、reported upper。
- CPU 可以实现真实加载器和拒绝测试、算术fixture、将未来 raw 合同冻结；本轮没有实测输入时不得生成实际 GPU-qualified 表。

## 现场条件借读

- `SystemSnapshot.load_signature` 在原 reactor 当前来源是 `SchedulerLoadObservation.signature`。它自己明确 `production_gpu_state_qualified=False`，不能仅重命名为正在执行的 GPU 条件。
- `native_full_step_collector.current_single_file_step` 借读真实已 prepared/awaiting_sample 帧和 open event pair：start.query 已完成、end record 尚未开始，返回 ordinal/start-record/start-completed-query/batch/active_decode/prefill/context。新薄适配应绑定实际 runner weak identity、设备/模型/layout/kernel/CUDA Event源、freshness，并在准入前后再次检查同一 active pair；结束/冲突/未知时回原路径。
- `existing_io` 应来自原 `_prefix_p4_existing_io_state` 的四阶段 StageAccounting inflight operations/bytes。它来自原 owner，并允许 unknown；不得硬填 ZERO。
- 原 GPU signature 为 model/GPU/layout/kernel/active_decode/batch/prefill/context/quantum；成本 key 再加 existing_io、stage、physical bytes。只消费 exact 匹配，不插值/泛化/上下文 bucket偷覆盖。
- 原 `issue_preview` 先检查 run、epoch、freshness、native ready、accepted-unsubmitted、generation，再处理 progress/max_wait。新增精确 I 适配必须保持该顺序及原容量、事件 fence、固定 ready.open_start 年龄；unknown/native fallback 和实际动作差异均计数。
- 原 full-step Event observer 的 max_steps 接口允许最多4096，可在新增薄 install 中设置冻结 cohort 的有界总步数；旧 FullStepCapture.export 则只断言128帧单请求，不能对全流直接照搬。

先交付可消费的精确加载/现场绑定 CPU 实现；真实 GPU 覆盖仍待测。若 effect 分支一直无条件 raise，即使所有 off/shadow 成功，也没有方法对照出口，属于未完成的 CPU 接线工作，而不是等待 GPU 数据即可解决的门。
