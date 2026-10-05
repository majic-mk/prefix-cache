本目录是正式输入的 CPU 候选接线，不是实际自然数据、GPU 结果或策略收益证明。

新增 `strong_trace_runner_v3.py` 从冻结的 V2 复制，只追加正式输入分派。资格工作负载仍走原 `validate_workload` 和旧 phase gate。正式工作负载只接受原 `natural_trace_workload_v1` 的完整字节重放、实际 CPU tokenizer 收据、原 parser、真实分区家族、独立 deadline 和隔离 namespace。原 `drive_original_engine` 的函数源码与 V2 完全相同；其 token-ID 分支收到选定分区的原 ID 列表、原全局顺序、原到达计划及完整 128 token 工作量。原完整 manifest 不重写、不改摘要。

正式开发支持 U/off 与 I/shadow。开发 I 必须消费原 guard 关闭后的同正式输入 U 基线，旧 12 条 P3 资格请求不能替代。两臂都先运行原有限来源/设备/domain/deadline gate；开发观测预算不能授权普通 I 动作。开发允许原协议允许的 SLO=None 诊断，但独立 deadline、实际 authority 和 host clock 仍必须存在。

正式效果支持 U/off 与 I/on；两臂要求原 private table 发行与原 native reserve 的只读重验。SLO 必须是开发之前独立声明的真实服务目标。当前没有这些真实输入，本目录没有生成可以直接执行的正式配置。

新 `namespace_bridge.py` 只对新加载的原 F 模块实例添加 metadata adapter。当前臂始终必须是新 namespace。首臂允许 U 或 I，两臂起初都须新；第二臂必须提供同分区实际 peer 的原 child、config、guard、完整输出、CUDA capture、native tail 和 source 继承闭合证明。证明通过后保留其路径；不删除、不清空、不重置。开发保持 U 先 I 后，效果保留固定 AB/BA 的两种顺序。只保存一个 PASS 标志不能放行。

推荐服务器映射：

- `strong_trace_runner_v3.py` → `artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/runner/strong_trace_runner_v3.py`
- `namespace_bridge.py`、`test_formal_runner_cpu.py` → 同 preparation 下的 `formal_runtime_bridge/`
- 搭配另一个候选目录 `formal_runtime/native_runtime_v3.py`，由根代理追加冻结清单、部署和核验。

新正式配置额外字段为 `formal_trace_binding_ref` 和 `formal_peer_closed_ref`。首臂的 peer 为 None；第二臂必须为实际已完成 peer 的精确冻结引用。`off_qualification_ref` 单独指实际正式开发 U 的 guard 关闭证明，不因 AB/BA 改为评估 peer。真实 peer parent 在原 child 字节上只允许新增 `completed_guard_ref`、`child_result_ref`、可选 `formal_parent_closure_schema` 和把原 `os_session_drained=False` 改成经原 guard 核验后的 True。

CPU 测试只检查分派、实际候选源码的字段/状态表达式、原 token-ID API 传递和拒绝行为。正向 spy 均明确标注 CPU fixture，不生成模型、CostTable、CUDA 帧、真实 trace、deadline、reserve 或 GPU 资格。

补充：namespace_bridge.verify_closed_peer_document 已暴露给根代理后置闭合 producer；parent 可只在内存中构造，但所有实际 raw/config/guard/source 叶已存在且精确核验，验证后单次 append，再追加 source lock 供消费者使用，不写 draft 再循环引用。最新本地测试为 32/32（LOCAL_CPU_TEST_RESULT_03.json）。
