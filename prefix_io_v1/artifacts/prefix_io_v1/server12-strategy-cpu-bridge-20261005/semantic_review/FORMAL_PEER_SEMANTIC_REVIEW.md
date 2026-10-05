# 正式 peer 接线的只读语义复核

2026-10-05。结论：本轮审查未发现实际 guarded CLI 能将旧 controlled U/off 当成正式 peer，或绕过当前正式 manifest/config/source/guard/capture 绑定的回归。此结论仅针对所列源码版本的静态语义，不是自然工作负载、GPU 或效果资格。

复核范围为 `namespace_bridge.py`、`strong_trace_runner_v3.py`、`native_runtime_v3.py`，并读取原 formal validator、guard validator 与 `reserve_join._capture`。6 份来源 SHA 前后相同，AST 解析通过；没有运行新旧测试、RPC、GPU、模型或原生库。`formal_close` 没有纳入冻结审查。

- 开发 I/shadow 要求实际正式 U/off development 结果，完整 manifest、formal binding、GPU 与共同运行域一致，并重新读取原 child、实际 config、guard、frontend 和完整 CUDA capture。旧 controlled 资格状态和工作负载 schema 无法满足这些条件。
- namespace 桥接允许 evaluation 的第一臂为 U/off 或 I/on；第二臂只有在另一臂原命名空间保留且 peer 已真实关闭时才能使用该 peer。当前臂始终要求新鲜命名空间，没有删除或重置路径。
- peer 的实际配置必须与原 child 的 runner/runtime/source-lock/workload/pair/formal 引用一致。旧源锁全部行必须逐行继承，实际 guard 的 command/config/permissions/reservation 与该次 child 对应。source-lock 可以追加来源，不能替换旧执行行。
- raw frontend/capture 是单独的真实字节引用，须与 parent 内对应记录一致。U 保持 off/bridge None；I 要有实际私有表、当前 owner 安装、原 shutdown/tail 证明。原 `_capture` 继续核对完整 128-token 输出与每个 CUDA witness；完整采集未知或不支持时拒绝，不从 CPU 标签推断成功。
- SLO 适配只作用于新私有 formal validator 实例，且仅 development 允许 `None`；evaluation 继续使用原严格 service SLO 校验。bridge 和原 validator 的实际 SHA 均由新 runner 明确固定。

内部 `native_runtime_v3.execute` 沿用原 V2 的可信 Python 调用边界：自身仅检查 reservation label，不重复 driver 的完整 active guard 校验。实际 CLI 的 `main` 在加载和调用 runtime 前核对原 guard、PID/SID/PGID、GPU UUID、配置命令、权限字节及环境，未发现 CLI 绕过。**所有实际和未来 GPU 作业仍必须通过 `strong_trace_runner_v3.main`，由原 guard 执行；CPU gates 或直接调用 execute 不构成 GPU 运行授权。**

桥接支持两种先后顺序，不自行发行 AB/BA 统计协议、原生成本或效果资格。真实自然数据、独立期限／正式 SLO、独立分区及两臂完整采集仍需后续实际验证。

逐项源码行与字节 SHA 见 `FORMAL_PEER_SEMANTIC_REVIEW.json`。
