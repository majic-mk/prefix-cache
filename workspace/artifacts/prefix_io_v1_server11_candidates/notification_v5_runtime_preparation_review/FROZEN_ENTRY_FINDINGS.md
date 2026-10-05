# 冻结入口与校准来源缺口

本轮只读查看并用 CPU 反例执行冻结接口；不修改这些文件。以下路径均相对 `artifacts/prefix_io_v1_server11_candidates/`。

| 位置 | 已确认的约束与最小后续准备 |
|---|---|
| `p4_single_file_runtime_v4/run_p4_single_file_experiment.py:471`、`:566`、`:578` | bridge 使用 job LABEL，capture 使用 request rid。C5 的 `reactor.py:2325` / `:2329` 要求 owner/capture 相等。新入口应令 capture.run_id=LABEL，原 request rid 保留；不能删除身份检查。 |
| `native_cost_v6/native_conditional_cost.py:125` `validate_capture` | 原函数已独立接受 run_id 与原 native request_id。CPU 合成 128 帧直接通过此原函数；错误 run、prepared/output request、缺帧均仍拒绝。新 runtime verifier 在旧 `verify_p4_single_file.py:274` / `:276` 分开传入两者即可，不必改原全帧定义。 |
| `p4_single_file_candidate_v5_cpu/.../p4_single_file_receipt.py:18`、`:19`、`:20`、`:89`、`:91` | C5 所复制 receipt 与 C4 字节相同，SHA `11312ab7a6f5c7823e77b22cf17c93bd255438af7f886568e693f76fc5b9825d`。仍固定 C4 overlay、native-v6 verifier/serializer。 |
| 同文件 `load_verified_single_file:215`，特别是 `:236`—`:245` | 限定旧 job06、单文件 917504 bytes、原 C4 reactor、parent cap=8、校准 bridge=None。新 C5 不能消费此凭据获得校准资格。 |
| 同文件 `_runtime_refs:173` / `:192`；`p4_policy.py:43` / `:53` | common ref 必须等于原校准 lock；策略要求规范模块中 ExactSingleFileReceipt 的 exact type。不能改运行时常量、私用 issuer 或构造假 receipt 绕开。 |
| `p4_single_file_runtime_v4/control_p4_single_file.py:99` `freeze_common`，`:114`—`:127` | 旧工厂固定 C4/V6。新的严格 ref 工厂应额外强制 runtime collector 的 overlay ref 等于 plan.collector_source_ref。不能直接搬到 common 组：冻结 `p4_bridge.py:128` 要求 capture 类源位于 runtime_overlay_refs，receipt 又禁止两个组重复。旧同路径祖先保护仍生效；最小新增是显式校准 collector 等值绑定，不改变分组或 bridge。 |
| `p4_single_file_runtime_v4/verify_p4_single_file.py:101` `validate_native_source_binding` | 要求实际 reactor 的 source/_run/module 闭包等于 receipt.calibration_native_source_sha256；只是改 config 路径不能迁移旧成本资格。 |
| `native_cost_v6/control_native_cost_job.py:86` / `:123` / `:152` | source lock、scope、guard 绑定旧目录、job、GPU 和预算。CPU 只能在新目录准备新的控制/锁模板及失败用例；原已执行 scope 不能重用。 |
| `native_cost_v6/prepare_and_verify_native_cost.py:72` `create_plan` | 已有最小范式：先冻结 wrapper/collector/实际 overlay/source、模型、Event、原估计器与输入，固定六窗口 AB/BA/AB、两组校准一组独立 heldout。新计划必须继续保持单文件、原公式、完整 128 tokens。 |
| 同文件 `validate_common_source_binding:149`、`serialize_runtime_record:181` | 实际 owner、加载模块、collector adapter 128 帧及 sourcebefore/after 都需重核；这些只能根据新真机 raw 记录签发，CPU 构造值不能替代。 |
| `p4_single_file_candidate_v5_cpu/single_file_runtime_binding.py:122` / `:177` | 真正运行时验证 actual model/GPU UUID/layout/kernel 与 receipt。新开卡的 UUID 未核实前不能宣称兼容，CPU 准备不探测 GPU。 |
| `p4_single_file_runtime_v4/verify_p4_single_file.py:38` / `:134` | 旧有限生命周期资格允许 on 的成本覆盖为 false；defer>0 只证明旧策略延期，不能证明通知实际启用或性能提升。新 CPU 接线需要独立的实际原 Queue wait/wake 有界证据。 |

未来最小闭环是：另建冻结 overlay 版本的新严格 receipt 工厂（可以逐字节继承冻结 C5 reactor/collector），先完成 CPU 入口与来源反例；在新的明确 GPU 授权和原累计预算内，用相同最终 overlay/collector、bridge=None/parent8 重新测原六窗口；新 raw 通过原估计式及独立 heldout 后，才产生新 receipt 与 off→shadow→on 的新共同锁。三臂共享底座，只有 on 安装通知。原 C4/v6、C5 CPU 结果与新的真机结果各自保留。

`test_frozen_entry_constraints.py` 的 8 项本地 CPU 检查已通过（0.004 秒）。这些检查证明入口身份和拒绝边界，不是 GPU/效果证据；后续新 adapter 的独立反例结果另存。
