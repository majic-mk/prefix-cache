# 正式自然 token-ID 输入的 CPU 合同桥接

本轮用户范围是先完成功能与成本验证。本目录只准备后续正式输入合同，状态 **UNBOUND**，不增加真实 trace，不整合或运行新的效果 GPU 入口，不修改旧 frozen 协议、runner9、模型执行器、缓存引擎、cal05 或原预算。

当前实际没有自然 dataset、CPU tokenizer/family receipt、正式三分区 namespace、独立数字 deadline/service SLO、真实 I development reserve。因此新增接口不能输出 GPU eligible，也不能把旧 12 行 qualification 重贴为 evaluation。保留 qualification/off 必须继续调用原 `strong_trace_runner.validate_workload` / `check_phase`，原语义和返回值保持。

## 正式 schema 与字节闭包

只接受已有 `prerental_protocol.freeze_trace()` 原样生成的 `natural_trace_workload_v1`。原完整 manifest 保留全部 calibration/development/evaluation 行、原 request_id/顺序/到达时间、正式 family、token IDs、声明/tokenizer/source digests。选取一个完整 partition 时不丢行、不重排、不改 ID、seed、计划到达或输出数。不接受旧 `controlled_original_p3_qualification_workload_v1` / `natural_off_qualification_workload_v1` 的改标签版本。

新增 `formal_natural_trace_binding_v1` 是缺失输入 refs 的只读连接描述符，不是新 workload。字段严格为 schema 和以下 refs：

```text
manifest_ref
dataset_ref
author_trace_ref
author_common_ref
protocol_source_ref
declaration_ref
tokenizer_receipt_ref
model_manifest_ref
namespace_contract_ref
independent_deadline_ref
strong_pair_validator_ref
```

每个 ref 都是精确 `{path, bytes, sha256}`，必须在实际项目内部且属于当前完整 source lock。原协议/author parser/strong helper 采用实际固定 SHA；所有叶子的真实 bytes/SHA 复核。CPU tokenizer 的 `tokenizer_source_refs` 和 `tokenization_result_ref` 同样进入完整闭包；原 receipt 内的历史绝对项目路径保留，不能为了通过 digest 改写原 receipt。原 helper 需要其 nested refs 是实际绝对路径，外层 descriptor 可以使用同项目相对路径。

验证器使用锁定的原 `freeze_trace(dataset, trace_source, common_source, declaration, tokenizer_receipt)` **完整只读重放**，要求结果与原 manifest 全字段/全类型相同。仅自报 family/source digest 不足以跳过原重放。32 MiB 以内完整自然输入、每分区最多 32 条、最多 8 并发、当前 Qwen token-ID 范围和原 collector 的 4096-step 有界合同分别检查；不满足时拒绝，不通过截取数据修复。

## CPU token-ID 域与 namespace

只消费真正经过 CPU tokenizer/family 冻结的自然 `prompt_token_ids`。U/I 均保持 `skip_tokenizer_init=True`、无 speculative decoding；原 strong helper 验证两臂 planner-on、break-even、预加载、共享 staging、资源和采样相同，U off、I interference，只有策略/独立 namespace/日志身份不同。不能切到 text runtime tokenizer lane，也不能把 cal05 的旧 exact cells 自动泛化到新 shape；真实 table/geometry/device/domain 匹配和 unknown→U 回退仍归原 finite issuer/binding。

`formal_trace_namespace_contract_v1` 严格字段：schema、workload_sha256、model_manifest_sha256、tokenizer_receipt_digest、common_runtime_domain_sha256、initial_cache_state、no_per_request_reset、partition_namespaces。partition_namespaces 包含 calibration/development/evaluation，各含 U/I 的实际绝对项目私有 storage 路径；全部六个路径不同。所选分区与实际 pair 配置一致且尚不存在；一条分区流内保持同 namespace，不逐请求 reset。

独立 deadline 必须是已有 `independent_deadline_declaration_v1`，带实际 authority_ref、同 host/boot clock 和 `independent_requirement_before_development` 来源。正式 service SLO 按原 `independent_service_SLO_v1` 的 TTFT_ns/request_ITL_P95_ns 前瞻合同闭合；没有独立 SLO 明确 CPU 失败，不给其任意数值，不从 A-max 或 B/on/evaluation 结果反推。

## 薄 wrapper API

```python
formal = validate_formal_workload(
    actual_manifest, root=root, workload_ref=config['workload_ref'],
    binding_ref=config['formal_trace_binding_ref'], refs=source_rows,
    pair=actual_strong_pair, partition='evaluation')

phase_metadata = validate_formal_phase(
    config, root=root, refs=source_rows, pair=actual_strong_pair,
    formal_workload=formal, finite_activation=original_activation_preflight,
    development_replay=original_read_only_native_reserve_replay)
```

第一接口输出原 selected partition 的 records，直接兼容现有 `drive_original_engine()` 的 token-ID 提交分支；时刻仍是原 author schedule，TTFT/ITL 仍按原实际 token 返回记录，不伪造 chunk 内 token 时刻。

第二接口允许正式 `(effect,off,U)` 和 `(effect,on,I)`，都要求同一冻结 evaluation families、同 strong domain、同实际 independent development budget/SLO，以及原 activation descriptor 的真实 development 原始证据和 `reserve_join.verify_existing_native_reserve()` 只读重放结果。U off 不能绕过开发前置，I on 不能用 CPU代数/fixture发 table。源码 ancestry 检查要求当前闭包逐行保留原 calibration/V9 immutable rows；加 source10 文件不替换旧源码，也不改变模型/collector/geometry 域。

所有 API 返回始终 `gpu_eligible=False`、`formal_goodput_allowed=False`、GPU操作0：这些是 CPU 输入/phase 元数据，不能取代原实际 guard、真实设备、private issuer、finite current conditions、生命周期、输出与全部已接受 I/O 结算。新纯 API 未整合到 GPU runtime。尤其旧 native_runtime 的 effect 分支仍要求 I bridge，不能因为本模块接受 effect/U/off 元数据就宣称 U 效果臂已可执行；后续需另外完成和验收薄 wrapper/runtime 的对应接线。

## 真实 CPU CLI

```text
python -B -I -S formal_trace_binding.py
  --project-root <实际项目>
  --source-lock <当前完整SOURCE_LOCK.json>
  --manifest-ref <含原manifest精确ref的JSON>
  --binding-ref <含连接描述符精确ref的JSON>
  --pair-ref <含原strong_native_u_i_cpu_configuration_v1精确ref的JSON>
  --partition calibration|development|evaluation
  --output <新文件>
```

CLI 不生成缺失的 trace/tokenizer/SLO，不调用 RPC/launch。没有闭合 inputs 时失败是当前预期，不是系统功能或成本验证的失败。完整原 guard、source ancestry 与现场 preflight 仍由调用的薄 wrapper 验收；本 CLI 只验证输入元数据。

CPU 测试只用内存 schema/namespace 元数据和真实本模块文件的 SHA 拒绝检查；不写 trace 文件、不创建 native/GPU 能力对象。actual测试命令/结果保留在 LOCAL_CPU_TEST_RESULT.json 与 stdout/stderr。当前 formal UNBOUND 原因见 CURRENT_FORMAL_INPUT_DECISION.json。下一允许工作仍是 root 当前的功能/成本验证；本合同供以后自然输入和独立 SLO 具备后使用。
