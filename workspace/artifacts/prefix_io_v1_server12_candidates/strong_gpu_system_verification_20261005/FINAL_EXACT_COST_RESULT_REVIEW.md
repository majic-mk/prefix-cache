# CAL05 真实精确成本单元结果复核

CAL05 已完整完成这次固定六窗采集，未修改的公开发行器实际通过并发行 **1 个进程内私有精确成本单元**。本说明只读核对保存的结果、计划、源核验与释放记录，不再次调用发行器、不构造私有表，不进行 RPC、GPU 或模型执行；此前失败记录和交付草稿保持原样。[公开发行结果](CAL05_ACTUAL_COST_ISSUER_REPORT.json)

## 完成了什么

六个原 fresh child 窗口 00–05 均 exit0、原 shutdown 返回；六个严格窗口检查均为 `PASS_NATIVE_COMPLETE_WINDOW_CAPTURE_IO`，分别记录 actual_frame_count=128、frontend cached512、initial execution pre-context496。固定顺序为 AB、BA、AB：前两对拟合，第三对独立验证。[完成监视与逐窗口严格检查](CAL05_ACTUAL_COMPLETION_MONITOR_RPC_RESULT.json)、[预注册计划](CAL05_ACTUAL_BOUND_PLAN.json)

原 guard exit=0、child_exit=0，未超时或中断，OS 会话排空。独立最终查询也记录会话成员为空、GPU compute PID 为空、账本 active=None；这次六窗有完整原 shutdown 和 guard 退出证据，不依赖此前 CAL04 的主动清理替代自然关闭。[原 guard](CAL05_ORIGINAL_GUARD_RESULT.json)、[独立最终释放](GPU_FINAL_RESOURCE_DRAIN_RPC_RESULT.json)

完成监视文件直接给出的是每窗 frame 数。每窗完整128 outputs及128 CUDA witnesses，则由源锁中未改的原 `validate_capture` 必需 `len(frames)==len(witnesses)==128`、`len(output_ids)==128`，且公开发行器实际成功消费这条核验路径来支持。本复核核对了该源码 SHA 与通过结果，但没有第二次逐帧读取21,396,151-byte parent-closed raw，也没有重新发行表。因此不得把本复核称为新增的一次原始采集或完整 raw 回放。[公开发行结果及 measurements_ref](CAL05_ACTUAL_COST_ISSUER_REPORT.json)、[原 validator](../gpu_prerental_preparation_20261004/activation/native_conditional_cost.py)、[原公开 issuer](../gpu_prerental_preparation_20261004/activation/source/prefix_io_control/gpu_cell_issuer.py)

## 精确资格范围与数值

| 字段 | 此次实际已发行单元 |
| --- | --- |
| table scope | `gpu_verified_exact_cells`，`these_actual_exact_cells_only` |
| GPU UUID | `GPU-b2de2c25-cdc7-a350-267f-56e7763a287f`，RTX 5090 |
| batch / active decode / prefill | 1 / 1 / 0 |
| 实际 decode pre-context | 527，固定 measured offset16 |
| I/O | 1次 SSD read，917,504 bytes；已有 I/O 四阶段均为0 |
| 成本口径 | `existing_io_plus_delta`，原 full_decode_step CUDA Event 时钟 |
| baseline | 13,085,264 ns |
| incremental_or_joint | 3,541,552 ns |
| uncertainty | 6,304 ns |
| 上界总计 | **16,633,120 ns = 16.633120 ms** |

数值之和已经只读复算：`13,085,264 + 3,541,552 + 6,304 = 16,633,120`。这不是仅 SSD 传输的独立耗时，也不是总请求耗时、主机控制 reserve 或 SLO；不把 baseline 成本读数改作服务 deadline。[实际 CostCell、load_signature 与 qualified_identity](CAL05_ACTUAL_COST_ISSUER_REPORT.json)

同一 plan/report/guard/before/after 的 UUID、common runtime domain、模型和 KV layout 一致。common domain 为 `f240c3738ac18ac304c3330612e03f272079fe434dc98ae61f65b94c99aa4d1c`；source lock V9 SHA 为 `c70041b407438fa9c16cf151d59049ae12774309319ed52ede5527368de2a0f3`。此次资格不能迁移到别的 GPU、source/domain、context、batch、非零已有 I/O 或不同物理负载。[实际计划](CAL05_ACTUAL_BOUND_PLAN.json)、[发行身份](CAL05_ACTUAL_COST_ISSUER_REPORT.json)、[before](CAL05_ACTUAL_SOURCE_BEFORE.json)、[after](CAL05_ACTUAL_SOURCE_AFTER.json)

## 独立 heldout 与源闭包

预注册 plan 明确 pair0/pair1 为 calibration、pair2 为 validation，顺序 AB/BA/AB，seed4029/4030/4031；验证对的 prompt、prefix family、trace SHA 与两拟合对不同，本复核重新核对这些摘要与实际固定 prompt tokens 对应。未改 issuer `_cell` 只把 `entries[:2]` 传入拟合，`_fit_calibration_only` 又按这些 fit IDs 过滤 A/B；heldout action 必须 `heldout[0] <= upper` 才能实际发行。结果记录 `independent_holdout_used_to_fit=false`，未为了通过而重拟合、抬上界或扩大单元范围。[计划](CAL05_ACTUAL_BOUND_PLAN.json)、[发行结果](CAL05_ACTUAL_COST_ISSUER_REPORT.json)、[公开 issuer 源](../gpu_prerental_preparation_20261004/activation/source/prefix_io_control/gpu_cell_issuer.py)

所要求的发行摘要与完成监视没有导出独立 heldout B 的具体 ns 数值；这里保留该字段 unknown，记录通过证据是原公开发行器的必需覆盖门。若最终完整 raw 归档后来补充该值，应注明真实 raw 来源，不从 upper 或其它字段反推。[发行结果](CAL05_ACTUAL_COST_ISSUER_REPORT.json)

服务器真实 before 与 after 源核验均覆盖 **4,876 个文件、16,049,247,604 bytes**，failed=[]，相同 plan/source-lock refs；本复核还核对本地计划和 guard 字节 SHA 与发行器引用一致，及实际 issuer/validator/CostTable/wrapper 字节与 plan pins 一致。完整4,876资产的逐字节核验是这两份真实服务器记录完成的；本复核没有再次扫描全部模型/SDK资产。[before](CAL05_ACTUAL_SOURCE_BEFORE.json)、[after](CAL05_ACTUAL_SOURCE_AFTER.json)、[本复核统计](FINAL_EXACT_COST_RESULT_REVIEW.json)

## 实际 GPU 预算与后续边界

CAL05 原 guard 实际计入预算 **807.2937119267881 秒**。最终原8小时账本累计 used **26,222.62820187537 秒**、remaining **2,577.37179812463 秒**，两者合计28,800秒；active=None、compute PID 空、独立 OS 会话成员空。此为保存的最终查询快照，后续作业仍需扣账。PRIMARY 剩余空间记录为52,134,408,192 bytes，tracked git status为空。[原 guard](CAL05_ORIGINAL_GUARD_RESULT.json)、[完成监视与账本增量](CAL05_ACTUAL_COMPLETION_MONITOR_RPC_RESULT.json)、[最终资源查询](GPU_FINAL_RESOURCE_DRAIN_RPC_RESULT.json)

用户选择先完成系统功能和成本验证。当前证明的是这个现场的原执行/缓存生命周期与有限精确成本发行路径已经真实通过；普通 I 候选上的 GPU 成本消费、完整主机控制开销、独立服务 deadline/SLO、自然负载性能或论文效果仍没有由这些结果证明。发行结果明确 `formal_SLO_qualified=false`、`strategy_effect_verified=false`；CPU runtime 接线测试也不能替代实际 I 激活证据。[公开发行结果](CAL05_ACTUAL_COST_ISSUER_REPORT.json)

这份 JSON 和说明是可归档证据，**不是可复用 GPU capability**。另一实际消费进程必须通过已核验公开入口回放原证据，保持同一实际 CostTable 类型并检查当前 GPU/source/context；不得从 JSON flag、摘要或本说明重建私有表。未知条件仍回到原路径，独立 deadline 和真实 control reserve 没有完成前不能声明预算准入或方法收益。[公开发行结果的 json_report_is_not_reusable_GPU_capability](CAL05_ACTUAL_COST_ISSUER_REPORT.json)、[成本消费 API 与边界](CPU_REVIEW_COST_ISSUANCE_AND_CACHE_SEMANTICS.md)

本次本地 Python `-B -I -c` 只读复核完成 **21项一致性检查，0失败**，11个输入证据/源码前后 SHA 相同；GPU/RPC/model/issuer calls 均为0，只描述本复核动作，不抹去CAL05真实GPU运行。[FINAL_EXACT_COST_RESULT_REVIEW.json](FINAL_EXACT_COST_RESULT_REVIEW.json)
