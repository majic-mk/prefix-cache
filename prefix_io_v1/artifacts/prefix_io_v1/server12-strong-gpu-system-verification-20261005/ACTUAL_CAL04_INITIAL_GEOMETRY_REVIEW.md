# CAL04 首次真实准备几何复核

本报告仅做本地原始文件和源码复核；没有 RPC、GPU 调用、源码修改或拟合/留出窗口 GPU 时长分析。

`CAL04_COMPLETED_FIRST_WINDOW_NATIVE_RESULT.json` 只有首个 A 窗口，实际前端输出完整 128 tokens。其 capture.valid 为 true，128 个实际模型帧及 128 个 CUDA witness 的原始 ordinal 连续为 214–341。首次 prepared 为 pre_context=496、prompt=512、scheduled=16，input_seq_lens=[512]；第二帧为 decode/context512；选定 offset16 为 decode/context527、scheduled1。前端 warmup 与正式输出的 num_cached_tokens 均为 512。初始 52 次 no-forward 是单独诊断，没有计作模型帧或重编号；首调用确为作者原类、模型工作字段为空、KV metadata 非空。

496 是本次原模型实际准备的值，原 511 预设与之不符。作者 `SOURCE_CACHE_SEMANTIC_scheduler.py` 629–633 行在初次匹配阶段设置 prefill_stats；`source_inputs/output_processor.py` 628–634 行把该统计值交给前端。因此报告命中量 512 不代表全部 512 个 token 最后都作为可用计算前缀进入 GPU。实际 frame adapter 182 行读取的是原 input_batch.num_computed_tokens_cpu；模型 runner 1402 行从原请求状态更新该数组，故 prepared496 是实际执行输入的证据。

两份真实 stdout 均在 41、44 行报告 `Recovered from KV load failure`，影响 16 tokens；28 行同时记录作者 OffloadingConnector 将其原配置中的 fail 改为 recompute。scheduler 2284 行遇到首个 invalid block 后把 num_computed_tokens 截断为 block_index × block_size；2111–2141 行的失败恢复分支保留已截断前缀，正常完整外部命中分支才执行 512→511。结合 block_size16、实际496+16=512，本次几何符合原加载恢复后重算最后一块。原 GPU 本地前缀查询也以 prompt−1 为上界并只接受完整块（kv_cache_manager 218–229 行），所以正常块对齐查询也可能得到496；不能仅凭496独立认定传输故障。

通用恢复警告也不能定位故障根因。与 CAL04 实际 loaded_native_modules 相同 SHA 的原 py-kvcache `reactor.py` 1431–1455 行可按 break-even 主动拒绝加载；`vllm.py` 630–638 行把 LoadDeclined 作为成功的零字节完成并发布 declined block IDs；原 offloading worker 531–538 行通过 get_block_ids_with_load_errors 把这些 IDs 交给 scheduler 触发重算。因此，现有证据证明发生了原路径的部分重算恢复，无法区分主动成本拒绝、实际加载失败或具体缺失块，更不能证明整个强 U 加载路径不可用。需实际 decline/transfer failure 事件与请求、块的关联才能定位原因。

这次完成的 A 窗口证明新被动观测可以在合法初始 no-forward 后保留完整真实模型帧。首窗口 original_engine_shutdown_returned 和 native tail 记录通过；根 guard 后续明确早停，最终会话排空、active=None。它们不等于所有 KV 加载成功，不构成方法收益证据。其 frozen plan 首次几何与真实值不同，CAL04 整体仍不具有成本单元资格：缺少完整六个新窗口与独立配对验收，不能追认、改写旧结果或拼入未来拟合/留出数据。

下一步只以新 V7 冻结准确的首次496/prefill16契约并在每窗早验，保留前端512报告、offset16/context527、原模型/缓存/LoadPlanner、估计器和策略预算。重新采集全部六个 fresh 窗口；任何窗口几何、原始事件或生命周期不符立即拒绝并停止。不得由本次 CAL04 回推 CAL03 未持久化的现场帧，也不得读取留出 GPU 时长来选参数。
