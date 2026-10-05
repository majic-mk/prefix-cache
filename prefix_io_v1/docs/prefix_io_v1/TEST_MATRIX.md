# 03 合同测试覆盖（本轮证据上限）

“CPU”包含真实 CPU 运算和明确的 fake-backend/fixture；不等于 GPU 或完整集成通过。最终日志 cpu-verified.xml 包含 328 通过、8 跳过。未完成类别仍保留，不用相似单测代替。

| ID | 本轮状态与证据 |
|---|---|
| T01 | 审计：本地状态留档；服务器原 tracked 文件未改变；最终保护检查 |
| T02 | BLOCKED：source Plan API 已确认，真实 vLLM/copy handler 未安装验证 |
| T03 | 源码限制已审计；真实 KV layout 集成待 P1 |
| T04 | 保留上游 identity；GPU 前缀/祖先/模型隔离验收未执行 |
| T05 | 上游 CPU 指针/拷贝 fixture 通过；同一生产 GPU KV 往返未执行 |
| T06 | GPU 模型输出/logits 未执行 |
| T07 | 上游 parent/copy CPU/mock 测试通过；真实部分恢复未验证 |
| T08 | 源码 end_event.query 保留；真实 CUDA 事件验收未执行 |
| T09 | 上游 slot/copy CPU/mock 测试通过；完整重复 completion 故障矩阵待验证 |
| T10 | CPU test_active_or_unknown_refs_never_release 通过；live 引用适配未接入 |
| T11 | CPU all_parent_conditions / incomplete_multi_guard_closure 通过 |
| T12 | CPU child_or_d2h_completion_is_not_parent_completion 通过 |
| T13 | CPU old_generation_and_old_run_fail_closed 通过；真实 generation 来源未接入 |
| T14 | 作者 shared-preload CPU/mock 套件通过；真实 DMA 验收未完成 |
| T15 | CPU native_clean_reclaim 及作者 clean-cache 单测通过 |
| T16 | 配置禁止门禁前启用限流；尚无 live quota，不声称任务限流完成 |
| T17 | 接受队列上限尚未接入；原队列未替换 |
| T18 | CPU actual_backing / min_slot_floor / preallocation_guard 通过；pinned allocator 总开销未测 |
| T19 | epoch quota 尚未实现／未接入 |
| T20 | 上游融合路径和 mock 测试保留；新额度计费未实现 |
| T21 | 共享读写额度／联合干扰表尚未实现 |
| T22 | 调用链已审计；真实 mandatory bridge 尚未实现／验证 |
| T23 | 原 _on_store_copy_done 续接函数未变；配额后的 GPU 进展待验证 |
| T24 | 新策略 age/progress override 尚未实现 |
| T25 | CPU snapshot freshness 拒绝过期、跨 run、未来时间；live fallback 未接入 |
| T26 | null 标定不视为 0，active interference 配置拒绝；真实表外策略尚未实现 |
| T27 | 源码取消／资源保护已审计；DMA 取消测试未执行 |
| T28 | 源码短读失败和 worker assert 已审计；真实 backend 故障验证未执行 |
| T29 | CPU failed_draining / failed_final 不计释放；真实异常排空未验证 |
| T30 | native-path-preservation.json + off/shadow CPU wrapper 测试；真实引擎尚未安装 |
| T31 | O_DIRECT 4096 字节探针成功；不等于真实模型 SSD cache hit |
| T32 | 完整请求 trace 尚未执行 |
| T33 | 没有逐 token 输出记录；ITL 未测得 |
| T34 | CPU preflight 保持 gpu_executed/gpu_verified=false；未伪造 GPU 证据 |
| T35 | 专属目录创建、默认权限和路径检查；无共享数据删除 |
| T36 | GPU overlap timeline 未执行，不以 non_blocking 标志代替 |
