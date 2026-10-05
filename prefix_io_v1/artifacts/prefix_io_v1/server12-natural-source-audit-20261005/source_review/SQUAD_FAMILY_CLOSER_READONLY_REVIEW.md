# SQuAD 家族闭合工具：只读语义复核

2026-10-05。复核 `close_squad_tokenizer_family_cpu.py`、adapter V2、原 `freeze_trace` 和 formal shape validator 的实际源字节；没有执行 tokenizer、CPU 测试、SSH 或 GPU，4 个源前后 SHA 一致。审计版本 SHA 和函数行号在同名 JSON。

未发现这些字节中的实际协议字段或家族来源阻断。

- 工具核验固定官方原始 SHA/Git blob，并用 adapter 重新构建全部 trace/mapping/report。家族来源是原 article/paragraph/context，全部接受记录逐序绑定，而非手填 family labels。
- 同一 article 或相同原 context 不允许跨区。真实 tokenizer 重编码全部 prompt，另独立编码 context；每条 prompt 的首个完整 16-token 块必须由原 context 覆盖，且首块在分区间不重复。在冻结 block16 精确缓存且没有隐藏追加前缀的条件下，更长公共缓存前缀必包含同一首块，因此该检查足以排除所选 trace 的跨区可缓存前缀。
- receipt/result 的 schema、输入/顺序/模型 SHA、真实来源 refs、四字段 token records 和 digest，均与原 `freeze_trace` 兼容。附加来源证据不改变正式 manifest 的精确字段。只读审计不代替父代理实际 close/consume 执行。

范围仅为固定选择的三个 article paragraph0 的完整派生 trace，不代表整个 SQuAD、真实生产会话或到达时序。本报告没有预测真实 tokenizer 首块检查会通过；若真实共享首块或边界不一致，现有工具应拒绝。CPU family receipt 不等于 GPU 成本、完整工作负载资格、正式 SLO 或方法收益。
