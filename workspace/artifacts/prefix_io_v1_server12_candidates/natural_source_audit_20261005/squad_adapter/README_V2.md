# SQuAD CPU adapter v2：保留原始文本与原作者边界语义

根执行者已先冻结选择规则并获取官方源。v1 的实际 CPU 运行因原 question 存在边界空格而拒绝；v1 源码、测试、交付清单和真实失败证据保留。

`squad_source_adapter_cpu_v2.py` 使用同一个前瞻合同、固定 commit、文章 0/1/2 各自 paragraph 0 和全部 QA 原序，未换篇、换段或筛掉请求。

`trace` 中每个 prompt 仍逐字符保存 `original_context + '\n\n' + original_question`。映射的 `raw_prompt_sha256`/`raw_prompt_utf8_bytes` 绑定该原始文本；`prompt_sha256`/`prompt_utf8_bytes` 绑定冻结原作者 `load_trace_prompts` 本来就会执行的边界 `.strip()` 所接受的实际 prompt。question、context 和 QA 元数据仍保留原文。逐行 `original_author_boundary_strip_applied` 与报告的布尔值/计数明确记录该现有行为，`original_raw_prompt_unchanged=True`。

这是沿用原作者解析语义，不修改原协议/分词器，不插入公共 prefix，不变更问题或删除慢请求。v2 移除“原 prompt 必须等于 `.strip()`”这一额外限制；未改先序选择规则。来源 mapping/report/replay schema 版本升为 v2，函数和 CLI 参数与 v1 一致。

20 项本地测试全部使用明确 synthetic fixture，覆盖 v1 原拒绝/回放范围，并实际使用已冻结原作者 CPU AST，证明带边界空格的 raw prompt 被完整保留且 accepted SHA 和原 `inspect_trace` 一致。未下载、SSH、分词或使用 GPU。结果只为来源适配，不是正式家族或 GPU 资格。

后续家族闭合工具必须绑定 v2 源 SHA、实际 mapping 的 accepted SHA 和实际 raw tokenizer inspection，同时单独保留原 raw SHA；不能直接把 v2 来源标签称作 token 前缀证明。
