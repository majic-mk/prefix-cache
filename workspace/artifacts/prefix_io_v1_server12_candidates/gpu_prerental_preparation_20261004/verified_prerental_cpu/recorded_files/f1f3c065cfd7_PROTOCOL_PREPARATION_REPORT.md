# 租用 GPU 前的协议与数据准备

本目录只新增 CPU 工具、原数据引用和测试。服务器实际运行与最终源锁由总交付核验；本目录没有 GPU 或模型操作，没有新租机、下载、驱动修改或删除。

## 已完成的输入

`CONTROLLED_P3_OFF_QUALIFICATION_WORKLOAD.json` 绑定原服务器 P3 的 `low_contention.json`，原文件 94,011 字节，SHA-256 `3ef3b7a65d87cb8f4d1cb88c885137ec29e1eebdfebe0aedf420e803d9d0a5d9`；原 index 658 字节，SHA-256 `ab21d8b18acf2328d8918f45db3d1a94823133bee0ed58251553209a610931fe`。

保留全部 12 条请求、每条 512 个原 token、原到达时刻和前缀复用关系，固定完整生成 128 tokens，并沿用原 4 并发上限。只读取已有 token，不重生成随机 token，不人为节流，不给任何请求注入额外前缀，不逐请求重置缓存。CPU 使用原作者 `generate_request_schedule` 完整函数 AST 验证每条原 request spec。原令牌、schedule 和源文件字节保持不变。

这是**原 P3 受控机制流**。它曾用于 development，不是新 heldout 数据，更不是生产自然 trace。机器合同始终设置 `natural_trace_bound=false`、`fit_or_evaluation_input_allowed=false`、`gpu_effect_qualified=false`、`formal_goodput_allowed=false`。可以用于强 U/planner-on 的完整输出和生命周期资格检查，不能用于论文主效果、成本重新拟合或独立评估。

总交付的 `ACTUAL_SOURCE_TREE_TRACE_DISCOVERY.json` 为真实服务器源目录搜索；现有候选为配置和 source examples，没有合格自然请求数据。本地两个 `raw_p3/*-trace.json` 为 profiler `traceEvents`，原作者 prompt parser 不会把它们识别成请求数据。不能将 profiler 时序或原人工 token 家庭冒充自然请求。

## 可执行工具

`prerental_protocol.py` 提供以下 CPU 子命令，输出用 exclusive-create，已有证据不能覆盖：

- `inspect`：对现有 JSON/JSONL/text 调用原作者纯 parser AST，计算源字节和原顺序摘要；不启动作者网络或 cluster 模块。
- `freeze-qualification`：在已有真实自然 trace 出现后，按事前声明的 first N（最多 32）冻结原文本和原作者 schedule。明确列出未选尾部，不作为拟合或评估输入。实际 token IDs 由真实模型输出捕获，CPU 不编造。
- `freeze-original-p3`：严格绑定上述既有 P3 流，用于当前可执行的受控 off 资格分支。
- `freeze-trace`：正式数据合同需实际 CPU tokenizer 输出的来源字节闭包、会话/文档/公共前缀家族闭包，以及事前固定的连续 calibration/development/evaluation 分割。原顺序保留，任何重复提示词或前缀家族跨分区均拒绝，不丢请求来修复泄漏。
- `freeze-budget`：只在独立 deadline 的实际来源文件存在且提前声明、同主机/boot 的 host CLOCK_MONOTONIC reserve 数据齐全时计算预算。合并 host 控制区间避免重复计量；拒绝混用 CUDA 和 host 时钟。SLO 是严格独立合同，不能由 A-max 或旧 B/on 结果抬高。该数学工具始终不授予正式 goodput、原生模型或策略资格。

作者 `shared_storage_trace_replay.py` 本身只读取 prompt，丢弃可能存在的 arrival 字段。它的 `build_global_specs` 默认 arrival rate 为 null，全部 scheduled time 为 0；指定正 arrival rate 时为作者的 Poisson 重放 schedule。工具据实标明 schedule 来源，不能宣称保留了数据集记录的自然到达时刻。

当前独立 deadline、实际 non-GPU reserve 和 service SLO 仍缺失。这些硬件或外部输入缺失不代表本目录 CPU 实现未完成；它们确实阻塞普通干扰策略和正式效果阶段，不能用旧 A-max 代填。

## 预算与先租卡的价值

原累计 GPU 限额保持 28,800 秒（8 小时），已知剩余快照 4,121.104761 秒。首轮只规划强 U/off **300 秒执行 + 20 秒收尾 = 320 秒**，成功后至多剩余 3,801.104761 秒；没有实际预留，也没有允许重试或扩大预算。

不再把旧 planner-off `cal01` 的 1,220 秒诊断加入新计划。该诊断的成本域不覆盖现在的强 U/planner-on；重复烧 GPU 并不能使该域合法。新强域成本、开发 reserve、shadow/on 和评估次数/上限尚需各自实际门禁及现有 ledger 再核验，不能声称剩余预算足够完成所有效果实验。

有界强 U/off 的价值是核验真实强基线输出、原生命周期和可追溯机制事实。CPU 准备、真实输入、完整源锁、原 guard 和现场资源条件全部通过后才可启动。受控 P3 流的通过不能证明自然目标问题存在，也不能证明我们的策略有收益。

保留原实施合同的 **约 10% goodput 改善继续投入门槛**和 **低争用退化约 2% 的工程目标**。这些不是成绩承诺或论文录用门槛。没有普通合法候选、正常目标问题不可重复、强简单基线已取得相同效果，或必须人工节流/扩研究范围才有收益时，保留负结果并停止相同配置的重复投入。

## 实际 CPU 测试

最终本机运行 30 项测试，0 failure、0 error、0 skip。8 个输入文件在测试前后 SHA 一致，GPU、模型和网络客户模块导入为 0。

测试覆盖原作者 parser/schedule、现有 P3 全部令牌/spec/arrival 保真、数据和源字节漂移、正式分区泄漏、tokenizer来源闭包、128-token 合同、人工注入/reset/throttle 拒绝、独立 deadline、reserve 区间合并、同 host/boot clock、无剩余预算、NaN/Infinity/bool 预算、SHA 非 hex/缺文件/错字节、未验证 SLO、自报非空 tokenizer refs，以及原 planner-off 旧成本域不能冒充 strong 域。

早先本机 22 项和 27 项结果及独立拒绝探针保留。独立探针揭示第一版租卡判定接纳 NaN/Infinity、只查 SHA 长度及自报 SLO 的问题；这些问题已修复，并新增实际拒绝用例。早期 pass 只代表当时用例通过，不覆盖后来发现的缺陷。

实际执行命令：

```text
python -B -I -S protocol/verify_prerental_cpu.py --source-dir ../i_pilot_cpu_preparation_20261004/source_inputs --output protocol/LOCAL_PROTOCOL_CPU_TESTS_FINAL.json
python -B -I -S protocol/prerental_protocol.py freeze-original-p3 --dataset source_inputs/low_contention.json --index source_inputs/index.json --source-refs source_inputs/P3_CONTROLLED_AND_CLOSURE_SOURCE_INPUTS.json --author-trace ../i_pilot_cpu_preparation_20261004/source_inputs/shared_storage_trace_replay.py --author-common ../i_pilot_cpu_preparation_20261004/source_inputs/prefix_cache_common.py --model-manifest-sha256 9b1c79986aa1fd62eb9064ecb8b4e2a03fc235a34ac2d69c33a4fcb531a91017 --output protocol/CONTROLLED_P3_OFF_QUALIFICATION_WORKLOAD.json
```

测试数据中用于拒绝测试的临时 JSON 是显式 CPU fixture，既不是实际自然流，也不构成原生运行资格。服务器测试次数、源锁和最终 byte join 以总交付中的实际服务器日志为准。
