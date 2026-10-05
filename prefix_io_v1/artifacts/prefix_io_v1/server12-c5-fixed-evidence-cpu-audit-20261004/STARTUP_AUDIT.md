实际启动审计已在本机读取已核验归档的扁平映射完成，未连接服务器，未导入模型、CUDA 包或共享库，GPU 操作为 0。现场执行脚本仍只做同样的 CPU 元数据读取，不重新运行实验，不发行缓存成本 receipt，不改阈值或研究策略。

本机正式结果是 `STARTUP_EVIDENCE_CPU_AUDIT_FINAL.json`。较早的 `STARTUP_EVIDENCE_CPU_AUDIT.json`、`STARTUP_EVIDENCE_CPU_AUDIT_V2.json` 保留为开发期成功审计记录；之后只加强冻结 source row、逐次账本折叠和输出目录校验，并区分缺失证据清单与下一轮最小观测。开发期首次执行因 manifest 不在 `files` 中而拒绝：实际归档将 manifest 单独放在 `manifest_local_path`，脚本已按其真实 bytes/SHA 校验后读取，未跳过校验或修改输入。所有输入和所用脚本的 SHA 都写入结果。最终源码编译及 18 项纯 CPU 路径／类型／JSON／区间拒绝检查通过。这里的 PASS 只说明已有启动／账本／日志记录相互一致。

三次真实记录分别显示：

| 观测 | off01 | off02 | off03 |
| --- | ---: | ---: | ---: |
| 完整入口校验（秒） | 24.068573 | 24.471002 | 24.265060 |
| 其中公共 receipt 回放（秒） | 19.490326 | 19.786572 | 19.617678 |
| LLM 初始化（秒） | 85.070050 | 85.412312 | 85.426268 |
| 原 guard 完整作业（秒） | 173.421246 | 175.386692 | 174.917104 |
| 公共 loader 调用 | 1 | 1 | 1 |
| 创建 context 后内部 metadata 重验计数 | 7 | 7 | 7 |

公共回放的起止时刻完整包含在约 24 秒的入口区间中，不能把二者相加成约 43 秒。源码中的成功 `ProcessEntryContext.create` 路径只有一个 `controller.load_receipt` 调用；后续计数随 metadata 重验递增。计数 7 指 context 创建后的内部重验，不表示整条入口只有 7 次授权／来源校验。它们是主机启动阶段的墙钟耗时，不是策略纯 CPU 成本，也不是 GPU kernel 时间。

日志分别报告模型权重加载约 4.315／4.183／4.252 秒，以及 engine profile／KV 初始化／模型 warmup 约 66.42／66.32／66.92 秒。这些日志阶段不能再与 85 秒 LLM 区间直接相加。日志还记录 FlashInfer autotuning，以及每次 3 条推理期 Triton JIT 警告，涉及 `_compute_slot_mapping_kernel`、`kernel_unified_attention`、`reduce_segments`。

记录不足以把 JIT 归因到选中的第 16 个测量步骤。警告只有日志墙钟时刻，没有请求 ID、native step ordinal 或和 CUDA Event 对应的编译起止记录。已有 1 次完整 128 token 热身和 128 帧 Event 捕获能够证明有限观测完整，却不能证明所有 kernel/shape 在选中步骤之前已完成热身。引擎 `enforce_eager=True`，并明确关闭 torch.compile 和 CUDAGraph；这里的原生 CUDA Event 捕获不是 CUDA Graph 捕获。

本轮三次作业在原账本中追加了 3 个唯一、成功、会话排空的 guard 事件，合计 523.725041784 秒。累计从 24,155.170197121 秒变为 24,678.895238906 秒，原 28,800 秒上限未扩大，剩余 4,121.104761094 秒。supervisor 记录的 1,189.626203258 秒还包括 CPU 资格回放与核验，不能把它等同于 guard 累计预算，也不能据差值证明 GPU 已释放或不重叠。

旧正常 off 的 guard 为 261.120514885 秒，旧 raw 没有本轮对应的启动区间或 loader 计数。新旧差额仅作描述：作业时间、JIT、主机和缓存条件没有构成配对控制，不能把约 86–88 秒差额因果归于入口去重，更不能当作调度方法收益。三次选中步骤分别为 16,266,945／15,458,848／19,185,921 ns；原 upper 16,238,752 ns 与 A-only budget 13,171,328 ns 均保持原值。启动优化没有消除成本越界，正常 gate、shadow/on 与 P4 提升资格仍未解锁。

下一轮有信息价值的最小观测，是把原路径上的 JIT 状态／编译起止与请求和 native step 绑定，并在相同测量窗口记录 GPU 温度、功耗、SM/显存时钟及降频原因，以及真实 SSD 读取的缓存状态或足以判断缓存状态的见证。现有记录缺少这些与测量帧绑定的观测；不可用绝对 CPU 时间戳补推 CUDA 时序、物理释放或 GPU 不重叠。无需为了完成本次审计租 GPU。补齐有限观测能力之后，才值得讨论下一次有固定次数的诊断；保持旧阈值、旧反例和原预算，不能不断重复同样实验直到碰到通过。

服务器命令（输出必须为新文件）：

```text
.venv/bin/python -B -I -S artifacts/prefix_io_v1/server12-c5-fixed-evidence-cpu-audit-20261004/audit_startup_evidence_cpu.py --project-root /root/autodl-tmp/prefix-io-v1-handoff/project --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-c5-fixed-evidence-cpu-audit-20261004/STARTUP_EVIDENCE_SERVER_CPU_AUDIT.json
```

本机使用相同脚本，额外传入 `--repeat-mapping native_gpu_delivery/repeat_fixed3_verified/LOCAL_REPEAT_DELIVERY_BYTE_VERIFICATION.json` 与 `--prior-mapping native_gpu_delivery/normal_off01_verified/LOCAL_NORMAL_DELIVERY_BYTE_VERIFICATION.json` 的实际路径。每个读取的文件均按映射和原 manifest 重新验证 bytes/SHA，并在审计结束再验证；不读取模型权重／SDK 二进制，不把本机读取宣称为服务器全量来源重验。
