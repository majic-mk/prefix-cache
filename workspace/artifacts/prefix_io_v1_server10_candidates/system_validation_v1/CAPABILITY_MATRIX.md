# 新服务器能力矩阵

2026-10-03 交付快照。以下路径均相对服务器项目 ROOT。PASS 只适用于列出的真实运行及有限案例；源审计、CPU 测试和编译链接不能替代 GPU 结果。

| 能力及归属 | 本轮状态 | 证据与适用边界 |
| --- | --- | --- |
| 精确 Prefix Cache、原模型执行器：既有能力 | 有限数值参考 PASS | 新机 native/paired 各 6 请求、guard exit 0；独立后处理为 `PASSED_EXACT_CACHED_REFERENCE`，6 个 cached 比较的 top5 差值均为 0。见 `artifacts/prefix_io_v1/server10-reference-postprocess-v1-20261003/REFERENCE_CPU_POSTPROCESS_RESULT.json`。 |
| 冷算与 GPU-hot 对照：既有路径 | 差异单列 | cold 与 GPU-hot 的 top5 存在差异；不得将此项写成零差，也不得与 cached 对 GPU-hot 的精确比较混同。原始行见 `experiments/prefix_io_v1/runs/server10-g3-reference-native-01/details/acquisition/result.json`。 |
| 原生存储/恢复、共享 staging、预加载及共享消费者：既有能力 | G1 off/shadow GPU PASS | 两种模式均为 `PASS_REAL_GPU_P4_NATIVE_OFF/SHADOW`，真实 CUDA、Linux AIO、`exact_content=true`；覆盖 store、age_store、restore、shared。见 `experiments/prefix_io_v1/runs/server10-p4-native-off-01/details/result.json` 与 `experiments/prefix_io_v1/runs/server10-p4-native-shadow-01/details/result.json`。 |
| 原生命周期、共享 staging 上限及排空：共同路径 | G1 有限案例 PASS | 四阶段 AIO 均 closed/drained；实际 staging 16,650,239 B < 16,777,216 B，accepted parents 最终为 0。证明这些对象和案例的排空，不等于完整物理资源释放模型。证据同上两份 G1 结果。 |
| 紧凑观测和 shadow 桥：观测增量 | G1 shadow 有限案例 PASS | 原执行路径保留，shadow 观测通过；off 路径可用。不得由 off/shadow 推断主动调度策略或普通额度为零时的行为。 |
| 完整模型冷请求/重复请求：G2 系统资格 | GPU PASS（完整输出和关闭） | off/shadow 各 cold/repeat 两请求，每请求完整 128 tokens，共 512 tokens；重复请求缓存 112 tokens。两模式原 shutdown 返回、guard exit 0、OS 会话排空。shadow 实际采到 256 帧，原报告仍未授予生产 collector/时钟映射资格。见两作业 `details/normal-model-lifecycle-result.json` 及本交付 `G2_ACTUAL_RESULT_SUMMARY.json`。 |
| 成本准入、复制合并、异步流水线：既有实现 | 保留；专项量化未完成 | 本轮未重写缓存引擎或执行器；不能仅凭 G1/数值参考判定成本模型、合并收益或流水线收益有效。 |
| T05 生产 KV 全字节一致性 | **未验证** | G1 使用合成 KV；当前缺少真实模型生产 KV 的 capture/逐字节证据。有限返回 token/top5 一致不能替代此项。 |
| 真实资源释放依赖、有限候选调度、干扰额度：研究增量 | **未验证主动策略效果** | 未形成新机有效成本曲线、完整物理释放证明或策略收益对照；不得宣称 P4 全部完成、生产资格通过或性能提升。 |

原参考父进程的 exit 78 记录保留：失败发生在 CPU 分析器来源检查，后续独立 CPU 后处理修复来源检查后读取已完成 GPU 数据；未改原数字比较器，也未将父进程原失败记录改为成功。

版本和克隆审计另见 `artifacts/prefix_io_v1/server10-system-validation-v1-20261003/SERVER10_VERSION_AUDIT.json`、`SERVER10_CLONE_SOURCE_AUDIT.json`；这些是来源证据，不是 GPU 能力结论。
