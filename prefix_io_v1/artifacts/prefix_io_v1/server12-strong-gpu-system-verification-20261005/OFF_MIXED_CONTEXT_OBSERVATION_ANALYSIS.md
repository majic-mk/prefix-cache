# 强 U/off 标量观测失效的独立 CPU 审查

真实模型输出和生命周期通过，完整 CUDA 观测资格没有通过。本审查 GPU/RPC 为 0，未修改旧源码、原始结果或创建 GPU 事件。

原始 off 有 12 个请求，每个完整生成 128 tokens，外层 guard 正常退出并排空 OS 会话。collector 只保留 52 帧（ordinal 0–51），全部属于请求 0；事件 witnesses 为 0。最后一帧结束于 25944890177465748 ns，请求 1 于 25944890177994617 ns 提交，随后请求 0 的第 53 个 token 与请求 1 的首 token 在同一前端步骤返回。

最强解释是已有精确标量 adapter 不支持新请求加入后的异构 cohort：原请求下一 pre-context 为 563，新请求报告缓存 448 tokens、prompt512，因而可能形成 decode 与 prefill 混合。冻结 adapter 明确拒绝不同 pre-context（`mixed context has no exact cell`）及混合 decode/prefill。本次 CPU 重放成功重现两个已保存帧，并确认两类混合形状都拒绝。该假设重放是 CPU 诊断，**不是实际失败帧**。

失效处的 prepared rows、scheduler output 和 adapter 原始 last_reason 没有被保存；connector 将具体错误压缩为 `observer:ValueError`，外层只保存“scalar observer invalid”。因此不能声称已直接证明哪一个分支触发。52 帧少于 4096 frame 上限和 128 pending 上限，不支持容量耗尽解释；正常模型全输出也不支持原采样器抛异常解释。事件 factory 的具体实现错误未得到证明。

这些 UNKNOWN 不影响已经证明的 baseline 模型可运行，但禁止授予多请求完整 CUDA 资格。保持原冻结 adapter；不把 52 帧补成完整流，不伪造时间或放宽 128 步门槛。

单请求成本采集必须由自身证明完整 128 closed frames、128 真实 CUDA witnesses、原始完整输出与 ID、offset16 的 batch1/decode/context527、真实 ordinary preload 与 SSDread1/917504B/ZERO existingIO、native drain、六个独立窗口 AB/BA/AB、拟合2对＋独立 heldout1，以及原 guard 和完整源 before/after。首个窗口资格失败立即停止，不把 off52帧当成本样本。当前无方法收益结论。

完整输入字节引用、实际事实、CPU 拒绝重放及缺证清单位于同名 JSON。
