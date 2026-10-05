# 租卡前独立审阅：已知风险与投资停止条件

本审阅只读原交接指令、上一轮服务器 CPU 交付和原作者源码，不执行 GPU、RPC、模型、缓存引擎或 JIT 库加载，不修改旧冻结文件。新 runner/protocol 尚待终版审阅；本报告不是 GPU 入口合格回执。

1. **先排除无效投资**。原 cal01 为 planner=off 的六窗口单文件机制诊断，最大预留 1,220 秒，约占剩余 4,121.104761 秒的 29.6%。其回执不覆盖 planner=on 强 U/I。原九槽 3,780 秒计划亦未包含这项强域资格。新 GPU 作业单必须把强域标定、问题存在性、shadow/on 生命周期和完整对照放在同一预算中，不能先跑旧清单再声称还有资金补强基线。保留所有旧产物；只新增明确替代计划。

2. **LoadPlanner 警告是有效性限制**。原 `_build_planner` 在本地 GPU Prefix Cache 开启时保留 `recompute is overpriced` 提示；原 `CostTables` 使用 recompute/SSD/memory 曲线，超出最后实测 knot 时还有外推警告。CPU parser/builder PASS 证明配置可构造，不能证明这些定价适配当前 GPU cache 状态，也不能称 fully tuned 强基线。两臂应保留相同原准入与全部特性；不能关闭 Prefix Cache、LoadPlanner 或改准入来消除警告。实际 GPU 必须留下成本域、曲线覆盖、缓存命中及原准入决定证据。无法在固定范围内得到可比强基线时停止收益主张。

3. **P4 成本和作者 break-even 成本是不同合同**。作者表的 block-granular recompute/SSD/memory 秒数不能直接替换完整 GPU `execute_model → sample_tokens` 当前 stream elapsed-time 的 P4 成本收据。自然请求流的 batch、context、KV layout、传输大小、planner 行为与旧单文件 offset16/context144 若不同，必须明确新的精确 cell 集。CPU 可以列举预期域、绑定源/模型/表和拒绝未知条件；不能生成实测上界或把未知 cell 当覆盖。普通候选始终为空或新策略没有实际动作差异时停止当前 pilot，不能扩域拟合、放宽旧门或增加人为 I/O 延迟。

4. **CPU 能完成 trace/deadline 合同，不能冒充实测**。可以绑定已存在数据与原作者 parser/schedule、固定请求顺序与 AB/BA、禁止 prefix injection/wipe，并预先声明独立 deadline。真实 control/output reserve 和成本覆盖仍需开发 GPU 数据。原 P3 prepare 明确生成 synthetic token 请求；其没有人为 I/O 限速，但不能改名为生产天然 dataset。若没有现成自然 trace，报告缺失或明确限定为正常机制受控 pilot；不得凭 CPU 声称真实资源等待或正式 service goodput。

5. **私有 JIT 准备要区分编译与运行**。CPU 可以验证工具链路径、源锁、私有缓存命名空间、编译/链接和可执行 argv。禁止为准备而加载需要 CUDA/native 初始化的共享库或导入真实模型；已编译不证明设备能力、核函数或 on 活性。两臂共享相同合法工具链与共同兼容修复，研究策略关闭时返回原执行路径。

预算投资门应在租卡前形成有界、机器可检验的作业单：各作业执行与收尾相加不超过剩余原账本，原 guard 独占预留、失败计费、无自动额外重试。先验证强域和正常机会；失败即停止后续预留。若达到预算边界、没有普通候选或资源等待、简单基线等效，或必须改引擎/attention/释放协议/新淘汰算法，应交付当前负结论，不继续租卡赌收益。持续 GPU 授权不需逐轮重复审批，权限并不替代科学与生命周期门。

本目录 `audit_prior_preparation_cpu.py` 会实际核验相关原文件 SHA、AST/合同边界、旧预算与资格判定，并写新的 CPU 结果。最终 runner/protocol 应另附审阅结果；本初审不预先宣布终版 PASS。
