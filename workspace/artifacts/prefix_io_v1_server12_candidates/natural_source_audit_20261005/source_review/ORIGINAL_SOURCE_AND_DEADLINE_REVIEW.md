# 原交接包：自然数据来源与开发预算要求复核

2026-10-05。以用户原 ZIP 的实际字节为准，完整读取 00/01/02/03/05、全部 5 个模板，并补读 04；11 份文件均列出 SHA 与原文行号。FULL_PLAN 仅作关键词交叉核对，规范冲突按 00:21 以分文件为准。本轮没有下载、SSH、GPU、模型/分词执行、造数据、签发 authority 或修改旧文件。8 份现有来源和 ZIP 前后 SHA 一致。逐条证据见 `ORIGINAL_SOURCE_AND_DEADLINE_REVIEW.json`。

**原包没有指定 ShareGPT 或其他具名自然数据版本。可以在结果产生前用固定规则选择有界真实公开样本，这是相容的实施解释；开发前外部独立 deadline/authority 是后期新增合同，不是原计划的必要输入。**

## 原文要求和后期合同必须分开

| 问题 | 原包实际要求 | 本轮判断 |
|---|---|---|
| 真实数据来自哪里 | 01:20、193 复用作者实验/replay；05:29–36 提到作者 dataset replay；03:138 要覆盖正常前缀问答、多轮、切换、低复用/低争用 | 没有具名语料、revision 或下载 URL。缓存 parser 支持 ShareGPT 不等于原计划指定它 |
| 可否选有界公开样本 | 03:142 提前冻结请求/前缀；03:207–215 使用完整固定 cohort；03:225–227 配对且报告全部结果 | 可以事先设计真实有界 cohort；这是实施解释，原文未逐字授权任意采样/下载 |
| 开发前必须外部独立 deadline 吗 | 02:316 内部 step budget 在开发集冻结并预留非 GPU 延迟；03:121 开发集选择 SLO；03:77 开发结束冻结策略/SLO | 不需要把后期外部 authority 当原包阻塞条件 |
| 正式评估能否任意填 SLO | 03:203 正式评估前固定每类 TTFT 与请求内 ITL 阈值，模板 null 不允许任意填后直接正式测试 | 不能。开发选择与正式评估预冻结是不同阶段 |
| 三套数据与家族 | 03:120 calibration 可用受控 token 负载；03:122 evaluation 未用于调参；03:124 会话/公共前缀家族不可泄漏 | 保持隔离和真实来源；原文未要求当前实现的所有固定 caps 或 exact family-proof 字符串 |

完整文本检索中 `ShareGPT/sharegpt/LongBench/HotpotQA/OpenOrca`、`full_control_window_deadline_ns`、`independent_deadline`、`authority`、`CLOCK_MONOTONIC`、`host_id/boot_id` 都为 0。这个检索只作补充；结论依据全文和正面条款。03:258 的 deadline/age 以及原 LoadPlanner 等待截止，是防止无限等待的运行机制，不能解释为外部服务 deadline。

## 内部经验预算与正式服务 SLO

02:201 的原公式保持：`T0_hat + Delta_hat + uncertainty_margin <= internal_step_budget`。02:316 明确写道：“内部 step budget 在开发集冻结，并预留 scheduler／sampling／输出延迟余量。”同一条禁止用 ITL 上限减最近一次 GPU step 声称确定安全 slack。01:173 也区分 GPU step 和请求 ITL；02:216 不承诺硬实时。

因此可以先在真实 development 上按**提前固定的有限规则**测量余量、确定经验预算和 SLO，再在正式 evaluation 前冻结。不能依据 evaluation/on 的结果放宽预算或 SLO；不能把已有 A-only 耗时改名为独立 deadline、制造 authority、声称确定安全余量。

当前 `prerental_protocol.py:328–366` 要求 `independent_deadline_declaration_v1`、`independent_requirement_before_development`、authority、同 host/clock 和 deadline-minus-reserve；`formal_trace_binding.py:321–334` 再核这些声明。这是后期更严格的实现合同。上一轮 `input_contract_review` 末尾把它列为“真正 development 前须完成”的材料，**只适用于那份实现合同，不是原包要求**；旧报告不改，本报告纠正归因。

若按原计划恢复推进，应在新追加的 CPU 开发合同中明确来源为“development 选择并在 evaluation 前冻结”，允许开发诊断暂缺正式 SLO/外部 deadline，保留原成本公式、物理边界、来源核验、实际 guard 和预算。不能删改旧冻结检查后声称其已通过，也不能给无正式 SLO 的开发输出签发正式 goodput/效果资格。

## 公开样本怎样选才保持真实

先选择真实来源和版本，在新的 GPU 输出出现前冻结选择规则。可按来源支持的完整 session/document/public-prefix 组，用固定 seed 或确定排序作有界选择；选择不依赖 GPU 代价、命中率或效果。保留原始下载字节、许可/版本、原行/数组位置、原 ID、所选完整组与实际 parser 接受记录的对应关系。报告只能覆盖这一固定 cohort。

所有被选请求、失败、慢请求、负结果、accepted I/O 和尾部 drain 都保留。不得截尾已有冻结 manifest、插入人工共享前缀、重复有利请求、制造 I/O 等待或按结果挑样本。新有界来源要在 parser 前用**追加的选择/来源清单**定义；当前旧 `freeze_trace` 的“全接受输入”仍完整适用于它声明的输入文件。

只分词不能产生真实家族证据。没有 source-backed 关系时应保持 `UNBOUND`，可做实际 CPU 检查和有界开发诊断，不能宣称独立 family-heldout 正式评估。若作者到达计划由 seeded replay 生成，应称“真实 prompt 的作者计划回放”，不能称生产到达时序。

数据方法允许选样，不自动等于下载许可。原模板只有 `allow_public_source_read: true`，下载/存储仍遵守当前有效权限与用户已给授权；本审计不新增权限，也不要求重复已给授权。

## 可以真实推进的最小动作

1. 选定一个实际可用的真实公开源/已有真实文件，在当前有效权限内准备原始字节、版本、许可和前瞻有界选择清单。无需等用户手工生成 family label 或 deadline authority。
2. 用原作者 parser 和真实 tokenizer 做 CPU 来源映射/解析/分词；从真实 session/document/public-prefix 证据闭合分区。缺失维度如实记 `UNBOUND`。
3. 追加源绑定的开发合同，恢复原计划的经验预算开发冻结流程，声明不签发正式效果资格。旧锁、代码、数据、严格合同完整保留。
4. CPU 来源与入口核验通过后，按实际授权和原 guard 做 U/off → I/shadow 开发观察；记录真实余量和机会，不把 shadow 反事实或启动差异当方法收益。
5. 开发结束冻结策略/SLO后，再做隔离 evaluation 的完整 AB/BA、token 时间线、失败/drain和统计。若真实机会不存在，报告否定结果并停止，不扩范围。

本只读审计**没有直接核验后续真实数据字节/家族来源映射或生成开发生产器**，不能因此推断项目仍无数据。父代理在审计完成后已报告：公开数据读取有现成授权，官方 SQuAD 固定 git blob 的 4,854,279 字节已核验，先声明的首 3 article 的 paragraph0 得到 40 QAs、30/5/5；真实 adapter 20 项 CPU 测试和完整映射回放通过。本代理没有重复这些测试，也不把它们计作自己的结果。这些父代理实证可继续来源支持的家族闭合，不需再要求用户手工提供同类输入。本报告自身不意味着正式效果资格。GPU 操作本轮为 0。
