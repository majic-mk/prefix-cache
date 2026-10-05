# v2 浅层可辨识性分析器

新增位置：`src/probekv/source_selection_identifiability_audit_v2.py`。

用途：补齐任务书 P1-E 的同池 QA 空间分析和 P1-M/T22 的浅层签名碰撞诊断。它不执行模型、不读数据集、不选择在线Source、不拟合阈值、不改变池，不授予任何GPU/P1资格。

## 为什么补这个工具

只统计“d1、d2选择了不同Source”不能证明多版本有价值；同分QA的Source ID不同也不是错误。需要真实QA矩阵回答：

1. 每请求最优Source相对整组固定一个Source，究竟有多少F1/质量合格覆盖空间？
2. 实际浅层选择保留了多少空间？
3. 浅层状态相同、完整Artifact或QA却不同的情况是否存在？
4. 差异来自评分实现/证据绑定不一致，还是可观察信息本身不足？

## 接口与输入

```python
from probekv.source_selection_identifiability_audit_v2 import (
    analyze_source_selection_identifiability_v2,
)

# frozen_matrix 由将来的合法QA/观测事件转换器提供；不是人工补齐结果。
report = analyze_source_selection_identifiability_v2(frozen_matrix)
```

输入是单个 content group 的JSON兼容对象，`kind=source_selection_identifiability_input_v2`。

| 部分 | 必要信息 |
|---|---|
| evidence origin | 明示CPU合成测试或声明的模型测量；不能将前者重新贴真实GPU标签 |
| cohort kind | 历史候选矩阵或受控E/M来源配对；后者不产生自然多Source的Go结论 |
| frozen contract | 模型、真实code commit与工作树摘要、patch、execution/repair policy、first reuse layer、timing scope、exact content、comparison depths、原先冻结的F1降幅标准 |
| Source catalog | 1–4个历史Source的完整Artifact digest、G0/G1、出生请求、发布epoch、各深度K内容digest和原始证据引用 |
| request row | request/input身份、严格递增epoch、当时可见池、dense和每个可见Source的真实F1或显式null、每Source mask身份 |
| depth observation | 当前K、所有可见Source K的内容digest、分数、所选Source或abstain、request/contract绑定、观测自身canonical JSON摘要 |

准确字段由函数校验及测试fixture定义。测试fixture只作接口示例，不能送入真实证据Gate。独立S0强控制不属于这1–4个历史候选，应继续保留在原P1矩阵及单独报告中，不能因该API不覆盖它而删实验。

## 严格拒绝和缺项语义

- Source必须在请求之前发布，当前请求自己出生的Source不可见。
- outcome须匹配该request/input、Source Artifact、执行合同及登记mask；不能把别的请求QA移入本矩阵。
- `source_outcomes`、观测K和分数的候选集合须一致；不同池、depth或policy不能静默合并。
- 策略阈值由输入原合同提供，不在工具内搜索或默认拟合。
- 不同请求可见池变化时保留逐行诊断，但全局固定池headroom为 `UNSUPPORTED_FIXED_COHORT`。
- QA缺失时显式 `NOT_EVALUATED` 和 `null`，不把缺失值当零、不只挑完整成功行冒充全矩阵。
- G2/unknown不属于此主池分析接口；可选传播风险实验仍按原任务书另行登记。
- digest只校验内嵌JSON一致性和外部引用格式。**本工具不打开或认证外部证据文件**，`references_verified=false`；完整P1消费者仍需独立校验原始文件、签名、数值与数据资格。

## 输出与解释

- `summary`：QA oracle、post-hoc best-fixed、F1 headroom、safe coverage headroom。
- `selection_by_depth`：实际所选Source的QA regret、tie-aware oracle命中、abstain与合格覆盖。
- `candidate_shallow_collisions`：相同Source浅层K签名但完整Artifact或QA不同的候选。
- `cross_request_shallow_equivalence`：相同可观察浅层状态下，多个请求是否仍存在共同QA最优Source。
- `requests`：保留每个请求明细，避免仅以均值掩盖失败。

post-hoc best-fixed是描述性强参照，不是上线可知策略。QA regret不是深层Residual-K regret，不能换名混用。相同K却给出不同score需先审计数值/证据一致性，不解释为信息不足。碰撞也不是质量失败的充分条件：多个Source可能QA同分，仍存在共同最佳选择。

可选duration只保持原timing scope报告，本工具不算完整系统速度Oracle、不扣去选择/物化开销，不给 `positive_net_gain` 结论。按G区分来源的受控E/M配对，也不能自动等同自然历史互补。

所有结果维持 `paper_evidence=false`、`gpu_runtime_qualified=false`、`gpu_execution_allowed=false`、`P1_execution_allowed=false`。人工声明的测量输入也不改变这些标志。

## CPU验证与下一阶段

针对性单测检查算术与fail-closed行为，不含真实模型。全仓CPU回归的新目录为：

`artifacts/decoupled-v2/novelty-audit-20260920/validation-v1/`

应读取该目录的实际 `correctness_report.json`、`test_results.json` 和日志确定是否通过，本文不代填通过结果。旧服务器P0部署包保持原样；新文件必须经新工作树摘要重新移交，不能套旧包SHA。

接下来仍先完成受控P0 GPU数值验证和P1证据消费者；只有合法真实QA矩阵产生后，才能用本工具回答多Source空间和浅层可辨识性。本轮未生成这类真实结论。
