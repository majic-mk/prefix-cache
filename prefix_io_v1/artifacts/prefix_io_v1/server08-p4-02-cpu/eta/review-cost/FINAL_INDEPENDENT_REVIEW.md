最终独立定向审查通过：p4_paired_measurement_verifier.py 860876ae2d6c6a0bceee52b07c1681f5c0a11b71822f58b775e5e638406f8b3f；p4_verified_cost_loader.py 23fddf7c159ff2bdcb98c000ad18f90c804183f0a5af9bf08be24ac3b54d368d。此文件更新原 INDEPENDENT_COST_AND_RECORDER_REVIEW.md 中的“待修复”状态；原反例/旧源/失败证据仍保留。

实际复用了 counterexamples-01 下六组原始 fixture，前后文件字节均未改变，不重新合成更容易通过的数据。相同 workload 内容改标签、两 cell 全局 calibration/validation 混用、active_decode 超 actual execution batch 三个漏洞现全部拒绝。raw source 与 GPU identity 漂移也拒绝。合法一致的 native_gpu_recording 标签只通过 CPU 语义重算，gpu_verified/production_qualified=false，production lookup=None。

公开 wrapper 的四个负例全部拒绝：原 None+FakeLookup、真实 semantic result+FakeLookup、真实 CostTable 错候选 source SHA、真实 CostTable 空 cells。PreparedCostTable 正式类型/source/cells 检查和 production lookup 直接返回 None 消除了原独审边界问题。证据为 final-counterexample-replay-03/result.json 与同名 executed.py。replay-02 审查脚本把原 CostTable 误当 dataclass，失败收据保留；03 改用其真实构造器。

本次只执行原反例和 wrapper 负例，没有机械重复 75 单元测试。CPU stdlib import guard 记录后端导入尝试为零，GPU 操作为零，ledger 和两份冻结源前后 SHA 不变。

raw-to-analysis builder 是已批准的 CPU 数据整理适配层，复用同一计算段，绑定四份实际 raw refs、冻结 plan、几何和 append-only analysis；它不会给 origin 文案真实性背书，不会补造缺失 GPU observations。validation 参与 empirical residual margin，严格称 margin validation 校验集，不能称独立效果评估/置信区间/泛化覆盖。

原 driver 仍不能直接产完整新 paired schema：已有 TTFT acquisition 使用 max_tokens=1；已有 concurrency wrapper 的 outstanding 请求数与 host engine.step 时段不证明实际执行窗口 active_decode、batch、context、既有 I/O、窗口新增接受 op ID/physical bytes。新 scheduler-work-v1 load signature 也不等于成本表的 exact9 actual execution signature。后续需要真实 native 执行窗口记录器及资格链；禁止将 cohort 汇总分摊为 window 字节、missing 填零、用 P3 条件收据宣称 P4 生产效果。生产能力未取得。
