# P3 已完成：系统路径可运行，研究效果未证明

**状态：P3_LIMITED_PILOT_COMPLETE_EFFECT_UNPROVEN；P3阶段已关闭，剩余P3 GPU实验为0。** 严格验收15项全部PASS、0项未满足，原始analyzer保持P3_EVIDENCE_COMPLETE_FOR_ROOT_REVIEW/p3_phase_closed=false，其任务只核验证据；本root裁决完成阶段关闭，没有重写analyzer或放宽门槛。

37次成功GPU stage、1次模型启动前路径错误失败，共38次尝试；其中20次完整模型对照，每次1280输出token精确一致。失败13.852598秒已计费保留。CPU1613去重passed、16 skipped，50 subtests单列，12个历史fixture适用范围保留。GPU累计16520.562887秒（4.589045小时），8小时上限；最终GPU空闲、38个session均消失，3048份注册源逐字节匹配。

|候选|两次cohort中位数/秒|相对原路径U时长|
|---|---:|---:|
|U|32.566494|+0.0000%|
|F16|34.856058|+7.0304%|
|P4|36.410556|+11.8037%|
|F8|33.372540|+2.4751%|
|P512|34.947063|+7.3099%|

**当前原路径U最好，B=U；新策略的有效提升未获证明。** 这不等于证明所有可能策略无效，也不授权扩大范围。每臂两次、已见开发域，只支持描述性结果，不提供置信区间、未见集/SLO或全局最优结论。容量960/l1和1024/l2独立报告；GPU-hot U/B/B/U实际为四次U。quiet观测+1.23%不能推广为正常I/O总开销≤2%。条件干扰表与host事件可见性资格不构成production lookup或真实DMA overlap资格。

真实普通pending flush目标在两次U中仍存在（0.408055/0.641355秒）。简单基线没有将此目标与整体速度同时改善到形成研究优势；不能据此开启提前GPU回收，unknown和gpu_release_credit=false保持保守。

本轮新增只有限定缓存合并、容量资格路径计划修正、剩余实际GPU对照、冻结锁及验收/交付文档。既有共同reactor/staging安全修复与fixed/pressure研究控制分开；关闭研究控制回到原stage签发，原精确Prefix、成本准入、共享staging、预加载、复制合并、异步流水线均保留。作者代码/配套vLLM路线未变，没有驱动/系统更改、新模型下载或预算扩展。

实际命令见p3-reproduction-final.json与*-command.json；严格验收使用analyze_p3_closeout_p316_v3.py，最新schema资格30passed及50subtests未重复加总。版本锁、能力矩阵和生命周期/修改清单已保留。完整细节见PILOT_PROBLEM_REPORT-final-evidence.md，严格证据见p3-closeout-analysis-final.json，阶段裁决见p3-final-root-review.json。

下一允许动作是P3证据归档与研究决策。可以依据真实剩余等待目标提出P4最小依赖调度CPU设计，但进入新阶段须单独授权和依计划设门槛；本次P4-P7保持关闭，没有自动GPU/生产策略激活。
