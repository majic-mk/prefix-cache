# Server07 P3 统一预算标定与长请求集成交付

本轮实际在 `connect.westd.seetacloud.com:38819` 的项目工作区执行。结论：既有 AutoDL + 显式 linux_aio 路线可以完成原始准入下的长前缀恢复与持续 decode，原始异步流水线确实存在设备重叠。**P3 尚未全部完成；没有证明新策略收益，joint 保持关闭。**

## 实际修改

- `acquire_native_aio_costs.py` 增加 paired-only 的 `--cached-reference-logprobs`，使用现成 SamplingParams 采集 top-5；标记为诊断，禁止混入成本拟合。
- 新增 `analyze_cached_references.py`：零容差比较原生 GPU 热命中与 SSD/staging 恢复的选中 token、top-5 token 集合和概率值，所有热身样本保留。
- 新增 `export_uniform_calibration_candidate.py`：按预先冻结的新参考协议校验两轮统一预算成本数据，调用作者原始 PCHIP、曲线解析器和 cost-table 构造函数，生成独立标定候选表。
- 新增 `long_calibration_contract.py` 与 `run_long_calibration_replay.py`：严格限定同预算、标定家族、原始准入、6 请求/每请求 128 token 的集成诊断，复用 vLLM LLMEngine.step，无第二套执行器。
- `run_p3_native_pilot.py` 的观测 helper 增加一个显式预算组合：2 GiB GPU KV / 1 GiB staging；原默认 256 MiB / 128 MiB 不变。
- 新增 `analyze_long_replay.py` 及拒绝/区间计数测试，核验真实 token 事件、GPU 活动区间、字节闭合和模型输出一致性。

作者 py-kvcache / vLLM 运行时代码未修改。P2 的 29 个锁定文件哈希全部相同；旧严格导出器哈希未变。准入算法、精确 Prefix 身份、共享 staging、预加载、复制合并和异步完成路径保留。没有启用 ordinary quota、fixed、pressure、dependency 或 joint。

版本仍为 py-kvcache `3abba7a502d553f6e7e2e58b92086487e3395d7e`、作者 vLLM `817a7e3124f817cd6e549581d3e5483207a753a4`。

## 数值比较协议及结果

先冻结 `cached-reference-plan.json`，再运行四个 GPU 诊断。统一 max_model_len / max_num_batched_tokens=16400、max_num_seqs=1、GPU KV 预算 2 GiB；外部路径 staging 预算 1 GiB，iodepth=4、共享 staging、LRU 和预加载开启。模型为锁定的 Qwen2.5-7B BF16。

2K / 4K / 8K / 16K 各三个前缀，SSD 和 staging 两种恢复，共 **24 个比较全部通过**：选中 token 与原生 GPU 热命中完全相等，top-5 token 集合及浮点概率值完全相等，容差为零；相同概率的 rank 顺序不作为数值差异。

12 个冷/热比较中，所有 top-5 概率都有差异；只有 4K rep0 的选中 token 不同，原始证据全部保留。旧导出器仍会拒绝该冷/热不等数据。新增候选表使用明确分开的协议：冷成本输出对应冻结的冷参考，恢复输出对应已严格验证的 GPU 热参考，**不再把独立冷重算逐 token 相等当作存储恢复是否正确的唯一判断**。这是单独版本的比较协议，不是对旧导出器悄悄调宽容差。

此结果只覆盖这些前缀和 top-5，不能宣称完整词表、所有输入或所有模型数值等价。同一生产 KV 字节一致性继续沿用此前独立 GPU 验证；本轮还检查了原已发布缓存文件未变。

证据：`artifacts/prefix_io_v1/server07-p3-03/cached-reference-result.json`，对应 run 的 frozen-config、result 和原始 trace。

## 统一预算成本候选

先冻结 `uniform-cost-plan.json`，关闭概率诊断后执行八个新 GPU 进程。每个长度、每条路径有两次独立引擎启动、共四个计入统计的样本；热身保留用于正确性检查。这里的独立性是进程启动，内容仍是两个相同的测量家族，不能把它们当作四个独立内容样本。

| Prefix | 冷重算中位数 | SSD 恢复中位数 | staging 恢复中位数 |
|---|---:|---:|---:|
| 2048 | 144.458 ms | 157.048 ms | 51.379 ms |
| 4096 | 301.437 ms | 299.174 ms | 71.705 ms |
| 8192 | 676.561 ms | 563.457 ms | 89.104 ms |
| 16384 | 1671.211 ms | 1185.686 ms | 171.682 ms |

全部 **72 行**（含热身）通过各自路径参考及 I/O 校验。SSD 波动仍明显，例如 16K 四次为 1.342103、1.189243、1.182129、1.168216 秒；未删慢样本。4K 两种路径接近，不能用中位数的微小差距宣称可靠收益。

候选曲线 SHA256：
`892a979b1b1136409e258537f85e9355f6c7bb88e5de1d3f636bf23caf119c2f`。

作者解析器与表构造校验通过。PCHIP 候选给出 SSD 阈值 2272、memory 阈值 2048；**2272 是插值候选，不是独立实测或正式可靠边界**。支持的诊断 Prefix 域为 2048–16384，禁止将小于首个测点的结果作为已有测量。候选未替换 P1 表，只被下述同预算标定集成驱动显式使用。独立内容和预测误差验证仍未完成。

## 长前缀持续 decode 与原始准入

冻结六条 16257-token prompt，三个既有标定 Prefix 家族各出现两次，每条生成 128 token，ignore_eos=true。到达来自作者调度生成器；这属于标定家族集成诊断，**不是独立 development / evaluation 数据**。

从已发布 SSD 语料开始，新引擎 GPU / staging 为空；热身按协议排空，服务流内没有 reset 或人工 I/O 延迟。私有 storage 通过 hardlink 复用原始文件，新增 suffix 文件仅在私有目录产生。前后逐文件 SHA256 校验 3072 个原文件，全都未变；没有复制一套额外的大缓存，也没有删除数据。

运行两次：普通原生 profiler 和额外 CUDA profiler。每次：
- 6 请求完成，实际 **768 个 engine token 时间事件**，每请求命中 16256 个 Prefix token；
- SSD 实读 **4,777,443,328 字节**，实写 **19,267,584 字节**；
- 实际 GPU KV storage **2,146,959,360 字节**，2340 blocks、28 层、1 KV group；
- 实际 pinned staging **1,073,483,775 字节**，均不超过冻结预算；
- 已接受 I/O 全部结算，尾部 drain 约 2.3 ms；
- pending flush wait 为 0，foreground slot unavailable 为 0。

两次对应请求的全部输出 token 完全一致。普通运行 cohort 含 drain 为 19.616 s；CUDA profiler 运行 22.148 s，后者只用于机制检查，不用于性能对照、观察器开销或 goodput 结论。

普通运行观测到原始 planner 的 plan=686、admit=327、defer=179 等事件；这些是重复规划事件数，**不是 327 次独立请求或传输**。说明真实原始准入路径被调用，未用伪造 admission 替代。

## CUDA 重叠证据与限制

CUDA trace 包含 251136 个 GPU kernel 活动、86784 个模型 GEMV 等线性计算活动。根据实际批量合并记录，筛选出 1637 个缓存复制活动；H2D 和 D2H 总字节与原生传输记录分别精确闭合为 4,777,443,328 和 19,267,584。

设备时间区间中，复制与 GPU kernel 的交集约 **1.016 ms**，与模型线性计算的交集约 **0.971 ms**；复制区间并集约 87.202 ms。H2D 在 stream 89，模型线性计算在 stream 7。这是实际 CUPTI 活动区间的非零交集，不是用 non_blocking 标志推测。

重叠占比很小，不能据此宣称强干扰、有效隐藏全部传输或新策略加速。原先按单层 32768-byte 页面筛选会漏掉批量记录；初版分析保留，已以全 block 聚合字节及精确总量闭合修正。最终使用 `long-replay-summary-verified.json`。

原始 CUDA trace 为 659,150,634 字节，SHA256：
`d1da3bd06b68921f461a96b890ed66a2f0615eba6b6576e9ab2ffa6955c6728d`。
其真实体积高于最初预留的 512 MiB，但实际磁盘仍在 8 GiB 保护线以上。已将后续 CUDA profiler 运行预留提高至 1 GiB；非 CUDA-profiler 运行预留 128 MiB。此最后的预检调整没有重新触发 GPU 作业，执行时源代码保存在每个 run 的 driver-source.py。

## 命令、测试与实际 GPU 预算

完整 argv、退出码：`artifacts/prefix_io_v1/server07-p3-03/commands.json`。复现说明：`REPRODUCE_SERVER07_P3_UNIFORM.md`。

最终 CPU 命令：
```text
.venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py -q --import-mode=importlib tests/prefix_io_v1_pilot tests/prefix_io_v1_native_costs --junitxml artifacts/prefix_io_v1/server07-p3-03/final-cpu-verified.xml
```

结果 **69 passed，0 failed，0 skipped**。本轮较前次增加 32 项；过程中的 48/58/63/68 均是中间重复测试，不相加。CPU 测试均在 GPU 计时作业结束后执行。P2 整套测试未重跑，29 文件未变另有哈希证据。

真实 GPU **14 次作业全部退出 0**：4 次概率参考、8 次纯成本、2 次长请求。均经现有预算包装器、授权 UUID 和累计账本运行；新 GPU 用时 **1084.473655 秒（18.07 分钟）**。累计 **4260.801090 秒（71.01 分钟 / 8 小时）**，剩余约 **6.816 小时**。本轮模型下载 0，累计仍为 19,422,798,722 字节。

收尾检查：GPU 0 MiB / 0%，无计算进程，所有作业 session 排空，active_reservation=null，/dev/shm semaphore=0。打包前磁盘空闲 9,097,043,968 字节。保留 resource_tracker 等原日志告警；无驱动/系统修改、无租机付款、无远端推送、无数据删除。

## 下一允许阶段与停止条件

仍在 P3：需要独立内容的数值/成本验证、独立持续 decode 开发负载，以及 fixed/pressure 薄桥接和强简单基线调优。当前数据不能跨分区冒充评估集，当前曲线不能直接作为正式生产标定。

短负载和本轮长标定流均未观测到目标释放等待，CUDA 复制与计算重叠也很少。这是必须保留的负结果；若独立正常负载及合理调优后仍没有可重复目标问题，应依 P3 停止条件结束研究策略投入。不得人工延迟 I/O 来制造收益，尚不能推进 joint 或宣称 P4–P7 完成。

新鲜长语料还受磁盘余量限制；后续必须先做明确的写入量与证据空间核算，不能绕过保护线或擅自清理既有数据。
