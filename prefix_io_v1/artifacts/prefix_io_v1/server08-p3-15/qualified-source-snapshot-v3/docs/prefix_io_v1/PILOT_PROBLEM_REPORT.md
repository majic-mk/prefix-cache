# P3 开发负载与长前缀成本筛查（Server07，2026-09-29）

本轮确认当前 AutoDL 的原生模型 + 可选 Linux AIO 缓存路径可持续运行；16K 前缀出现值得进一步验证的 SSD 复用区间。**P3 尚未完成，新增调度策略收益仍未验证。** fixed/pressure、interference、dependency_only、joint 均未接入生产路径。

## 实际实现

- 新增 token_timeline.py：记录原生引擎输出事件对应的 token 时刻，拒绝时间倒退/输出改写；一个事件含多个新增 token 时标记缺失逐 token 证据，不编造时间。
- 新增 prepare_p3_pilot.py：复用作者 common.prefix_cache_common.generate_request_schedule，冻结到达时间、复用关系、精确输入 token 与停止规则。
- 新增 run_p3_native_pilot.py：使用现有 LLM/LLMEngine.step，不替换模型执行器。旁路统计原生 jobs_to_flush 的实际等待、槽位申请失败、AIO 数据读写字节及有界快照；保留原调度、准入、预加载、复制合并与流水线。
- 新增 analyze_p3_pilot.py、CPU 测试及三份开发 manifest。
- 扩展 acquire_native_aio_costs.py 的显式 --domain：1024/4096/8192/16384，默认仍为1024。较大范围仅允许成本采集，禁止直接进入 planned 模式；增加 8GiB 磁盘余量检查和原生 GPU 冷/热诊断选项。未改运行时成本表。
- P2 source-lock 中 29 个文件逐个 SHA256 核对，全部保持原样。本轮没有编辑作者 vLLM/py-kvcache 执行器。

## 持续 decode 与资源等待

固定 Qwen2.5-7B BF16；GPU KV 配置256MiB、实际267911168字节；staging配置128MiB、实际133959679字节；单KV组、4并发、iodepth=4。每条请求固定生成128 token、ignore_eos=true，明确属于受控合成 token 开发流，不是真实问答质量或最终评估集。

| 运行 | 请求数 | cohort 秒数（含末尾排空） | 完成请求/秒 | 请求内ITL P95的中位数 ms | cohort 外部写入字节 |
|---|---:|---:|---:|---:|---:|
| mixed-02 | 24 | 13.0979 | 1.8324 | 24.3923 | 742260736 |
| mixed-03（补齐槽位计数后重跑） | 24 | 12.5526 | 1.9120 | 34.9892 | 745013248 |
| lowreuse-01 | 24 | 14.2759 | 1.6812 | 26.1097 | 1211105280 |
| lowcontention-01 | 12 | 30.9246 | 0.3880 | 17.5016 | 198180864 |

合计84条成功请求、10752个真实逐 token 输出时刻。所有样本均保留，mixed-03的ITL较差也没有删除。预先规定的热身单列，测量中的JIT尖峰保留。

- 四轮 cohort 的实际 SSD 读取均为0，原生 pending jobs_to_flush 等待均为0。原LoadPlanner真实执行：混合流分别有255/256个break_even拒绝事件，另有missing拒绝；没有改准入来强制读取。
- mixed-02 尚未安装槽位失败计数，记为null。mixed-03、lowreuse、lowcontention 的cohort槽位申请分别812/1320/216次，失败均为0。
- staging free_slots最小值都达到0，但原生可回收缓存机制仍然满足前台申请，不能把这个“0”当作写回依赖阻塞。
- 实际发送相对计划到达最大迟到分别255.93/41.19/36.95/12.29ms，已记录client_queue；不是无误差的open-loop到达。
- 末尾排空约2.57–3.47ms；所有已接受AIO均回收，engine shutdown完成。
- 这些是引擎输出事件的token时间，非客户端网络SSE时间；未冻结SLO，因此只报告完成吞吐，不报告SLO goodput。P1单请求曲线在4并发下的预测准确性尚未重新标定。

## 2K–16K 单token成本筛查

完全相同的 N+1 输入、一个输出 token；每点3次，其中第0次是预定热身，剩余2次测量。f为冷计算TTFT，g_ssd为真实SSD恢复，g_mem为同一请求紧接SSD恢复后的staging恢复。此处GPU cache reset仅用于明确的成本诊断，持续请求流没有逐请求reset。

| 可复用前缀 | 冷计算 f 中位数 ms | SSD恢复 g_ssd ms | staging恢复 g_mem ms |
|---|---:|---:|---:|
| 2048 | 141.777 | 179.851 | 53.388 |
| 4096 | 301.030 | 304.850 | 43.309 |
| 8192 | 676.885 | 691.672 | 114.613 |
| 16384 | 1671.169 | 1282.912 | 179.024 |

16K 两个测量样本均有SSD复用优势；中位TTFT减少约23.23%（约388ms）。这是**原有缓存路径的初步复用优势**，不是新调度策略收益，也不能外推为持续decode或整条trace加速。

配对阶段共12次SSD恢复和12次staging恢复；SSD实际读字节覆盖合计5284823040，staging路径无SSD读取，均有原生成功load/H2D事件。另有populate阶段的staging验证。此处证明真实路径及字节覆盖，不把它冒充新的全量KV逐bit比较。

预算与上下文分别冻结，不能跨组拼接曲线：
- 2K/4K：max_model_len=4112，GPU KV配置1GiB，staging512MiB（实际536743935字节）。
- 8K：max_model_len=8208，其余预算同上。
- 16K：max_model_len=16400，GPU KV配置2GiB，staging1GiB（实际1073483775字节）。
- staging均为pinned且不超预算。成本筛查没有新增逐tensor GPU storage清点，GPU KV值在此列作配置预算；短请求流的实际GPU storage已读取核验。
- 不同上下文/预算及仅2个测量样本，不满足新的完整运行时成本表发布条件。没有覆盖/替换P1曲线，没有外推未测成本。

## 4K 输出差异与独立对照

4096、rep=0 的原始冷计算和store输出为token1095；staging、SSD及配对staging输出为70。该热身样本保留，不能因不计性能均值就忽略正确性差异。

随后在外部connector完全关闭（kv_transfer_config=null）的原生GPU路径重复同一输入：
- 冷计算输出1095；
- 原生GPU Prefix热命中4096 token，输出70。
- 其余5个2K/4K对照pair输出一致。

这证明相同差异可以在不经过外部I/O时出现；不能归因于新增AIO路径。具体数值机制尚未确认，也没有据此宣称任意差异都可接受。

未改动严格导出检查。实际调用原exporter时：
- 2K/4K数据被 token mismatch 拒绝；
- 16K数据被 invalid sample（测量次数不足）拒绝。
两项均按预期失败，目标曲线目录没有创建。native-gpu-control.json保存独立对照证据。下一次正式数值验证需要控制计算形状、记录logits/裕量并预先约定比较口径，不通过删除样本或事后放宽阈值获得通过。

## 测试、失败与资源收尾

最新CPU验证 **24 passed，0 skipped**，覆盖真实事件时间口径、分块事件不得伪造ITL、扩展域不得误启用planned、样本/范围限制，以及原成本导出测试。此前CPU运行是其子集或同套重复，不累加为新增测试数。

真实GPU共15轮，14轮按各自验证合同完成，1轮失败：
- mixed-01 的调用方ID/内部ID映射错误触发KeyError；修复实验驱动后重跑成功。未修改缓存执行器；失败现场、源版本和预算均保留。
- 原生GPU冷/热诊断成功指路径和记录完成，不表示其中所有pair输出相同。
- 部分模型子进程退出时有Python resource_tracker semaphore回收警告，日志保留。最终/dev/shm/sem.*计数0，GPU计算进程为空，显存0MiB、利用率0%，budget active_reservation=null。

本轮GPU累计1065.5845秒（17.76分钟）；全项目累积2664.5266秒（44.41分钟）/8小时，剩余约7.25985小时。新增模型下载0字节，累积19422798722字节。无驱动/系统修改、租机、推送或共享数据删除。结束时项目盘可用约9.94GB（十进制），后续较大采集须继续满足8GiB余量检查。

## 版本、命令与证据

服务器根目录：/root/autodl-tmp/prefix-io-v1-handoff/project，SSH端口38819。
GPU UUID：GPU-f8744916-1693-fa6a-93b6-7f503c03459c。
作者锁：py-kvcache 3abba7a502d553f6e7e2e58b92086487e3395d7e；vLLM 817a7e3124f817cd6e549581d3e5483207a753a4；kvcache-experiments 0e023a84a21246b9bbc06266fa8070397eccbdc9。
使用P2隔离worktree py-kvcache-p2-aio，原始io_uring仍为默认，可选linux_aio仅显式选择。

所有实际命令/退出码/stdout/stderr在 artifacts/prefix_io_v1/server07-p3-01；GPU完整原始结果在 experiments/prefix_io_v1/runs/server07-p3-*；精确输入在 manifests/p3-development-01；脚本源快照、源哈希及commands.txt随包交付。

核心命令入口：
- .venv/bin/python experiments/prefix_io_v1/scripts/prepare_p3_pilot.py
- .venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py -q --import-mode=importlib tests/prefix_io_v1_pilot tests/prefix_io_v1_native_costs
- .venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py --label <唯一标签> --seconds 300 -- .venv/bin/python experiments/prefix_io_v1/scripts/run_p3_native_pilot.py <完整冻结参数>
- 同一GPU wrapper调用 acquire_native_aio_costs.py，分别cold/populate/paired、--domain 4096/8192/16384；native-hot对照只允许cold。
- .venv/bin/python experiments/prefix_io_v1/scripts/analyze_p3_pilot.py --output artifacts/prefix_io_v1/server07-p3-01/qualification-summary.json

省略参数的入口不是完整复现命令；使用commands.txt和launch JSON，改为新的唯一label/output，不覆盖原始证据，不绕过GPU wrapper。

## 下一允许阶段与停止边界

当前仍处于P3。现有短请求流没有证明释放依赖调度所针对的阻塞；16K筛查说明长前缀原缓存复用值得进一步研究，尚不足以启用joint。

下一允许工作：处理原生冷/热数值比较口径；以相同预算、充分重复和独立验证建立长前缀成本表；再冻结真实读写并发的持续decode开发流，补充干扰标定与fixed/pressure强基线。任何生产额度接线仍须验证该具体边界的mandatory和下游续接；真实owner/freshness控制器桥接及阻塞witness仍是dependency/joint前提。

P4–P7未完成。不能仅凭16K原缓存的优势宣称新策略有效；若正常长前缀负载仍没有目标问题，应缩小或停止研究策略扩展，如实交付负结果。


## 2026-09-29 P3 复测增量

详见 [SERVER07_P3_RECHECK_REPORT.md](SERVER07_P3_RECHECK_REPORT.md)。新增仅限采集诊断、离线校验及 CPU 测试；29 个 P2 锁定文件未变。4K 原生冷/热候选概率排序/并列值变化已观察；16K 三组复测均观察到原有 SSD 恢复获益，但存在明显延迟波动。未导出长域运行时曲线，P3 尚未完成。


## P3 统一预算与长请求集成增量

详见 [SERVER07_P3_UNIFORM_REPORT.md](SERVER07_P3_UNIFORM_REPORT.md)。24 个缓存参考比较零容差通过；生成单独的标定候选表并完成两次原始准入下的长请求集成诊断，每次 6 请求/768 token。实际 CUDA 复制与计算有约 1.016 ms 重叠，pending flush 等待仍为零。候选仅用于标定集成，独立数据与强简单基线未完成；P3 继续，joint 关闭。


## Server07 P3 两并发与 I/O 深度验证（2026-09-29）

本轮观察到跨配置相似的小 store 等待形态及 I/O 与 token 间隔的时间相关性，但单配置重复性不足、并发混杂因素未排除。简单 iodepth=8 已出现描述性改善，后续新策略必须与更强候选基线对照。60/60 连续输出正确；深度 2 的成本失败完整保留。当前证据不支持宣称新增研究收益或直接开启 P4。 详见 [本轮报告](SERVER07_P3_CONCURRENT_REPORT.md)。


## Server07 P3 mixed recurrence / 2026-09-30

See SERVER07_P3_MIXED_RECURRENCE_REPORT.md and server07-p3-10/qualified-analysis.json. Frozen depth8 repeats both matched a restore-destination fence (0.401557/0.808202 s), involving four protected single-file stores. This is destination reuse protection, not additional free GPU bytes. Depth4 repeats had no such wait. Seven real model jobs passed 70 full-output references. The mixed start-budget screen was slower than off: 43.0639/35.8550 vs 28.4126 s (one fixed/pressure/off run each). No strategy speedup, full P3 completion or P4 activation is claimed. Next: full stage budget contract and ready-state cause diagnostics.


## P311 增量记录

P311: 新增一次 depth8 被动混合复现，10 请求/1280 token 全部精确匹配。首次相关 D2H 在 wait 后 0.598843761 s。支持检验 mandatory 优先的强简单基线，尚无策略收益；下一步仍 P3，不进入 P4。见 SERVER07_P3_DISPATCH_READINESS_REPORT.md。


## P312 更新（2026-09-30）

P312：固定 off/pressure/pressure/off 四轮 cohort=32.704260/27.200766/29.057094/25.223704 秒；全部 40 完整输出精确。两个 off 无目标等待；首个 on 未排序，第二个 on 重排两次并等待 32.171 ms。平均差 2.883% 不能作为稳定提速证据。计划不变但实际 SSD read 与时序变化。保持 P3 gate，下一步先定义可复现的目标等待资格和停止条件，不能挑样本。

详见 [本轮报告](SERVER07_P3_MANDATORY_ORDER_REPORT.md)。
