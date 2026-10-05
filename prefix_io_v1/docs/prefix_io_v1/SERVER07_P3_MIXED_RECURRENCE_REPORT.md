# Server07 P3：混合等待复现与额度集成实测

本轮在 AutoDL server07 完成 7 次真实 RTX 5090 / Qwen2.5-7B BF16 运行。环境能够运行作者 py-kvcache + 配套 vLLM 的原生异步路径；Linux AIO 路线继续有效。没有修改驱动、系统或模型执行器，没有新增模型下载。

本轮首次在同一冻结配置的两次重复中，将真实 flush 等待唯一匹配到具体 GPU 目标块复用依赖。简单固定/压力“文件链启动额度”在混合负载中均未带来本次观测到的加速。**完整 P3 仍未完成，P4–P7 仍未启用，研究方法的有效提升尚未证明。**

## 实际执行与边界

服务器项目：`/root/autodl-tmp/prefix-io-v1-handoff/project`。
当前证据：`artifacts/prefix_io_v1/server07-p3-10/`。
原始模型结果、逐 token 时刻、native trace 和缓存：`/root/prefix-io-v1-validation/runs/server07-p3-10-*/details/`。

先完成上一轮冻结清单的 10,457 个私有缓存副本合并，实际回收冗余分配 9,594,339,328 B。四次复测之后，又取得用户本轮明确授权，对另一份仅包含这四个完成运行的清单合并 8,226 个副本，回收 7,547,387,904 B。两份清单、授权、逐文件事务日志及最终 SHA-256/同 inode 核验均保留。18,683 个目标路径和字节内容全部保留；未替换已有共享目标或模型文件；后续将这些归档视为不可就地修改的共享内容。

第二份审计完整扫描曾发现另 6 个旧运行私有副本；它们被排除在申请和执行范围之外。实际授权清单 SHA-256：`41e197dce9235c0776e2e1ac72f694a8839ccebcec1d462b0cad10d5109f259f`。没有将一次授权扩大为今后任意缓存整理授权。

为避免主盘接近 8 GiB 下限时继续写大日志，只调整归档工具的审计/日志路径验证：允许项目证据目录或已授权辅助盘的 `audits/`，拒绝重定向和旧日志；日志预留按实际落盘位置检查，仍需准确清单哈希和明确授权。没有扩大 20 GiB 上限或降低 8 GiB 可用空间下限。

## 第一组：冻结 ABBA 复测

沿用 P306 已合格的 depth 4/8 配置及参考输出，固定顺序 8、4、4、8。每组相同被动 flush 原因观测；没有普通额度、人工延迟、缓存重置或研究策略。每轮 10 请求，16,257 输入 token，128 输出 token，最多两序列；2 GiB 配置 GPU KV / 1 GiB staging。实际 GPU KV 2,146,959,360 B，实际 staging 1,073,483,775 B。

| 运行尾名 | I/O 深度 | 含排空 cohort 秒 | 待完成 flush 次数 | 等待秒 | 占 cohort |
|---|---:|---:|---:|---:|---:|
| mixed-d8-01 | 8 | 27.4782 | 1 | 0.401557 | 1.4614% |
| mixed-d4-02 | 4 | 28.1037 | 0 | 0 | 0 |
| mixed-d4-03 | 4 | 27.8421 | 0 | 0 | 0 |
| mixed-d8-04 | 8 | 26.4143 | 1 | 0.808202 | 3.0597% |

40/40 请求的全部 5,120 输出 token 与冻结 cold/GPU-hot 参考一致。深度 8 两次均出现唯一匹配的 `restore_destination` 原因，达到本次事先冻结的开发复现条件；这是两次开发重复，不能外推普遍发生率。深度 4 两次未出现该等待。两种深度 cohort 中位数为 27.9729 与 26.9462 秒，仅作描述，不能据此给出稳定性能排序或置信区间。

每次等待均涉及 parent 22/24/26/28，共 4 个 917,504 B 的单文件 store（合计 3,670,016 B）。GPU block 1313/311/309/307 已有新的分配代际和 active ref=1，仍受旧 store parent 保护；完成回执之后保护集合清除，但块仍由新请求使用，未进入 free queue。这证明的是**目标块覆盖前的原生保护依赖**，并非新增可用 GPU 容量。可立即复用字节仍标为 unknown，不能把 store 字节当释放收益，也不能提前 complete_store。

首个 D2H 的 host enqueue 分别在等待开始后 0.378286 / 0.772267 秒，约占等待 94.2% / 95.6%。从首个 enqueue 至 wait 返回仅余约 23.3 / 35.9 ms。单文件 SSD write host interval 约 10.8–22.7 ms。**尚无 ready-state / 未签发原因的完整时间序列，不能断言前段延迟全部来自队列优先级。** CUDA trace 的 ts 是 host enqueue 时间，dur 是 CUDA Event elapsed；两者不能合成精确 GPU 执行起止时刻或设备队列顺序。

原生代码核对：scheduler 在新分配的恢复目标块仍关联旧 store 时加入 jobs_to_flush；worker 等待原 parent Future；reactor 保留 D2H→SSD write→完成链；scheduler 收到完整完成回执后才调用 complete_store 并清理保护。原有预加载、共享 staging、复制合并、成本准入和异步流水线均保留。

## 第二组：相同混合负载的启动额度集成

沿用 P309 已通过 13 项真实 CUDA/AIO 原语检查的配置，不调整参数：
fixed = 10 ms 每 epoch 4 个文件链 start，最大普通等待 100 ms，预留 0 slot；
pressure = 同上，预留 1 slot。off 无新增预算/记账对象。三组均来自同一 P3 隔离工作树；使用 depth 8 冻结请求和初始源缓存。
此处只有文件链启动额度与被动四阶段记账，**尚非完整四阶段累计字节 / in-flight 限额，也非 dependency_only/interference/joint**。

| 模式 | cohort 秒 | 相对 off 耗时变化 | TTFT P95 秒 | 请求内 ITL P95 的中位数 ms | flush 等待秒 |
|---|---:|---:|---:|---:|---:|
| off | 28.4126 | — | 19.5518 | 41.8112 | 0.732476 |
| fixed | 43.0639 | +51.5659% | 33.1934 | 113.1016 | 0 |
| pressure | 35.8550 | +26.1938% | 27.1291 | 114.6343 | 3.522881 |

30/30 请求、3,840 输出 token 均与冻结参考及 off 一致。所有启用额度的阶段记账有效且排空，未完成记录为 0，失败操作为 0；off 无额度/记账状态。每组 native AIO accepted/completed/reaped 一致，pending/ready/unreaped/outstanding 为 0；每个模型引擎已正常关闭，外层进程 session 全部排空。

压力组实际触发 1,657 次 mandatory 文件链启动和 4 次续接豁免，仍正常完成与排空；fixed/pressure 均无 fallback。拒绝计数是重复 pump 判定次数，不是被丢弃请求。四阶段记账包含 warmup，因此不能直接与 cohort I/O 差分混用；通用对象中的 gpu_qualified=false 不是运行失败，真实资格由冻结源码哈希对应的外部 GPU 回执提供。

固定额度没有 flush 等待却整体更慢，说明不能将减少 flush 数量代替端到端效果。压力额度在本次既更慢又有更多等待。这里只各一次且固定运行顺序，未隔离新增被动记账开销，缓存命中及预加载时序也会自然改变；不能把差值认作纯限流因果效应、稳定退化百分比或正式 goodput。没有放宽数值判定，没有丢弃慢请求，没有追加运行挑选好结果。

## 代码改动、CPU 检查

新增：
- `experiments/prefix_io_v1/scripts/analyze_blocked_store_chains.py`：有界单文件原生 store fence 分析；按 parent/request/process 匹配，缺失、截断、方向/字节/代际不符均不认定。
- `experiments/prefix_io_v1/scripts/analyze_mixed_start_budget.py`：混合集成与参考输出、SSD 源保全、AIO 平衡检查。
- `experiments/prefix_io_v1/scripts/verify_private_cache_archive.py`：两份已批准归档的只读内容与路径核验。
- `tests/prefix_io_v1_pilot/test_blocked_store_chains.py`、`test_mixed_start_references.py`。
- 两个准确清单绑定的授权记录及本轮冻结计划、运行/分析回执。

修改仅有归档工具及其路径测试。GPU 使用的缓存/模型/策略源码本轮未变。共同兼容修复、P2 观测、P3 隔离额度仍按既有工作树和 patch 分离。最终定向 CPU 检查 **51 passed，0 failed，0 skipped**（其中本轮新增 30 个用例；初次 36 pass 是其子集，不重复累计）。未重跑无关完整回归；上一轮全矩阵保留为历史证据。独立 runtime 身份读取确认 CUDA 未初始化，真实 GPU 运行单独由预算包装器记账。

## 实际命令与证据

所有命令在服务器项目根执行。完整 argv、环境和 stdout/stderr 见本轮 `*-command.json`、两份 `*-run-plan.json` 与 `gpu-job-ledger-slice.json`，包含 7 次模型命令的全部模型、清单、校准和授权路径。

```bash
.venv/bin/python experiments/prefix_io_v1/scripts/consolidate_private_cache_copies.py --audit artifacts/prefix_io_v1/server07-p3-09/private-cache-duplicate-audit.json --authorization experiments/prefix_io_v1/configs/authorizations/server07_p309_archive_dedup.json --journal artifacts/prefix_io_v1/server07-p3-10/dedup-approved-journal.jsonl --apply

.venv/bin/python experiments/prefix_io_v1/scripts/consolidate_private_cache_copies.py --audit /root/prefix-io-v1-validation/audits/server07-p3-10-four-runs-audit.json --authorization experiments/prefix_io_v1/configs/authorizations/server07_p310_archive_dedup.json --journal /root/prefix-io-v1-validation/audits/server07-p3-10-next-dedup-journal.jsonl --apply

.venv/bin/python experiments/prefix_io_v1/scripts/analyze_blocked_store_chains.py --plan artifacts/prefix_io_v1/server07-p3-10/run-plan.json --out artifacts/prefix_io_v1/server07-p3-10/qualified-analysis.json
.venv/bin/python experiments/prefix_io_v1/scripts/analyze_mixed_start_budget.py --plan artifacts/prefix_io_v1/server07-p3-10/start-mixed-run-plan.json --completed 3 --out artifacts/prefix_io_v1/server07-p3-10/start-mixed-analysis-after-3.json

.venv/bin/python -m pytest -q tests/prefix_io_v1_pilot/test_blocked_store_chains.py tests/prefix_io_v1_start_budget/test_archive_consolidation.py tests/prefix_io_v1_pilot/test_flush_cause_matching.py tests/prefix_io_v1_start_budget/test_start_model_analysis.py tests/prefix_io_v1_pilot/test_mixed_start_references.py -p no:cacheprovider --basetemp=/root/prefix-io-v1-validation/cpu-evidence/server07-p3-10-final-unit --junitxml=artifacts/prefix_io_v1/server07-p3-10/cpu-final-targeted.xml
```

上述 apply 命令是历史执行记录，**不可原样重跑已完成清单或覆盖旧 journal**。只读分析输出也以 exclusive create 保留旧结果，复现时需新输出路径。

GPU UUID `GPU-f8744916-1693-fa6a-93b6-7f503c03459c`；驱动 580.105.08。本轮 GPU 包装器计时 720.364856 秒，含模型装载等作业时间；累计 9,588.929878 秒，即 2.663592 小时，8 小时授权尚余 5.336408 小时。7 次均 exit 0；最终无活动预约，无 GPU 计算进程，显存 0 MiB。没有追加长任务或后台监控。

打包前辅助目录占用 17,260,265,472 B，所在文件系统可用 14,075,351,040 B，仍在 20 GiB/8 GiB 合同内；下一次 3 GiB 预留检查通过，但不得把一次检查当无限后续容量。主盘可用 8,603,336,704 B，距下限约 12.8 MiB，继续保持大 trace、日志和交付包在辅助盘。

版本：项目 `cc7898b1ba59d89ce7fdbb186ded21880f1adf08`；py-kvcache `3abba7a502d553f6e7e2e58b92086487e3395d7e`；配套 vLLM `817a7e3124f817cd6e549581d3e5483207a753a4`。工作树改动由源码哈希/补丁补充；commit SHA 本身不代表未改动版本。

## 下一允许阶段

继续 P3：补齐四阶段累计字节与 in-flight 额度合同，验证 mandatory/续接不能被普通额度阻断；补充“已有 store 为何尚未可签发”的紧凑只读原因观测，区分队列、I/O 深度、源事件和真实资源依赖。保持 off 作为当前最快的已测可比参照，不能选择明显受限的 fixed/pressure 来制造研究收益。

本轮已补上目标复现的一部分证据，但仍缺完整 P3 强简单基线与预算合同验收，不能直接进入 P4 主策略或 P5 正式效果实验。当前结论是：**系统可运行；存在可重复但占比有限的恢复目标保护等待；这两组简单启动额度未显示优势；联合研究方法仍待验证。**
