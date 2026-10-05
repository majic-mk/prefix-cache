# Server07 P3 独立内容验证交付

日期：2026-09-29。服务器：connect.westd.seetacloud.com:38819。项目根目录：/root/autodl-tmp/prefix-io-v1-handoff/project。

本轮结论：**2K 独立内容的缓存数值检查通过；冻结成本表的预测验收失败。P3 未完成，不进入 P4。** 原生缓存链路在当前 AutoDL 的显式 linux_aio 路线可以运行，本结果不证明新增调度策略有收益，也不证明研究路线不可行。

## 实际修改及保留边界

- acquire_native_aio_costs.py 增加可选 --prompt-manifest。未传该参数时继续使用原标定 prompt；未改变缓存身份、LoadPlanner、缓存引擎或模型执行器。
- 新增 heldout_manifest.py：校验验证分区、token 范围、样本覆盖、家族重复及与标定前缀首块的隔离；生成数据前检查预计 KV 写入、额外块、2% 元数据、64 MiB 日志预留和 8 GiB 磁盘底线。
- 新增 prepare_heldout_validation.py：离线 tokenizer 生成三个本地编写的文档家族，一个热身、两个测量。内容属于受控合成文档，未用于拟合，也不是正式 evaluation 集。
- 新增 validate_heldout_costs.py：严格缓存数值比较、冻结配置与来源核验、独立点预测检查；不输出新曲线。
- 新增 analyze_heldout_drift.py：保留失败后追加的原标定内容对照和主机时间线摘要。
- 新增两个 CPU 测试文件，共 20 个测试；更新状态、证据及复现文档。
- P2 锁定的 29 个源文件全部未变，原严格曲线 exporter 未变，原候选表 SHA256 未变。fixed／pressure 普通额度及 dependency／joint 仍未接入生产 reactor。

脚本位于 experiments/prefix_io_v1/scripts/；测试位于 tests/prefix_io_v1_pilot/。本轮增量补丁为 artifacts/prefix_io_v1/server07-p3-04/acquisition-driver.patch。没有修改驱动或系统配置，没有下载新模型，没有删除或移动旧结果。

## 冻结合同

模型为本地 Qwen2.5-7B-Instruct，BF16；作者 py-kvcache 3abba7a502d553f6e7e2e58b92086487e3395d7e，作者 vLLM 817a7e3124f817cd6e549581d3e5483207a753a4。

保持 max_model_len=max_num_batched_tokens=16400、max_num_seqs=1、KV 2 GiB、共享 staging 1 GiB、iodepth=4、预加载、复制合并和异步执行。成本采集使用原有诊断置冷流程，每请求只输出 1 token，不能当成持续 decode 或 goodput 主结果。

独立内容前缀 N=2048，请求长度 N+1。prompt 清单 SHA256：
9499988298349508cba88ded3611eb1d05228333259d8bbd1ed38c5a5270f600。

原候选表 SHA256：
892a979b1b1136409e258537f85e9355f6c7bb88e5de1d3f636bf23caf119c2f。
表仍仅具有先前的标定集成诊断资格，没有升级为正式运行表。

## 数值与成本结果

原生 GPU 命中对照与 SSD／staging 恢复共 6 组比较：选中 token、top-5 token 集合及 logprob 全部零容差相等。冷计算与缓存计算的差异另外保存，不以跨 batch 重算逐位相等作为 KV 字节正确性的定义。此次不是全词表或所有输入的保证。

独立成本共检查 18 行（含热身正确性），每条路径 4 个测量值，来自两个文档家族和两次独立引擎启动。诊断标记及 logprob 采样未进入成本统计。验收前固定相对预测值偏差上限为 25%。

| 路径 | 原表预测 ms | 独立内容中位数 ms | 偏差 | 结果 |
|---|---:|---:|---:|---|
| 冷计算 f | 144.458 | 144.240 | -0.15% | 通过 |
| SSD g_ssd | 157.048 | 172.536 | +9.86% | 通过 |
| staging g_mem | 51.379 | 64.879 | +26.27% | **失败** |

staging 测量值为 66.961、66.189、63.568、32.167 ms；全部保留。没有改阈值、丢弃慢样本、重新拟合或将失败包装为通过。这是成本预测失败，不能据此认定缓存恢复内容错误。

失败后冻结并执行两次原标定 2K 内容的诊断对照。SSD 中位数为 199.387 ms（比旧表高 26.96%），staging 为 56.608 ms（高 10.18%）。因此旧表在原标定内容上也存在时间漂移；该后置、未交错的对照不能确定内容因果关系。

独立 staging 的主机 forward 区间为 17.507–42.864 ms，缓存 transfer 区间为 6.413–16.914 ms。它们来自原作者 profiler 的主机时间线，包含嵌套与异步边界，不能相加当作互斥耗时，也不能当作 CUDA kernel 执行时间。本轮未新增 CUDA timeline 或持续 decode 实验。

## 测试与真实资源

最终 CPU 回归：**89 passed，0 failed，0 skipped**。GPU 初始化在 CPU 测试中被禁止；cpu-gpu-guard.json 记录 cuda_initialized=false。中间测试计数不重复相加。

真实 GPU：**9 次预算包装运行全部 exit=0**，其中 3 次内容参考／缓存生成、4 次独立成本、2 次原标定内容漂移对照。成本分析器按约定 exit=1，记录预测验收失败；这与 GPU 作业是否成功完成分开报告。

48 行采集记录全部保留；恢复路径累计可核验读取 1,761,607,680 字节。native handler 的 accepted=completed=reaped，未结算队列为零；引擎退出完成，所有作业会话排空。收尾 GPU 0 MiB／0%，无计算进程，无遗留 sem.*。

本轮 GPU 631.005 秒（10.52 分钟）；累计 4891.806 秒（81.53 分钟）/8 小时，剩余约 6.64 小时。模型累计下载账本仍为 19,422,798,722 字节，本轮新增 0。没有重置累计账本。

## 证据和命令

完整证据目录：artifacts/prefix_io_v1/server07-p3-04/。

- reference-plan.json、heldout-prompts.json、heldout-source-texts.json：事前冻结内容和数值协议。
- cost-plan.json、heldout-cost-result.json：事前 25% 阈值及真实失败。
- heldout-reference-result.json：6 组严格数值比较。
- drift-control-plan.json、drift-result.json：失败后的有限对照及原始主机区间。
- final-cpu.xml、cpu-gpu-guard.json、final-health.json：测试、预算和排空。
- commands.json、environment-overrides.json：实际执行命令和环境。
- acquisition-driver.patch、source-lock.json、workspace-audit.json：修改与源码审计。
- experiments/prefix_io_v1/runs/server07-p3-heldout-* 及 server07-p3-2k-drift-control-*：原始输出、冻结配置、每次运行的 driver 源码和 traces。

实际 GPU 命令均通过 run_gpu_stage.py --label <已记录标签> --seconds 180 -- .venv/bin/python experiments/prefix_io_v1/scripts/acquire_native_aio_costs.py ... 执行，完整参数见 commands.json。详见 REPRODUCE_SERVER07_P3_HELDOUT.md。

## 下一允许阶段

P3 保持进行中。下一步应先处理成本表的时间波动与预测可靠性，再完成长前缀独立内容、持续 decode 开发负载及强简单基线；不能用这一个 2K 点宣告全域通过。fixed／pressure、生产 owner/freshness 适配以及真实阻塞证据仍欠缺，P4–P7 仍未获阶段资格。

当前数据盘只剩约 8.09 GiB，接近既定 8 GiB 底线，不能在不增加空间的情况下保存新的完整长前缀数据。系统盘约有 29 GiB 空闲，已准备 system-disk-proposal.json：仅新建 /root/prefix-io-v1-validation，最多使用 20 GiB，并保留至少 8 GiB 空闲。

**该目录尚未获新增授权，也没有创建或使用。** 需要用户确认的原因是 experiments/prefix_io_v1/configs/permissions.yaml 当前 approved_experiment_root 仅覆盖数据盘项目 runs 目录。GPU 和下载总预算保持原值。若允许使用新存储位置，必须记录实际介质并重新验证该位置的成本，不能直接拿不同位置的数据作性能归因。旧证据全部保留。
