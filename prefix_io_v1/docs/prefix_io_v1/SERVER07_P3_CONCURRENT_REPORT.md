# Server07 P3 两并发与现有 I/O 深度验证

日期：2026-09-29。服务器：connect.westd.seetacloud.com:38819。项目根：`/root/autodl-tmp/prefix-io-v1-handoff/project`。

## 结论与阶段

AutoDL 上的现有作者缓存流水线已在真实 RTX 5090 上完成本轮有界并发验证。6 轮连续请求共 60 个完整输出、7680 个实际 token 时刻，全部与原生冷/热参考匹配；没有把 CPU/mock 结果计作 GPU 结果。

现有 iodepth 从 4 调至 8 后，混合请求两轮中位耗时由 29.570 秒降到 25.037 秒，描述性降幅为 15.33%。这是简单基线参数的变化，不能归因于尚未安装的新调度策略。每档仅两轮，深度 4 测量更早，未给出置信区间；这不是 SLO goodput 结论。深度 8 是后续需要对照的更强候选基线，仍需重复验证其稳定性。

深度 2 数值正确，但成本门槛失败：SSD 中位耗时 1.770091 秒，相对冻结预测的误差 +39.3746%，超过原定 25%。未生成许可、未运行预定的两轮连续重放；失败样本全部保留，没有放宽门槛、重拟合曲线或填造性能数据。

当前为 P3 部分完成。生产 fixed/pressure 普通额度、dependency_only、interference、joint 均未接入；P4–P7 未开启。系统能运行与新方法有研究收益是两个不同的验收结论，后者尚无实测证据。

## 实际修改与版本边界

- 已有文件仅修改 `experiments/prefix_io_v1/scripts/acquire_native_aio_costs.py` 和 `validate_heldout_costs.py`：加入显式 max_num_seqs=2 和 iodepth=2/8 的受限实验入口及对应严格校验。默认 max_num_seqs=1、iodepth=4 保留；旧单请求成本结果原样复现，旧并发资格仍通过。
- 新增 7 个实验脚本：concurrent_pilot_contract、prepare_concurrent_pilot、qualify_concurrent_pilot、run_concurrent_pilot、run_concurrent_native_reference、analyze_concurrent_pilot、analyze_fixed_io_baselines；新增 2 个 CPU 测试文件。离线分析不控制运行时调度。
- 本轮未编辑缓存引擎、reactor、模型执行器、Prefix 身份、LoadPlanner、复制合并或父任务完成协议。上一交付 90 个锁定文件中 88 个哈希不变，另 2 个为上述实验脚本；最终源锁覆盖 99 个文件。
- 公共 AIO 兼容修复、P2 观测补丁沿用旧版；新策略未安装。关闭观测仍沿用既有公共修复后的原始派发路径。原 io_uring 默认设置及显式 linux_aio 选择边界未修改。
- 作者 py-kvcache：`3abba7a502d553f6e7e2e58b92086487e3395d7e`；作者 vLLM：`817a7e3124f817cd6e549581d3e5483207a753a4`。Torch 2.11.0+cu130、Python 3.12.3、driver 580.105.08。真实 GPU UUID `GPU-f8744916-1693-fa6a-93b6-7f503c03459c`；精确版本见 version-lock.json。

## 冻结条件与能力矩阵

| 项目 | 本轮事实 | 限制 |
|---|---|---|
| 模型/精度 | 固定 Qwen2.5-7B-Instruct，BF16，离线模型 | 没有新增模型下载 |
| 并发 | max_num_seqs=2，各轮均观测到一步返回两请求 | 不是更大并发范围验证 |
| 显存与 staging | GPU KV 实际 2146959360 B；staging 实际 1073483775 B | 上限分别为 2 GiB / 1 GiB，不能外推完整显存容量 |
| 流水线 | 原始准入、预加载、共享 staging、融合复制及异步路径保留 | 没有普通研究额度 |
| 请求 | 每轮 10 请求，每请求 16257 输入 token、128 输出 token | 人工编写的受控文档，非生产代表性 trace |
| 到达 | 作者 Poisson 到达工具，rate=1.0、seed=1704，固定 manifest | 突发且有排队，不能称正常稳态负载 |
| 初始缓存 | 同一 3048 文件 SSD 语料硬链接克隆，GPU/staging 冷启动 | 每轮后续实际命中与读量可变化，均完整记录 |
| 数值 | 三档共 18 项缓存 top5/输出零容差比较；连续请求 60/60 完整输出相同 | 不等价于不同 batch BF16 全词表逐位相同 |
| 成本 | 深度 4、8 在冻结长前缀点通过；深度 2 失败 | 单请求无载成本，不是并发负载预测资格 |
| 释放证据 | 捕获原生 jobs_to_flush、父任务和传输/写入时序 | 缺少本轮具体触发原因、GPU 物理块 generation 与全部保护者证据 |
| 策略收益 | 未测量 | 不能从原有缓存收益或参数改善推导 joint 收益 |

同一 trace 内未逐请求重置缓存，未添加人为 I/O 延迟。all_hit 标签表示初始 SSD 内容已存在，不能替代实际命中计数。mixed_readwrite 在三个已存在家族之外加入两个初始不存在家族，后续复用和所有写入均计入。

## 实测结果

| 运行标签后缀 | 整轮含 drain 秒 | 全体 ITL P95 毫秒 | pending flush 等待秒 | SSD 读字节 | SSD 写字节 |
|---|---:|---:|---:|---:|---:|
| all-hit-01 | 22.257 | 31.55 | 0.000000 | 5474746368 | 19267584 |
| mixed-01 | 29.308 | 59.25 | 1.079710 | 6255542272 | 1896480768 |
| mixed-02 | 29.833 | 89.38 | 0.000000 | 6254624768 | 1896480768 |
| all-hit-02 | 20.793 | 30.66 | 0.000000 | 5475663872 | 19267584 |
| d8-mixed-01 | 26.674 | 43.67 | 0.000000 | 6399590400 | 1896480768 |
| d8-mixed-02 | 23.400 | 47.36 | 0.351732 | 5983961088 | 1896480768 |

每轮 10/10 完整输出通过；表中 ITL P95 来自全部 1270 个 token 间隔，未剔除慢请求。每次末尾 drain 约 2–3 毫秒。所有 AIO accepted/completed/reaped 相等、pending/ready/unreaped/outstanding 为零，随后模型引擎 shutdown 完成，19 个 GPU 包装会话均已排空。final_probe 是引擎关闭前的快照，不能把其中 closed/drained 标志改写成已关闭。

深度 8 的实际读量分别为约 6.40 和 5.98 GB，深度 4 两轮约 6.25 GB；这些差异保留在结果中，不假设每轮动态缓存路径完全相同。预定 8/2/2/8 顺序因深度 2 门槛失败变为仅执行 8/8，原计划、阻塞理由和实际顺序分别留档。

## 真实释放依赖与干扰边界

四轮混合重放中有两轮出现原生 pending flush：深度 4 一次 1.079710 秒，深度 8 一次 0.351732 秒。两次都涉及同一 greenhouse 家族的小后缀 store，分别 3 / 4 个父任务，每个 917504 字节。相关 D2H 在等待末段才出现，随后 SSD 写入和父任务回调完成。这是跨配置重复出现的形态，但每个配置都只有一轮出现，尚非稳定瓶颈。

前一次等待期间还记录到其他写入在推进。scheduler 源码中 jobs_to_flush 有目标块重用冲突、分配块重叠、抢占和已结束请求排空等来源；本轮没有逐触发原因和真实 block generation 快照，不能确定具体来源，也不能用 pending bytes 直接推导可释放 GPU 字节。V1 仍等待原 parent 完成；不引入 early complete_store / source-safe ack。

两次已观测等待分别占对应整轮耗时 3.68% / 1.50%。这只是该次等待的占比，不是全部策略的收益上界。无等待的轮次也可能更慢，消除一段 wait 并不自动等于同等端到端收益。

真实 I/O 时间窗与较长 token 间隔相关；部分去掉冷 prefill 时间窗后仍可观察到相关性。但调度阶段、批形状、排队及缓存路径均可能混杂。本轮没有新的 CUPTI 因果干扰验证，不能将相关性当成策略收益。

## 命令、测试与证据

服务器根目录下实际运行的主要入口如下；所有 19 个 GPU 作业的完整参数、逐次退出码及准确命令见 `commands-executed.json`、`gpu-job-ledger-slice.json` 和每个 `*-launch.json`，CPU 命令见 `*-command.json`。

```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
.venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py -q --import-mode=importlib tests/prefix_io_v1/test_config.py tests/prefix_io_v1_pilot tests/prefix_io_v1_native_costs --junitxml artifacts/prefix_io_v1/server07-p3-06/io-baseline-cpu.xml
# GPU 作业通过 run_gpu_stage.py --label <唯一标签> --seconds 240 -- .venv/bin/python <采集或重放入口> <冻结参数>
# 采集入口：acquire_native_aio_costs.py；完整参考：run_concurrent_native_reference.py；连续重放：run_concurrent_pilot.py
.venv/bin/python experiments/prefix_io_v1/scripts/analyze_fixed_io_baselines.py --plan artifacts/prefix_io_v1/server07-p3-06/io-baseline-eligible-plan.json --out artifacts/prefix_io_v1/server07-p3-06/io-baseline-result.json
```

上述输出已存在，复现应使用新输出位置和唯一 GPU 标签；不得覆盖旧证据。环境变量记录于 execution-environment.json，GPU 包装器继续执行既有预算和指定 GPU 授权。离线和 CPU 命令使用 CUDA_VISIBLE_DEVICES 空值。

- CPU 单元测试：182 passed，0 failed，0 skipped（JUnit：io-baseline-cpu.xml）。历史中间测试不累计相加。
- 离线资格：深度 4 和 8 通过；深度 2 一项成本资格失败，原始退出码为 1；这不是模型数值错误，也没有计为通过。
- 原结果回归：legacy-single-regression-command.json、io-baseline-legacy-regression-command.json。
- 冻结配置/结果：preregistration.json、io-baseline-preregistration.json、各 manifest、reference-result.json、cost-result.json、replay-result.json、io-baseline-result.json、d2/replay-blocked.json。
- 生命周期证据：native-flush-source-audit.json、flush-parent-timeline.json、flush-overlap-detail.json、native-wait-recurrence.json；保留首次并发实验代码快照 before-io-baselines。
- 版本/补丁：version-lock.json、source-lock.json、runtime-preservation-audit.json、existing-source-changes.patch；交付清单记录所有文件 SHA256。

## GPU、磁盘与下一允许阶段

本轮实际 GPU 作业 19 个，全部成功结束，耗用 1580.151 秒（26.34 分钟）。累计 2.276664 / 8 GPU 小时，剩余 5.723336 小时；这是包装器记账时长，不等价于云平台账单。新增模型下载为 0。收尾时 active_reservation=null，nvidia-smi 无计算进程。

打包前辅助存储使用 15.030 GiB / 20 GiB，所在文件系统空闲 14.155 GiB，仍高于 8 GiB 底线。打包后另行核验。权限文件哈希保持不变；未修改驱动或系统，未新租机、付款、远端推送或删除历史数据。

下一允许阶段仍是 P3：以 iodepth=8 作为需要复核的更强候选基线，补充有界的原生 flush 触发原因与 owner/generation 观察，区分释放等待、冷 prefill 和 I/O 相关 token 间隔，并在冻结负载与预算下检查稳定性。fixed/pressure 及普通额度接入仍须满足 mandatory 排空和实际所有权证据门槛，不能借本轮结果直接开启 joint。

若在合理强基线及正常负载中仍不能找到可重复的目标问题，应按交接包停止复杂策略投入并报告负结果。本轮证明了有界环境可行，尚不能证明新增研究方法有效，也不据此宣布方法在所有场景不可行。
