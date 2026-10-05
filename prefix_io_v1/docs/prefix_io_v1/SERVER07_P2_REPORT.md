# Server07 P2 观测与进展保护交付（2026-09-29）

本次在 westd:38819 的现有服务器完成 P2 有界验证。真实 GPU/AIO 路径可运行，off 可回到原生执行路径；没有启用生产普通额度、dependency_only 或 joint。研究收益尚未验证。

## 实际改动与边界

- 新建隔离 worktree `third_party/work/py-kvcache-p2-aio`，沿用 P1 的显式 linux_aio 公共兼容补丁，再应用 mandatory bridge 和有界快照补丁。未更改默认 io_uring，没有自动后端回退。
- `src/prefix_io_control/native_options.py` 与 `0003-native-options-adapter.patch`：严格验证项目自有 off/shadow 配置，原生 PyKvCacheOffloadingSpec 仅传入可选观察器。未配置或 off 时无 publisher、无 mandatory 状态分配。此配置不是作者原有 API。
- `src/prefix_io_control/native_owner.py`：可卸载、同 owner 线程的有界诊断观察器。观察原生分配、释放、store parent 创建与所有 worker 确认后的完成，不改引用计数、fence、complete_store 或分配队列。
- 新增两个 GPU 验证驱动及 `analyze_p2_qualification.py`；新增配置适配与 owner 生命周期 CPU 测试。
- 未修改作者 vLLM 的模型执行、attention、Prefix 身份、原 LoadPlanner 成本准入、共享 staging、预加载、复制合并或异步流水线。P1 基线 worktree 保留。

版本：py-kvcache `3abba7a502d553f6e7e2e58b92086487e3395d7e`；vLLM `817a7e3124f817cd6e549581d3e5483207a753a4`。真实源文件 SHA256 位于本次 `source-lock.json`。补丁边界与完整工作区 diff 均留存。

## 验证结果

CPU 三组共 **398 passed / 16 skipped**：合并原生/AIO/progress/observer 测试 380/16，新增 options 8/0、owner 10/0。GPU 初始化被 CPU guard 禁止；CPU 测试不是 GPU 证据。补丁 reverse-check 与两个 worktree 的 git diff --check 均通过。

真实 RTX 5090 UUID：`GPU-f8744916-1693-fa6a-93b6-7f503c03459c`。最终 mandatory 验证 `server07-p2-progress-05` 通过：

- off 原生路径无观察状态。
- 在生产写入事件尚未完成时接受 store，test-only 零普通额度下 handler.wait 发布 mandatory 信号后完成 store/load；16 块、每块 917504 字节，真实 CUDA/AIO 往返字节完全一致。
- 两个已接受 store parent 在 shutdown 下均排空。
- 私有测试文件短读确实失败，观察到 FAILED_DRAINING；全部已接受 AIO 回收、outstanding=0、ring drained/closed、无残留 worker。
- 人工 CUDA 延迟和零额度仅为正确性诊断，不作为正常负载或性能收益证据。此处张量为真实 GPU 上的合成 KV 布局；同一生产 KV 的往返证据仍引用 P1 报告。

真实 Qwen2.5-7B BF16：2 轮 off、2 轮 shadow、1 轮独立 owner 诊断，各 32 条请求、每条生成 16 token。对应输入、输出与命中数全部一致。每条冷请求 cached=0，热请求 cached=128；没有逐请求 reset。每轮 128 个原生 AIO 操作接受并回收，最终 engine shutdown 完成。GPU KV 实际 66977792 字节，staging 实际 133959679 字节，分别在 64MiB/128MiB 配置内。

同一第 3 号请求族的冷/热生成不同，在全部 off/shadow 臂完全复现。原因尚未归因；不能宣称所有冷/热计算逐 token 相同，也不能把独立计算形状差异直接当成存储拷贝错误。

## 真实释放依赖证据

owner 诊断最多追踪 8 个源块、64 个 generation、每块 32 个 parent，留存 96 条最近快照。只有当前 generation、ref_cnt=0、非 null、原生 free queue 双向链接正确、原生 pending fence 为空、观察到的全部 parent 经原生回调确认后才标记可复用。

此次观察到 **13 次满足该条件后的实际原生重新分配**、1424 次活跃引用观测；可复用块总量最高 7340032 字节。owner errors=0、faulted=false。这里“释放”指 KV 槽可覆盖复用，不是 cudaFree 或系统空闲显存增加。

带 fence 的 free block 观测为 **0**。该有界低压力请求流尚未展示真实资源等待问题，不能据此虚构 dependency/joint 的收益。reactor 的 GPU reusable 字段继续为 null：它没有原生 allocator 所有权。诊断探针尚不是带 freshness/epoch 合同的生产 controller 适配器。

## 观测开销

保持 10ms 原生 snapshot 采样，owner 诊断不混入 M4 开销：

| 运行 | 28 条非热身请求 generate 调用总秒数 | 快照数 |
|---|---:|---:|
| off-01 | 5.894072 | 0 |
| shadow-01 | 6.051154 | 61 |
| shadow-02 | 5.974771 | 66 |
| off-02 | 5.994430 | 0 |

两对相对耗时变化 +2.665%、-0.328%，合并 +1.156%。这些是短诊断请求的端到端调用耗时；没有把请求间写报告及末尾 drain 纳入 cohort 分母，不能称为正式 goodput。首末 token 跨度/15 只是平均 decode 间隔，不是逐 token ITL P95。两轮和固定输入不足以证明稳定 2% goodput 目标。

CPU microbenchmark：off hook 47.5ns，未到采样时间 355.3ns，1ms 周期诊断 370.0ns，强制每次完整捕获 41429.6ns（每组 7×20000，报告的是每轮均值的中位数）。合成 32 parent/64 I/O/64 copy 元数据，无 GPU/模型调用，不能替代真实 E2E 结果。

## 失败记录

- progress-01：测试错误地把 store-only 的 _schedule_one 当成 load 入口；修正测试入口，没有修改原生调度。
- owner-01：过强的冷/热输出完全相同断言失败；off 对照复现差异，后续独立报告而非隐藏。
- progress-03/04：未满足 pending producer 的测试前提，保留为失败。预热 fill/event 后，progress-05 在 enqueue、submit、wait 三个边界均观测 pending，完整字节与排空验证通过。没有为了通过而删除 pending 断言。
- progress-02 为早期通过的 mandatory 诊断，尚未充分证明 pending producer 写入的依赖；05 是最终强化证据。

## 命令与证据

所有计算、修改、测试均在服务器项目根目录执行。准确 argv、stdout、stderr、退出码与 GPU wrapper 收尾记录保存在：
`artifacts/prefix_io_v1/server07-p2-01/*-launch.json`、`*-cpu-tests.json/xml`、
`experiments/prefix_io_v1/runs/server07-p2-*/`。

核心入口：
```text
PYTHONPATH=third_party/work/py-kvcache-p2-aio:src
.venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py -q --import-mode=importlib ...
.venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py --label <unique-label> --seconds <bound> -- .venv/bin/python experiments/prefix_io_v1/scripts/qualify_p2_gpu_progress.py --output <private-run>/details
.venv/bin/python experiments/prefix_io_v1/scripts/qualify_p2_native_observer.py --mode off|shadow --output ... --model-dir ... --model-plan ... --curves ...
.venv/bin/python experiments/prefix_io_v1/scripts/measure_observation_cpu.py --output artifacts/prefix_io_v1/server07-p2-01/observation-cpu.json
.venv/bin/python experiments/prefix_io_v1/scripts/analyze_p2_qualification.py --output artifacts/prefix_io_v1/server07-p2-01/qualification-summary.json
```
模型命令也始终经 run_gpu_stage 调用；上方省略参数不可直接作为复现命令，复现应使用保存的完整 launch JSON。禁止直接绕过 UUID/预算/离线检查。

本阶段 11 次真实 GPU 运行（含失败），合计 **429.5989 秒**；累积 **1598.9421 秒 / 8 小时**，剩余约 **7.55585 小时**。本阶段新增模型下载 0 字节，累积 19422798722 字节。没有修改驱动或系统。

## 阶段结论与下一允许动作

P2 有界正确性与观测验证完成，允许进入 P3 开发集的持续 decode 干扰标定和正常请求流 pilot。生产普通额度接线前仍须对具体接线重复 mandatory/下游续接验证；当前未安装额度。

joint 仍不允许启用：需生产 owner/freshness 适配器与真实等待 witness，而不仅是离线诊断快照。P3 应先确认正常负载有可重复的资源等待，冻结开发配置，补齐实际逐 token 事件、完整 cohort/drain 口径、fixed/pressure 强基线。无目标问题或简单基线已消除问题时应停止研究扩展，如实报告。P4–P7 未完成；不预设收益。
