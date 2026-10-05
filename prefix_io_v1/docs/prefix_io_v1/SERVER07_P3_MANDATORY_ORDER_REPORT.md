# P3 第 12 次交付：必需写回优先基线及真实模型 ABBA 对照

## 结论
**功能可运行；尚未证明稳定的端到端收益。** 在原有 AutoDL、作者 py-kvcache 和配套 vLLM 上完成了可关闭的有限候选写回排序，未重写缓存引擎或模型执行器。真实 GPU 基础验证与四次 Qwen 模型回放均通过。

关闭策略的两次 cohort 为 32.704260、25.223704 秒；开启策略为 27.200766、29.057094 秒。均值差为 2.883%（开启较快），但相邻配对方向相反：一次快 16.828%，另一次慢 15.198%。两次关闭均没有目标 flush 等待；第一次开启也没有触发排序。故均值差不能作为策略收益、等待被消除或研究方法有效的证据，不做置信区间或 SLO goodput 声明。

P3 仍未完成，P4–P7 保持依赖门禁。此处配置名 pressure 专指本轮必需写回优先的简单基线，不表示已实现完整干扰额度或 joint 策略。

## 实际修改
- 创建隔离工作树 `third_party/work/py-kvcache-p3-order-cpu`，分支 `codex/prefix-p3-mandatory-order`；先从原 P3 工作树复制已验证的共同修复和 P2/P3 增量，逐个验证 31 个 Python 源文件一致，再作本轮修改。原 P3/P2 工作树、vLLM 作者工作树未修改。
- `src/prefix_io_control/store_order.py`：仅查看 native active 列表的前 32 个 parent，先遍历当前 mandatory 集合内且仍有未提交文件的 store，其余 parent 仍在同一次 native pass 内按原顺序遍历。不另建 I/O 队列，不保存跨轮 job/Future 引用，不产生额度、事件查询、提前释放或完成回调。
- 新工作树 `py_kvcache/reactor.py`：只在原 store 遍历处接入可选排序；构造参数验证及 coordinator 转交。原 load 优先级、物理资源检查、D2H→SSD 已接受续接、预加载、复制合并、parent 完成及 shutdown 不变。关闭时返回原 active 列表；可选排序异常标记 fault 后退回原路径，实验拒绝把 fault 算作通过。
- `order_options.py` 和新工作树 `py_kvcache/vllm.py`：严格的显式参数解析，默认关闭；enabled 必须有 progress identity、候选窗口固定为 32，禁止与旧 start-budget 同时启用。
- 模型实验采用新的 `run_store_order_pilot.py`、`store_order_worker_probe.py`、`store_order_model_contract.py`，复用未改的 `run_concurrent_pilot.py`。两组热身均关闭排序，热身排空后在尚未提交 cohort 请求时设置一次排序对象，两组均用相同的 flush/readiness 观测。结束后原生 shutdown，再读取计数。原 result.json 的 shadow 指观测模式，另存 order-result.json 明确实际排序模式。
- 新增基础 GPU 验证、固定 ABBA 分析、CPU 矩阵入口和 65 项 CPU 测试。共同兼容补丁未改；本轮增量单列在 baselines/0004 和 0005，未混入研究策略补丁。

精确 Prefix、LoadPlanner 成本准入、共享 staging、preload、fusion、原生异步流水线继续保留。四阶段 DispatchBudget 仍是 CPU 合同，没有接入 native intake/接受点，不能将其写成已执行的完整字节/在途限额。

## CPU 结果与补丁验证
| 实际执行 | 通过 | 跳过 | 失败 |
|---|---:|---:|---:|
| 原生排序定向测试 | 27 | 0 | 0 |
| 完整固定矩阵 | 822 | 16 | 0 |
| 排序与模型边界套件 | 59 | 0 | 0 |
| 分析/驱动恢复测试 | 6 | 0 | 0 |
| 按测试身份去重，取最新结果 | **860** | **16** | **0** |

全部 CUDA guard 为 initialized=false、GPU workloads=0。16 个跳过包括 13 个 no-vLLM fallback 测试（当前加载真实 vLLM）和 3 个原 io_uring EPERM 测试，不算通过。新增覆盖已接受续接、资源不足、窗口外 parent、共享预加载/融合、短写/重复 CQE、关闭还原、fault 回退、线程/身份/重复激活、来源锁及实际模式校验。AST 对照确认移除新排序接入后整个 reactor 等同前一版本，vLLM 只改变适配器导入/名称。

两个补丁在辅助盘空白目录完成 git apply --check、实际 apply 及逐文件内容比较，覆盖 14 个文件。首个项目补丁生成时缺少 new file mode 头，检查失败；已修正补丁格式并重验通过，未改变实验运行源码。证据：`patch-roundtrip.json`，初版补丁备份保留在 before/。

## 真实 GPU 执行与结果
固定 UUID 为 GPU-f8744916-1693-fa6a-93b6-7f503c03459c，RTX 5090，driver 580.105.08。所有 GPU 操作通过 run_gpu_stage.py 执行，逐次检查 UUID、无其他 GPU 进程、源码 SHA、剩余 8 小时总预算，以及辅助盘和主盘预留。

| 标签后缀（均以 server07-p3-12- 开头） | 内容 | GPU 预算计时 |
|---|---|---:|
| order-primitive-01 | off/pressure 各 store、restore、shared，共 6 阶段 | 24.459798 s |
| mixed-order-01-off | 第一轮关闭 | 106.209787 s |
| mixed-order-02-pressure | 第一轮开启 | 102.866795 s |
| mixed-order-03-pressure | 第二轮开启 | 105.028876 s |
| mixed-order-04-off | 第二轮关闭 | 101.230866 s |

基础验证使用真实 CUDA 与 Linux AIO，来源相同的字节精确比较；关闭时大/小 parent 完成顺序为 [1,2]，开启为 [2,1]，已接受工作继续完成。每组 65 个文件、8,519,680 B；共享恢复读一次，实际 H2D 字节 17,039,360 B。全部会计计数闭合，AIO 和线程排空，无人工设备延迟。

四次模型实验在运行前固定 ABBA 顺序和参数，全部完成，没有按耗时挑选、补跑或临时调参。模型为 Qwen2.5-7B BF16，depth=8、max_num_seqs=2、2 GiB GPU KV、1 GiB staging，冻结的混合请求流每轮 10 请求×128 token。四轮 **40 个完整输出、5,120 个 token** 对照原生 cold/GPU-hot 参考完全一致；每轮 3,048 个源缓存文件保持不变，AIO 接受/完成/回收一致，无观测或排序 fault，engine shutdown 完成。

| 运行顺序 | 排序 | cohort 含 drain（s） | pending flush 等待（s） | 重排 pass | SSD read（B） |
|---|---|---:|---:|---:|---:|
| 1 | off | 32.704260 | 0 | 未安装 | 6,399,590,400 |
| 2 | pressure | 27.200766 | 0 | 0 | 6,399,590,400 |
| 3 | pressure | 29.057094 | 0.032171 | 2 | 6,254,624,768 |
| 4 | off | 25.223704 | 0 | 未安装 | 5,980,291,072 |

每轮 cohort SSD write 均为 1,896,480,768 B。计划参数相同，但实际读字节/调度形态存在差异；两个关闭重复之间相差 7.480556 秒，远大于两组均值之差 0.835052 秒。第一组开启较快却根本没有触发排序，第二组开启实际触发却慢于配对关闭。这些数据不支持“已验证稳定提升”，也不足以宣布方法无效。

第三轮的唯一 qualified fence 仍保护 parents 23、25、27，各 917,504 B，合计 2,752,512 B。首次相关 D2H host enqueue 在等待后 10.045722 毫秒，等待返回共 32.170609 毫秒。保留的一组 pre-enqueue 采样同时显示前置背景 parent 22、depth 已满；它不是连续等待积分或设备时间轴。目的块仍属于新 allocation generation，active_refs=1；解除该保护不等于新增加可用 GPU 容量。本轮不能与 P311 的单次 616 毫秒等待直接算加速比。

## 实际命令和证据
服务器项目根目录：`/root/autodl-tmp/prefix-io-v1-handoff/project`。原始命令、环境差异、stdout/stderr、退出码保存在本轮 *-command.json，GPU 环境在 gpu-environment.json。以下是已完成命令，旧标签为 append-only，勿直接重复启动。

```text
.venv/bin/python experiments/prefix_io_v1/scripts/run_order_cpu_qualification.py --label server07-p3-12-order-matrix
.venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py -q --import-mode=importlib tests/prefix_io_v1_store_order --basetemp /root/prefix-io-v1-validation/cpu-evidence/server07-p3-12-model-targeted --junitxml artifacts/prefix_io_v1/server07-p3-12/cpu-model.xml
.venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py -q --import-mode=importlib tests/prefix_io_v1_store_order/test_order_analysis.py --basetemp /root/prefix-io-v1-validation/cpu-evidence/server07-p3-12-analysis-targeted --junitxml artifacts/prefix_io_v1/server07-p3-12/cpu-analysis-first.xml
.venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py --label server07-p3-12-order-primitive-01 --seconds 120 -- .venv/bin/python experiments/prefix_io_v1/scripts/qualify_store_order_gpu.py --output /root/prefix-io-v1-validation/runs/server07-p3-12-order-primitive-01/details
.venv/bin/python experiments/prefix_io_v1/scripts/analyze_store_order_model.py --plan artifacts/prefix_io_v1/server07-p3-12/model-plan.json --completed 4 --out artifacts/prefix_io_v1/server07-p3-12/model-analysis.json
.venv/bin/python artifacts/prefix_io_v1/server07-p3-12/fence-analysis-command.py
```

四轮模型的完整 argv 分别见 model-1-plan.json 至 model-4-plan.json，实际回执见对应 model-N-command.json；每次外层 --seconds=240，并使用 run_store_order_pilot.py 的 --order-config、原冻结曲线/manifest/permit、--flush-cause-probe、--store-readiness-probe。未修改原固定 3 GiB 存储预留。基础验证预留为 128 MiB；CPU 矩阵为 640 MiB。

主要证据目录：`artifacts/prefix_io_v1/server07-p3-12/`。包括 cpu-summary.json、primitive-result.json、model-analysis.json、fence-analysis.json、gpu-summary.json、patch-roundtrip.json 及源码锁。原始 GPU 输出、完整 token/trace 位于 `/root/prefix-io-v1-validation/runs/server07-p3-12-*/details/`；矩阵日志/XML/guard 位于 `/root/prefix-io-v1-validation/cpu-evidence/server07-p3-12-order-matrix/`。交付 ZIP 收录这些原始证据和代码，不收录模型权重和缓存 .bin。

## 预算、版本与下一允许阶段
本轮 GPU 增量 **439.796122402 秒**，累计 **10,156.083615508 秒 / 2.821134338 小时**，剩余 **5.178865662 小时**。五次 wrapper 均 exit=0、无超时、session_drained=true；结束后 GPU 0 MiB/0%，无活跃预算预约。没有下载模型、改变驱动/系统、外部 push 或新租机。

本轮后辅助盘计入用量约 19,853,131,776 B（打包前），尚未超过 20 GiB，但加上下一次固定 3 GiB 模型实验预留会超限，且会跨越辅助盘 8 GiB 剩余空间门槛。故**下一轮 GPU 模型实验当前受存储门槛阻塞**；不得降低预留或把以前按 SHA 批准的合并范围扩展到本轮新缓存。CPU 工作可在重新预检后继续。主盘仅约 10 MiB 高于 8 GiB floor，大产物仍放辅助盘。本轮没有额外合并或删除缓存。

根项目 HEAD cc7898b1ba59d89ce7fdbb186ded21880f1adf08；py-kvcache author 3abba7a502d553f6e7e2e58b92086487e3395d7e；vLLM author 817a7e3124f817cd6e549581d3e5483207a753a4。GPU 前锁 239 项、模型前锁 253 项，最终锁与工作树 dirty 状态见 source-lock.json/version-lock.json。

下一阶段仍为 P3：
1. 保留此次全部零触发和负向配对结果，先用 CPU 验证有限候选排序与四阶段额度/原生接受点的接入边界，保留已接受续接和真实资源约束。
2. 为性能验证预先定义能稳定出现目标等待、且不加人工设备延迟的开发负载与停止条件；冻结之后才评估。不能在当前四轮中挑最快样本，不能用一次旧等待作反事实。
3. 下一次 GPU 模型运行前需解决明确的存储门槛，并完成相应 CPU/真实 GPU 正确性资格；仍需补足 P3 最强简单基线、观察开销及阶段额度资格，才可进入 P4 研究策略。
