# Server11 实际系统验证交付 — 2026-10-03

本轮在用户指定的新服务器完成了环境审计、真实模型与原生 I/O 校准，以及限定 P4 单文件干扰控制的 GPU 验证。**当前服务器可以运行原系统；新增控制的功能链路通过，但本次条件下没有形成性能提升。完整 P4 与正式 P5 未完成，不能以功能通过替代性能通过。**

最后状态核查：2026-10-03T12:32:11.744847+00:00。全部作业已结束、原会话已排空，GPU 计算进程为空；本轮不再启动 GPU。可先关闭 GPU 节省租费，无需继续扩容当前数据盘。

## 1. 最终结果及证据边界

最新 v3 使用同一 Qwen2.5-7B、固定 129-token 输入、相同 seed=2829、单请求、完整生成 128 tokens，逐项比较输出 token IDs。单次后台读取为原模型生成的 KV 文件，917,504 字节；测量包含全部 128 步、原 I/O 完成和最终排空。

| 版本 | 模式 | 选定步 GPU 时间 ms | 完整请求加排空 s | 相对同版 off | 本臂限定资格门禁 | 冻结成本覆盖 |
|---|---|---:|---:|---:|---|---|
| v1 | off | 20.354 | 2.192960 | 基线 | 通过 | 通过 |
| v1 | shadow | 46.586 | 9.847907 | +349.07% | 失败 | 失败 |
| v2 | off | 20.846 | 2.317297 | 基线 | 通过 | 通过 |
| v2 | shadow | 21.780 | 2.417813 | +4.34% | 通过 | 通过 |
| v2 | on | 62.081 | 2.533853 | +9.35% | 通过 | 失败 |
| v3 | off | 19.238 | 2.263030 | 基线 | 通过 | 通过 |
| v3 | shadow | 21.493 | 2.363657 | +4.45% | 通过 | 通过 |
| v3 | on | 66.240 | 2.424484 | +7.13% | 通过 | 失败 |

最新版 on 比同版 off 慢 **7.13%**，比 shadow 慢 **2.57%**；选定步 66.240 ms 超过冻结上界 24.999 ms。on 的 `qualification_passed=true` 只表示预先定义的功能与生命周期资格；其 `frozen_cost_migration_pass=false`、`strategy_effect_verified=false`、`production_qualified=false` 均保留。未事后调大阈值，未挑选较好步骤替代完整请求。

v3 实际记录 152 次阻止提交、151 次观测复用，同一候选的首末延期跨度 60.864 ms；三臂输出一致、单文件读取完成、原 shutdown 返回、OS 会话排空。读取接受时间相对 host end/event record 的位置不等于设备已无重叠，报告不声称 GPU 零重叠或资源已释放。

这些是固定单条件、每臂一次的开发验证，未做统计重复、正常业务负载或业务 SLO 验收。该负面结果支持“当前实现与当前条件未显示优势”，不能推出整个研究方向永远不可行，也不能称完整 P4/P5 已完成。

原始验证：[汇总 JSON](C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/server11_results/SERVER11_FINAL_EVIDENCE.json)、[v3 off](C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/server11_results/p4_v3/QUALIFICATION_off.json)、[v3 shadow](C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/server11_results/p4_v3/QUALIFICATION_shadow.json)、[v3 on](C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/server11_results/p4_v3/QUALIFICATION_on.json)。

## 2. 新服务器审计与版本锁

服务器工作区：`/root/autodl-tmp/prefix-io-v1-handoff/project`。SSH 端口 26909；凭据不写入报告或证据包。

| 项目 | 实际锁定/审计结果 |
|---|---|
| GPU | RTX 5090，GPU-38c1d1c7-2aba-3fa9-e1cd-fe4d3ae569c9 |
| 驱动 | 595.71.05；libcuda SHA-256 76e0d9678d41cf6b6ae71d18549d88a963d3d434f08108baa12457eb53ca88d6 |
| CPU/内存配额 | 25 核 / 90 GiB |
| 项目 HEAD | cc7898b1ba59d89ce7fdbb186ded21880f1adf08 |
| 作者 py-kvcache | 3abba7a502d553f6e7e2e58b92086487e3395d7e |
| 作者 vLLM | 817a7e3124f817cd6e549581d3e5483207a753a4 |
| Python / Torch | 3.12.3 / 2.11.0 |
| Triton / Transformers / Flashinfer | 3.6.0 / 4.57.6 / 0.6.11.post2 |
| 工具链 | 复用已有私有 CUDA 13 SDK；与本机 libcuda 精确绑定；没有安装系统软件或修改驱动 |
| 克隆核验 | 继承 server10 的 4,112 项源/资产核验通过 |
| v3 运行源锁 | 4,271 项，每臂执行前后逐项核验通过；SHA-256 5f3dd39331f17b9d6d0a6b94e03e3052fb3eb3fc25d4f2be380bae969faab276 |

版本审计原件位于第一份归档中的 `server11-migration-audit-v1-20261003/CLONE_SOURCE_AND_VERSION_AUDIT.json`。模型、SDK、作者源码和新叠加层分别记录实际路径及 SHA；没有把旧 reactor 的 SHA 冒充为新运行代码。

## 3. 能力矩阵和生命周期

| 能力 | 本轮证据与边界 |
|---|---|
| 作者模型执行器、Prefix Cache、共享 staging、预加载、复制/异步路径 | 沿用现成实现；真实 Qwen 模型与原生 SSD I/O 已执行；关闭新策略时 bridge=None，回到原 dispatch |
| 成本校准 | v4 八文件及 v5 单文件各完成六个真实窗口；每组前两对拟合、最后一对留出，未把八文件成本除以八冒充单文件 |
| 单文件干扰观察 | v5 留出 A=12.673 ms、B=20.418 ms，冻结预测上界=24.999 ms，留出未低估；仅覆盖该有限条件 |
| 单文件有限 I 控制 | 实际延期、原 IO 完成、完整输出和 shutdown 均通过；最新版总体耗时仍退化 |
| 成本迁移到 active 控制 | on02、on03 均不通过；不得将原校准覆盖扩大成控制后保证 |
| D/J 真实资源释放与 ETA | 仍未获得完整真实压力工况的 GPU 资格；释放额度、通用生产成本表保持关闭 |
| 正式性能/业务 goodput | 未验证；正式 TTFT/ITL SLO 仍为空，不编造服务目标 |

生命周期保持原 owner：模型建立前缀、预热并排空后，利用实际 runner 步信息触发原 preload；原 open/ready、slot reserve、AIO 接受与完成、store 和最终 drain 全部沿原路径。延期时原 slot 正常归还；新缓存只保存不可变观测值，不持有 Future、job 或 slot owner，不产生新工作队列。mandatory、下游推进、关闭、上下文失效与 100 ms 原等待上限保留。

历史 server10 的字节往返与正常模型证据继续保留；本报告没有将这些历史运行冒称为在新服务器重新执行。当前单文件实验使用原有 `load_planner='off'` 开关隔离 I/O 干扰，因此不构成正常生产成本准入开启时的效果证据。原成本准入实现未被重写。

## 4. 实际修改及失败原因

所有实验叠加层另存于新 artifact 目录，作者基线和既有证据未覆盖。原 vLLM 模型执行器、CUDA 内核、缓存引擎未重写。

- 共同兼容设置：修正已有私有 SDK 路径绑定；使用原生支持的 `disable_log_stats=True` 避开可选统计结构不兼容。三种对照一致。
- 共同观测修复 v2：空候选时跳过完整 P4 快照构建。CPU 原路径回放 10,000 次空调度的 collect 从 30,000 次降至 0；真实 shadow 完整请求从 v1 的 9.848 s 回落至 v2 的 2.418 s，但仍不证明达到观察开销门槛。
- 共同观测修复 v3：同一个原 ready、同一真实 step 的不可变观测复用；状态变化即时失效，不刷新旧采样时间，每次仍执行原 preview。CPU 回放和真实 GPU 均确认复用发生，完整请求仍比 off 慢。
- 有限研究策略：只为 v5 的单文件、零在途 I/O、batch=1、decode=1、context=144 条件签发绑定原始证据的限定资格。通用成本资格、资源释放信用和 D/J 未据此放开。

主要修改位置：[reactor.py](C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v3/source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py:2107)、[p4_policy.py](C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v3/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_policy.py:275)、[p4_bridge.py](C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v3/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_bridge.py:388)、[证据工厂](C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v3/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_single_file_receipt.py:211)、[完整步采集](C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v3/native_full_step_collector.py:253)、[运行身份绑定](C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v3/single_file_runtime_binding.py:122)、[运行器](C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server11_candidates/p4_single_file_runtime_v3/run_p4_single_file_experiment.py)、[独立验证器](C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server11_candidates/p4_single_file_runtime_v3/verify_p4_single_file.py)。

[基线→v2 补丁映射](C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/server11_results/patches/PATCH_MAP.json) 把共同空观测修复与策略分开；[v2→v3 补丁映射](C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/server11_results/patches_v3/PATCH_MAP.json) 单列重复观测修复及计数。每份补丁已逐 hunk 重建并核对目标字节，不重复应用综合差分。

最初三次 native-cost 启动失败分别是私有 SDK 缓存路径、可选 Prometheus 统计兼容、以及输入恰好满 block 时缓存命中数量假设错误；失败原件全部保留。后续统一使用冻结 129-token 条件。v1 shadow 则因观察路径明显变慢未获下一臂资格，因此没有运行 on01。

当前能确定的是主动等待期间仍有显著耗时增长。反复完整收集确实存在且已消除，但 v3 仍退化，所以不能把所有损失归因于这一项。Python reactor 重试与模型 CPU 提交之间的竞争属于待量化解释，本轮没有伪造因果剖析结果。

## 5. 实际执行命令与测试

以下命令均在服务器工作区执行；完整 argv、环境覆盖、stdout、stderr、退出码和耗时保存在对应目录的 `*_COMMAND.json`、`*_RESULT.json`、`*_STDOUT.log`、`*_STDERR.log`，并进入归档。

```text
.venv/bin/python -B -I -S C3/run_original_cpu_regression.py --project-source /root/autodl-tmp/prefix-io-v1-handoff/project
.venv/bin/python -B -I -S C3/test_single_file_retry_observation.py
.venv/bin/python -B experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py -q --import-mode=importlib --basetemp C3/cpu-native-tmp --junitxml C3/cpu-native.xml tests/prefix_io_v1_p4_policy tests/prefix_io_v1_p4_02_bridge tests/prefix_io_v1_p4_stage tests/prefix_io_v1_p4_eta tests/prefix_io_v1_p4_measurements tests/prefix_io_v1_p4_load tests/prefix_io_v1_p4_next_day tests/prefix_io_v1_p4_hybrid tests/prefix_io_v1_p4_observation
.venv/bin/python -B -I -S D3/control_p4_single_file.py freeze
.venv/bin/python -B -I -S D3/control_p4_single_file.py prepare --mode off
.venv/bin/python -B -I -S D3/control_p4_single_file.py launch --mode off
.venv/bin/python -B -I -S D3/control_p4_single_file.py after --mode off
```

其中 C3=`artifacts/prefix_io_v1/server11-p4-single-file-candidate-v3-20261003`，D3=`artifacts/prefix_io_v1/server11-p4-single-file-runtime-v3-20261003`。shadow/on 依次执行同一 prepare/launch/after 流程，每臂均先通过独立原始结果验证。控制器真正启动的是既有守护器：

```text
.venv/bin/python -B experiments/prefix_io_v1/scripts/run_gpu_stage.py --permissions-path experiments/prefix_io_v1/configs/permissions.server11.native-cost.yaml --label server11-p4-single-file-on-03 --seconds 300 -- .venv/bin/python -B D3/run_p4_single_file_experiment.py --execute --config D3/CONFIG_on.json
```

服务器 v3 候选 CPU 167/167；原生回归 547/547（3 条既有 JUnit 格式警告）；运行器/控制器/验证器 31/31；独立安全反例 12/12。这些组存在覆盖重叠，不把它们相加冒充唯一测试数量。原生 CPU guard 明确 `cuda_initialized=false`、`gpu_workloads_run=0`。GPU 各臂前后 4,271 项源核验全部通过。

真实 GPU 守护作业记录如下。**13 次启动中 10 次退出 0、3 次早期失败；不称 13 次成功实验。** 校准作业 04/05 各包含六个独立模型窗口。

| 作业 | 计账秒数 | 退出码 | 原会话 |
|---|---:|---:|---|
| server11-native-cost-six-window-01 | 0.885 | 1 | 已排空 |
| server11-native-cost-six-window-02 | 105.525 | 1 | 已排空 |
| server11-native-cost-six-window-03 | 107.878 | 1 | 已排空 |
| server11-native-cost-six-window-04 | 663.189 | 0 | 已排空 |
| server11-native-cost-six-window-05 | 658.042 | 0 | 已排空 |
| server11-p4-single-file-off-01 | 122.000 | 0 | 已排空 |
| server11-p4-single-file-off-02 | 123.327 | 0 | 已排空 |
| server11-p4-single-file-off-03 | 123.399 | 0 | 已排空 |
| server11-p4-single-file-on-02 | 123.058 | 0 | 已排空 |
| server11-p4-single-file-on-03 | 123.110 | 0 | 已排空 |
| server11-p4-single-file-shadow-01 | 130.910 | 0 | 已排空 |
| server11-p4-single-file-shadow-02 | 123.786 | 0 | 已排空 |
| server11-p4-single-file-shadow-03 | 123.627 | 0 | 已排空 |

## 6. 预算、存储和备份

本轮新增守护器计时 2528.737 s（42.15 分钟）；继承 18,829.894 s；累计 21358.632 / 28,800 s，剩余 7441.368 s（2.07 小时）。这是任务 GPU 作业预算，不是 AutoDL 整台实例的租用计费时长。

归档后数据盘 150 GiB，剩余 57.75 GiB；保留 8 GiB 底线。未删除原始数据，未下载模型，未改系统、驱动或已安装包。当前不需要继续扩容。

原始 KV、日志、源码、失败结果、权限和预算快照已分五包保存到本机；各包逐成员 SHA-256 全部通过。排除可重建 runtime-cache、pycache 和软链接别名，不删除服务器原文件；模型权重仍在服务器，归档保留其资产锁而不重复打包权重。

| 本机归档 | 文件数 | SHA-256 |
|---|---:|---|
| SERVER11_RAW_EVIDENCE.tar.gz | 711 | fa9ef08980dc64670af8a66f0ad433ba4b1224bf636c6c46dfac2bd7615ba32e |
| v5/SERVER11_RAW_DELTA.tar.gz | 404 | 37cbc1e46c7971fc39bb32d2cb40c1e395953e5874bce58d4f324c9b932b8f94 |
| p4_v1/SERVER11_RAW_DELTA.tar.gz | 1050 | d08a5a6f3c55225c177129e72a8964fdd2811ad593a22c3cc10603f3bc4b6db8 |
| p4_v2/SERVER11_RAW_DELTA.tar.gz | 1147 | d155a89f6724e46a3824ea749ff94e10dc5e23eca26397f7342bf3fe4efbfac7 |
| p4_v3/SERVER11_RAW_DELTA.tar.gz | 1172 | 47063c595abd84c3521b925b02fabf6a7849c7562d60c889ab0061ca53486132 |

文件数按各包计，包含重复的权限/台账等，不表示唯一文件总数。各包旁有原始 manifest、receipt 和 LOCAL_ARCHIVE_VERIFICATION.json。最后状态见 [FINAL_SERVER_STATE.json](C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/server11_results/FINAL_SERVER_STATE.json)。

## 7. 下一允许阶段与停止决定

本轮预先限定的最后三臂运行已完成，依据未获得净收益的结果停止 GPU 调参。默认路径保留 off；有限 on 仍属开发实验，不具备生产效果资格。

下一阶段仍为 P4 开发层的证据补全：若继续研究，应先在 CPU 上明确剩余热路径、冻结有重复与顺序平衡的对照方案，并针对 D/J 建立真实资源等待和释放证据。业务 SLO 和正常负载尚未确定，不能直接报 P5 goodput 或宣称论文级提升。既有 P3 强简单基线的不利结果也未被本轮推翻。

当前结论足以停止为本轮验证继续付费：**环境和有限功能链路可运行，当前方法在已测试条件下没有加速优势；完整研究有效性仍未被证明。**

