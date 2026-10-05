# C5 通知候选：服务器 CPU 验证交付

本轮完成了受限范围内的实现、独立审查及一次冻结 CPU 对照。主场景的控制线程 CPU 开销明显下降，模拟语义全部通过；完整升级资格未通过，原因是当前半核配额使两个较长场景的延迟判断资源受限。保留 off 默认，不启动 GPU，不选择性删样或重复正式运行。新候选尚不能证明端到端推理加速、完整 P4 或论文级收益。

## 实际实现与边界

服务器为用户提供的 server11；项目目录 `/root/autodl-tmp/prefix-io-v1-handoff/project`。C5 位于 `artifacts/prefix_io_v1/server11-p4-notification-candidate-v5-cpu-20261003/`，C4 冻结基线和作者源文件没有改动。

实际运行时改动只有候选 overlay 的 `py_kvcache/reactor.py` 与 `native_full_step_collector.py`。复用原 Queue 和原完整 pump，在已发生延期且仅有一个安全可选 ready 工作的窄域等待原期限以内的通知；唤醒后重新走原 intake、live 状态和资源检查。结束 record、观测故障、detach、scheduled-load 变化可解除等待，mandatory 与 STOP 保留原接口和完成链。通知没有代替 CUDA 完成、资源释放或准入决定。

原 `_pump_once`、调度、完成、成本策略、缓存引擎和执行器保持原源方法；候选原方法修改限于 `_run`、`_intake`、`_prefix_single_file_retry_key`、`publish_p4_scheduled_load`，另加通知辅助方法。共同 C4 修复与新研究候选分开，新接口须显式安装。off/shadow/未安装不进入新等待，保留原行为路径；每轮仍有新的 helper/getattr 快速分支，以及部分初始化/type-check 差异，不能说源码逐字相同或零开销。本轮未单独计量 off 微开销。

## 服务器实际命令与测试

所有命令使用 `ROOT/.venv/bin/python -B -I -S`，`ROOT` 为上述绝对项目目录，且 `CUDA_VISIBLE_DEVICES=''`。完整参数、环境、stdout、stderr 和真实退出码保存在每组 `_COMMAND.json`、`_RESULT.json`、日志中；复现环境见候选 `SERVER_INVOCATION_NOTES.md`。

| 运行 | 实际结果 | 范围 |
|---|---:|---|
| `run_cpu_candidate_qualification.py --author-root ROOT --previous-root C4 --v3-root C3 --output NEW.json` | 106/106 | 候选等待、策略、观察复用、绑定及原功能 |
| `run_original_cpu_regression.py --project-source ROOT` | 101/101 | 实际 C5 overlay 上的原 policy/ABI；无失败、无跳过 |
| `run_review_cpu.py --candidate C5 --previous-candidate C4 --output NEW` | 32/32 | 独立竞态、mandatory/STOP、错误传播、会计与资源反例 |
| `test_analyzer.py -q`，指定服务器 smoke 数据 | 16/16 | 拒绝漏计费用、假总量、错来源、失效阶段、非法期限及缺资格 |
| 冻结 C4 的 `test_retained_idle_wait.py -q` | 15/15 | 单独保留的原版本断言，不冒称 C5 新测试 |
| `cpu_full_pump_compare.py --mode smoke ...` | 12/12 | 非计分完整控制流程 smoke |
| `cpu_full_pump_compare.py --mode score ...` | 132/132 功能通过 | 一次正式比较；48 次预热与84 次计分，AB/BA固定顺序 |

首轮语义运行保留 `SERVER_CPU_QUALIFICATION.json`：两个旧 empty-window 测试因历史 v1 源码路径环境变量缺失失败。补全 `SERVER11_PREVIOUS_CANDIDATE_DIR=v1` 后，在新结果 `SERVER_CPU_QUALIFICATION_EXPLICIT_PROVENANCE.json` 中106项通过；运行时源码没有修改。v1 历史 fixture 路径与 `--previous-root=C4` 的新对照路径不可混用。正式 CPU 比较没有失败重跑。

## CPU 结果及判定

A=C4 主动轮询，B=C5 通知。实际执行完整原 `_run`/`_pump_once`、原入队/调度/完成链，独立 producer 驱动外部时间。两个线程分别用 `thread_time_ns` 采样；producer 包含每 step 的 `after_prepare` 注册、end.record 与通知开销。共同透明计数插桩开销也计入，初始化及进程导入不计入。物理 AIO、CUDA、staging pool 和 payload 为明确 CPU fixture，未执行真实物理 I/O。

| 场景 | 配对数 | A 两线程 CPU 中位数 | B 两线程 CPU 中位数 | 配对 CPU 中位差 B−A | 两臂均无限流配对 |
|---|---:|---:|---:|---:|---:|
| step end 40 ms，主结果 | 12 | 33.081 ms | 8.747 ms | −24.310 ms | 12/12 |
| step end 20 ms | 6 | 17.839 ms | 8.552 ms | −9.261 ms | 6/6 |
| step end 80 ms | 6 | 51.810 ms | 8.514 ms | −43.264 ms | 1/6，未达4/6 |
| mandatory 20 ms | 6 | 18.008 ms | 9.009 ms | −9.028 ms | 6/6 |
| STOP 20 ms | 6 | 17.670 ms | 8.471 ms | −9.147 ms | 6/6 |
| 原100 ms deadline | 6 | 63.876 ms | 8.867 ms | −55.073 ms | 2/6，未达4/6 |

主结果12/12对两线程总CPU下降，前后半均通过；两臂中位数相比较约下降73.6%，仅为该合成控制负载的描述值。主场景完整 pump 次数中位数165→16，wall 配对中位增量0.150 ms。mandatory 和 STOP 入队到原进展的配对中位增量分别0.090/0.102 ms；均未出现提前进展导致测量资格不足的样本。原到达时刻起100 ms期限没有重置或延长，候选没有早于期限提交。

11项单独的 CPU/延迟数值门槛均通过，独立竞态源锁通过。但当前 `cpu.max=50000 100000`（0.5核）环境中，80 ms场景与100 ms deadline场景两臂均无新增限流的配对数不足。冻结协议要求各至少4/6，因此 `CPU_and_latency_gates_passed=false`、`CPU_candidate_qualified=false`。原始限流样本全部参与统计，没有筛掉重算。此项属于环境资格不足，不能说候选语义失败，也不能越过门槛宣告通过。

分析真值为 `SERVER_SCORE_ANALYSIS.json`：`functional_pass=true`、`scored_complete=true`、`trial_count=132`、`latency_measurement_ineligible=[]`、`decision=RETAIN_OFF_STOP_OR_RESOURCE_LIMITED`。冻结协议 SHA-256 为 `6c00cf687f4903988a363e18a178ccca413076ced81920271bf9c962abde1218`。

## 证据、真实 GPU 情况与下一允许阶段

三组新增服务器目录分别保存候选源码与测试、独立审查、CPU benchmark。完整 patch、阶段收尾核验、可复现结果与日志已封存到 C5_CPU_EVIDENCE.tar.gz，448 个数据文件、4,179,417 字节原始数据均已在本地逐文件 SHA 核验；本地对原始数据重新执行冻结分析器，与服务器全部配对、门槛和判定一致。

本轮实际 GPU 操作 **0**，未导入 torch/vLLM/CUDA 后端，未加载模型、下载、删实验数据、修改系统/驱动/配额或租用 GPU。最终收尾核验确认：原 GPU 预算账本保持原始字节、累计22380.561257688794秒和无活动预留；不能用旧C4 receipt替代C5资格。剩余原8小时预算约6419.439秒，本轮不消耗它。

当前不建议为这轮结果租用 GPU。按冻结协议保留 off，并停止当前半核环境下的升级及重复比较。若继续研究，先明确能提供不触发该限流的现有CPU环境，并以新的预注册版本补齐环境资格；保持所有已得数据和阈值，不修改本轮结果。GPU投入仍需独立确认真实可控瓶颈、C5真实生命周期/成本路径资格和限定授权；当前均没有补齐。CPU省时不能直接折算成推理吞吐提升。此前GPU负结果和完整P4/P5未证实的状态继续保留，不能以本轮CPU结果覆盖。

## 已完成的交付封存

归档 SHA-256：`05f6cbacd2728434d6213b28187fa685d1c42bf94cb615c163df092310e03848`；嵌入清单 SHA-256：`b1304e585ae70e5f6d61301f67c479e59eca4c5080f2fd7bbb177b85c51c6107`。归档大小483,941字节。服务器与本地保存同一封存快照，448份文件全部一致。源补丁位于 `server_replay/delivery/C4_TO_C5_CPU_CANDIDATE.patch`，服务器实际配对结果位于 `server_replay/benchmark/SERVER_SCORE_ANALYSIS.json`，最终阶段核验位于 `server_replay/delivery/SESSION_AFTER_CPU.json`。本地字节核验 `LOCAL_DELIVERY_VERIFICATION.json` 及确定性分析回放 `LOCAL_SCORE_REPLAY.json` 为封存后的追加回执，已同步回服务器；它们不属于448份原始快照，未重新计算或替换原归档。
