# 本轮 GPU 系统交付草稿：待 CAL05 最终记录补齐

本文件是交付结构草稿，不是 CAL05 成功报告。基线与 CAL01–CAL04 的已知事实如下；CAL05 的最终退出、六窗、源码核验、公开发行器与预算结果必须由根代理依据新记录填写。起草动作没有 RPC、GPU 或模型执行，不能把本动作 GPU=0 写成整个任务 GPU=0。

## 已经取得的结果

当前服务器实际运行了原 U 基线，12 个 512-token 请求各完成 128 个输出 tokens，共 1,536 tokens；原缓存/planner/preload/共享 staging 路径存在，记录了实际缓存命中和 SSD 写入，原 shutdown、native tail 与作业结束后的 OS 会话排空有记录。基线的成本、I 策略和性能提升资格仍为 false。[基线原输出](OFF_ACTUAL_REQUEST_OUTPUTS.json)、[原 child](OFF_ACTUAL_CHILD_RESULT.json)、[原 guard](OFF_ORIGINAL_GUARD_RESULT.json)、[parent join](OFF_GUARD_CLOSED_QUALIFICATION.json)

CAL01–CAL03 均真实尝试并失败，没有完成六窗校准。具体停止于 config 相对路径、cached frontend 512 对预期 496 的契约，以及 collector completeness 检查；CAL03 没有保存 capture，不能定位坏帧或发行成本资格。原始失败记录、账本计时和释放结果保留。[前三次原始证据复核](GPU_BASELINE_AND_FIRST_THREE_ATTEMPTS_REVIEW.md)、[统计与全部文件 SHA](GPU_BASELINE_AND_FIRST_THREE_ATTEMPTS_REVIEW.json)

CAL04 已真正保存首个 A 窗口的 128 个 model frames 和 128 个 CUDA witnesses。现场 frontend cached 报告为 512，但真实首帧 execution pre-context 为 **496**、prefill 为 **16**；measured offset16 的 context 仍为 **527**。与原先预登记的 initial pre-context 511 不一致，原严格 validator 拒绝 `actual complete cold/decode load differs`。这属于条件契约不符，并未得到性能失败结论，也没有使用 CAL04 Event duration 来选择参数或调上界。[实际布局读数](CAL04_FIRST_WINDOW_REAL_GEOMETRY_RPC_RESULT.json)、[原严格校验与停止理由](CAL04_ACTUAL_EARLY_STOP_RPC_RESULT.json)

【待根代理补齐】CAL04 的第二个 B 窗口已完成退出，但其 capture、原 I/O 与 lifecycle 是否通过严格回放，需列出实际验证结果及证据 refs。不能只凭 exit0 宣称发行资格。第三个进程已启动后被 SIGTERM 停止，未产生 child report；不能声称第三窗口自然 shutdown 或六窗完成。

CAL04 的原 guard 最终 exit=143、child_exit=-15、interrupted_signal=15，未超时；计入原 GPU 账本 **286.4334052987397 秒**，OS 会话排空，账本 active=None，随后 nvidia-smi compute 查询为空。这里的 OS 清理证据不替代第三个模型进程自然 shutdown 证据。停止时历史剩余额度为 3,384.665510051418 秒，之后的 CAL05 另行扣账；该数值不能写成最终当前剩余额度。[实际停止与排空](CAL04_ACTUAL_STOP_DRAIN_RPC_RESULT.json)

## 实际代码修改与保留边界

本轮对现成 py-kvcache、作者 vLLM 和既有校验器进行增量接线，旧冻结版本与作者源保留。按最终文件锁确认以下具体改动及对应 SHA：

- 相对 config 路径传给原 guard 管理的子入口，仍拒绝逃出 ROOT 的路径。[相对路径 CPU 结果](../gpu_prerental_preparation_20261004/raw_relative_config/LOCAL_CPU_RESULT.json)
- UUID 与实际分配的 NVIDIA 数字设备节点绑定，保持资源、原权限与预算边界。[设备绑定 CPU 结果](../gpu_prerental_preparation_20261004/raw_device_binding/LOCAL_CPU_RESULT.json)
- collector 对作者源码认证的无 scheduled-work 调用保留独立 NO_FORWARD 诊断，并委托原 collector；不伪造 model frame/CUDA Event，不改实际 model ordinals、sample 或 native ownership。实际 collector 与其原 delegate 都进入冻结闭包。[NO_FORWARD CPU 结果](../gpu_prerental_preparation_20261004/raw_no_forward_observation/LOCAL_CPU_RESULT_FINAL.json)、[两个实际 source leaves 的绑定结果](../gpu_prerental_preparation_20261004/raw_no_forward_binding/LOCAL_CPU_RESULT.json)
- `runner/strong_native_cost_runner_v7.py` 将 frontend cached512 与本现场实帧 initial execution496 分开预登记；首步 scheduled/prefill16，prompt512、offset16/context527、128输出、917,504-byte 单 SSD read 和两拟合对/一独立 heldout 保持原定值。
- V7 在每个原 child 已完成后、下一子进程启动前，调用既有严格 capture、native I/O、shutdown/drain 校验，条件错误立即停止后续窗口。该早验只核验真实记录，不拟合成本，不改原估计器或 heldout 覆盖门槛。[当前 V7 源](../gpu_prerental_preparation_20261004/runner/strong_native_cost_runner_v7.py)、[针对测试源](../gpu_prerental_preparation_20261004/raw_initial_execution_contract/test_initial_execution_contract_cpu.py)

【待根代理补齐】最终 wrapper、collector/delegate、issuer/validator/estimator、source lock、before/after proof 与权限 refs 的真实 SHA，以及源码/AST 差异审核报告。不要按 wrapper 文件名数字推断对应 source-lock 版本。

共同兼容修复与 I 研究策略保持分开：原精确 Prefix Cache、成本准入、load planner、共享 staging、预加载、复制合并、异步流水线、原模型执行和 owner 生命周期继续复用；关闭 I 策略时保持原 U 路径。没有修改驱动、系统、已安装作者文件或原成本数学；没有通过扩大上界、改变预算或把 heldout 纳入拟合来使资格通过。每条保留结论应由最终实际源锁/AST 审核补齐，而非仅由此草稿宣告。

## 执行命令与 CPU 检查

已实际执行的原 U 子命令来自原 guard 记录，工作目录是 `/root/autodl-tmp/prefix-io-v1-handoff/project`：

```text
.venv/bin/python -B artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/runner/strong_trace_runner.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --config artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/live-off01/CONFIG.json --execute
```

CAL04 的实际子命令同样受原 guard 管理，不能绕过 guard 直接复跑：

```text
.venv/bin/python -B artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/runner/strong_native_cost_runner_v6.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --config artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005/raw04/CONFIG.json --execute
```

【待根代理补齐】CAL05 原 preflight 的 `original_guard_command` 完整 argv、原 guard 限时/预留，以及真实 freeze/config/preflight/source-before/source-after/公开 issuer CLI 命令。可从 `GPU_STRONG_RAW_20261005_05_COMMAND.json` 和对应 CPU 日志复制实际执行参数；文件不存在时保持待验证，不能将建议命令写作已执行。

已经保存的 CPU 检查分别为设备绑定8项、相对路径3项、早期缓存语义5项、失败诊断5项、NO_FORWARD16项和两个 collector leaves 的绑定5项；这些是各动作的测试结果，不是合并后的一次 GPU 资格测试。早期 511 测试验证的是作者外部异步全命中分支预期，没有证明本现场 measured 首帧必为511；CAL04 的真实496据以修正当前现场契约。上述 CPU 动作 GPU=0 只描述其自身。[各项 CPU 结果见上一节链接]

【待根代理补齐】V7 针对测试的实际本地/服务器计数与 returncode；当前源码列有8项，但未有结果时不能直接写8/8。最终 source-before/source-after 实际逐文件 SHA 数量、漂移/失败列表、封存文件计数与 archive SHA 另行填写。

## CAL05 必须补齐的最终判断

| 项目 | 目前状态 / 最终填写要求 |
| --- | --- |
| guard 实际运行 | 待真实 exit、child_exit、elapsed、signal、session_drained、active=None 与账本增量 |
| 六个 fresh processes | 待逐 child 命令、PID/SID、exit、原 shutdown、native tail；不能把 OS 清理当自然 shutdown |
| 完整测量与输出 | 待每窗128输出/frames/witnesses、AB/BA/AB顺序、输出一致、实际首帧496/prefill16及offset16/context527 |
| 严格条件与 I/O | 待原 validator 对全部帧、真实 zero-existing-I/O、独立物理负载/CQE/staging/释放的回放 |
| 校准来源 | 待仅 CAL05 同一冻结版本的六窗；CAL04 失败/停止数据不复用为 fit，heldout 永不拟合 |
| 公开成本表发行 | 待真实 parent-closed raw/source/guard 与公开 issuer 判定，保留拒绝理由；JSON 摘要没有进程内私有能力 |
| 成本值 | 只有公开发行回放完成后才能填写 baseline/incremental-or-joint/uncertainty/upper 和独立 heldout coverage，不在运行中凭部分 duration 调参 |
| 方法效果 | 无独立策略实验、完整控制成本与独立 SLO 时保持未验证；单元成本发行通过也不等于方法提速 |

若 CAL05 完整记录通过，能够证明该现场精确条件下的有限 SSD read 成本单元与原 engine/cache 生命周期可验证；不能自动泛化到所有上下文、batch/concurrency 或自然流量。若失败，只报告实际失败的执行、条件或 heldout 门槛，不将环境/契约失败称为方法永久不可行。

## 下一允许阶段与预算

用户持久 GPU 授权继续有效，原总8小时预算不扩大；每个新作业仍须实际设备/源/权限/guard/空间核验，失败按有限协议停止。最终交付填写所有已结束实际 guard 的累计和新增秒数、最终 remaining、未授权系统操作为0，不用 CPU 结果的 GPU=0 省略已发生的 GPU 尝试。

独立 deadline/SLO 仍是策略阶段的缺失前提，不能从基线、A-max、CAL05上界或已观察结果倒推。无独立 deadline 时，现有普通 shadow 可以保持 `cost_table=None` 检查观测与回退生命周期，但不能声称真实 finite I 预算消费或策略效果；真实成本消费的 development shadow 需要独立 prospective deadline 与实际表，on 还需要真实控制 reserve 及覆盖子集。无 natural trace 可保留明确标记的受控原 P3 功能检查，不能宣称自然负载 goodput/TTFT/SLO收益。[成本消费接口与边界说明](CPU_REVIEW_COST_ISSUANCE_AND_CACHE_SEMANTICS.md)

最终用户答复应先说明 CAL05 的实际结果和能证明的范围，再给具体改动、运行/测试、证据与剩余前提；不得把本草稿或待补字段作为完成系统验证的证据。
