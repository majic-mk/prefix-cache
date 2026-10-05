# 下一个真实 I 开发与效果入口：只读审查

日期：2026-10-05。此次仅读取本机实际源码和已有证据，并新增本报告及对应 JSON；源码修改 0、RPC 0、GPU 操作 0。父任务报告 cal05 六窗成本采集正在运行；本审查未取得完整 V9 锁和 cal05 GPU 结果，因此不认定其发行成功。机器可读证据为 [NEXT_ACTUAL_STRATEGY_ENTRY_REVIEW.json](</C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server12_candidates/strong_gpu_system_verification_20261005/NEXT_ACTUAL_STRATEGY_ENTRY_REVIEW.json>)，包含 9 份实际源码的字节数、SHA-256 和具体行证据。

即使 cal05 成功，当前入口也不能直接启动正式 I/U 效果实验。首先还须补齐 collector V2 接线、真正的正式 trace 消费与 U 对照入口、独立数字 deadline/SLO，并取得真实 I development shadow 的控制预留证据。这些缺口不能通过重标签旧资格请求、表 JSON 的 ready 标志或 CPU 测试成功替代。

## 1. 当前入口中的实际硬缺口

| 缺口 | 实际源码或证据 | 对下一阶段的影响 |
| --- | --- | --- |
| collector 不一致 | [strong_native_cost_runner_v7.py](</C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server12_candidates/gpu_prerental_preparation_20261004/runner/strong_native_cost_runner_v7.py:83>) 将校准锁定到 `bounded_native_full_step_collector_v2.py`；[native_runtime.py](</C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server12_candidates/gpu_prerental_preparation_20261004/runner/native_runtime.py:280>) 仍选旧 collector，302 行要求其 ref 与成本 descriptor 严格相等 | 当前 I 入口将在模型导入后拒绝这个真实 V2 成本表。应在 CPU 上新增 runtime 文件，选取同一个已校准 V2，保留旧文件字节；不得放宽相等检查或另选旧表 |
| 正式 manifest 没有消费者 | [strong_trace_runner.py](</C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server12_candidates/gpu_prerental_preparation_20261004/runner/strong_trace_runner.py:169>) 只接受两类 qualification schema；176 行强制 `fit_or_evaluation_input_allowed=false`，223 行强制 `split=qualification`、family 为 null | 已存在的正式 `freeze_trace` 输出不能直接由当前 runner 执行；必须新增仅负责输入验证和调度原 engine 的薄 wrapper |
| U 效果入口缺失 | 同文件 248–250 行仅允许 qualification/off/U、shadow/shadow/I、development/shadow/I、effect/on/I | 不能完成真正相同自然工作流的 I 与强 U 评估。需新增 effect/off/U 合同并验证同源、同初始状态、相同到达顺序和冻结配对顺序 |
| 独立数字 deadline 未绑定 | [activation_request.py](</C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server12_candidates/gpu_prerental_preparation_20261004/runner/activation_request.py:43>) 要求前瞻独立 deadline declaration 与 byte-bound authority | 用户同意推进不等于提供时延要求。不得从 Amax、已观察性能或调整后预算反推 deadline |
| 自然数据与正式 SLO 未绑定 | 原完整自然数据、实际 tokenizer receipt、family provenance、连续且互斥的 calibration/development/evaluation 切分尚不存在；TTFT 和 request ITL P95 也没有独立数字 | 目前 12 条原 P3 数据仍是 controlled qualification，不能成为论文效果数据。缺 SLO 时即使诊断开发能运行，也不能计算正式 goodput |
| 真实 I reserve 尚缺 | [reserve_join.py](</C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server12_candidates/gpu_prerental_preparation_20261004/activation/control_observation/reserve_join.py:139>) 要求真正 I lookup/binding 路径及非空、完整的自然 eligible preview 子集 | 必须先进行实际 development/shadow/I，再原 guard 完成后 CPU join。没有自然可用 preview、覆盖丢失或未知时停止；不得注入候选、删帧或假造控制窗口 |

新增 development config/namespace、cal05 原 plan/raw/guard/intent refs、独立 deadline、当前源闭包、历史与覆盖合同也尚需实际绑定。development 的所有后置 reserve/proof refs 应为 null；effect 则必须引用实际完成的开发输出，纯读重验原 reserve，不得在 effect 启动时临时创建资格。

目前旧 U 资格证明了原模型 12 个请求各生成完整 128 tokens 和正常退出；它没有证明完整观察资格。`OFF_MIXED_CONTEXT_OBSERVATION_ANALYSIS.json` 记录原 engine 1047 次 step、仅保留 52 帧、CUDA event witness 为 0、`full_capture.valid=false`。cal05 使用的 V2 只处理第一模型帧之前可证明的零工作调用，不意味着后续混合请求流的全帧采集已成立。真实开发仍须完整验证所有支持的原调用；额外零调度步、空输出 prefill 或其他不支持形状应保守 UNKNOWN，不删步制造通过。

## 2. V9 成本表和新增 V10 host wrapper 的边界

**仅因为新增 V10 host 文件使整个运行锁 SHA 改变，不需要重新 GPU 校准。** 现有合同将成本资格锚定在原校准 V9 锁，同时允许后续源锁完整继承 V9 的全部原行并新增文件。

具体依据：

- [gpu_cell_issuer.py](</C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server12_candidates/gpu_prerental_preparation_20261004/activation/source/prefix_io_control/gpu_cell_issuer.py:306>) 将 plan 的原校准 `source_lock_ref.sha256` 与独立期望值比较；367 行将它保存在私有 identity。370–371 行明确将 `runtime_pair_ref` 排除在必须逐叶一致的 identity tuple 之外。
- [activation_request.py](</C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server12_candidates/gpu_prerental_preparation_20261004/runner/activation_request.py:39>) 要求新 runtime refs 中每个 V9 原路径/字节数/SHA 完全一致；它不是要求新整体锁 SHA 等于 V9。
- [finite_current_binding.py](</C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server12_candidates/gpu_prerental_preparation_20261004/runner/finite_current_binding.py:173>) 仍要求原 calibration lock 等于私有 identity 锚、该锁文件也在新 refs 中、原全部叶精确继承，以及所需 issuer/native/collector/model/geometry 叶完全一致。

因此安全实现是新增不同文件名的 host wrapper、runtime、配置与当前 V10 锁，保留全部 V9 原文件；启动时重新从真实 cal05 六窗 raw、独立 heldout、原 guard/source/SDK 证据发行私有 table，实际 runtime 再校验它。`ACTUAL_COST_ISSUER_REPORT.json` 只是报告，不能还原进程内 capability。

若修改同一路径 V9 文件，其祖先行相等检查必失败；若改变实际 model、geometry、collector V2、CUDAEvent、native owner、GPU UUID、eager/kernel、KV/cache/staging 或 common engine/sampling 域，旧表也不能复用。新增 wrapper 不等于授权跨域。

有一处容易误改：现有自然 text lane 要求 `skip_tokenizer_init=false`，而 cal05 的原 token-ID lane 为 true；common-domain 归一化不排除此字段。将它改为 false 会改变真实域。若使用真正来源闭合、实际 tokenizer 结果中的自然 prompt IDs，可以在新薄消费者中保留原 token-ID lane 与 true，但必须重放 tokenizer/source/family 合同；不能随意生成 IDs 或忽略域字段。这个路线仍只使用原精确 cells，任何未匹配状态回到 U。

## 3. 下一次真实 I development 的资格与预留

独立 deadline 必须是正整数 `full_control_window_deadline_ns`、来源于开发前独立要求、authority 文件字节闭合；声明时间须在同一 host/boot/CLOCK_MONOTONIC 域且先于开发。formal goodput 另需独立正整数 `TTFT_ns` 和 `request_ITL_P95_ns`。仅诊断开发时 SLO 可以为 null，但 genuine deadline 仍是硬条件。父任务已询问具体数字，本审查不重复询问或替用户选择。

真实 shadow 必须消费真实 table，执行同 I 的 finite lookup/binding/controller 路径，仅观察并返回原 native fallback。完成原 guard、OS 会话排空后，`reserve_join.verify_development_inputs` 重验原 plan/intent、完整 outputs/CUDA witnesses、实际 SDK/driver、四类 host 控制区间、全部原步骤与自然出现的 eligible preview 窗口。preview 是原自然尝试的完整非空子集，不要求每个步骤都有 preview，也不允许遗漏实际尝试。

reserve 使用全体真实 host 区间保守 union 最大值加前瞻不确定度；没有用 wall step 减 CUDA 时间混合时钟。最终外部 deadline 减 actual reserve 得到控制预算，且仅应用于同一已覆盖精确 cell。`verify_existing_native_reserve` 为纯读重放入口，effect 应消费它的结果而不是自建 native=True 的 receipt。

真实 cell/fallback 率、自然可用 preview 数和性能收益目前未知。若没有正常可用的 ordinary candidate，应交付“在该冻结流与精确域没有机会”的结论；不能改到达率、增加暖机、改变初始缓存、重排请求或人为 throttle 获得收益。原 10% 投入门槛和 2% 低竞争工程 guard 继续有效，均是决策阈值，不是承诺结果。

## 4. 已有命令与还未成为可执行配置的部分

服务器项目根为 `/root/autodl-tmp/prefix-io-v1-handoff/project`。下面命令是从已有 CLI 核实的接口说明，**本审查未执行这些服务器命令**。含占位符的命令只有相应真实证据存在并闭合之后才能执行。

cal05 六窗实际成功、原 guard 完成和会话排空后，可调用当前 CPU 关闭器：

```sh
CUDA_VISIBLE_DEVICES= .venv/bin/python -B -I -S \
  artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005/close_actual_exact_cost_cpu_v1.py \
  --project-root /root/autodl-tmp/prefix-io-v1-handoff/project \
  --attempt 5 --wrapper strong_native_cost_runner_v7.py \
  --expected-plan-sha256 1538d6529653a17bd1b8322b40b6b21589b5db79837544dd50f3d8981211776c \
  --expected-guard-sha256 <真实已完成-cal05-guard-SHA256>
```

已核实 actual CPU config metadata 中 raw05/BOUND_PLAN.json 为 166947 bytes、上述 SHA，raw05/CONFIG.json 为 2600 bytes、SHA `73e859330320250754a67aff383cb79fa59a0110e9cf5ca5cbf7ff17821c92a7`。该 metadata 标明 model execution 尚未开始，不是 cal05 完成结果。

当前 trace runner 的 CPU 入口为 `--project <ROOT> --config <new-config> --preflight` 或 `--full-source`，GPU 入口为 `--execute`，由原 `run_gpu_stage.py --permissions-path <same-effective-permission> --label <fresh-run> --seconds 300 -- <execute-command>` 包住。已有 `entry_control_v4.py` 的 launch 仍是固定旧 off01 namespace，不是 development/evaluation orchestrator。新增正式 consumer/runtime 后应 CPU 检查新文件及新配置，再使用其实际输出的 guarded command；不能把当前未修入口描述为 ready。

协议已有实际自然数据检查与冻结 CLI：

```sh
CUDA_VISIBLE_DEVICES= .venv/bin/python -B -I -S \
  artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/protocol/prerental_protocol.py \
  inspect --dataset <实际原数据> --author-trace <原shared_storage_trace_replay.py> \
  --author-common <原prefix_cache_common.py> --output <新inspection>

CUDA_VISIBLE_DEVICES= .venv/bin/python -B -I -S \
  artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/protocol/prerental_protocol.py \
  freeze-trace --dataset <同一实际原数据> --author-trace <同一原parser> \
  --author-common <同一原common> --declaration <前瞻独立切分声明> \
  --tokenizer-receipt <实际闭合tokenizer结果> --output <新正式frozen-manifest>
```

当前这些自然数据占位符尚无可填真实 refs。父任务已分派 additive 正式 consumer 工作；本报告只记录审查时缺口，不宣称其实现或测试已完成。

## 5. 原累计预算与可串行推进的条件

父任务给出的 cal05 前剩余约 56 分钟属于近似快照，本审查未读取实时 ledger。cal05 执行上限 900 秒加原 guard 收尾预留 20 秒，共 920 秒；按最大使用推算后约余 2440 秒，即 40 分 40 秒。它不是下一次 launch 的授权数据；须等会话排空后读取原实际 ledger。

当前 `strong_trace_runner.MAX_SECONDS=300`，一次短作业最多预留 **300+20=320 秒**。400 秒执行加 20 秒预留的配置会在当前入口 CPU 拒绝，不能按 420 秒伪称已有支持。

旧 `i_pilot_protocol` 的 9 槽 3780 秒由 1 次 1200+20 校准及 8 次 300+20 短作业组成；它不是 9×420，也不是当前新强域的完整计划。其旧 context144/planner-off 校准不能追加烧卡或替代 cal05。

原 [run_gpu_stage.py](</C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server12_candidates/gpu_prerental_preparation_20261004/source_inputs/run_gpu_stage.py:176>) 在锁内逐作业检查没有 active reservation，且 `used+seconds+20 <= 28800`。268 行按真实 monotonic elapsed 含收尾累加，271 行只在实际会话排空后清除 reservation。因此可以在所有其他资格满足时，逐次完整预留下一项 320 秒；未用的计划时间不会扣成实际用时。无需先声称剩余够旧全组 3780 秒，也不得扩大原 8 小时预算。

这只能说明某一次合法下一作业可以串行启动，不能保证整个开发加正式多组 I/U 对照有足够预算。须在真实 cal05 使用确定后冻结新的先决关系、配对次数、单项上限与失败分母；若下一项完整预留不够或条件失败，停止。部分对照结果不能称为论文级完成证据。

实际审查执行了 PowerShell `Get-Content` 和 `rg -n`，核实入口、schema、身份继承和原预算代码。此轮没有运行或重复 CPU 测试；父任务的 CPU 171 项通过属于父任务证据，不作为此次新增测试结果。本报告对正式方法效果的结论仍为 **未验证**，并未将源审查、CPU 准备或系统资格标记为性能提升。
