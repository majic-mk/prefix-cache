# I-only pilot CPU 前瞻准备

本目录完成协议、预算清单、真实逐 token 时刻校验与实际入口证据的 CPU 连接验证器。它没有运行 GPU，也没有把当前失败资格转成策略效果。

`I_PILOT_PROTOCOL.json` 固定新请求家庭、运行顺序和全部停止规则。校准首 token 为 40100/41100/42100；开发为 43100/44100；评估为 45100/46100。每个家庭的首 token 不同，使完整精确前缀家族分开；输出固定 128 tokens 是机制诊断，不是自然问答。校准仅拟合前两对，第三对只验证。旧 18100/19100/20100/28100 记录和旧 held-out 不参与新拟合。

旧 U=16.238752 ms、A-only 工程预算=13.171328 ms 和失败记录原样保留。新表另建、条件另锁。新结果不能使旧失败变成通过。开发的独立 deadline、控制/采样/输出余量和 internal budget 目前均为 null；必须有前瞻声明及开发实测，且在首次 on 前冻结。缺独立 deadline 时仅允许 off/shadow 诊断，不运行普通 on 效果实验。不得拿 A-max 直接当服务 SLO，也不得观察 B/on/评估结果后放宽 deadline。

正式 TTFT/ITL SLO 仍为 null，因此禁止正式 goodput 结论。后续可以在门槛通过后报告配对的原始 TTFT、请求内 ITL P95、完整排空 makespan 和吞吐；两对结果不能证明论文收益、总体尾部保证或两类正常负载的重复优势。

## 实际现有驱动的限制

保留的校准驱动确实有 3 对、6 个独立模型子进程窗口，顺序 A/B、B/A、A/B。所有子进程继承唯一原 GPU guard 的 session；原 guard 负责累计 ledger、超时和安全收尾。保留 single-file 驱动在源码中存在 off/shadow/on，但最新 repeatability controller 只允许 off，且配置固定在旧失败 gate。不能直接把这些旧脚本变成新的效应资格。

校准及目前 single-file 驱动的 `config_for_engine()` 设置 `load_planner='off'`，只用于直接 preload 的机制诊断。它们不能代表完整、充分调优的 U 基线。未来 U/I 配对必须在两臂统一保留原 LoadPlanner、break-even、Prefix identity、预加载、共享 staging、复制合并和异步流水线；该完整请求流入口尚未绑定。F/P 强简单基线仍是论文比较前的独立要求。本轮 CPU_READY 只覆盖新校准入口。

现有 frontend collector 为累计输出返回中的每个新增 token 填同一个接收时间。只有后续核验每次原生输出恰好新增一个内容 token，才可以使用其中逐 token ITL。`token_metrics()` 显式检查原生 step 的累计计数、序号、时刻和完整 128 个输出事件；一次返回多个 token 必须拒绝个体 ITL，不拆分 chunk gap，也不重新 tokenize 造时刻。时刻是原生 frontend 输出接收时间，不能冒称客户端发送/网络时间。GPU Event 完整 execute_model→sample_tokens 成本是不同口径。

## 前瞻执行清单与预算

1. `cal01`：迁移后的现成校准驱动，off，6 窗口，执行 1200 秒 + 20 秒收尾。
2. `dev01/dev02`：2 个独立开发家庭 off，分别 300 + 20 秒。
3. `shadow01`：开发家庭 dev0 的 shadow，300 + 20 秒。
4. 独立 deadline、开发余量与成本覆盖冻结；无普通可行候选即停止。
5. `on01`：开发家庭 dev1 的 on 生命周期资格，300 + 20 秒。
6. 仅在先决条件通过后执行 eval0 的 off→on、eval1 的 on→off，各 300 + 20 秒。

九槽最高预留 3780 秒，初始原 ledger 剩余 4121.104761 秒，计划余量 341.104761 秒。这是计划上限，不是运行承诺，也不是第二个 ledger。每次必须读取原累计 ledger 和原 permissions 并由原 guard 预留；失败消耗该槽，重试/替换槽为零。PRIMARY 单个短作业预留 128 MiB、至少保留 8 GiB 空闲；无下载、系统/驱动修改或已有数据删除。用户持续 GPU 授权保持，不按每个源锁重新申请授权。

九槽中的 dev/on/eval 当前只有清单，尚无已绑定的效果 runner。当前可执行入口准备位置是 `artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004/calibration_v2/`。校准配置和完整 before/launch/after 文件名沿用现成 controller，未创建新 GPU 所有权层。

```text
.venv/bin/python -B experiments/prefix_io_v1/scripts/run_gpu_stage.py
  --permissions-path artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004/calibration_v2/EFFECTIVE_GPU_PERMISSION.json
  --label server12-i-pilot-cal01 --seconds 1200 --
  .venv/bin/python -B artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004/calibration_v2/run_native_cost_experiment.py
  --execute --config artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004/calibration_v2/NATIVE_COST_CONFIG.json
```

上述仅是确切 guard 命令模板，不能在缺 site binding、有效权限、真实 GPU、目录空间或 source proof 时直接运行。实际校准 controller 仍负责 freeze/bind/scope/before/launch/after 原链路；本目录不会调用 launch。

## 可执行 CPU 验证

`prepare_protocol_cpu.py` 读取完整规范的保留文本，核验 bytes/SHA，AST 检查已保留真实 runner，再以新文件写入协议/能力矩阵。`i_pilot_protocol.py --protocol ...` 验证精确家庭、顺序、原预算、无重试及无改旧阈值，并输出 `CPU_PROTOCOL_COMPLETE_RUNTIME_BINDING_REQUIRED` / `BLOCKED_FOR_EFFECT`。

提供 `--entry-references`（四个实际服务器文件 refs：source_lock_ref、source_proof_ref、cpu_test_result_ref、preflight_ref）和 `--project-root` 后，验证器核验真实字节、必要 runner/guard pin、当前模型 manifest、完整 source rows proof、实际 CPU 测试输出，以及原 idle ledger 可预留 cal01。只有这条字节连接通过，才输出 `CPU_READY_FOR_GPU_CALIBRATION`。设备可仍不可用；该状态表示 CPU 校准准备完成，GPU launch 仍须现场原 guard 复核。单纯布尔声明不能取得该状态。

`validate_development_freeze()` 检查前瞻 deadline 在开发记录前声明、开发余量被扣除、freeze 在 on 前完成、无 held-out/evaluation 拟合、无事后放宽。upper 超出预算时输出 `STOP_NO_CURRENT_INTERFERENCE_ADMISSION_OPPORTUNITY`，不调阈值强行准入。

`test_i_pilot_protocol.py` 所有事件时钟与 entry join stub 都明确是 CPU 拒绝/算术 fixture；它们不写入校准表，不生成 GPU 记录，不充当方法效果。实际测试数量和命令见 `CPU_PROTOCOL_FINAL_TEST_RESULT.json` 及 stdout/stderr 文件。

下一允许阶段是：在真正冻结的新校准源码、CPU 验证与现场 guard preflight 通过后，仅执行 cal01 的 GPU 校准诊断。正式效果阶段继续阻塞在独立开发预算、普通可行候选、on 生命周期、完整 U/I 请求流、自然负载和配对评估证据。I-only 不要求先完成依赖排序，但不能用这一机制诊断证明真实 GPU ownership 释放收益。
