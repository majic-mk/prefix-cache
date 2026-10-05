# 独立租卡前审阅

本轮审阅仅在本机读取原交接、保留源码和新准备目录；只向 `review/` 新增测试与证据。没有服务器 RPC、GPU、真实模型或原生 I/O 操作，没有修改原冻结文件。此报告不授予 GPU 资格，也不证明方法有或没有性能收益。

## 实际执行的独立核验

| 独立核验 | 本机结果 | 最新证据 |
|---|---:|---|
| 原交接/LoadPlanner/成本域/预算与旧资格合同 | 16 checks 通过，8 个保留输入前后不变；不是单元测试计数 | `PRIOR_PREPARATION_REVIEW_CPU_RESULT_V2.json` |
| 协议拒绝边界 | 7 tests 通过，0 failure/error/skip | `PROTOCOL_INDEPENDENT_REJECTIONS_FINAL.json` |
| 原作者请求 ID 函数正文回放 | 6 tests 通过，internal/external ID 映射正确，原源码不变 | `ORIGINAL_REQUEST_ID_CPU_REPLAY.json` |
| 有限 GPU capability 拒绝及 heldout 算术 | 9 tests 通过，真实已发行能力与 GPU-qualified cells 均为 0 | `FINITE_CAPABILITY_INDEPENDENT_REFUSALS_FINAL.json` |
| 新 collector 安装参数对原 event observer 构造口 | 3 tests 通过；512/4096 steps 保留 pending≤128 | `OBSERVER_CAPACITY_INDEPENDENT_FINAL.json` |
| 真实旧 raw 窗口形状的 external/native ID 适配 | 3 tests 通过；保持原 raw frames 不变，漂移拒绝 | `RAW_REQUEST_ID_MAPPING_INDEPENDENT_FINAL.json` |
| 紧凑 host 四类观察拒绝边界 | 6 tests 通过；CPU fixture 无 GPU 或 reserve 发行资格 | `HOST_CONTROL_REFUSALS_INDEPENDENT_FINAL.json` |
| 可选 host boundary 的 ordinal 失效回退 | 2 tests 通过；原 callable 仍执行且观测失效 | `HOST_BOUNDARY_FALLBACK_INDEPENDENT_FINAL.json` |

合计是 **36 个本机独立单元测试 + 16 个本机审计 checks**。重复复验不重复计数；服务器复验必须由其单独运行记录证明。全部独立执行的 GPU/RPC 数量为 0。初版非有限预算/声明 SLO 问题、review 自身旧路径与本机路径拼写问题、collector 容量错误和 host boundary 阻断原 callable 的错误记录保留，没有将失败记录删除或改成成功。

运行使用 Python 3.12.14、`-B -I -S`；脚本分别为 `test_prerental_rejection_review.py`、`test_original_request_id_review.py`、`test_finite_capability_review.py`、`test_observer_capacity_review.py`、`test_raw_request_id_review.py`、`test_host_control_refusal_review.py`、`test_host_boundary_fallback_review.py` 与 `audit_prior_preparation_cpu.py`。协议测试提供 `--source-dir`，容量和 host 观察测试提供 `--observer-source`，raw ID 测试提供 `--runner-source`，可在服务器显式绑定真实原源文件；保留本机默认值。所有输出为新文件。

## 已确认的证据边界

新 CostTable 公共 mock/conditional 构造仍不具 production 资格。私有能力只在独立冻结的计划/intent/实际原 guard 引用、实际源闭包、完整 128 输出/CUDA 因果见证、原 owner journal/CQE/drain 和两个 calibration + 一个独立 heldout 重验成功后发行。原数值 AST 仅拟合 calibration；heldout 超界拒绝，不抬高上界重新拟合。私有 Python 能力用于防止意外把 CPU 摘要升格，不是硬件签名或任意同进程代码的安全沙箱。

首版真实有限域是 batch=1、active_decode=1、prefill=0、冻结 context、同模型/布局/GPU/source/eager 域、SSD read、ZERO existing I/O 及完整原单位；其他条件为 UNKNOWN 并回原路径。ZERO 必须由实际 owner journal 证明。原 policy 精确 CostTable 类型保留；原 choose_batch 的 production 标记仍关闭，报告明确不声称新 production batching 或融合已经激活。首轮现场策略限定为普通原 preload 的单次 issue/deferral。

现场 binding 已修正 eager 标识、issuer runtime-ref 三元结构、实际 model directory，并增加真实 Triton backend 来源与完整层覆盖检查。源文件的这些门可在 CPU 实现和拒绝测试，当前真正 GPU runtime identity 仍未验证。

同源 query-only collector 默认仍要求 128 步标定证据，4096 仅扩大有界观测。其原 prepare/current-step/start.query/end.record/detach/等待正文保持；新 install 使用 `max_pending=min(128,max_steps)`。没有新增 GPU synchronize，增加观测容量不扩大成本 cell 资格。最终 collector 为 25,070 bytes，SHA-256 `9915ca0c18147eb26f44e2292f541da95e33d0321c6236b224d19d4446e09e9d`。

## 下一步和停止条件

先使用原 guard 执行唯一有界 strong U/off 正常输出与生命周期资格作业，最多执行 300 秒、收尾 20 秒。原 GPU 预算余量快照为 4,121.104761 秒，实际启动必须重新读原累计账本。保留原精确 Prefix Cache、LoadPlanner、成本准入、预加载、共享 staging、融合和异步 I/O。原 LoadPlanner 的 recompute-overpriced 和外推限制仍是基线有效性问题，CPU builder PASS 不代表作者成本曲线已适配当前设备。

不先运行旧 planner=off 的 cal01，不将其单文件 context144/offset16 回执用作 planner-on 强请求流资格。首个方法对照宜为最小 I-only；必须先证明真实 ordinary 候选、实际干扰或资源等待、独立冻结的 deadline/实测 host control reserve 和同域精确成本。没有普通机会、全部 UNKNOWN、只有 mandatory/max_wait、无法形成真实动作差异、heldout 超界、输出改变、收尾失败或预算不足时停止当前 pilot，交付对应的限定负结论。不能据此断言方法永久无效，也不能扩大原研究范围赌收益。

当前受控 P3 输入可用于正常机制与生命周期资格，不能改称自然生产 heldout。自然数据、正式服务 SLO、强域实测成本、真实 on 输出及对照效果尚未完成。即使一个 off 作业成功，也不等于已验证效果；不承诺剩余约 68.7 分钟内完成全部阶段或论文级实验。

当前无卡系统 driver library 为 0-byte 挂载占位，不能报告当前 full SDK/runtime PASS。私有 SDK 源与编译/链接资产的 CPU 核验只能证明对应准备；真实 driver、UUID、CUDA Event/native 初始化仍需 GPU 模式重验，不通过修改系统/驱动或放宽旧 pin 来伪造资格。

## 普通 preload 接线与观测复核

已只读复核有限普通 preload 路和 startup 的实现：原 reactor 的 unbound ordinary preload 原先仅在 single_file receipt 路调用 preview；新接线借读原 _ReadyFd sequence/open_start_ns/hash/preload_info，仅修改原方法中的一个资格 if。私有真实 CostTable 必须先由原 raw verifier 完整发行，然后在模型构造前替换原工厂的 bridge 参数，使用原 owner snapshot 触发 drain 后的附着，原 owner shutdown 后恢复。原容量、ready、progress、max_wait、owner、release、FIFO/融合主体保留；UNKNOWN 回原路径，不注入新 preload。真实普通机会与 exact cell 的交集仍未在 GPU 验证。

旧 raw probe 的 window.request_id/capture.run_id 是外部 ID，prepared rows/outputs 是带 UUID 的 native ID；新打包已分离外部 run_id 与 native request_id，禁止改写原 frames，CPU 回放覆盖两种外部 ID 漂移拒绝。该修复不代表旧原始数据已成为新的 GPU 合格证据。

紧凑 host 观察借原 schedule、refresh_metadata、process_outputs 与原 frontend 书记方法；没有包装整个 step/execute_model/sample_tokens，也不新增 GPU 同步。原 execute_model 在 refresh_metadata 前递增 _profile_step，现有 scheduler-before、sampling/output-after ordinal 映射与原作者源相符。可选 ordinal 不可用时继续执行原函数，同时令观察失效；跨线程、溢出、漏类、混时钟及 CPU fixture 不能形成实际 reserve。

已复审封存的 control observation 接口及最后的 effect 消费口。实际 development reserve 仍需要四类真实 host control 区间、独立 deadline、实际原 guard、完整 CUDA/output/SDK/source closure，以及真实有限 I controller 的 observe-only 路径；CPU 摘要不能发行这份资格。新 reserve_join 将所有原步骤、所有原尝试和完整 128 输出保留，真实 preview interval 与 attempt 一一对应，以 source-bound 当前 tuple、原 CUDA 因果事件和原 owner StageAccounting 查验有限 eligible 子集。不用整段 host step 减去 GPU elapsed，不从 A_max 反推 deadline，不注入 preview 填满 128 decode 步；空覆盖、丢记录、UNKNOWN、overflow 和条件漂移关闭资格。

effect 消费口重新纯读验证真实 raw/guard/source/reserve 三件套，再把非空、去重且属于原 issuer.cells 的 covered signature 子集传入 FiniteStartup 与 CurrentFiniteBinding。Startup 在任何运行期 patch 前检查该子集，现场 snapshot 在 production lookup 前先检查子集；未被实测 reserve 覆盖的状态回 U。development 可观测原 issued cells，但 ordinary 返回 native fallback；effect 不允许默认全部 cells。没有新释放 credit、资源 owner、GPU 同步或原生 batching 激活。

封存控制接口的 observer 为 18,796 bytes，SHA-256 `04b618db4e816f30231164a503adcdc2a55369f603350f107c0e48c90ec985ec`；reserve_join 为 32,772 bytes，SHA-256 `92996cb217bb11a85e237991cb82b37c1b1d10e26f65889338f030a0d46342ff`。其单独 CPU source manifest 为 `activation/control_observation/CONTROL_OBSERVATION_SOURCE_FREEZE.json`，记录该 agent 的 29 个本机 tests 与 GPU/真实 reserve 发行均为 0；这 29 项不混入本报告的 36 项独立测试计数。

额外无 CUDA frame 的 zero-scheduled metadata 步、empty-output prefill 或四类区间不完整仍按封存合同 UNKNOWN 拒绝，不删步、不填零、不改模型执行器。这是本次有限验证范围的实际边界；遇到这类失败只能报告相应的限定阻塞，不能将它解释成方法永久无效。

本轮独立 CPU 薄接线、拒绝边界和源口复审已完成。父流程仍负责服务器复验、完整 source/asset freeze、纯 CPU 模板预检和原输入 AFTER 保护证据；本报告不把其他 agent 尚待执行的服务器步骤写成成功。下一允许 GPU 阶段仍仅为通过真实设备/driver/UUID/原账本门后的唯一有界 strong off 资格作业。真正的有限 GPU cell、development reserve、现场动作和 U/I 效果都必须分别用真实运行证明，当前均未测，独立执行 GPU/RPC 为 0。
