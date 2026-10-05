# 真实成本表发行与缓存语义的只读 CPU 复核

本说明审查 2026-10-05 本地保存的源码与 CAL02 失败证据，不连接服务器、不执行 GPU，不发行成本能力或策略资格。旧冻结代码与记录保持原字节。下面的 API 是当前 V3 源码的真实接口；下一版本必须冻结自己的更新源码与实际记录，不能把本说明当作新版运行成功证明。

## CAL02 已知结果与缓存语义

`CAL02_result.json` 记录一次真实 GPU 尝试，144.24609911069274 秒，exit/child_exit 均为 1，未超时，原 OS 会话已排空。子记录状态为 `FAILED_NATIVE_FULL_STEP_DIAGNOSTIC`，错误是 `original hot warmup cached_tokens expected496 actual=512`。子记录 `windows=[]`、`warmups=[]`，测量尚未开始；独立保存的 primer/hot-warmup 输出不能代替六个测量窗口、CUDA 见证或已发行表。

缓存数量有两种不同含义：

| 情况 | 前端缓存命中报告 | 首次模型计算前的上下文 | 首步待计算 tokens |
| --- | --- | --- | --- |
| 本地块缓存，512-token prompt，block=16 | 本地最多 496 | 496 | 16 |
| 外部 KV 异步完整命中，512-token prompt | 可以为 512 | 源码重算最后 token 后为 511 | 1 |

作者 `SOURCE_CACHE_SEMANTIC_kv_cache_manager.py` 的 `get_computed_blocks` 将本地命中长度限制为 `request.num_tokens - 1`，块对齐时为 `16*floor(511/16)=496`。这是本地路径的规则，不能用于所有外部 KV 命中。作者 `SOURCE_CACHE_SEMANTIC_scheduler.py` 先记录 `num_local_cached_tokens` 与 `num_external_cached_tokens` 的 prefill stats；在 `_update_waiting_for_remote_kv` 的成功分支缓存已接收 KV 后，当 `num_computed_tokens == num_tokens` 时将它改为 `num_tokens - 1`，保留最后 token 的 logits 计算。因此 frontend 命中 512 与模型首步 pre-context 511 可以同时成立。

CAL02 的真实 hot-warmup 记录确实报告 `num_cached_tokens=512`。它的 `output_token_ids` 是 completion 列表再嵌套 token 列表：外层长度 1，唯一内层长度 128；primer 相同结构的内层长度为 1。因此 hot-warmup 生成完整 128 tokens，但仍不是正式测量窗口或 CUDA capture。上述 511 是作者源码语义对完整外部异步命中的预期；由于本次在 warmup 校验就失败，没有 measured capture，不能声称本次已实测首帧 511，也不能推断每种连接器路径都必然这样运行。

当前 V3 发行器 `_cell` 要求 `cached_prompt_tokens < prompt_tokens`。它又把同一字段交给 `validate_capture(cached_tokens=...)`，后者要求首帧 `context_length=cached_tokens`、`prefill_tokens=prompt_tokens-cached_tokens`。只把 `CACHED_TOKENS` 从 496 改为 512，会先违反发行器规则，且可能要求不存在的首帧 context=512/prefill=0。

下一版本的最小契约修复应区分实际 frontend cached 报告与 initial computed/context 值，并逐项匹配真实首帧的 prepared rows、scheduled tokens、input_seq_lens 和输出。对于完整外部异步命中，可以冻结 frontend 512 / initial computed 511 的源码预期，随后由真实 capture 验证，而不从前端命中报告推导实际帧。保持 prompt=512、measured offset=16、完整输出=128、原作者缓存/模型路径、估计数学与准入门槛；第 16 个 decode offset 的原有 pre-context 仍是 `512+16-1=527`。

## CPU 发行接口与进程内成本消费

真实公开接口位于 `gpu_prerental_preparation_20261004/activation/source/prefix_io_control/gpu_cell_issuer.py`。应由原有受检 startup 路径导入一次 `prefix_io_control.gpu_cell_issuer` 及同一包内 `p4_cost_table`，保持与 policy/bridge 相同的 `CostTable` 类型。`activation/gpu_cell_issuer.py` 是相同字节的另一份文件，不能再以另一个模块别名导入，造成两个私有能力注册表。

当前接口没有发行 CLI；以下是已核对的函数调用签名，变量必须来自独立、已冻结且真实核验的引用，不能用模板或自行填入成功标记：

```python
table = issuer.issue_verified_gpu_table(
    project,
    plan_ref=plan_ref,
    measurements_ref=parent_closed_measurements_ref,
    guard_ref=completed_original_guard_ref,
    intent_ref=prelaunch_intent_ref,
    expected_plan_ref=independently_pinned_plan_ref,
    expected_guard_ref=independently_verified_guard_ref,
    expected_intent_ref=independently_pinned_intent_ref,
    expected_runtime_domain_sha256=expected_common_domain_sha256,
    expected_gpu_uuid=actual_gpu_uuid,
    expected_source_lock_sha256=expected_full_source_lock_sha256,
    expected_collector_source_ref=actual_bounded_collector_ref,
)
identity = issuer.qualified_identity(table)
```

必须显式传入本次 `expected_collector_source_ref`：默认分支锁定旧 collector SHA，不适用于当前已冻结的 bounded collector。不得从事后可变 plan/intent 的同一内容临时生成全部 expected 值并称其独立预注册。

实际 GPU 原记录成功且原 guard 已退出并排空后，可以先调用当前 runner 的 `close_actual_raw(root, pending_ref=..., plan_ref=..., before_ref=..., after_ref=..., guard_ref=..., output=新文件路径)`。该函数核验六个真实 child 状态、原 shutdown 返回、`load_planner=on`、fresh process、相同 guard SID，以及 before/after 完整源验证，再生成 parent-closed raw。它只关闭来源链，不发行资格。当前 CAL02 失败，不满足这一步和上述发行 API 的前提。

公开发行 API 是 CPU 数据回放：核验原 guard、独立计划/intent、完整来源、128 帧与 CUDA Event 见证、原 native I/O journal、实际 shutdown/drain、两对拟合数据和一对独立 heldout。拟合只使用前两对，第三对不用于调整上界。`activation/native_conditional_cost.py` 中的 capture/I/O/guard/估计器纯函数是这条路径的核验来源；旧 `verify_native_cell` 包含旧单文件目录、job 和负载假设，不能直接用于新的 strong-domain raw。

返回的表是同一进程内真实私有能力，`scope='gpu_verified_exact_cells'`。表的 `production_qualified` 只表示已验证的有限成本单元能被同一实际 controller 消费，不表示整套策略、SLO、自然负载效果或论文收益已验证。导出的 JSON 摘要没有发行能力；另一进程必须通过公开 API 重新回放真实证据，不能反序列化布尔字段、调用私有 issuer 或伪造类型实例。

有限 cell 身份包含 GPU UUID、common runtime domain、model/layout SHA、batch、active decode、prefill、实际 context 和物理 bytes。当前边界为已有 I/O 四阶段均为零的 SSD read，operations 为 1/2/4/8，单物理块为 917,504 bytes；未覆盖的真实条件回到原路径。不能把唯一 context527 单元泛化为完整请求期间的所有上下文或并发条件。

## 必须归档的实际证据

- GPU 执行前：独立冻结 plan/intent 及其 expected refs，当前 UUID/domain/model/layout、完整 source lock 与实际 before SHA 核验、原 guard 源、有效权限与预算引用、固定六窗顺序 AB/BA/AB、独立 prompt/prefix/trace/seed 和测量 offset。
- 执行后：原 guard 的 exit、child_exit、elapsed、reservation/session/permission refs、OS 排空结果、完整 after SHA 核验；六个独立 child 的命令、输出、原 shutdown/native drain、fresh PID/SID、请求 ID、128 outputs、128 frames 与 CUDA Event 原始见证、I/O journal 与 CQE/staging 生命周期记录；保存 pending 与 parent-closed raw 的真实字节 refs。
- CPU 发行后：实际 issuer/CostTable/validation/estimator/collector 源 refs，公开 API 回放结果，有限 table scope、每个完整 cell signature 与成本口径、baseline/incremental-or-joint/uncertainty/total，原数学证明、heldout 实值与 coverage 判定、`holdout_used_to_fit=False`。保留失败记录，不改上界使 heldout 通过。
- 策略阶段另行保存：独立 deadline/SLO 来源、真实主机控制 reserve 及覆盖单元、真实 ordinary candidate/preview/commit/defer/fallback 记录、持续来源与当前条件绑定、完整输出与尾部排空。未有这些证据时，策略收益、完整 runtime cost、正式 goodput/SLO 资格保持未验证。

`verify_running_identity` 会读取真实 torch/GPU 属性并绑定已构造的原 worker，它不是无卡 CPU 发行 API。`FiniteStartup` 只能在未来原 guard 管理的实际模型进程中使用真实表，校验当前 owner/thread/request/context，关闭时恢复原 startup 方法；本只读复核没有调用它。实际文件名是 `runner/finite_startup.py`，当前目录不存在 `finite_startup_binding.py`。

## 没有独立 deadline / natural trace 时的检查边界

当前 `activation_request.verify(..., phase='development')` 要求 `independent_deadline_declaration_v1`，来源为 `independent_requirement_before_development`，正整数 full-control-window deadline、对应独立 authority、clock scope 与 service SLO。`FiniteStartup` 即使 `observation_only=True` 也要求真实表和正整数独立预算。不能从基线观测值、拟合 A-max、当前上界或已经看过的结果倒推 SLO。因此没有独立 deadline，当前真实成本消费的 development I shadow 路径不能启动。

已有 `phase='shadow', mode='shadow', arm='I'` 是另一条普通观测路径：需要真实同域 off 完成证据，配置保持 `cost_table=None`，没有实际 finite I 激活，可检查原模型/owner/shutdown/观测与回退生命周期。它不证明成本表消费、干扰额度准入、策略效果或控制 reserve；不能将其命名为已资格化的 I 策略功能测试。

无 natural trace 本身不阻止明确标记的原 P3 controlled workload 的机制或生命周期检查；源码允许 `controlled_original_p3_qualification_workload_v1` 并保持 `natural_trace_bound=False`、`fit_or_evaluation_input_allowed=False`。如果后来存在独立 deadline、真实 finite table 与覆盖的 ordinary 候选，可进行受控 development shadow 功能检查，但仍不能宣称自然流量下的 goodput/TTFT/SLO 改善。候选条件不匹配或没有实际 finite preview 时应报告未覆盖 / NOT_EXERCISED，不能制造候选或声称零控制开销。

本复核没有新增 GPU 作业、成本表、权限或资格；下一允许动作是独立冻结缓存语义修复并完成针对 CPU 契约检查，根代理再按持久授权和原预算决定一次真实采集。成功之后才可按以上公开 API 发行并消费实际有限表。没有独立 deadline 时，保留普通观测检查的有限结论。
