# 有界 P0 原生请求批执行入口

## 本轮人工管理计费的明确时长授权（2026-09-20）

用户已明确表示自己管理实例停止，并追加“先运行一小时”。可使用显式
`billing_mode=operator_managed_time_cap`、`operator_managed_billing=true`，
`maximum_gpu_hours=1` 和当前实例/GPU/审批窗口。未知的 `unit_price_per_hour`、
`maximum_cost` 必须为 null，输出费用估计也为 null，不能伪填零或历史单价。
动作上界、初始化/清理和整个 E/M 共用窗口仍受硬时间约束。该模式不能从缺字段自动推断。
历史 manifest 仍默认 `metered_caps`，保持正数单价、费用/时长双限的原规则。
这只响应本次明确时长授权，不授予租卡/付款/关机权，也不解除任何数值或阶段门槛。

此入口接通现有 target capture、K-only 比较、SSD winner 准备、FinalCommit、原生 finish 和原子发布。它不是 P1 实验入口，也不是 P0-E/P0-M 数值资格的替代品。

## 当前可执行边界

- 仅当前操作者明确授权的本机 GPU；无 SSH、下载、租赁或付款逻辑。
- 使用既有 built-in native factory 核验模型、安装补丁与服务器依赖；不接受任意 Python factory。
- 仅既有隔离 v2 Source store/manifest registry；不能把找不到的旧池自动替换为空池。
- 单活跃请求、完整 inventory；多 Segment 不自动扩展动作清单。
- fixed15 normalized K/V；不在 P0 加 Query×D。
- 每个目标的候选为 0..4 个已存在 Source 或预先登记的更早 action 发布结果。禁止未来引用、隐式构建或临时补 Source。
- 每个 action 显式重建空原生 Prefix 状态，不复制 CUDA 指针；reset 成本计入批动作预算，不能把此模式的耗时当作 Prefix-warm 性能。
- 独立 K SelectionState 比较，不恢复 CFO、前文 KV 所有权或默认 Prefix shadow。
- 缺匹配成本时仍可完成 dense、capture 和正常规则发布；没有任何“强制生产 commit”开关。
- 真实 CUDA、混合来源模型数值、r1 对照、性能均尚待验证。

## 文件与绑定

`--manifest` 为独立冻结的 `bounded_native_p0_batch_v2` JSON，完整内容摘要写入 `manifest_sha256`，其计算不包含该字段自身。

必备：

1. `phase=P0`、`locked_test_accessed=false`、`automatic_rental_allowed=false`。
2. `native_manifest_sha256`：既有原生环境 manifest 文件真实摘要。
3. `binding`：code_commit、runtime_digest、patch_sha256、model_signature、tokenizer_hash、gpu_uuid、instance_id、input_manifest_sha256、initial_pool_sha256、initial_registry_sha256。
4. `authority`：approval_reference、instance_id、gpu_uuid、starts_at_unix、expires_at_unix、unit_price_per_hour、maximum_cost、maximum_gpu_hours。无缺省价格或历史授权继承。
5. `limits`：initialization_upper_seconds、cleanup_seconds、maximum_actions、host_capture_bytes、host_comparison_bytes、cuda_comparison_bytes。
6. `registry_budget`：max_bytes、max_manifest_bytes，沿用该隔离 registry 的明确预算。
7. `jobs`：冻结顺序的 action_id、request、request_sha256、input_origin、upper_seconds 和明确操作参数。历史省略 `operation` 时仍只解释为 `source_request`，需要 sources_by_segment、comparison_profile；独立诊断支持 `exact_capture_control`、`explicit_mixed_reference`、`mixed_sparse_control`，见下节，不接受任意操作名。
8. 可选 `exact_capture_pairs`：结果前冻结的 T20 配对与数值政策；省略时不做数值通过判断。

`input_origin` 只能是 `controlled_provenance_diagnostic` 或 `frozen_development`。这个标签本身不是数据资格证据；原始分区、lineage、tokenizer 仍须单独审计。受控诊断不能改标为自然多 Source 机会。

`input_manifest_sha256` 绑定按顺序的 action_id/request_sha256/input_origin 三元记录列表。请求还必须明确 max_new_tokens。

Source 引用有两种：

- 已发布对象：`{"source_id": "实际对象ID"}`；
- 本批更早动作发布：`{"birth_action_id": "较早动作ID", "segment_id": "目标ID"}`。

第二种是冻结出生配方引用，不是预编造 Source digest。如果前一动作没有发布对应对象，后续动作失败并停止，不偷偷构建、替换或者继续用次优 Source。

## 命令结构

仅在实际文件和权限齐备后代入路径。不要用占位 hash、假授权或参考测试文件执行。

```text
python scripts/server/run_decoupled_v2_p0.py
  --manifest <冻结P0清单>
  --native-manifest <核验后的原生环境清单>
  --store-root <隔离v2池>
  --registry-root <对应共享manifest目录>
  --instance-id <当前明确授权实例>
  --output <全新输出目录>
```

默认只验证输入，不触发 CUDA/模型初始化。它会读取模型审计文件及 hash；结果是 `INPUTS_VALIDATED_GPU_NOT_RUN`，不是 GPU readiness。

显式增加 `--execute` 才会在本机核验 GPU UUID、调用真实 native factory 和执行清单。开始前仍需人员核实 approval_reference 的当前授权来源及限额；一个填写了字符串的文件不是付款授权。

初始化和测量/清理预算保留；额度不足执行下一完整 action 时 `BUDGET_STOP`。模型每层检查 deadline，但这里不承诺能从 CUDA hang 中强制安全回收，也不销毁实例。费用仅为经过时长乘已声明单价的估计，不冒充平台账单。

## 原始证据与状态

- 外层 `preflight.json`：输入、资产绑定和是否观测过当前 GPU。
- `batch/manifest.json`：不可覆盖的实际动作清单。
- `batch/actions.jsonl`：追加、fsync、hash 链事件；半行、断链、不同 binding 均拒绝读取。
- `batch/<action_id>/request.json`：答案/token IDs、执行/选择/发布/清理明细。
- `logits.npy`：仅保存已经显式捕获为 CPU FP32 的原始 logits，禁 pickle；保存真实文件摘要、shape。
- `record.json`：原始文件摘要及 evidence_origin。
- `result.json`：completed / failed / pending 独立列表；不确定发布和模型失败立即停止。

`COMPLETED` 只表示请求动作走完。所有记录初始 `numerical_verdict=NOT_EVALUATED`。
原始数组比较工具没有默认 BF16 容差；必须提供结果前冻结的容差、位置数与文件 hash。其 raw numeric pass **不包含输入/teacher/mixed参照一致性证明，也不能自动解锁 P0/P1**。

### 实际 teacher 输入对齐（2026-09-19 补充）

原生 `finish_from_prefill_hidden` 在 `capture_logits=true` 时记录 `decode_input_trace_v2`：原始输入摘要、实际 Prefix token 数、成功 decode step 真正送入的 token、每行 logits 的绝对位置以及 argmax 输出。teacher 输入与 argmax 输出分开；未完成的 teacher 序列不生成完整记录。关闭 logits capture 时不增加这项诊断工作，也不添加 GPU 同步。

`compare_p0_logit_actions` 从两份 manifest、record、request 和 `.npy` 原始文件重新校验摘要与输入，再比较数值。它拒绝不同 teacher、Prefix 条件、模型/patch/runtime/device、错误位置、缺失 trace、失败清理或实际行数不符。普通 QA 的 greedy logits 不可当作匹配 teacher 证据。

这只补齐 **conditioning alignment**，不是完整 **execution recipe alignment**：上游真实 Source/注入、mask、目标全层账本、显式 mixed reference 仍需独立核验。即使数值相同且来源标签为 CUDA，输出仍保持 `native_runtime_qualified=false`、`P1_execution_allowed=false`。不得用整请求 dense 代替 mixed 的实现参照。

没有自动恢复：成功事件前缀不等于存在可恢复的 Source/原生 allocator 状态。失败、部分文件和旧目录全部保留，用新目录重跑有界动作。

### T20 exact capture 开／关执行与配对

`operation=exact_capture_control` 是独立 P0 诊断，不调用生产 selector、FinalCommit 或 Source 发布，也没有“强制生产 commit”含义。

- `capture_enabled` 必须为布尔值，`target_ids` 为明确的非空唯一目标列表。
- 请求绑定同一原始 tokens、位置、Segment、模型与 teacher 序列；`capture_logits=true`，teacher 数量严格为 `max_new_tokens-1`。
- 禁止 Source/comparison 参数、Prefix 命中、旧整请求 capture、Prefix shadow、hot cache、prefetch 和 native dense continuation。
- 两 arm 均显式启动同一 resumable 全行执行器，每层恰好完整执行一次；off 不捕获，on 只捕获目标切片。
- on 的 manifest registry 为受预算约束的私有内存对象，不能指定磁盘目的地。完成后复制出轻量证明/摘要、等待清理并丢弃 tensor 候选，**不会成为后续 action 的 Source**。
- 没有完整捕获、模型错误、first-token 端点缺失或清理失败时，保留失败，不补跑第二次 forward。

每项 `exact_capture_pairs` 必须且只能包含：

```text
pair_id
reference_action_id                 # capture_enabled=false
candidate_action_id                 # capture_enabled=true
relative_l2_limit                   # 结果前显式确定，无默认 BF16 数值
minimum_positions                   # 不大于两 arm 的 max_new_tokens
require_predicted_token_ids_equal   # 结果前显式确定
```

两请求除 request_id 外须完全相同，目标列表也相同。批末校验事件链绑定的 record、manifest、request 与 logits 摘要、实际 teacher 输入、全层全行执行配方、G0 目标证明及零 parent/shadow 所有权。只比较原始 logits 和声明的 token 策略，不把 teacher-forced 输出当自由 greedy QA。

状态区分：未执行任一 arm 为 `PENDING`；CPU fixture 即使数值一致也只为 `CPU_ONLY`；真实 CUDA 的 `PASSED` 也仅限 **T20 capture 开关不改变该配方数值**。损坏、配方不一致或数值不通过为 `FAILED`，请求动作即便完成，批结果仍为 `NUMERICAL_PAIR_FAILED`。原始记录保留，所有全局 P0/P1/正式 GPU 资格仍为 false。

此对照不证明 resumable 等价于原生 monolithic dense，也不证明 mixed 或 r1 正确。reset、hook/copy/hash/证据落盘都属于诊断费用，不能拿它直接做在线性能收益结论。

## 仍需完成才能宣称完整 P0

1. exact capture-on/off、独立 explicit-mixed 参照与实际稀疏执行入口已接通 CPU 接口测试，仍需当前模型/GPU 的真实数值；不得把接口测试当作 T20/T21/T31 GPU 通过。
2. frozen development/S0/E/M1 输入资格和构建动作预算绑定属于 P1 前置，不阻止使用隔离受控输入的 P0 数值测试；不由此入口自行选择自然数据。
3. 当前模型/设备真实执行，以及数值/完整性/显存生命周期的独立验收。

当前无卡测试只能验证接口、文件、预算/错误状态；不能声明 native 正确、Source 互补、selection 有效或端到端加速。

### T21/T31 的独立 mixed 参照侧（CPU 已接线，非完整配对资格）

现有生产 P0 driver 在缺成本支持时会合法 dense，不能靠填写 PASS 强行把它当 mixed 对照。新 `explicit_mixed_reference` 采用明确隔离的诊断配方：固定上游 Source、boundary 与逐层 repair mask，用普通全行 decoder 的 scoped QKV hook 只替换上游未修复位置的 pre-RoPE K/V，目标所有行仍逐层完整计算。`mixed_sparse_control` 通过现有 CacheBlendV6OnlineEngine 独立执行相同配方，不借用 full-query hook，也不放宽生产准入。

必须先接类型化诊断来源标记，禁止参照路径默认发布 exact Prefix；prefill 后移除所有替换 hook 再做 teacher decode。参照多算的 query 行单独记账，不发布 Source、不复用生产来源 ledger 冒充其数学假设；生产 ledger 约束不可削弱。必须观测非零上游历史替换，不能两边实际 dense 却声称 mixed PASS。目标 r1 对照应使用相同上游 mixed 参照，而不是整请求 dense。

每项参照动作另需：

```text
operation: explicit_mixed_reference
target_id: 原请求内完整目标 Segment ID
upstream_segment_id: 严格位于目标之前的完整 Segment ID
source: {source_id: 已发布 G0 ID, artifact_digest: 实际摘要}
first_reuse_layer: 明确的一基层号
repair_positions_by_layer: 从上述层到模型最后层的绝对位置列表
```

只接受一个已发布、可见且 token/模型/geometry 完全匹配的 G0 Source，不隐式构建，也不接受更早诊断动作的私有 capture。每层 upstream repair 数严格为 `ceil(0.15*N)`，位置唯一、有序、不可 reentry；目标全部位置、全部层完整投影。GQA 映射、RoPE、causal attention 和原生 block 引用由正常模型层执行；该额外 full-query 参照不使用 selective executor。

请求必须显式 teacher logits，保持 fixed15、无 Prefix 命中/整请求 capture/shadow/hot cache/prefetch。`limits` 另需正整数字节上限 `mixed_reference_host_bytes`、`mixed_reference_cuda_bytes`。host 上限包含紧凑目标 K/V 与逐层 Source/hash 临时空间；CUDA 上限包含固定 Source 的诊断副本与 projection/target scratch，不是无限显存许可。执行前持有 Source lease 与 HBM reservation；成功 fence 后才释放，失败 fence 保留资源并阻止下一动作。

记录 `extra_forward_count=1`、真实逐层 query/替换行、Source before/destination/after/destination-after 摘要、诊断 hash/D2H/捕获耗时。带类型的诊断来源标记使 native endpoint 禁止 exact Prefix、warm history、canonical export 与 Source 发布；它不赋予执行或 GPU 资格。

目标证据逐层写入 `target-kv/layer-0001-K.npy` / `V.npy`，用 CPU BF16 的原始 int16 位模式保存，不合并全层、不偷偷 D2H、不转成近似浮点。`record.json.target_kv` 绑定 encoding、shape、层数、payload bytes 和各文件摘要。`read_p0_target_kv` 必须提供 record SHA 与明确内存上限，拒绝坏摘要、缺层、异形、非有限值、路径重定向或超预算；读取结果只是原始证据，不是 Source 也不是 PASS。

本轮 CPU toy 模型检查了 full-query 参照与独立 reduced-query 计算的目标数值，并验证其确实不同于 whole-request dense。这个结论仅用于数学/接口单元测试。**尚未执行真实 Mistral/Qwen；尚未形成 T21/T31 native mixed/r1 成对数值报告；P0-M 和 P1-M 仍未解锁。**

### 独立稀疏候选与 raw 配对验收（2026-09-19）

`mixed_sparse_control` 使用相同请求、上游 Source、boundary、fixed15 mask、teacher，
但实际执行现有 CacheBlendV6OnlineEngine，而不是再次调用 full-query reference。
第一 selective 层记录 `projected_positions=active_before` 和
`attention_query_positions=active_after`，后续层按上一层保留行投影；不能把两个集合混为一谈。
每层必须观测真实投影、attention 输出行数、原生状态及历史 KV 安装，不接收人工成功标记。

每项 `mixed_reference_pairs` 明确包含：

```text
pair_id / reference_action_id / candidate_action_id
relative_l2_limit / minimum_positions / require_predicted_token_ids_equal
target_relative_l2_limit / target_absolute_max_limit / target_read_bytes
```

所有数值阈值在结果前写入清单，无默认 BF16 容差。reader 预算同时涵盖两 arm 的
原始 BF16 数组及逐层数值比较 scratch。检查器重算文件、hook recipe、实际捕获摘要，
核对 GQA geometry、逐层 projection 次数、所有 token 位置类型及两侧执行条件。
logits 和每层目标 K/V 均通过才产生限定配方的结果。CPU 结果固定为 `CPU_ONLY`；
真实 GPU 的 scoped pass 也不自动授予完整 P0 或 P1 资格。

### 目标 r=1：不可用 target-full 改标签

两 arm 可显式增加：

```text
target_execution: R1_ALL_LAYERS
target_source: {source_id: 实际已发布 G0, artifact_digest: 实际摘要}
```

默认仍为 `FULL_ALL_LAYERS`，不接受无目标 Source 的 r1 声明。reference 绑定并保护同一
目标 Source，但仍独立完整计算目标，不注入它；candidate 真正准备第二个目标 ticket，
在同一 boundary 调用 target commit，repair positions 为全部目标 token。
之前层也完整计算目标，所以不存在先丢行后假称 full-all-layers。

候选侧在原生 attention backend 调用后检查目标 working K/V 等于本层实际 post-RoPE
current K/V。逐层输出 `target_r1_commit_active`、`target_r1_repair_positions`、
`target_r1_source_installed`、`target_r1_current_kv_writeback_verified`。
上游仍必须真正 fixed15 selective，不能用整请求 r1/dense 代替混合对照。
两 Source 分别记录 before/destination/after 完整性；主存/显存预算包含两者；
失败 fence 隔离两份租约与缓冲，成功 fence 后才释放。没有通过接口诊断发布 Source。

这个配对只覆盖 T31 的 mixed-target-r1 子场景，不等于已通过所有 r0/r1、模型或 Prefix 条件。

### 分阶段输入检查与租卡前阻塞

`scripts/check_decoupled_v2_stage_readiness.py` 只读取证据并写入全新报告目录：

```text
python scripts/check_decoupled_v2_stage_readiness.py --stage CONTROLLED_P0_M
  --evidence <实际证据引用JSON> --output <不存在的新目录>
```

支持 `CONTROLLED_P0_E`、`CONTROLLED_P0_M`、`P1_E`、`P1_M`。受控 P0 不要求先有
自然 P1 cohort；P1 则不能继承 CPU 数值或未绑定 slots 作为数据/历史 Source 资格。
证据引用为实际 `path` 与 `sha256`，包含 CPU report、worktree_files、test_results、commands、
batch_manifest、native_manifest、patch_audit、patch_manifest、initial_pool、initial_registry，
以及当前 installed_runtime_root、instance_id。缺项逐条 BLOCKED，不写假 hash。

新的 model geometry preflight 在加载权重前核对审计绑定 config 的真实层数、词表、上下文，
拒绝仅覆盖三层的 toy mask、越界 teacher token 和没有 d+1 投影的深度。
输入通过也只是 `INPUTS_VERIFIED_RUNTIME_RECHECK_REQUIRED`，不是实际 GPU 数值通过。
真正执行还需当前实例、GPU、补丁 import、审批窗口、单价、时长与费用上限全部核验。
本工具不 SSH、不租卡、不付款，也不会将旧服务器授权自动延用到本轮。

### 受控 P0 的两批构建与启动（2026-09-19 收口）

`scripts/prepare_decoupled_v2_controlled_p0.py` 补齐命令入口，但不调用模型、GPU、SSH或付款。
它不生成示例 Source、不补默认容差、不替用户填写授权。所有命令均写入新的目录。

第一步 `init-pool`：核对实际 native manifest 文件 SHA 和模型资产，只建立空的隔离
`store/catalog.json`、`registry/registry.json` 及准备报告。参数为：

```text
init-pool --native-manifest <实际文件> --native-manifest-sha256 <实际文件SHA>
  --pool-spec <明确容量/来源策略/权限域JSON> --output <新隔离目录>
```

pool-spec 的 `store` 必须明确 authorization_domain、policy、max_bytes、staging_bytes、
max_variants、probation_opportunities；`registry_budget` 明确 max_bytes、max_manifest_bytes。
模型/tokenizer 身份只能来自核验过的 native 附件，不能由 pool-spec 覆盖。
新目录不得嵌入已有 Source store 或 registry，不会修复或覆盖旧目录。

第二步 `freeze-batch`：从实际空池/已有池读取并核验 catalog、registry、模型配置和已发布
Source 的 KV/SelectionState 文件，再派生逐动作清单与 SHA。参数为：

```text
freeze-batch --native-manifest <实际文件> --native-manifest-sha256 <实际文件SHA>
  --store-root <隔离store> --registry-root <隔离registry>
  --instance-id <当前授权实例> --recipe <显式配方JSON> --output <新清单目录>
```

recipe 顶层必须完整提供 authority、limits、registry_budget、numerical_policy、
exact_controls、mixed_controls、birth_controls。空分支显式用空列表。
numerical_policy 含 policy_id、rationale、exact、mixed，未使用的数值分支为 null；
容差、teacher、位置数、原始 K/V 比较内存限额须在观察本轮结果前冻结，不使用历史默认值。
每个 arm 有单独的 upper_seconds，maximum_actions 必须等于实际派生动作数，
初始化与清理时间计入授权。修改清单不会扩大费用/时长权限。

最小的一个受控场景可以分成以下两批；这是动作结构，不是已经冻结或执行的清单：

| 批次 | 动作 | 允许发布的内容 |
| --- | --- | --- |
| CONTROLLED_P0_E | T20 capture-off、capture-on；另一个普通 source_request 在空池构建 U/T | 只有普通请求完成后发布的两个真实 G0，且仅在隔离 P0 池 |
| CONTROLLED_P0_M | T21 full-query reference / sparse；同配方 target-r1 reference / sparse | 均不发布；仅原始目标 K/V、logits、来源、资源和计时证据 |

E 批的 `birth_controls` 是普通 request，不能含 teacher/logit diagnostic；每个目标的候选
列表为空，真实 miss 不需要成本表，也不会读取当前 K 来虚构一次比较。
目标必须来自原请求，完整重算后才发布。T20 的额外参照前向输出不能拿来供池。

M 批必须读取 E 批实际发布成功的 Source ID 与 Artifact digest 后再冻结；
不允许假 ID、同批未来 birth 引用或从 teacher/reference 导出 Source。
两批之间只绑定输出身份，不能根据结果换样本、改 teacher、mask、容差或预算。
若 E 失败，M 不执行。这个受控流程不证明自然 workload 的 Source 机会，也不解锁 P1。

最后仍先运行 `check_decoupled_v2_stage_readiness.py`，再由服务器入口核对当前 GPU、
import/patch、代码、预算窗口。`freeze-batch` 成功只是
`BATCH_FROZEN_RUNTIME_RECHECK_REQUIRED`，不等于准许 `--execute`，更不等于 GPU 数值通过。

### 持久目录格式与证据范围修正

真实 Source catalog 是 `{payload, digest}`，不是平铺的 `{config, ...}`。
准备 CLI、readiness 和服务器入口统一使用 `parse_target_catalog_v2` 先验证封装摘要，
再读取 config；旧平铺 fixture 不再被当作真实存储格式。这个解析只证明元数据封装，
实际 store 打开与清单冻结仍独立验证 backing、manifest、预算和来源。

CPU 报告中的旧总括 false 字段保留兼容，但新增 `scoped_cpu_bridge_evidence`，从本次
精确测试 ID 的记录区分“本地链路已实现/测试”与“真实硬件未验证”。
普通生产 factory 的 v2 Pool、Prefix-hit proof adapter 和完整 P0/P1 资格消费者仍未接通，
不能把受控 P0 builder 或 CPU PASS 写成这些模块已完成。

### 2026-09-21 当前版本 GPU 证据补充（非 P1 解锁）

服务器37581已完成当前 `ac1ea7f9…` 工作树的受控 E/M/r1（10动作）、
自然 h0/S0（6动作）、自然 E/M 来源及混合数值（7动作）、原冻结传播/六段交错/
九前段长输入复验（8动作）。各批保留独立清单和原始身份；不得将这些动作数
理解为独立质量样本或论文实验数。

详情见 `artifacts/decoupled-v2/p0-em-acceptance-20260921/REPORT.md` 和
`artifacts/decoupled-v2/p0-natural-mixed-20260921/REPORT.md`。后者保留了一个重要零结果：
首个自然 E/M 配对的上游因果前文、位置相同，导致完整 Artifact 内容相同；
不能用它证明深浅状态差异、混合质量风险或多 Source 互补性。

本次默认 Prefix-off 的数值和来源范围通过，不代表可选 Prefix-on、GPU 故障注入、
P1 自然消费 QA 或生产经济性已通过。`P1_QUALIFICATION_CONSUMER_NOT_IMPLEMENTED`
仍是真实阻塞：后续需由统一消费者复核 P0 原始数组与事件、数据隔离、来源出生和
构建清单，不能只删除阻塞项或接受手写 `passed=true`。
