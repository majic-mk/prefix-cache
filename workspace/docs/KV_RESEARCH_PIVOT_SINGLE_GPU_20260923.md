# 单卡 KV 缓存研究转向方案：PaceKV（暂名）

日期：2026-09-23。原始提案保留；当前实施范围以 [R0/R1 执行合同](PACEKV_R0_R1_IMPLEMENTATION_20260923.md) 为准。用户已要求无卡实施；尚未运行新 GPU 实验。

硬件决定已于 2026-09-23 更新：首批真实验证改用单张 RTX 5090，A800 仅作可选跨硬件迁移。新的可执行环境合同见 [5090 租机预检](PACEKV_5090_RENTAL_PREFLIGHT_20260923.md)。历史 A800 无卡交接仍保留，但不能用其旧环境或结果解锁 5090 运行。无需训练大模型，无多机、多卡、RDMA 或 GDS 前置要求。

## 1. 决策与可信边界

建议停止旧的多历史 Source 选择路线，保留代码、原始结果和失败证据，不再为它追加 GPU 试错。已有结果支持停止当前实现和实验路线，但不能推导“所有多 Source 方法在所有 workload 上都无效”。

推荐的新候选不是“更好的 Source 评分”，而是：

> 面向单卡 LLM 服务，以实际解码干扰和未完成写回占用为反馈，联合控制精确前缀 KV 的恢复与可选持久化，提高同时满足首 token 和后续 token 时延要求的有效吞吐。

英文工作题目：Interference- and Write-Debt-Aware Exact KV Caching for Single-GPU LLM Serving。

PaceKV 只是工作名，不代表名称查重或创新性已经认证。

最重要的决定：**先做有明确停止条件的小规模瓶颈实验，再开发完整系统。** 不以“可以保底 SCI 二区”作为立项依据。期刊录用、分区和性能结果都无法预先保证。这里优先降低的是质量风险和无止境研发风险，而不是承诺正结果。

SCI/JCR Q2 与中科院二区不是同一个概念；正式选刊时应核验当年的收录、学科分区、范围和版面要求，本方案不冒认某刊当前分区。

## 2. 为什么改研究问题

旧路线有三个串联的经验假设：历史版本存在足够互补性、浅层评分能预测任务质量、修复与选择的成本低于节省。这些条件任一不成立，整个收益链就中断。

新路线只复用相同完整前缀的缓存，不改变 attention、不近似替换历史上下文、不选择 repair token。研究难点转为可以直接观察的硬件资源竞争：

- 新请求恢复 CPU/SSD 中的 KV，可能影响正在 decode 的请求；
- 将刚计算的 KV 写回 CPU/SSD，也消耗传输、内存和队列资源；
- 延迟写回不一定免费：D2H 未完成前，相关 GPU 块可能不能回收；
- 全部停止写回又可能增加后续 miss 和完整重算。

需要检验的核心假设是：**分别优化读取和写入，会遗漏它们经 GPU 块生命周期、pinned buffer 和在途 I/O 对其他请求造成的代价；联合控制能否优于充分调优的简单策略？**

这不是已成立结论。若异步 DMA 对真实 decode 几乎无影响，或者一个固定限流器已经解决问题，就应停止本题。

## 3. 创新重叠审计

检索截止本日期。学术连接器需要重新认证，本轮改用作者论文、会议、出版社和官方仓库核验；不是穷尽式系统综述。2026 年部分工作为预印本，不能写成已经同行评审。

| 已有工作 | 已覆盖的部分 | 本项目不能这样宣称创新 |
| --- | --- | --- |
| [Cake](https://arxiv.org/abs/2410.03065) | 重算与加载并行、decode 优先、适应计算/I/O变化 | 首次 load/recompute overlap 或首次 decode 优先 |
| [CacheFlow](https://arxiv.org/abs/2604.25080) | 多维度和 batch-aware KV 恢复 | 首次按 batch 决定加载与重算 |
| [Tutti](https://arxiv.org/abs/2605.03375) | GPU-centric SSD I/O、读写争用/SM占用治理、按执行空隙延迟写回 | 首次保护前台、后台写回或 slack-aware I/O |
| [Cascade](https://arxiv.org/abs/2608.06557) | 共享延迟预算下的调度、恢复、预取、重算；估计纳入active decode和内存总线争用 | 首次 SLO-aware KV cache |
| [py-kvcache](https://arxiv.org/abs/2609.11744) | 单卡 CPU/NVMe 缓存、有界 staging、异步加载、盈亏点 | 首次消费卡外部 KV、有限 buffer、异步预取 |
| [Bidaw](https://www.usenix.org/conference/fast26/presentation/hu-shipeng) | 分层缓存、KV 大小/位置感知调度和选择性缓存 | 首次联合存储层级与请求调度 |
| [SuperInfer](https://arxiv.org/abs/2601.20309) | 特定超级芯片上的 TTFT/TBT SLO 与双向 KV 传输 | 首次同时关注首 token 和 decode SLO |
| [HiCache](https://docs.sglang.io/docs/advanced_features/hicache_design) | 预取超时、选择性写穿/写回 | 首次热门才写、超时重算 |

拟争取的贡献必须收窄为以下三项，且都需实验支持：

1. **可复现的问题刻画**：普通 PCIe 单卡上，混合恢复/持久化的在途资源占用如何影响持续 decode 的尾延迟，而不只测生成一个 token 的 TTFT。
2. **联合反馈机制**：以其他活跃请求实际承受的干扰，以及未完成写回占住的 GPU/pinned 资源为状态，共同决定传输批次和可选写回；不是两个互不通信的读写优化器。
3. **生命周期净收益证据**：包含当前写入、后续 miss、缓存容量竞争和完整排队时间，证明对比简单固定配额、读优先、选择性写回仍有优势。

现有工作已经联合控制相关资源。“联合”二字不是本题的创新。待验证差异仅是：匹配条件下建立的decode干扰估计，以及GPU写回保留成本，共同驱动闭环控制。增加两个状态变量本身也不构成创新。

仅仅把已有模块组合起来不等于创新。必须增加四臂消融：基础、仅干扰控制、仅写回控制、二者联合。消融能够解释收益来源，不能单独证明新颖性；仍需逐项对照最近工作的具体算法。如果只有工程移植价值，没有机制差异和稳定收益，不应包装成新算法。

不推荐同时启动的备选：页级 repair/省读规划。它仍有任务质量风险，且与 MEPIC、CacheSlide、SparseX 的块级机制接近；当前不投入实现。

## 4. 目标工作负载与精确性合同

### 4.1 适用场景

- 多轮对话：同一会话的既有前缀在后续轮次再次出现；
- 文档/代码问答：固定材料在问题之前，多个请求共享相同材料前缀；
- agent/tool 工作流：相同系统说明和累计历史形成可复用前缀；
- GPU cache 容量不足，但 CPU/SSD 保存了仍可能命中的前缀。

不承诺解决：任意不同前文下相同文档的精确复用、无重复前缀的一次性短请求、只运行一个请求且无读写竞争的全部场景。没有前缀重复时应退化到正常引擎，报告额外开销。

### 4.2 精确身份

每个缓存块的身份使用完整祖先链，而不是仅对本块/文档 tokens 做 hash：

```text
ExactBlockKey
  model_weights_revision
  tokenizer_and_template_signature
  adapter_and_attention_signature
  position_rope_signature
  dtype_layout_backend_signature
  tenant_namespace_or_salt
  parent_prefix_digest
  current_block_token_ids
  valid_token_count
```

缓存只在全层完成、传输完成和身份校验后可见。不完整块、跨 tenant 或不同模型/位置语义的块不得命中。第一版只处理支持的标准文本模型，混合 attention、MLA、多模态等未适配形式明确拒绝。

同一文档在不同前文下产生的 KV 不相同。旧系统的 DENSE_EXACT 来源标签不能替代这里的完整前缀身份。

精确指“不引入 KV 近似”。不同 batch/chunk/kernel 的浮点顺序仍可能不同，所以验证分三层：无损传输字节一致、同执行计划数值/生成一致、并发计划改变后的数值与任务回归。不能用 F1 容差掩盖缓存错配。

## 5. 系统结构和算法

```text
真实请求队列
    ↓
原生精确 Prefix lookup
    ├─ GPU hit → 原生执行
    ├─ CPU/SSD hit → 等待恢复任务
    └─ miss → 原生 chunked prefill
                       ↑
        联合干扰/资源控制器
          ↙           ↘
   分块恢复队列      可选写回队列
          ↘           ↙
        原生 block allocator + 有界 staging
                       ↓
             原生 attention/decode
```

不重写模型 forward，不增加 Source selection，不恢复旧三道 Gate，不保留 15% repair；原来 gamma=0.8 也不是这个新课题的准入目标。新目标是完整请求流的质量不变与时延/有效吞吐，而不是强制每次恢复都快20%。以上都是待用户确认的新路线边界。

### 5.1 最小数据结构

```text
BlockRecord
  exact_key, generation, valid_layers
  gpu_refcount, backing_locations
  readiness_event, in_flight_owners

TransferTicket
  block_keys, direction, total_bytes, issued_bytes
  pinned_slot_lease, gpu_block_lease
  submitted_at, completion_event, cancellation_state

WritebackDebt
  pending_gpu_bytes          # D2H未完成而被保留的GPU块
  occupied_pinned_bytes      # 尚不能复用的buffer
  h2d_inflight_bytes
  d2h_inflight_bytes
  ssd_read_write_queue_depth
  oldest_pending_age

ServingObservation
  active_decode_count, active_context_tokens
  latest_decode_step_times
  waiting_request_deadlines
  writeback_debt, allocator_free_bytes
  engine_generation

ActionEstimate
  supported
  expected_progress
  empirical_conservative_decode_stall
  predicted_completion_ms
  added_gpu_retention_byte_ms
  added_pinned_byte_ms
```

“写回债务”是以上可测资源的向量，不把毫秒、字节和队列长度无量纲相加。byte-ms 指资源占用随时间的积分，不是累计传输字节。

### 5.2 两个耦合动作，避免机制堆叠

**动作一：恢复传输配额。** 在原生调度迭代边界，根据 active decode、已提交读写和剩余资源，从小型离散 tile 集合选择下一批恢复量。tile 与原生缓存块对齐；例如测试1/4/16/64 MiB级别，最终集合在开发集冻结，不做无限搜索。

**动作二：可选写回处理。** 在相同状态下选择现在写、短暂推迟，或放弃尚未提交的缓存副本写入。放弃的是未来复用副本，不是当前推理需要的 KV。延迟写造成 GPU 块不能释放时，必须纳入判断，而不是把“延后写”当免费操作。

第一版缓存淘汰固定使用基线的 LRU/访问次数策略，不同时开发新的复杂淘汰算法。写入候选可使用开发集冻结的简单二次访问规则；独立的该规则也必须作为强基线。

### 5.3 目标与可计算规则

定义请求 i 的首 token 延迟 TTFT_i，后续相邻 token 到达间隔 ITL_i,k。预先固定服务类别的首 token 阈值和每请求 P95 ITL 阈值：

\[
G = \frac{\#\{i: TTFT_i\le D_i^{first},\ Q_{0.95}(ITL_i)\le D_i^{token}\}}{T_{evaluation}}.
\]

这称为双 SLO goodput。少于两个输出 token 的请求没有可计算 ITL，单列，不自动记为满足 decode SLO。另行报告跨请求 token 的 P95/P99 ITL，不能用平均 TPOT 替代。

对动作a，在匹配的decode batch/context长度桶、相同读写在途状态下，离线成对测量“无该I/O”和“执行该动作”的step时间，得到干扰差值分布和预冻结安全余量，形成经验保守估计 \(\hat I^{cons}(a,x)\)。在线并不能同时观测无I/O的反事实，不能把所有ITL波动归因于缓存动作；在线只用预测误差更新保守余量，连续超限退回固定保守配额。

只有资源租约可获得且该经验估计处于预留范围内，才签发该批I/O。它不是统计或实时硬上界。缺少支持状态不能填0；采用保守固定配额或暂停可选写回，并记录unsupported。长上下文增长、batch变化和冷启动不能混入同一个估计桶。

主策略先采用可解释的约束规则：

1. 原生 decode 正常推进，不能因等待缓存写入而停下。
2. 优先恢复接近首 token deadline 且能取得进展的请求，加入等待年龄，避免无限饿死。
3. 从大到小尝试合法 tile，考虑**当前读写共同状态**对 decode 的影响。
4. 可选写回只有在不挤占必要读取且资源保留代价可接受时提交；超过债务水位时取消未提交部分或及时完成并释放，不能无限推迟。
5. 第一版仅在首次恢复copy提交前选择加载或原生重算；提交后不动态切换到覆盖同一目的页的重算。缺支持/不合算时在提交前选择原生路径。

恢复异常时必须有切换屏障：取消尚未提交的读取，已提交目的页等待完成事件后才能重算，或改用互不重叠的新页；撤销尚未完成的restored/computed-token状态，只保留已验证的连续前缀。generation检查不能阻止已发出的DMA覆盖数据，不能替代此屏障。

```python
def on_engine_boundary(observation):
    reap_completed_io_and_release_owners()
    x = refresh_generation_and_resource_state(observation)
    let_native_scheduler_advance_decode()

    reads = rank_restore_requests_by_deadline_and_age(x)
    for read in bounded_front_of(reads):
        action = largest_supported_tile_that_fits(read, x)
        if action is not None and acquire_all_owners_atomically(action, x):
            submit(action)
            x = update_inflight_state(x, action)
        else:
            defer_or_choose_recompute_before_first_submission(read)

    for write in bounded_write_candidates(x):
        decision = evaluate_optional_write_with_retention_cost(write, x)
        if decision == SUBMIT and acquire_all_owners_atomically(write, x):
            submit(write)
            x = update_inflight_state(x, write)
        elif decision == DROP_NOT_SUBMITTED:
            cancel_only_unissued_persistence(write)
        else:
            preserve_owners_and_account_wait(write)
```

实现优先使用少量测量单元、EWMA和滞回，第一版不用神经网络预测器。只有 pilot 证明读优先规则有缺口，才比较联合候选动作；不预先写复杂全局组合求解器。

重要限制：经验保守预测不等于分布无关的实时保证；CUDA stream priority 也不代表可以抢占已启动 copy。SLO违约如实计入，不丢弃慢请求改善统计。

## 6. 消费级硬件可行性

建议主机：Linux、RTX 5090 32GB、至少64GB主存（优先128GB）、普通NVMe、足够CPU核数。记录PCIe实际协商宽度/速率、NUMA、SSD型号/温度/后台占用，而不是只记录GPU型号。[5090官方规格](https://marketplace.nvidia.com/en-us/consumer/graphics-cards/nvidia-geforce-rtx-5090/)

主模型先用 Qwen2.5-7B-Instruct；第二模型使用有合法访问权限的 Llama-3.1-8B-Instruct，或另一个引擎支持的7–8B模型。一次只加载一个模型。先BF16，不把新量化方法带入主贡献。

GQA模型的KV理论字节数：

\[
B_{KV}=2 L H_{KV} d_{head} N b.
\]

依据Qwen官方配置，L=28、H_KV=4、d_head=128、BF16 b=2，所以每token约56 KiB，8192 token约448 MiB，32768 token约1.75 GiB。这只是KV本体，不含权重、workspace、分页浪费和临时缓冲。[模型配置](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)

7–8B BF16权重大致14–16GB，实际显存可容纳范围需现场测定。因此最长输入与并发数不做完整笛卡尔积；超过真实token容量的点报告不可执行，不能靠删记录隐藏OOM。

主实验不依赖A800；A800只作为可选跨硬件迁移验证。不能用A800结果代替消费卡结论。

### 后端选择

优先在现代vLLM的原生KV offload接口上做外部插件，保持attention/decode主体不变。官方接口已有CPU和经CPU staging的多级存储能力，这部分是工程底座而不是本项目创新。[官方文档](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)

先固定一个实际支持5090及选定模型的Torch/CUDA/vLLM组合和完整commit，验证扩展二进制兼容，不直接照搬旧A800环境，也不盲目追最新版本。若选择的接口不能实现受控签发，先做一个边界清楚的adapter；若必须大改attention和逐层执行才能继续，应重新估算并停下来复审选题。

## 7. 实验顺序：先决定值不值得，后决定怎么发论文

以下数值是建议的内部投入门槛，不是预期结果、论文录用标准或已获用户授权的GPU预算。看结果前应正式冻结。

### R0：文献与协议，1–2个工作日

- 将上表扩展为逐机制对照，重点核对Tutti、Cascade、py-kvcache、Bidaw。
- 冻结exact语义、计时端点、双SLO、真实/受控负载标记。
- 若发现拟议联合反馈与已有方法没有实质区别，则不开发，仅保留复现实验。

### R1：现代后端与问题存在性，预计1–2周内做决定

先完成单模型dense/native-prefix/CPU/SSD roundtrip和真实并发。随后进行约4–8 GPU小时的首批稀疏pilot；该时数不含首次环境编译和调试，不是自动运行授权。

实验只回答：相同持续decode期间，增加以下工作后发生什么？

```text
decode alone
+ CPU cache restore
+ cache persistence
+ restore and persistence
+ real NVMe staging
```

主输入长度2K/8K/16K/32K，输出256或512 token；active decode为1/4/8中实际可容纳的组合。先少量锚点，不全量交叉。

必须测优化后的异步基线，再比较固定在途配额、读优先、选择性写回，不能仅打败故意无界或同步的弱基线。真实机器正常状态为主要证据，人工限速/压力注入单独报告。

建议继续条件：至少两个正常负载区间存在稳定decode尾延迟或goodput损失；联合决策的可实现调度回放显示约10%以上改进空间，且简单固定策略尚不能消除。该回放只是开发阶段收益空间估计，不是在线方法成绩或严格全局oracle。

以下任一发生则停止：无稳定干扰；只在极端人为限速下有效；固定配额/读写分离已达到同样结果；所需改动是重建整个执行器。

### R2：最小联合控制器，约1–2周

- 只接入传输签发预算和可选写回债务处理。
- 加入generation、in-flight所有权、取消/失败恢复。
- 对相同trace做四臂实验，记录全部前台与后台费用。
- 至多两轮由独立开发集支持的机制修订；若仍无稳定收益，不通过无穷调参继续。

### R3：双模型与真实请求流，约1–2周

至少两类自然复用工作负载：多轮对话、共享文档/代码前缀QA。第三类可用公开agent轨迹，前提是数据许可及构造规则清楚。

可选用[SCBench](https://arxiv.org/abs/2412.10319)的多轮/共享上下文任务检验质量及缓存复用，不能把任意单轮QA随机拼接后称为自然生产trace。若没有真实到达时间，则保留文本/会话顺序，明确标记为“公开请求内容＋合成到达过程”。

fit与test按会话/文档分组隔离。工作集跨越GPU cache容量，并分别测CPU容量充足、CPU不足需SSD、重复前缀很少、热门集合切换。不得只展示为新方法定制的极端工作集。

### R4：完整证据与论文，约1–2周

- 冻结参数，执行未使用过的测试trace；不根据test回调阈值。
- 核对正确性、成本、稳定性、局限，公开可复现配置和原始指标。
- 有条件增加A800迁移实验；不以扩展硬件替代5090失败结果。

整体工程预估约5–8周，后端接口有风险时更久。前1–2周应能决定继续还是止损，不能等数十小时优化后才查核心假设。最终双模型实验先估40–100 GPU小时，须在pilot实测后重算ETA；这里不授权租卡、付款或执行。

## 8. 对照、指标与1+1证据

### 基线层级

1. 原生引擎：GPU prefix cache，无外部offload。
2. 同引擎默认CPU/SSD offload。
3. 可运行的优化异步系统，如LMCache/py-kvcache。
4. 同引擎充分调优的固定配额＋读优先＋选择性写回。
5. 最接近机制：Tutti/Cascade/Cake中与本硬件兼容的策略。无法运行原系统时，标为“机制复现”，不能冒认原论文实现成绩。
6. 候选联合控制器。

CacheBlend/SparseX/QCFuse不是本题主对照，因为它们的近似非前缀复用语义不同；可以在相关工作解释边界，不为凑表强行比较。

### 公平性

- 同模型、精度、GPU/CPU/SSD字节预算、输入和到达序列。
- 统计pinned buffer、排队中KV和未完成写回保留的GPU空间。
- native baseline与新系统使用同等调优预算。
- cold start、steady state、工作集切换分开。
- warm TTFT不包含上一请求构建，不代表构建免费：另报含全部写入、队列、后续miss和期末必要drain的trace总成本。
- 真SSD读取与OS页缓存命中分开核验。
- profiler与正式计时分开；host关键路径不重复叠加CUDA重叠时间。
- 全部到达请求入账；超时/失败/未完成请求在goodput中不算成功，报告数量，不从延迟表悄悄删除。

### 主指标

双SLO goodput、TTFT P50/P95/P99、ITL P50/P95/P99、总输出tokens/s、拒绝/失败/超时率、CPU调度费用、实际H2D/D2H/SSD字节、pinned/GPU峰值与byte-ms、后续缓存miss、重算token数。

正式主要arm建议至少1000个请求并有5次独立完整trace复验；对于尾分位这仍可能有较宽区间，应报告区间而不是把数字当高精度真值。以会话/trace为重采样单元，不能把同请求的每个token当独立实验。需求量和运行时由pilot估计后冻结。

### 联合机制是否真的1+1>2

令Y为相同负载下的goodput，四臂为Y00、Y10、Y01、Y11。报告交互项：

\[
\Delta_{interaction}=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
\]

只有该量及其配对不确定性支持正交互，才能在此指标上声称协同收益。若联合仅等于最强单机制，就如实写成一个机制有效，不拼凑两项贡献。

建议正式继续投入门槛：至少两类正常工作负载相对最强可比基线的goodput改善约10%以上，配对95%区间不跨0；无争用场景附加成本不超过约3%；正确性全部通过；不能依靠牺牲长期命中率或丢弃慢请求获胜。门槛失败不等于“永远不能发表”，而是本项目不再继续重投入。

## 9. 正确性与安全测试

CPU测试：完整prefix key、模型/tenant/position隔离、部分块不可见、generation失效、原子租约、双缓冲复用、未提交/已提交取消区别、GPU回收等待、损坏文件隔离、崩溃恢复、所有资源有界、aging。

GPU测试：

- 相同执行计划下CPU/SSD往返原始KV字节一致；
- 关闭offload与无损恢复时greedy token一致，并保存teacher-forced logits；
- 对调度变化单独建立native同类变化的数值基线，预冻结误差标准，不在发现失败后现场放宽；
- 实际连续batch下lookup/import/export与请求结束正确；
- 取消、超时、磁盘读失败后不得读取未完成页或提前释放buffer；
- 对任何缓存错配视为硬失败，不以“最终F1没变”解释通过。

正式性能路径禁止每token/每copy执行完整KV SHA256；资格测试可全量验证。对象创建/发布时的校验和所有后台证据写入仍需计费。

## 10. 现有代码怎么复用

本轮是只读审计。工作区有大量已有未提交修改，全部保留。本提案不部署、不删除、不合并、不推送。

| 现有模块 | 可提取能力 | 必须重做的部分 |
| --- | --- | --- |
| `v8_schema10_staging.py` | 实际pinned buffer与CUDA事件归还 | 从BF16 K/V层形状改为原生缓存块 |
| `v8_schema10_layer_storage.py`、`v8_schema10_storage.py` | 分层寻址、事务/故障恢复思路 | 旧pre-RoPE文件不能直接当现代paged KV |
| `source_store_v2.py` | 原子catalog、校验和与发布 | 替换target-only身份和历史variant语义 |
| `v8_leases.py`、`v8_schema6_hbm.py` | 所有权和有界资源账本 | 绑定新引擎实际allocator，不把逻辑预算当实物 |
| `request_wallclock.py`、`v8_schema10_cost_collection.py` | host/CUDA分离计时 | 增加逐token、读写队列、byte-ms |
| `v8_schema10_event_log.py` | 证据链和失败记录 | 避免每事件同步fsync卡主路径，仍保留写日志成本 |
| `v8_schema10_experiments.py` | 到达trace和聚合壳 | 真实连续batch接口，不复用旧单活跃请求限制 |

旧`v8_schema10_native_factory.py`硬绑定A800、Torch2.2.1/CUDA12.1、vLLM0.4.1和CacheBlend补丁。**不能删除guard就宣称5090可用。** 旧Source捕获、d1/d2选择、repair、Prefix shadow、mixed来源P0/P1和历史Profile均归档，不纳入新运行路径。

### 建议新模块（尚未创建）

```text
src/pacekv/
  exact_key.py               # full-prefix与backend身份
  engine_adapter.py          # 现代原生KV offload边界
  block_ownership.py         # 引用、generation、完成事件
  transfer_budget.py         # 在途读写统一资源预算
  writeback_debt.py           # GPU/pinned保留与取消
  interference_profile.py    # 支持/缺测量状态、反馈
  joint_io_policy.py          # 唯一主控制策略
  evidence.py                 # token事件与成本闭合

scripts/pacekv/
  audit_environment.py
  run_exact_roundtrip.py
  run_interference_pilot.py
  replay_serving_trace.py
  aggregate_results.py

tests/pacekv/
  test_exact_key.py
  test_inflight_ownership.py
  test_cancel_and_failure.py
  test_resource_bounds.py
  test_no_future_trace_leakage.py
  test_complete_cost_accounting.py
```

优先out-of-tree插件和小型adapter；不以不断新增schema或修改老执行器作为进度。

## 11. 论文成立条件与停止条件

一个可投系统论文的最小证据包应该包含：清晰且真实的未解决问题、区别于最近工作的机制、两个模型和多类负载、强基线、公平资源预算、正确性、完整成本和失败边界。

若在此基础上有跨硬件/跨负载稳定的显著收益、可解释协同和公开复现，可考虑更高目标；若只有一个构造样例或未调优基线上的改善，不能认为已达到二区要求。

第一阶段应交付而不是口头宣布成功：

```text
novelty_overlap_matrix.md
environment_lock.json
exact_roundtrip_report.json
interference_raw_events/
strong_baseline_tuning_manifest.json
pilot_go_no_go.md
```

只有pilot GO后才开发全系统。NO-GO时交付负结果、原因和代码边界，停止本候选，另经用户确认选题。不会自动回到旧多Source路线，也不会自动把备选机制全部叠上去。

## 12. 本次已完成与未完成

已完成：原始文献重叠审计、消费卡约束规划、现有代码只读复用审计、完整候选方案与分阶段止损计划。

未完成/未执行：新系统实现、5090/A800数值验证、干扰存在性测量、性能收益、创新性最终确认、任何论文分区/录用保证。

本轮仅新增此提案文件；未连接服务器，未启动GPU，未修改运行时代码，未购买或租用资源。
