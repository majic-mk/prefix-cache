# 02 控制算法与工程合同

## 1. 必须先厘清的事实

以下均为 V1 必须遵守的定义，不能让 Codex 自行更改语义。

| 概念 | 定义 |
|---|---|
| 逻辑 Prefix block | 原缓存引擎的身份与管理单位 |
| storage unit | 底座文件／存储块的最小合法工作单位，可包含多个 GPU blocks |
| I/O batch | 本次批准的若干合法 storage units 或复制映射 |
| parent job | 原 connector 对外认可的完整 load／store 任务 |
| child work | parent 内部的一项文件 I/O 或一组复制，不自行改变 parent 完成条件 |
| submitted | 已交给 CUDA／I/O 后端，包含设备排队，未必正在传输 |
| in-flight bytes | 已提交尚未完成的字节，不能等同于瞬时带宽 |
| source-safe | 源数据不再被相关异步操作读取，是否能复用仍取决于原生命周期协议 |
| buffer-reusable | 原执行器确认这批物理内存没有其他保留／在途使用者 |
| cache-visible | 底座已按既定协议发布可命中的外部缓存 |
| durable | 满足选定 fsync 等持久性语义，不等于任意 write 返回成功 |

`source-safe`、`buffer-reusable`、`cache-visible`、`durable` 可以发生在不同时间。V1 不改变它们的上游协议。

## 2. 关键实现限制

### 2.1 不能提前把 D2H 完成当成父任务完成

已核对的作者 worker 通过 completed job 更新状态，源块复用通过 `jobs_to_flush` 等协议保护。py-kvcache 的 store parent 则在文件子任务终结后汇总。[S4][S5]

因此 V1 不新增“D2H 完成就直接 complete_store”的快捷路径。

只有 P0 证明所选实际后端已有独立 source-safe 确认，才能把它用于观测和原生回收。若无此接口，GPU 释放收益必须估算到原父任务／全部相关保护完成。

将来改造两阶段完成协议，应作为单独的协议优化项目，并对基线和新策略采用相同协议。本轮不隐式加入。

### 2.2 不能把原执行器续接硬拆成新的独立队列

已查 reactor 中 D2H 完成会继续签发 SSD 写入。第一版保留这种已接受链路的续接。[S5]

只有已经在原执行器中存在、尚未签发的 ready work 才可重排。不能为了制造“先写 SSD 再 D2H”的策略空间，故意把自然续接变成额外等待。

如果一批 SSD 写入已在设备队列里，策略只能减少其他新流量、等待它完成，或者选择其他合法可推进工作。不能声称 Python 调整优先级已经重排设备队列。

### 2.3 真正的 source fence 可能有 job 粒度

若一个父 job 覆盖多个文件，只有整个 job 完成才解除一组块的 fence，则完成一个文件不会立即释放这些 GPU 块。

依赖分析必须把这组完成条件作为闭包。跨两个 store job 受保护的同一物理块，只有全部相关保护结束后才能计为可用。仍有活跃请求引用时，GPU 释放收益仍为零。

## 3. 新增模块的数据结构

这里的类名均为项目拟新增名称，不是声称 vLLM 已有同名 API。Codex 应在 P0 输出实际字段与接口映射。

### 3.1 SystemSnapshot

| 字段组 | 最少内容 |
|---|---|
| 版本 | run_id、snapshot_epoch、monotonic_ns、字段能力掩码 |
| 推理负载 | active decode 数、实际 batch、prefill token 数、上下文长度分位、近期 step 时间 |
| 请求 | 有界 waiting request 列表、原生优先顺序、到达／等待时间、已有准入决定 |
| GPU | 原 pool 总量、可立即覆盖量、带 fence 的候选量、restore 预留量 |
| staging | 实际总分配字节、空闲 slots、各阶段占用、可驱逐干净缓存量 |
| I/O | SSD read/write 未完成数与字节、H2D/D2H 未完成字节、待签发量 |
| 生命周期 | 选定 job 的块有效期、活跃引用摘要、保护者集合、完成依赖 |
| 安全进展 | 原生 jobs_to_flush、被 wait 的 parent、最后进展时间 |

不能把“free list 中存在但仍需 fence 的块”直接计为可以无等待覆盖的块。两个指标分别记录，避免总容量虚增。

调度器线程提供 GPU 所有权摘要，reactor 线程提供实际 staging/I/O。跨线程只能使用发布快照，不能在执行器锁内反向调用调度器进行复杂查询。

### 3.2 WorkDescriptor

包含 parent_id、child_id、原请求／共享消费者集合、方向与 stage、合法传输字节、原顺序、创建时间、已提交状态、所需 slots、目标块有效期、已有源事件、后端最小单位。

集合可以是有界摘要和数量，不把任意长度用户文本或完整 KV 放入策略。

### 3.3 ReleaseWitness

用于描述一次候选完成闭包能带来的新增可用资源。

- 对象包含物理资源 ID、generation／等效生命周期标识、原 ownership 来源。
- 条件包含仍需完成的所有 I/O／引用事件。
- 效果分为 GPU 可复用字节、CPU 可复用字节、直接完成某个 restore，不能无单位混加。
- `evidence_level` 区分实际已观测阻塞、原 allocator 尚未选择 victim 的候选释放路径、无法证明。
- `release_scope` 指明 per-copy、per-file、parent-job 或多保护者闭包。

未能证明释放条件的对象不参加依赖升优先级，不以“所有 pending bytes 都能释放”替代。

### 3.4 DispatchBudget

包含 epoch、到期时间、累计 SSD read/write 签发数与字节上限、累计 H2D/D2H 签发字节、各方向 in-flight 上限、staging 预留限制与共同队列上限。

“本轮还能提交多少”和“全系统现在最多在途多少”必须分别检查。一次预算发放只消费一次，不随 `_pump_once()` 重置。

控制器禁止把 MiB/s、bytes 和 operations 混为同一个数。

## 4. 资源预算的硬边界

### 4.1 CPU staging

`StagingPool.compute_slot_count()` 采用包含 `min_slots` 的下限逻辑，实际内存可能高于用户提供的 staging 配置。[S8]

P0/P1 必须记录实际 tensor storage 字节，计入共享 cache、copies 占用和真实 allocator 开销。若最低 slot 需求超出批准预算，启动失败并报告所需值，或在冻结前调低 iodepth 重新配置。不能静默超配。

容量修复是共同基线修复，不能只在新策略里执行。

### 4.2 GPU

统计模型、KV pool、workspace、CUDA graphs 和 I/O 相关临时内存。所有实验臂固定相同实际 KV pool 预算和模型设置。

同一物理块多请求共享只计一次。GPU source bytes、restore reserved bytes 与 active bytes 可能重叠，不能直接相加作为总使用量。

### 4.3 SSD

py-kvcache 当前没有完整磁盘文件淘汰与 GC。[S1]

第一版目标为“有限 GPU／CPU 缓冲、磁盘足以容纳该轮完整数据集”。每次运行预估全部唯一前缀 storage units、对齐、临时文件、元数据和日志，留明确空间余量。数据写到项目专属根目录。

达到设定 disk guard 时停止接纳新测试请求并完成合法排空，标记本轮无效或未完成。不能让满盘后静默跳过写入成为方法收益。

不运行后台手写 GC，不在正文宣称已研究 SSD 淘汰最优策略。需要 SSD 满容量场景时，以单独后续阶段评估可复用的成熟后端。

### 4.4 已接受队列

保留原队列和 parent Futures，不再默认添加一层无限等待的“第二父任务队列”。若原队列缺乏合理界限，在共同 admission 入口添加容量保护，并让所有策略共享。

队列满在接受前表达延期。不能让 `transfer_async=False` 代表“已经接受，留待以后”。所查 worker 中有 `assert success`，这会把限流当成失败。[S4]

## 5. V1 策略的确定形式

### 5.1 实验模式

| 模式 | 行为 |
|---|---|
| off | 完整保留上游动作，策略路径不产生额外资源占用 |
| shadow | 执行动作仍是上游，旁路记录候选和成本 |
| fixed | 固定批量／额度，仍保留所有正确性进展 |
| pressure | 固定额度基础上加水位、age、防饥饿与共同缓冲预留 |
| interference | 以同样基础根据 decode 干扰调整额度，不利用真实释放依赖排序 |
| dependency_only | 依赖感知排序，普通额度固定 |
| joint | 同时启用依赖排序与干扰额度 |

这些是项目自定义字段。不要把它们当作 py-kvcache 自带选项传入。

### 5.2 候选边界

建议起始实现最多看 32 个前台／已接受 parent，最多检查 64 个待签发候选，依赖追踪最多经过 3 个 I/O 阶段或生命周期事件，单次生成最多 8 个闭包。它们是控制器计算量上限的起始设计，不是性能最优值。

这些观察上限不等于全系统允许的请求并发数，也不允许丢弃窗口外的已接受任务。原队列中的其他工作按公平轮转或原顺序继续获得服务。

一个闭包可以包含同一 parent 的多个文件完成条件。不能把“深度最多3”误写成“最多处理3个文件”。超出可准确分析范围时回退到原有策略，记录 reason。

### 5.3 先选择前台目标

先处理真正的 `jobs_to_flush` 与必要 completion，无须性能优化评分。

对普通工作，使用原生优先级与已有规划器决定的等待顺序。先考虑已接近等待上限的任务，再考虑有实际资源阻塞的任务。首版不再额外设计全局请求重排算法。

等待年龄用于防止无限让路，不允许按结果随意丢弃慢请求，也不允许为了提高 goodput 提前结束长请求。

### 5.4 判断零 I/O 的解阻机会

当缺少 staging 时，先沿用底座现成的干净 cache／预加载回收机制。能够安全驱逐就不需要先额外写盘。不能把正常缓存占用伪装成必须写回的债务。

GPU 同理，存在原生可无等待复用的块就不触发所谓 emergency writeback。

只有既有办法仍不能提供资源时，才计算需要传输完成的候选闭包。

### 5.5 比较释放闭包

对选定等待目标，得到当前需求缺口 `need_gpu`、`need_slots` 或 `restore_remaining`。针对可用闭包估计：

1. 何时按原协议能获得所需资源或完成 restore。
2. 完成后的新增可用资源是否足以满足缺口。
3. 新签发动作对 decode 的附加干扰估计及其测量支持。
4. 该路径是否依赖仍然活跃的计算、尚未完成的父任务或已经不可调度的设备工作。

第一版采用可解释的字典序选择。

- 优先保持原等待目标和原生服务优先级。
- 在可满足同一目标的支持路径中，优先预测合法解阻时间较早者。
- 时间处于标定误差范围内，选择预测干扰较小者。
- 再按新增传输字节少、原队列顺序确定平局。

不对 GPU bytes、CPU bytes、ms 任意加权相加。若需要多任务才能满足缺口，只使用受限闭包估计并去重。无法估计时不承诺解阻，用回退路径推进并记录。

这里的优先级只会作用于尚未签发 work。路径中已经提交的阶段保留原执行顺序，其剩余时间计入估计。

### 5.6 选择批量与额度

对选中的方向和工作集，只比较几个按实际 storage unit 对齐的合法批量。建议初始候选为 1、2、4、8 个 storage units，实际总字节和最小可用批量在 P0/P3 冻结。它们不是固定 token block size。

令 `s_t` 为当前负载，`i_t` 为已经存在的 I/O，`a` 为本轮新增动作。

`T0_hat(s_t,i_t)` 表示不再增加此动作时，下一控制窗口的预计基线耗时。

`Delta_hat(s_t,i_t,a)` 表示相对该基线的附加成本。

候选必须同时满足硬资源限制、累计签发额度和经验时延预算，例如：

```text
T0_hat + Delta_hat + uncertainty_margin <= internal_step_budget
```

若用的是“无 I/O 基线＋联合 I/O 总成本”，就不能再额外加入同一笔已有 I/O 的成本。两个估计口径必须二选一且固定。

可行候选中优先推进足以解除所选短路径阻塞的批量。尚无解阻目标时，在当前允许范围内使用较大合法批量。不要把“最大可行批次”称为全局最优。

实际选定结果一次性消费共享读写预算。不能由 H2D、D2H 两个独立模块各自认为自己拥有整份剩余额度。

### 5.7 没有可行普通动作

暂停新的推测性预加载和未接受的可选写入。继续收取 completion、既有链路续接和合法缓冲回收。

对于等待过久或必须保持系统前进的已接受工作，允许使用事先验证过的最小进展额度，记录 `progress_override`。它可以超过经验干扰预算，但不能突破物理容量和依赖。

所有超预算造成的 token 延迟都计入统计。它只保证在可执行条件下不因本策略无限停发，不保证硬实时完成，也无法保证故障设备必定响应。

## 6. 为什么要先做 shadow

shadow 使用真实当前状态生成拟动作，却不改变底座调度。它能检查：

- 是否真的出现候选依赖。
- 预测释放条件与后续实际回收是否一致。
- 控制器是否主要在修改本来就会执行的同一操作。
- 是否把原生可直接驱逐的缓存误判成需要写回。
- 候选完成粒度是否与真正 parent completion 一致。

shadow 不能证明采用拟动作后的反事实性能收益，也不能把原来发生的时延直接算成新策略可消除的时延。

## 7. 完成状态机

### 7.1 恢复任务

```text
LOOKUP/PLAN
  ├─ DECLINE → 原生计算
  ├─ DEFER → 原有有界等待与可选预加载
  └─ ADMIT → 原任务保护/分配
                ↓
           ACCEPTED
                ↓
    READ_PENDING / READ_INFLIGHT
                ↓
          DATA_IN_STAGING
                ↓
       COPY_READY / H2D_INFLIGHT
                ↓
      原协议的全部子任务完成
                ↓
         PARENT_COMPLETED
                ↓
      请求ready，只通知一次
```

staging hit 可以跳过文件 read，不跳过 H2D 源数据保护。单个文件到达、CUDA API 返回、event 已创建均不等于完成。

### 7.2 写回任务

```text
CANDIDATE
  ├─ SKIP → 不新增外部副本
  ├─ NOT_ACCEPTED → 有效期内可继续评估
  └─ ACCEPTED → 原源块保护协议
                  ↓
          WAIT_COMPUTE_SAFE
                  ↓
        D2H_READY / D2H_INFLIGHT
                  ↓
        原有SSD写入续接路径
                  ↓
         FILE_PUBLISH_COMPLETE
                  ↓
        原父任务完成和保护解除
```

完成后 CPU 副本是否仍可缓存由原策略决定。store 完成不等于必须把 CPU 数据立即丢弃。

### 7.3 取消、失败与终止

候选取消不动当前计算 KV。已接受任务只用上游已有的取消／拒绝路径。客户端离开也不能导致 GPU 目标块在 DMA 尚未结束时被复用。

真实 I/O 错误、短读、进程退出和 parent failure 的处理，按底座实际能力 fail-closed。未提供完整在线恢复能力时，在报告中明确服务失败并保留证据，不添加看似成功的重试假象。

至少区分 `FAILED_DRAINING` 与 `FAILED_FINAL` 的观测状态。该观测状态不替代执行器内部状态机。部分子任务失败时，也必须防止剩余复制写入已经转交新请求的目标块。

## 8. 强制排空的最小对接

所查 worker 在 `handle_preemptions` 中调用 `worker.wait(jobs_to_flush)`。[S4]

加限流前先验证这条 wait 如何到达具体 handler。若 wait 只是等待 Future，新增额度可能阻止 Future 取得进展，需要最小桥接：在等待之前把受影响 parent IDs 交给 reactor 的紧急集合，reactor 独立推进其必要未提交 work。

该桥接是拟新增能力，不能假装已存在某个 `force_drain` API。Codex 需要给出真实调用链和测试，再设计窄接口。

紧急集合使用 parent ID 与 run/generation 防止串用。即使 scheduler 停在 wait，reactor 仍能收取 completion 并续接。不能持有 reactor mutex 等待 reactor 自己完成。

清理紧急集合以真实任务完成为准，不以新一轮 scheduler 到达为准。测试 shutdown、重复通知、空集合、已经完成 job 和不存在 job 的行为。

## 9. 标定与在线反馈

### 9.1 标定内容

沿用作者 break-even 数据生成脚本估算恢复与 prefill 路径。[S3]

本题额外标定持续 decode 的干扰。固定模型、上下文分布、batch、prefill 混入、kernel 模式和起始缓存状态，做成对重复测量。方向至少包含 H2D、D2H、联合读写和真实 SSD 链路。

初始状态采用稀疏设计，先识别主要变量，再增加边界。避免把所有计数器都做成全笛卡尔积。

A/B 次序随机化或 ABBA，包含热身，报告方差和置信区间。输入／输出 token 工作量固定的机制实验与自然结束的质量实验分别保存。

### 9.2 step 时间与 ITL

`decode step` 只作内部信号，客户端 ITL 是另一口径。若一次流式消息打包多个 token，记录为 chunk gap，不能给其中每个 token 伪造相同时间并宣称测到 token ITL。

主 ITL 使用可核验的原生逐 token 输出事件，客户端同时报告 TTFT 与 chunk gap。无法获得逐 token 时刻时标记 ITL 未测得，不能用 TPOT 均值替代。

内部 step budget 在开发集冻结，并预留 scheduler／sampling／输出延迟余量。禁止使用 `ITL上限 - 最近一次GPU step` 就声称存在确定的安全 slack。

### 9.3 在线误差

维护对当前总耗时预测的残差与保守分位余量。线上无对照状态时，这个残差只能说明预测失配，不能声称得到 I/O 的因果干扰。

观测不匹配、混入大 prefill、数据过期或表外状态时，不更新原因归属的干扰表，使用保守回退。线上只调整预先规定的 margin／quota，不能训练新网络，也不能修改最终 SLO 阈值。

配置变更、模型变更、传输粒度或明显硬件环境变更后，旧表失效。原始表不可原地改写，在线余量另存带 run_id 的记录。

## 10. 可执行伪代码

以下为拟实现逻辑，不是现成 API。真实函数名由 P0 的接口映射决定。

```python
# scheduler / worker boundary: publish a compact immutable snapshot.
def at_scheduler_boundary():
    publish_snapshot(collect_supported_state())
    publish_native_mandatory_jobs()  # must happen before a blocking wait

# reactor: remains the sole owner of transfer resources.
def reactor_pump():
    poll_native_io_and_cuda_completions()
    run_native_completion_and_continuation_paths()

    if has_mandatory_jobs():
        progress_mandatory_closures_with_hard_limits()

    snapshot = latest_compatible_snapshot()
    if policy_mode == "off":
        run_original_dispatch_path()
    elif not snapshot_is_fresh_and_compatible(snapshot):
        run_validated_fallback_without_blocking_progress()
    else:
        if budget_epoch_needs_refresh(snapshot):
            candidates = observe_bounded_existing_ready_work()
            witnesses = build_verified_release_closures(snapshot, candidates)
            choice = select_target_and_closure(witnesses, candidates)
            grant_epoch_budget(choose_supported_batch(choice, snapshot))
        issue_legal_work_with_remaining_credits()

    run_native_fusion_and_submit_pending()
    finish_native_parent_jobs_once()
```

不能在每次 pump 里重发整份额度。不能通过调整循环顺序破坏原 `_flush_copy_batch()` 的复制合并。融合提交本身也必须对物理总字节验额度，既不能重复扣账，也不能绕过额度。
