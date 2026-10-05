# V5 通知候选：预先定义的 CPU 比较协议

状态：在正式计分运行前冻结；这是 CPU 工程门槛，不是 GPU 性能协议。比较对象为共同 C4 底座中仍存在的 active 延期轮询，与相同底座上的薄通知候选。不得改变原 policy、原成本判定、原期限、已接受工作续接、完成条件或关闭语义。

## 测量范围和真实执行要求

必须执行两份候选源码中的完整 `_run` 和 `_pump_once`，使用独立 producer 线程、真实 `queue.Queue` 和真实线程调度。允许 AST 提取原方法，但必须记录源文件 SHA、每个方法 AST SHA、实际调用次数、绑定/替换列表。不得把 `_pump_once` 改成直接调用新等待 helper 或只做一个 preview。

C4 原 pump 顺序必须保留：`_drain_cuda_copies` → `_poll_ring_completions` → `_schedule_work` → `_flush_copy_batch` → `ring.submit_pending` → `_finish_jobs`，随后无进展且仍有工作时原 `time.sleep(0)`。原 `_drain_incoming`、STOP 和 mandatory 的原 `_intake` 分支必须真实执行。正常调度应至少真实经过 `_schedule_work` → `_drain_ready_preload_fds` / `_drain_ready_load_fds` 中实际候选所属分支 → `_prefix_ready_read_decision` / `_prefix_stage_decide` → bridge preview。不得把真实接受点替成“看到通知便宣告完成”。

CUDA、AIO、文件打开、staging pool 等物理接口只允许显式 CPU fixture；必须记录每个 fixture 的输入/输出和限制。模拟 I/O 使用原 dispatch 和完成分支，或者明确列出未执行的原生分支，不得把 mock success 当 native I/O 资格。无 torch、vLLM、CUDA 动态库导入，`CUDA_VISIBLE_DEVICES` 为空。GPU 操作数必须为零。

两个臂的输入、外部事件顺序、step 持续时长、原 `_Ready` 身份/到达时间、资源 fixture 和完成通知相同。producer 使用绝对 monotonic 时间驱动外部状态；不能等 worker 进入 wait 后才释放事件。仅线程启动的共同 barrier 可同步。另由独立 race 套件使用人为 barrier 穷举切换位置，该套件不混入性能数据。

生产函数调用可以产生候选自己的通知；额外 queue put、generation 更新等开销应计入候选 producer 线程，不能替候选免费执行，也不能给 C4 人为添加同样费用。

## 固定场景和顺序

主场景 `step_end_40ms`：单个 917504-byte 可选 ready 工作在真实原策略路径被延期；外部 step 从共同零点持续 40 ms，然后 producer 发布 step end。原始 max-wait 和 sample max age 均保持 **100000000 ns**，不延长、不因效果改变。随后通过 CPU 物理 fixture 的真实接受/完成消息闭合，原 STOP 与排空自然返回。

敏感性场景只改外部 step 持续时长为 20 / 80 ms，不替代主结果，不挑最好者汇总。若具体 fixture 无法支持原单文件、完整 live-state/preview 路径，则该场景报告 unsupported，不能降低路径要求后称通过。

每个场景先各臂 4 次 warmup，按 ABBAABBA 顺序；warmup 全部保留但不计分。主场景 12 对，敏感性各 6 对。成对顺序固定为 AB、BA 交替；A=C4，B=V5。前后两半单独报告。每次新建相同 fixture，旧 Queue、Future、owner、deadline 均不能带入下次。

补充功能时序场景各 6 对、同样 AB/BA 交替：

- `mandatory_20ms`：step 保持至 80 ms，20 ms 从原接口放入 mandatory；记入队→原 intake、原进展接受时间。
- `stop_20ms`：step 保持至 80 ms，20 ms 放入原 STOP；已接受工作必须沿原完成链排空。
- `original_deadline_100ms`：step 保持至 120 ms；100 ms 原 deadline 应先于 step end 解除等待。记录实际策略理由（max-wait、freshness fallback 等），不得为了拿到某个理由更新旧时间戳。原 deadline 从原到达时刻计算，不能从每次重试或通知重置。

没有测量过程中自适应的次数、参数、数据剔除或重跑。启动/fixture 错误不成为策略负结果，但会关闭本轮比较资格；修复须保存失败并在新协议版本中重新冻结，不能静默替换。

## 每次记录

1. worker 和 producer 各自 `thread_time_ns` 起止差，以及两者合计；在自己的线程内采样，不用其他线程读到的 CPU 时间代替。主结果不运行 cProfile、trace profiler 或每次 pump 日志落盘。
2. 完整 wall、外部事件计划/实际时间、producer 迟到、入队→intake、入队→原 progress 接受、STOP→自然 join、原 deadline→实际解除的延迟。
3. pump/真实各阶段调用次数，collect、preview、live-state borrow、freshness/资源检查、延期决定、reserve/release、实际模拟 submit/complete/retire、控制事件入队/消费次数。
4. 输出 fixture payload/hash、完成状态、无丢失/重复、Future 结算一次、原队列及在途为空、线程退出、STOP 原回收路径；区分“模拟完成正确”和“真实 I/O 正确”。
5. OS、Python、CPU/内存 cgroup 限额与 Linux `cpu.stat` 前后差（如可读），进程 CPU 为补充值；不得将相互重叠的统计求和。当前半核/2 GiB 可以运行，但调度噪声必须保留。

检查在每次真正进入 preview 和真实接受前仍发生，不能预先缓存动作或因通知省略 native live/physical check。允许减少没有状态变化期间的决策次数；不要求 V5 和 C4 重试次数相同。必须证明状态变化后重新判定，而不是复用“允许提交”。

## 固定判定与止损

**硬功能门槛**：任一 lost wake、错误提前提交/释放、漏做接受前 live/资源检查、修改/延长原 deadline、STOP/mandatory 错误、挂起、重复完成、原有活跃进展被错误 park，立即失败，停止正式比较。每次全局 watchdog 为 2 秒；触发只能记失败，不能以 watchdog 当正常完成。独立 race 资格必须全通过，不能以性能较快抵消。

**CPU 机制门槛**：主 40 ms 的 worker CPU 成对中位差 < 0，且 worker+producer 合计 CPU 成对中位差 < 0；合计 CPU 至少 9/12 对下降，前六/后六对各自中位差 < 0。20 / 80 ms 两组的合计 CPU 中位差都不能 > 0。严格零或计时分辨率不足不算改善。此规则用于筛掉没有明确 CPU 收益或把费用搬给 producer 的实现，不是论文收益阈值或统计显著性结论。

**延迟护栏**：主场景候选 wall 成对中位增量不得超过 25 ms；mandatory/STOP 的“实际入队→原进展”中位增量分别不得超过 25 ms；deadline 解除不得早于原阈值，候选“超过原阈值”中位数不得超过 50 ms。它们是半核 CPU 工程护栏，不是服务 SLO。对存在 cgroup CPU 配额的 Linux 环境，主场景至少 9/12 对、每个其他场景至少 4/6 对的两臂 `nr_throttled` 均无新增，才给延迟资格；读不到该计数则记 unknown。未达此条件不筛掉 throttle 样本重算通过，只标记延迟判断资源受限并关闭本轮升级资格。全部样本仍参与前述中位数与 CPU 统计；未配置 CPU 配额的环境单独记 unlimited。

若功能通过但 CPU/延迟门槛不通过，保留 off，停止本候选投入。若全部通过，只能说“在指定 CPU 合成工作负载下省去了主动重试开销并保持模拟语义”；不能说 GPU 变快、真实瓶颈可调度、D/J 获得释放信用、完整 P4 通过或自动启动 GPU。后续决定仍须独立研究价值审查，不在本协议内扩大范围。

## 根实现者检查表

- 优先原方法：`_run`、`_has_work`、`_has_poll_work`、`_drain_incoming`、`_intake`、`_pump_once`、`_schedule_work`、ready drain、`_prefix_stage_decide`、bridge/policy preview、settle/finish、原 STOP cleanup。
- 最危险 fixture：把 `_schedule_work`、`_finish_jobs` 或 `_has_work` 写成常量；让 fake ring 同步完成从而绕过原完成路径；用 main thread 直接清 `_ready` 代替状态推进；producer 等候新 waiter 形成才发通知；伪造新 snapshot 时间掩盖原过期；把 mandatory / STOP 转为新旁路事件；在 CPU 计时之外执行额外通知成本。
- 如只能保留完整 pump 调用而某个物理阶段不能真实执行，明确报告覆盖层级，不能用更深层结论补齐。每个源码/fixture 白名单在正式运行前冻结。
