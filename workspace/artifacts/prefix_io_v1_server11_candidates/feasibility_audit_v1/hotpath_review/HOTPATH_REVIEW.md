# C4 active 重试：最后一轮无卡可行性审计

审计结论：**当前冻结实现确实会对延期的 ready preload 重复运行原 pump；现有通知尚不足以安全地把这一路径改成阻塞等待。存在一个仍在增量范围内的窄改法——让已有观测层的 host-step 失效边界唤醒原 incoming Queue——但它需要新增可证明的唤醒协议，不是已有接口的直接配置，也尚未实现或证明有性能收益。不能只把 ready 从 `_has_poll_work` 删掉。**

本审计未连接服务器、未使用 GPU、未修改任何冻结代码。服务器复验应只运行独立 `audit_notifications_cpu.py`；它不是候选修复。

## 已有实际证据与微观成本

已独立重算 v4 归档的整包 SHA、manifest SHA 和 9 个使用成员的字节/SHA。归档 SHA 为 `a52b9de286c46fd342592565bb9dd0550d1c82791ca0a8e94e2c9e7cd8f1e8b6`。具体原件引用和调用位置在 `AUDIT_EVIDENCE.json`。

- on04 实际延期 **181 次**、不可变观测复用 **180 次**。首次至末次延期 **64,898,699 ns**；180 个相邻间隔的均值 **360,548.33 ns** 是墙钟间距，不是线程 CPU 时间。
- 选定 GPU Event 步骤为 **68,464,035 ns**，比 shadow 多 **52,827,907 ns**。完整请求比 shadow 多 **112,501,682 ns**；请求加排空多 **123,576,034 ns**。因此不能把完整退化全部算在选定步的 181 次循环上。
- 历史服务器 CPU 数据为 **155 次**重试/组、41 组、完整 live-borrow 合成元数据，未加 profiler 的线程 CPU 中位数 **7,259,697 ns**。这不是实际 runner 的线程耗时，也不能乘以 181/155 后声称解释了 64.9 ms。
- 本次另做 **181 次合成 CPU 重试**，仅用于核对路径计数：181 次 preview/defer/reserve/release、180 次复用、1 次 collect、545 次容量检查、363 次 retry-key、2 次 `dataclasses.replace`。live accessor 在此计数回放中明确为 scalar stub；没有报告延迟或因果比例。

GIL、线程调度、主机提交间隙或某个 kernel 各占多少，目前都没有直接测量。GPU Event 时长也不是纯 kernel 执行时间。

## 源码证明的循环路径

冻结 C4 reactor SHA：`96bfd88dcee7f9c7996518ffe762b87be5ad81ef11d598329d41145e2b01fd87`。

1. `reactor.py:3054 _drain_ready_preload_fds` 在原队头取得 ready，原 reserve 成功后调用 `_prefix_ready_read_decision`。返回 defer 时，原 slot 立即 release；ready 保留在队头；函数返回本轮没有推进。
2. `reactor.py:1040 _has_poll_work` 包含 `_ready_fds_preload`。即使没有真实在途 I/O，这个延期 ready 仍让它为真。
3. `reactor.py:1058 _run` 因而使用 `_drain_incoming(block=False)`，每轮继续 `_pump_once`。
4. `reactor.py:1443 _pump_once` 仍处理 copy、CQE、调度、提交、finish。没有推进但 `_has_work` 为真时执行原有 `time.sleep(0)`。这是主动重复检查并让出调度机会，不是等待特定完成通知。
5. `reactor.py:2125 _prefix_stage_decide` 每次执行原实际状态/进度检查；`2256 _prefix_single_file_retry_key` 前后重新核验。缓存的是不可变观测，不是最终许可。C4 bridge 的派生快照缓存没有改变以上循环。

这一来源链能证明循环为什么存在；不能单凭它证明 GPU 退化的全部因果。

## 可复用和不能直接复用的通知

| 来源 | 冻结实现实际通知对象 | 对 reactor incoming 等待的意义 |
|---|---|---|
| `submit_job`（620）、`enqueue_preload`（725） | 原 `_incoming.put(...)` | 可唤醒；原 FIFO 保留已接受工作顺序 |
| `request_mandatory`（703）、STOP（739） | 原 incoming Queue | 可唤醒；必须继续无条件推进原 mandatory/shutdown |
| snapshot、P4 inspect/publication（640、2817、2834） | 原 incoming Queue | 可唤醒，但不能滥用 mandatory/publication 当虚构的 step 完成消息 |
| `_parent_cv.notify_all` | 等待 parent 准入的提交线程 | 不是原 incoming Queue 的 condition，不能唤醒其 get |
| `publish_p4_scheduled_load`（2487） | 只在 `_submit_lock` 下更新标量 | **没有入队通知**；新 load frame 也会令既有 retry reuse 失效 |
| `EventProxy.record`（collector:49） | 写 `record_before_ns`，调用原 event.record | **没有入队通知**；恰是现有 live 条件变未知的一个边界 |
| Linux AIO `_complete`（250） | `_done` + AIO 自身 `_cv.notify_all` | 不是 reactor incoming；reactor 仍由 `poll_all` 取完成 |
| Linux AIO `_run`（273） | 内部 selector + eventfd | 内核 CQE/元数据完成可唤醒 AIO worker，不能据此认定 reactor 已被唤醒 |
| `_drain_cuda_copies`（1477） | 对原 copy 的 `end_event.query()` | 当前接口没有向 reactor Queue 发送完成通知；有 pending copy 时仍必须保留原轮询 |

Linux AIO 并非整个后端都忙轮询：内部已有真正的 eventfd/selector 等待。把它笼统称作“SSD 完成只能轮询”不准确。准确说法是：当前 reactor 消费完成的接口是 `poll_all()`，而该后端的通知未桥接到 reactor incoming。CUDA copy 的实际完成在当前 reactor 则只有 `query()` 这个非阻塞观测路径；本审计未证明可用另一种 CUDA 回调替代。

另外，`current_single_file_step`（collector:253）只借用标量，不调用 CUDA query。它在 end event **开始记录**时返回 unknown；这不证明 GPU 已完成。若未来通知这个失效边界，只能令原策略重新评估/回退，绝不能发放 GPU 完成或资源释放额度。

## CPU 验证的边界

本机独立 8 项测试通过，冻结源码未改：

- 使用实际原 `_run/_drain_incoming/_intake` 与真正 `queue.Queue` 的 CPU 夹具，确认 submit、preload、mandatory、STOP 唤醒；“检查状态后、进入 get 前入队”也不丢失消息。
- 使用原 `EventProxy.record/current_single_file_step`，但 raw Event 是明确的非 CUDA 对象：record 一次使 live 值失效，却没有向 incoming 添加消息。随后调用真实原 snapshot producer 才唤醒等待。这证明现成 step-end 接线缺失，不是修复后的 GPU 证据。
- 从冻结 Linux AIO 源提取原 `_complete`：合成 completion 进入 `_done`，其自身 condition 得到通知；reactor Queue 没有消息。没有执行系统 AIO。
- 合成 ready 即使标作 defer，当前 `_has_poll_work` 仍为真。没有修改这一判断来伪装安全停车。

`audit_notifications_cpu.py` 支持 `--candidate-root C4 --author-root PROJECT --output NEW_JSON`。服务器上 C4 含完整包，不需要 `--aio-source`；本机只存差分时可显式指定同 SHA 的原文件。源 SHA 不一致则失败。该脚本不读取大归档，不导入 Torch/vLLM，不进行 SSH。

## 窄改法是否仍在原范围

**可能在范围内，但当前仅是有源码依据的工程候选；不能立即认定安全或直接租 GPU验证。**

已有 `_prefix_single_file_retry_key` 只允许一个 ready preload、零原生在途 I/O、无 pending copy/其他 ready/accepted parent、无原 preload waiter、合法未过期 live step。这给出一个可限制的等待域。未来可在 slot 正常 release 后，仅对这个域暂停再次调度该 ready，复用原 incoming Queue；原队列及所有权保持不变。消费者被唤醒后仍重走原 reserve/preview，不缓存允许提交的动作。

还缺少必须实现并证明的协议：

1. 现有观测层 host-step 失效、异常和 detach 必须发送**有界、可合并、带 run/step 标量身份**的控制唤醒。无需重写原模型执行器或改变 GPU kernel；可放在已有观测 wrapper，但它是新增通知接线。
2. 需要 arm/check/enqueue/get 的无丢唤醒证明。必须覆盖“结束在注册前/中/后发生”、旧 step 消息、重复消息、原 STOP/mandatory 并发。不能仅用 event.clear()/wait() 猜测正确。
3. 等待不得超过原 sample freshness 与 max-wait 的较早截止；原 0.5 秒 Queue 超时明显大于 100 ms 策略时限，直接套用是不安全的。需按原截止时间等待/重新评估，不能用新增固定 sleep 调参。
4. 任意其他工作、真实 I/O、pending copy、容量/进度不确定或通知故障必须回到原 pump。不可让一个可选策略阻塞唯一 owner 的完成/释放职责。
5. 默认 off 不安装控制等待；共同修复与策略接线仍分开。新控制消息不能成为第二套工作队列、缓存引擎或 release-credit 来源。

如果这些约束无法在现有观测层和 reactor 原 Queue 内实现，而必须重写 executor、统一所有 I/O/GPU event loop 或取消原 owner checks，那就超出本次增量路线。当前没有必要为了此窄域改造 Linux AIO 或 CUDA copy 完成机制，因为窄域本身要求这些在途工作为空。

因此，本审计支持“存在一个值得先证明的窄通知方案”，不支持“再调几次参数就能改善”，也不支持“当前代码已经有可直接复用的 step 完成通知”。继续实验的前提是先完成上述纯 CPU 协议资格；是否还有性能空间和最终是否值得新增 GPU 对照应由整体可行性审计另外判断。
