# P0 lifecycle audit; not P2 GPU acceptance
Store success: scheduler _build_store_jobs → worker prepare_store_kv delays submission until next step → handler.transfer_async → TransferCoordinator.submit_store records compute_event → D2H waits that event → _on_store_copy_done immediately queues native SSD write → _on_write_complete checks full size, publishes, settles slot → _file_terminal completes success only after all files → worker completed_jobs → scheduler.update_connector_output complete_store and fence removal.

Active requests protect blocks by ref_cnt. Finished full-attention requests register each pending parent in _block_id_to_pending_jobs and return False, allowing free-list entry. Reallocation of these blocks creates jobs_to_flush before overwrite. Free-list presence therefore differs from immediate overwrite safety. All protecting parents must finish; an active reference still means zero newly reusable bytes. The map does not supply allocation generation: live observation must add an owner-observed identity lifetime, never infer it from recycled block_id alone.

Mandatory chain: handle_preemptions submits pending stores first; OffloadingWorker.wait delegates to handler.wait/futures_wait. Native reactor runs independently. There is no quota-aware mandatory signal yet. Ordinary throttling remains disabled.

Load ready requires all selected files/copies. _drain_cuda_copies queries end_event; event creation or API return is insufficient. Shared slots stay protected while cached or copies_inflight > 0. Existing clean eviction may supply space without I/O.

source-safe, buffer-reusable, cache-visible and durable are distinct. No independent D2H source-release ACK exists. Files may become visible before parent completion. sync_on_store=false gives no fsync durability promise.

Failure: _file_terminal resolves an exception immediately; other copies/files can still run. _finish_jobs removes a failed parent only after drain. The worker asserts success, so the supported failure outcome is experiment/service termination, not transparent recovery. Failed Future completion must never be promoted to a release witness.

At the original P0/CPU snapshot no GPU/model qualification existed. Current native GPU-only model results are recorded below; actual GPU release witness, logical-KV round trip, overlap timeline, token ITL and interference evidence remain absent.

## Continuation 01 CPU evidence

The isolated progress worktree now carries a pre-wait marker over the native queue. Its reactor-instance token and actual Future identify accepted parents; an early failed Future does not retire the marker until native I/O drains. STOP marks all accepted parents. Fatal termination only drops added marker references and preserves native error handling; it does not prove physical drain or source release. Source fence/complete_store semantics remain unchanged. Twenty-three CPU/mock tests pass; real GPU/io_uring and vLLM worker qualification remain blocked.

## Continuation 02 — native release proof is separate

A successful parent Future is insufficient to claim immediate resource reuse before scheduler fence retirement. ReleaseAnalysis now requires explicit native_reusable=True from the resource owner, in addition to complete parent conditions, zero active references and current identity. Without that confirmation, completed parents imply only potential release. No live GPU owner adapter exists. Parent.lifecycle_state distinguishes FAILED_DRAINING/FAILED_FINAL as observation only. Latest snapshots retain no completion history: a parent disappearing from the bounded window is not a completion or release witness.

## New server 02 — copy qualification is not a release witness

Real author H2D/D2H microcopies now pass, with explicit stream completion and guard checks. These are raw test buffers, not the same production KV through store/restore. The test never constructs the reactor or native cache owner and cannot prove source-fence retirement, safe reallocation, failed-parent drain, shared-slot retirement, SSD publication or overlap. The lifecycle requirements above remain mandatory. No early complete_store or new ownership system was added.

## New server 03 — native integration source audit

The frozen author's actual `_on_write_complete` calls `_release_or_cache` after successful write completion; with native LRU enabled, cold store can warm CPU staging. The old fs_config comment claiming store buffers are never cached is not a valid premise. Shared-slot retention, all-parent fences and active-request references remain separate release conditions. No early complete_store, live owner adapter or ordinary quota was introduced.

For complete successful foreground parents, the original reactor accepts only CQE res equal to frozen io_size before emitting file samples: successful native file_read event count times io_size proves that successful subset of completed SSD bytes. Its logical num_bytes alone is insufficient. Shared preload events carry actual CQE nbytes and a shared read is counted once, not per waiter. Failed-parent traces cannot be promoted to complete failed-I/O accounting. Native profiling must be active and fully flushed.

A one-time native RPC in the GPU-only diagnostic is prepared to read at most 28 tensor storage metadata records and deduplicate shared backing. Tensor backing nbytes is distinct from CUDA allocator reserved memory and physical release. This diagnostic adds no cache hooks or resource ownership system. Its actual execution status and result are reported separately in NEW_SERVER_03_REPORT.md.

The original benchmark server helper starts a new session; it cannot be launched unchanged under the current session-based budget runner. The reviewed offline uni LLM uses normal multiprocessing within the child session. The actual SSD path remains blocked by io_uring setup EPERM and has not received runtime lifecycle qualification. Exact source references/hashes and pending native integration requirements are in artifacts/prefix_io_v1/new-server-03/native-integration-review/.

## New server 03 — actual native storage metadata and source audit

native-prefix-04's one existing worker RPC measured28 CUDA BF16 unique tensor storages totaling66,977,792B. cuda_allocator_reserved_bytes=null and physical_release_witness=null. Engine shutdown/process-session drain passed; neither proves per-cache-resource reuse. No production KV bytes were read back, stored to SSD or compared. Parent/fence/shared-slot/failure-drain requirements still lack real cache qualification.

Current source _on_write_complete calls _release_or_cache after successful write: native LRU store can warm CPU staging, despite a stale fs_config comment. Successful complete foreground file_read count × frozen io_size can prove that subset of full CQE read bytes; logical num_bytes alone cannot. Shared preload raw nbytes counts a shared read once. Failed traces cannot be promoted to complete failed-I/O accounting. Exact references/hashes:artifacts/prefix_io_v1/new-server-03/native-integration-review/.

Author benchmark server helper creates a separate session and cannot be run unchanged under the current runner's cleanup. The reviewed offline uni LLM uses normal multiprocessing in its child session. Native io_uring remains EPERM;no early complete_store,new ownership system,live resource-owner adapter or ordinary quota was introduced.

## Server07 P2 实证更新（2026-09-29）

最新结果见 SERVER07_P2_REPORT.md；本节更新早期“未跑 GPU”状态，不覆盖历史证据。
真实 CUDA/AIO 的 pending producer → D2H → SSD → H2D 字节往返、零普通额度测试门下 mandatory wait、双 parent shutdown、短读 FAILED_DRAINING 后排空均已验证（progress-05）。诊断零额度不是生产调度器。
原生 allocator/connector 同 owner 诊断见 native_owner.py：按 generation 区分块，ref_cnt=0、非 null、free queue 链接、无 pending fence、全部已观察 parent 经原生确认后才记录 reusable。13 次后续原生重分配证实该有限样本可复用；不是归还显存。活跃引用观测 1424；本流 fenced-free 观测 0，不证明真实阻塞存在。
reactor 只能报告 staging、copy、I/O、parent；GPU reusable 仍为 null。生产 owner freshness/epoch/控制器适配尚未实现，joint 保持关闭。未观察到的资源或未来阶段成本仍是未知，不填零。

## P3 独立长请求的生命周期证据（2026-09-29）

两次连续 cohort 各 6 请求/768 token，原异步执行器和原 LoadPlanner；无逐请求重置/人为暂停。每轮 SSD 实读 4777443328B、实写 19267584B；3048 个来源文件哈希不变，关闭时 accepted=completed=reaped、全部在途/待确认计数为零，进程会话已排空。pending_flush_wait 与前台 staging 失败均为 0，仅限 max_num_seqs=1。本轮不提供新的生产 owner/freshness 或并发释放阻塞证据，不改变早先 mandatory/failed-draining 有界资格。详见 server07-p3-05/long-service-result.json 与本轮报告。


## Server07 P3 两并发与 I/O 深度验证（2026-09-29）

4 次混合请求重放中 2 次观测到小后缀 store 父任务的原生 flush 等待（1.079710 s / 0.351732 s）；分别涉及 3 / 4 个 917504 字节父任务。具体 scheduler 触发原因、GPU 物理块 generation 和全部保护者未记录，不能据此推导提前可释放字节。未提前 complete_store。所有已运行作业排空成功；生产 owner 控制器仍未安装。 详见 [本轮报告](SERVER07_P3_CONCURRENT_REPORT.md)。


## Real destination-fence evidence / server07-p3-10

Two GPU replays uniquely joined native pending waits to restore_destination parents 22/24/26/28, block IDs 1313/311/309/307. All eight observed source instances already had a different allocation generation and active ref=1 at the fence; the old job remained in the protection map. Parent completion later removed its protection while the new generation remained active and outside the free queue. Do not award free-capacity credit or complete_store early. Offline evidence is in server07-p3-10/qualified-analysis.json; omitted or ambiguous witnesses remain unknown. CUDA host enqueue timestamps plus device event durations are not a global device execution timeline.


## P311 增量记录

P311: 真实混合等待 0.616136868 s，3 个 destination fence parent；保留采样显示待提交小 store 与大背景 store / full iodepth 共存。目的块 active_refs=1，仍不可作为新增 free GPU bytes。free staging=0 但原 reserve 失败数=0，可回收量未知。完整性与时序限制见 SERVER07_P3_DISPATCH_READINESS_REPORT.md。


## P312 更新（2026-09-30）

P312：仅在 native store pass 调整既有未发完 parent 的遍历顺序；已接受 D2H→SSD、load/preload/fusion、真实物理条件与完整 parent completion 保留。实际 fence 保护的 3 个目的块 active_refs=1，解除保护不等于新增加空闲 GPU 容量。模型热身先排空，再一次设置可选排序，线程 shutdown 后读取统计；无新队列或常驻 Future 引用。

详见 [本轮报告](SERVER07_P3_MANDATORY_ORDER_REPORT.md)。

## Server08 P313 acceptance audit (2026-09-30)

Native resource ownership and parent completion remain unchanged. queue_read/queue_write returned successfully means accepted/pending backend work; swap_blocks_batch returned successfully means accepted CUDA work even if the subsequent end_event.record fails. Neither is physical completion, reusable space, cache visibility or durability.

The new resource-free audit consumes the issuing epoch on API acceptance (including would-defer previews), never refunds on later failure, and reports uncertain acceptance/completion separately. Native owner four-stage StageAccounting supplies in-flight counts; free reclaimable staging and the complete incoming/active/failed-draining parent count are None. The observer does not infer them from free_count or len(_active).

Normal real-GPU off/shadow store/restore/shared cases drain independently verified worker, AIO and in-flight resources. CPU event-failure injection verifies conservative classification and preserved original exceptions only. Real partial-DMA-failure cleanup safety remains unqualified; audit complete flags are not a drain witness. No early complete_store, new resource credit, queue or cancellation protocol.

## P314 来源与原 native 排空边界

固定AUX origin只读 → PRIMARY逐字节复制 → CPU时点注册（无GPU/无复制/无资源信用）→相同source/new UUID reference与成本许可→原os.link私有副本→原native LoadPlanner/共享staging/preload/fusion/异步model chain→drain/shutdown→来源后验SHA。3,048既有private paths与PRIMARY source同inode，suffix写入私有树；origin及PRIMARY来源最终全部hash保持。

注册函数 original_payloads_read=false仅指未读AUX origin payload，PRIMARY payload确实全hash；整轮复制/保全读过origin。registration不是writer lock或释放依赖证明。单model 10请求1280token输出精确、1270 ITL无歧义、最终AIO accepted=reaped/outstanding0/engine shutdown。没有以这些结束检查提前兑换source/staging/release credit。

全parent accepted/incoming/failed-draining统计、unknown-aware budget owner和完整admission协议仍待CPU资格；Future.done不代表物理释放，len(_active)也不涵盖incoming；原必需continuation、consumer与jobs_to_flush保护不能被策略停住。P314没有安装full stage caps或新DispatchShadow model策略，没有研究收益结论。
