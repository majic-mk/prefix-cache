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
