# P0 lifecycle audit; not P2 GPU acceptance
Store success: scheduler _build_store_jobs → worker prepare_store_kv delays submission until next step → handler.transfer_async → TransferCoordinator.submit_store records compute_event → D2H waits that event → _on_store_copy_done immediately queues native SSD write → _on_write_complete checks full size, publishes, settles slot → _file_terminal completes success only after all files → worker completed_jobs → scheduler.update_connector_output complete_store and fence removal.

Active requests protect blocks by ref_cnt. Finished full-attention requests register each pending parent in _block_id_to_pending_jobs and return False, allowing free-list entry. Reallocation of these blocks creates jobs_to_flush before overwrite. Free-list presence therefore differs from immediate overwrite safety. All protecting parents must finish; an active reference still means zero newly reusable bytes. The map does not supply allocation generation: live observation must add an owner-observed identity lifetime, never infer it from recycled block_id alone.

Mandatory chain: handle_preemptions submits pending stores first; OffloadingWorker.wait delegates to handler.wait/futures_wait. Native reactor runs independently. There is no quota-aware mandatory signal yet. Ordinary throttling remains disabled.

Load ready requires all selected files/copies. _drain_cuda_copies queries end_event; event creation or API return is insufficient. Shared slots stay protected while cached or copies_inflight > 0. Existing clean eviction may supply space without I/O.

source-safe, buffer-reusable, cache-visible and durable are distinct. No independent D2H source-release ACK exists. Files may become visible before parent completion. sync_on_store=false gives no fsync durability promise.

Failure: _file_terminal resolves an exception immediately; other copies/files can still run. _finish_jobs removes a failed parent only after drain. The worker asserts success, so the supported failure outcome is experiment/service termination, not transparent recovery. Failed Future completion must never be promoted to a release witness.

No actual GPU release witness, logical-KV round trip, model result, overlap timeline, token ITL or interference result exists. CPU/mock evidence is reported separately.

## Continuation 01 CPU evidence

The isolated progress worktree now carries a pre-wait marker over the native queue. Its reactor-instance token and actual Future identify accepted parents; an early failed Future does not retire the marker until native I/O drains. STOP marks all accepted parents. Fatal termination only drops added marker references and preserves native error handling; it does not prove physical drain or source release. Source fence/complete_store semantics remain unchanged. Twenty-three CPU/mock tests pass; real GPU/io_uring and vLLM worker qualification remain blocked.

## Continuation 02 — native release proof is separate

A successful parent Future is insufficient to claim immediate resource reuse before scheduler fence retirement. ReleaseAnalysis now requires explicit native_reusable=True from the resource owner, in addition to complete parent conditions, zero active references and current identity. Without that confirmation, completed parents imply only potential release. No live GPU owner adapter exists. Parent.lifecycle_state distinguishes FAILED_DRAINING/FAILED_FINAL as observation only. Latest snapshots retain no completion history: a parent disappearing from the bounded window is not a completion or release witness.

## New server 02 — copy qualification is not a release witness

Real author H2D/D2H microcopies now pass, with explicit stream completion and guard checks. These are raw test buffers, not the same production KV through store/restore. The test never constructs the reactor or native cache owner and cannot prove source-fence retirement, safe reallocation, failed-parent drain, shared-slot retirement, SSD publication or overlap. The lifecycle requirements above remain mandatory. No early complete_store or new ownership system was added.
