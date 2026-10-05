"""Compose existing read-only resource witnesses with the new native owner snapshot."""
from store_readiness_worker_probe import worker_probe as original_probe

def worker_probe(worker, action, limits=None):
    from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group
    result = original_probe(worker, action, limits)
    cw = get_kv_transfer_group().connector_worker
    snapshots = []
    for h in cw.worker.handlers:
        r = h.coordinator.reactor
        if action == "finish":
            if r._worker.is_alive():
                raise RuntimeError("common native handler not shut down")
            controller = r._prefix_dispatch_controller
            snapshot = dict(admission=r.parent_admission_snapshot(),
                controller=controller.snapshot(native_shutdown=True) if controller else None,
                stage_accounting=r._prefix_stage_accounting.snapshot(),aio=r.ring.snapshot())
            accounting = snapshot["stage_accounting"]; aio = snapshot["aio"]
            snapshot["physical_drained"] = (not r._active and not r._inflight and not r._pending_copies
                and not r._copy_ready and not r._ready_fds_load and not r._ready_fds_preload
                and aio["closed"] and aio["drained"] and aio["outstanding"] == 0
                and aio["accepted"] == aio["reaped"] and accounting["valid"]
                and accounting["outstanding_records"] == 0
                and all(v["inflight_ops"] == 0 and v["inflight_bytes"] == 0 for v in accounting["stages"].values())
                and snapshot["admission"]["accepted_parents"] == 0
                and (snapshot["controller"] is None or not snapshot["controller"]["pending_attempt"]))
            if not snapshot["physical_drained"]:
                raise RuntimeError("common native physical drain not proved")
        else:
            snapshot = r.inspect_snapshot(timeout=5)
        controller = r._prefix_dispatch_controller
        if controller is not None and controller.config.byte_quantum != r.file_store.io_size:
            raise RuntimeError("frozen performance quantum differs from native storage I/O unit")
        snapshot["native_io_size"] = r.file_store.io_size
        snapshots.append(snapshot)
        if r.actual_staging_bytes > (limits or {}).get("staging_budget_bytes",1073741824):
            raise RuntimeError("actual staging over approved model budget")
    result["simple_native_control"] = snapshots
    return result
