"""Use existing model probe, then snapshot P3 state after native shutdown."""
from run_p3_native_pilot import worker_probe as original

def worker_probe(worker,action,limits=None):
    result=original(worker,action,limits)
    if action=="finish":
        from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group
        cw=get_kv_transfer_group().connector_worker
        result["start_budget_final"]=[]
        for h in cw.worker.handlers:
            h.shutdown()
            r=h.coordinator.reactor
            assert not r._worker.is_alive() and not r._inflight and not r._pending_copies
            b=r._prefix_start_budget.snapshot() if r._prefix_start_budget else None
            a=r._prefix_stage_accounting.snapshot() if r._prefix_stage_accounting else None
            assert not b or not b["faulted"],"start gate faulted"
            assert not a or (a["valid"] and a["outstanding_records"]==0),"stage accounting invalid"
            if a:
                assert all(v["accepted_bytes"]==v["transferred_bytes"] and v["failed_ops"]==0
                    and v["inflight_bytes"]==0 for v in a["stages"].values()),"stage completion mismatch"
            result["start_budget_final"].append(dict(budget=b,accounting=a,
                reactor_module=type(r).__module__,drained=True))
    return result
