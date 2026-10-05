"""Experiment-only composition of existing flush and readiness observers."""
from flush_cause_worker_probe_capacity_p316 import worker_probe as original
from prefix_io_control.store_readiness import StoreReadinessProbe

def worker_probe(worker, action, limits=None):
    from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group
    cw = get_kv_transfer_group().connector_worker
    if action == "install":
        result = original(worker, action, limits)
        worker._readiness_probes = []
        return result
    if action == "start":
        if worker._readiness_probes:
            raise RuntimeError("readiness cohort cannot start twice")
        # Warmup has already been drained by the existing driver.
        for h in cw.worker.handlers:
            r = h.coordinator.reactor
            if r._active or r._pending_copies or r._inflight or getattr(r, "_prefix_start_budget", None):
                raise RuntimeError("readiness diagnostic requires a drained native baseline")
            if not r.progress_bridge_enabled:
                raise RuntimeError("mandatory progress identity is required")
        result = original(worker, action, limits)
        for h in cw.worker.handlers:
            r = h.coordinator.reactor
            probe = StoreReadinessProbe(run_id=r._prefix_progress.run_id,
                interval_ns=1_000_000, mandatory_only=True)
            old = r._observation_sink
            def combined(reactor, old=old, probe=probe):
                if old is not None:
                    old(reactor)
                probe(reactor)
            r._observation_sink = combined
            worker._readiness_probes.append((h, r, old, combined, probe))
        result["store_readiness_installed"] = True
        return result
    if action == "finish":
        # Unwrap ours first so the existing driver's own restore chain stays valid.
        entries = worker._readiness_probes
        for h, r, old, combined, probe in entries:
            if r._observation_sink is combined:
                r._observation_sink = old
        result = original(worker, action, limits)
        result["store_readiness"] = []
        for h, r, old, combined, probe in entries:
            h.shutdown()
            if r._worker.is_alive() or r._active or r._inflight or r._pending_copies:
                raise RuntimeError("native readiness diagnostic did not drain")
            observation = probe.export()
            if observation["faulted"]:
                raise RuntimeError("readiness observer failed: " + str(observation["error"]))
            result["store_readiness"].append(observation)
        return result
    return original(worker, action, limits)
