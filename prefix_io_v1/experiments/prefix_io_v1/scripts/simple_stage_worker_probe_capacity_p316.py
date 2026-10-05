"""P316 read-only witnesses plus optional bounded pure-metadata CPU timing."""
import copy, functools, inspect, threading, time
from pathlib import Path
from store_readiness_worker_probe_capacity_p316 import worker_probe as original_probe

def _worker_probe(worker, action, limits=None):
    from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group
    cw = get_kv_transfer_group().connector_worker
    if action == "install":
        if (limits or {}).get("kv_budget_bytes") != 2147483648 or (limits or {}).get("staging_budget_bytes") not in (1006632960,1073741824):
            raise RuntimeError("registered capacity probe budget required")
        worker._p316_capacity_staging_budget_bytes = limits["staging_budget_bytes"]
        expected = Path(__file__).resolve().parents[3] / "third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py"
        if not cw.worker.handlers or any(Path(inspect.getfile(type(h.coordinator.reactor))).resolve() != expected.resolve()
                                         for h in cw.worker.handlers):
            raise RuntimeError("P316 native reactor source mismatch")
    cpu = getattr(worker, "_p316_metadata_cpu", None)
    sampling = getattr(worker, "_p316_optional_sampling", None)
    enabled = action == "install" and bool((limits or {}).get("native_cpu_probe", False))
    original_limits = dict(limits) if limits is not None else None
    if original_limits is not None:
        original_limits.pop("native_cpu_probe", None)
        original_limits.pop("native_cpu_phase", None)
        original_limits.pop("native_observation", None)
    if action == "install":
        mode = (limits or {}).get("native_observation", "on")
        if mode not in ("on", "off"):
            raise RuntimeError("native optional sampling mode differs")
        worker._p316_sampling_mode = mode
        sampling = OptionalSampling() if mode == "off" else None
        worker._p316_optional_sampling = sampling
        if enabled:
            if (limits or {}).get("native_cpu_probe") is not True:
                raise RuntimeError("native CPU probe must be an explicit boolean")
            cpu = MetadataCpu()
            worker._p316_metadata_cpu = cpu
        else:
            worker._p316_metadata_cpu = None
    elif cpu is not None:
        if action == "start":
            cpu.unwrap_sinks()
            cpu.phase = "cohort"
        elif action == "drain":
            phase = (limits or {}).get("native_cpu_phase", "tail")
            if phase not in ("warmup", "tail"):
                raise RuntimeError("bounded CPU phase required")
            cpu.phase = phase
        elif action == "finish":
            cpu.phase = "tail"
            cpu.unwrap_sinks()
    if sampling is not None and action in ("start", "finish"):
        sampling.restore_sinks()
    result = original_probe(worker, action, original_limits)
    if sampling is not None and action in ("install", "start"):
        sampling.suppress([h.coordinator.reactor for h in cw.worker.handlers])
    if action == "install" and cpu is not None:
        for h in cw.worker.handlers:
            cpu.install(h.coordinator.reactor)
        cpu.wrap_sinks()
    elif action == "start" and cpu is not None:
        cpu.wrap_sinks()
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
        if r.actual_staging_bytes > worker._p316_capacity_staging_budget_bytes:
            raise RuntimeError("actual staging over approved model budget")
    result["simple_native_control"] = snapshots
    result["optional_native_sampling"]=dict(mode=worker._p316_sampling_mode,
        optional_sink_suppressed=sampling is not None and action!="finish",
        finish_restore_outside_cohort=action=="finish",
        physical_proof_requires_optional_sampling=False)
    if cpu is not None:
        result["native_metadata_cpu"] = cpu.export()
        if action == "finish":
            cpu.restore()
    return result



class OptionalSampling:
    """Suppress only the optional sink; restore the existing nested probe chain."""
    def __init__(self):
        self.sinks = []

    def suppress(self, reactors):
        if self.sinks:
            raise RuntimeError("optional sampling already suppressed")
        for reactor in reactors:
            old = reactor._observation_sink
            self.sinks.append((reactor, old))
            reactor._observation_sink = None

    def restore_sinks(self):
        failure = None
        for reactor, old in reversed(self.sinks):
            if reactor._observation_sink is None:
                reactor._observation_sink = old
            else:
                failure = RuntimeError("optional sampling sink replaced while suppressed")
        self.sinks.clear()
        if failure is not None:
            raise failure

CPU_PHASES = ("warmup", "cohort", "tail")
CPU_METHODS = ("_prefix_stage_decide", "_prefix_stage_settle", "_prefix_stage",
               "_prefix_stage_copy", "_prefix_capacity_snapshot",
               "_capture_owner_snapshot")
CPU_REGIONS = CPU_METHODS + ("observer_sink",)

class MetadataCpu:
    """Fixed-size counters; diagnostic failure never changes the native call."""
    def __init__(self):
        self.phase = "warmup"
        self.local = threading.local()
        self.counts = {p:{k:dict(calls=0,thread_cpu_ns=0) for k in CPU_REGIONS}
                       for p in CPU_PHASES}
        self.restores = []
        self.sinks = []
        self.reactors = []
        self.publish_calls = 0
        self.publish_thread_cpu_ns = 0
        self.valid = True
        self.error = None

    def invalidate(self, exc):
        self.valid = False
        if self.error is None:
            try:
                self.error = (type(exc).__name__ + ": " + str(exc))[:160]
            except BaseException:
                self.error = "metadata CPU diagnostic failed"

    def measured(self, region, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if not self.valid:
                return fn(*args, **kwargs)
            outer = False
            start = None
            phase = self.phase
            try:
                depth = getattr(self.local, "depth", 0)
            except BaseException as audit:
                self.invalidate(audit)
                depth = 0
            if depth:
                return fn(*args, **kwargs)
            try:
                if self.valid:
                    self.local.depth = 1
                    outer = True
                    start = time.thread_time_ns()
            except BaseException as audit:
                self.invalidate(audit)
            try:
                return fn(*args, **kwargs)
            finally:
                try:
                    if outer and start is not None and self.valid:
                        elapsed = time.thread_time_ns() - start
                        if elapsed < 0:
                            raise ValueError("metadata CPU clock moved backwards")
                        row = self.counts[phase][region]
                        row["calls"] += 1
                        row["thread_cpu_ns"] += elapsed
                except BaseException as audit:
                    self.invalidate(audit)
                finally:
                    if outer:
                        try:
                            self.local.depth = 0
                        except BaseException as audit:
                            self.invalidate(audit)
        return wrapped

    def install(self, reactor):
        if not all(callable(getattr(reactor, key, None)) for key in CPU_METHODS):
            raise RuntimeError("P316 pure metadata hooks not available")
        self.reactors.append(reactor)
        for name in CPU_METHODS:
            old = getattr(reactor, name)
            wrapped = self.measured(name, old)
            setattr(reactor, name, wrapped)
            self.restores.append((reactor, name, old, wrapped))

    def wrap_sinks(self):
        if self.sinks:
            raise RuntimeError("CPU sink already wrapped")
        for reactor in self.reactors:
            old = reactor._observation_sink
            if old is not None:
                wrapped = self.measured("observer_sink", old)
                reactor._observation_sink = wrapped
                self.sinks.append((reactor, old, wrapped))

    def unwrap_sinks(self):
        failure = None
        for reactor, old, wrapped in reversed(self.sinks):
            if reactor._observation_sink is wrapped:
                reactor._observation_sink = old
            else:
                failure = RuntimeError("unexpected metadata CPU sink replacement")
        self.sinks.clear()
        if failure is not None:
            raise failure

    def export(self):
        try:
            if not self.valid:
                raise RuntimeError(self.error)
            start = time.thread_time_ns()
            phases = copy.deepcopy(self.counts)
            totals = {p:dict(calls=sum(r["calls"] for r in rows.values()),
                             thread_cpu_ns=sum(r["thread_cpu_ns"] for r in rows.values()))
                      for p, rows in phases.items()}
            self.publish_calls += 1
            self.publish_thread_cpu_ns += time.thread_time_ns() - start
        except BaseException as audit:
            self.invalidate(audit)
            phases = totals = None
        return dict(schema_version=1, clock="thread_time_ns",valid=self.valid,error=self.error,
            counting="outermost_only; nested inclusive hook costs are never added twice",
            phase_regions=phases, phase_totals=totals,
            publication_calls=self.publish_calls,
            publication_thread_cpu_ns=self.publish_thread_cpu_ns,
            scope="bounded pure metadata/control/observer hooks; includes timer overhead",
            excludes=["native GPU/IO launch and wait", "model execution CPU",
                      "frontend/core whole-process CPU", "unwrapped cache bookkeeping"],
            algorithm_cpu_claim=False, full_production_cpu_breakdown=False)

    def restore(self):
        failure = None
        try:
            self.unwrap_sinks()
        except BaseException as audit:
            failure = audit
        for reactor, name, old, wrapped in reversed(self.restores):
            if getattr(reactor, name) is wrapped:
                setattr(reactor, name, old)
            else:
                failure = RuntimeError("unexpected metadata CPU method replacement")
        self.restores.clear()
        if failure is not None:
            raise failure


def cpu_delta(before, after):
    """A bounded start/end difference; callers keep phase boundaries explicit."""
    if before is None or after is None:
        return None
    if before.get("valid") is not True or after.get("valid") is not True:
        raise ValueError("metadata CPU evidence invalid")
    if before["clock"] != "thread_time_ns" or after["clock"] != "thread_time_ns":
        raise ValueError("metadata CPU clock differs")
    phases = {}
    for phase in CPU_PHASES:
        phases[phase] = {}
        for region in CPU_REGIONS:
            row = {}
            for key in ("calls", "thread_cpu_ns"):
                a = before["phase_regions"][phase][region][key]
                b = after["phase_regions"][phase][region][key]
                if type(a) is not int or type(b) is not int or not 0 <= a <= b:
                    raise ValueError("metadata CPU counter went backwards")
                row[key] = b - a
            phases[phase][region] = row
    totals = {p:dict(calls=sum(r["calls"] for r in rows.values()),
                     thread_cpu_ns=sum(r["thread_cpu_ns"] for r in rows.values()))
              for p, rows in phases.items()}
    return dict(phase_regions=phases,phase_totals=totals,clock="thread_time_ns",
                counting=after["counting"],algorithm_cpu_claim=False,
                full_production_cpu_breakdown=False)


def worker_probe(worker, action, limits=None):
    try:
        result = _worker_probe(worker, action, limits)
    except BaseException as original:
        cpu = getattr(worker, "_p316_metadata_cpu", None)
        if action in ("install", "start", "finish"):
            for cleanup_fn in (
                    cpu.restore if cpu is not None else None,
                    getattr(worker, "_p316_optional_sampling", None).restore_sinks
                    if getattr(worker, "_p316_optional_sampling", None) is not None else None):
                if cleanup_fn is not None:
                    try:
                        cleanup_fn()
                    except BaseException as cleanup:
                        original.add_note("optional diagnostic cleanup also failed: " + str(cleanup))
        raise
    return result
