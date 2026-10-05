"""Bounded native full-step evidence, using the frozen G2 observer unchanged.

This is a thin observer of original execute/prepare/sample calls. It neither
implements model execution nor consumes native completions. CUDA event durations
and host I/O membership remain separate. A conservative causal interval can
prove an owner acceptance happened inside a selected GPU event pair, without
pretending that the GPU and host clocks have an exact absolute mapping.
"""
from dataclasses import asdict
import hashlib
import inspect
import math
from pathlib import Path
import time
import weakref


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def primitive(value, depth=0):
    require(depth <= 6, "bounded scalar metadata depth")
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is str:
        require(len(value) <= 2048, "bounded scalar metadata string")
        return value
    if type(value) is list:
        require(len(value) <= 128, "bounded scalar metadata list")
        return [primitive(item, depth + 1) for item in value]
    if type(value) is dict:
        require(len(value) <= 32 and all(type(key) is str for key in value),
                "bounded scalar metadata mapping")
        return {key: primitive(item, depth + 1) for key, item in value.items()}
    raise ValueError("metadata must not retain tensors, futures or resource owners")


class EventProxy:
    """Delegate each original event operation once; retain its host bounds."""
    def __init__(self, raw, clock):
        self.raw, self.clock = raw, clock
        self.record_before_ns = self.record_after_ns = None
        self.completed_query_ns = None
        self.elapsed_ns = None
        self.elapsed_target = None

    def record(self):
        require(self.record_before_ns is None, "event record must occur once")
        self.record_before_ns = self.clock()
        result = self.raw.record()
        self.record_after_ns = self.clock()
        require(type(self.record_before_ns) is int and
                type(self.record_after_ns) is int and
                0 < self.record_before_ns <= self.record_after_ns,
                "real monotonic event-record bracket")
        return result

    def query(self):
        ready = self.raw.query()
        require(type(ready) is bool, "CUDA event query must be an exact bool")
        if ready:
            now = self.clock()
            require(self.record_before_ns is not None and type(now) is int and
                    now >= self.record_before_ns, "completed event host upper bound")
            if self.completed_query_ns is None:
                self.completed_query_ns = now
        return ready

    def elapsed_time(self, other):
        require(type(other) is EventProxy and self.completed_query_ns is not None
                and other.completed_query_ns is not None,
                "both actual events must have completed before elapsed time")
        elapsed = self.raw.elapsed_time(other.raw)
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0,
                "positive real CUDA event elapsed milliseconds")
        self.elapsed_ns = round(elapsed * 1_000_000)
        require(self.elapsed_ns > 0, "positive CUDA event nanoseconds")
        self.elapsed_target = weakref.ref(other)
        return elapsed


class BoundedEventFactory:
    def __init__(self, event_class, *, clock=time.monotonic_ns, max_steps=128):
        require(type(max_steps) is int and 1 <= max_steps <= 4096,
                "bounded full-step event count")
        self.event_class, self.clock = event_class, clock
        self.max_steps, self.events = max_steps, []

    def __call__(self, *, enable_timing):
        require(enable_timing is True and len(self.events) < 2 * self.max_steps,
                "timed event capacity")
        event = EventProxy(self.event_class(enable_timing=True), self.clock)
        self.events.append(event)
        return event

    def witnesses(self, ordinals):
        require(len(self.events) == 2 * len(ordinals), "complete event/frame bijection")
        rows = []
        for index, ordinal in enumerate(ordinals):
            start, end = self.events[2 * index:2 * index + 2]
            require(start.elapsed_target is not None and start.elapsed_target() is end,
                    "elapsed duration belongs to this exact event pair")
            require(all(type(value) is int and value > 0 for value in (
                start.record_before_ns, start.record_after_ns, start.completed_query_ns,
                end.record_before_ns, end.record_after_ns, end.completed_query_ns,
                start.elapsed_ns)), "every actual event pair resolved")
            require(start.record_after_ns <= end.record_before_ns and
                    end.record_after_ns <= end.completed_query_ns,
                    "event recording/query ordering")
            rows.append(dict(native_step_ordinal=ordinal,
                start_record_before_ns=start.record_before_ns,
                start_record_after_ns=start.record_after_ns,
                start_completed_query_ns=start.completed_query_ns,
                end_record_before_ns=end.record_before_ns,
                end_record_after_ns=end.record_after_ns,
                end_completed_query_ns=end.completed_query_ns,
                gpu_elapsed_ns=start.elapsed_ns,
                event_elapsed_source="torch.cuda.Event.elapsed_time",
                cross_clock_absolute_mapping=False))
        return rows


def event_class_source(event_class, origin):
    require(origin in ("native_gpu_recording", "cpu_fixture"), "explicit capture origin")
    if origin == "cpu_fixture":
        return dict(origin=origin, native_class_qualified=False)
    require(event_class.__module__ == "torch.cuda.streams" and event_class.__name__ == "Event",
            "original installed torch CUDA Event class required")
    source = Path(inspect.getsourcefile(event_class)).resolve()
    raw = source.read_bytes()
    require(0 < len(raw) <= 4 * 1024**2 and all(
        Path(inspect.getsourcefile(getattr(event_class, name))).resolve() == source
        for name in ("record", "query", "elapsed_time")),
        "all delegated event APIs belong to the same installed source")
    return dict(origin=origin, path=str(source), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                module=event_class.__module__, class_name=event_class.__name__,
                native_class_qualified=False)


class FullStepCapture:
    def __init__(self, *, run_id, origin, selected_offsets, action, clock=time.monotonic_ns):
        require(type(run_id) is str and 0 < len(run_id) <= 128, "bounded run identity")
        require(type(selected_offsets) is tuple and 0 < len(selected_offsets) <= 8 and
                all(type(item) is int and 1 <= item < 128 for item in selected_offsets) and
                tuple(sorted(set(selected_offsets))) == selected_offsets,
                "frozen bounded pure decode selection offsets")
        require(action is None or callable(action), "optional native preload trigger")
        require(origin in ("native_gpu_recording", "cpu_fixture"), "explicit capture origin")
        require(origin != "native_gpu_recording" or clock is time.monotonic_ns,
                "native host boundaries use the real builtin monotonic clock")
        self.run_id, self.origin, self.selected_offsets = run_id, origin, selected_offsets
        self.action, self.clock = action, clock
        self.valid, self.failures, self.actions = True, [], []
        self.observer = self.factory = None
        self.event_source = None
        self.runner_ref = None
        self.original_prepare = self.prepare_wrapper = None
        self._detached = False

    def fail(self, reason):
        self.valid = False
        if len(self.failures) < 16:
            self.failures.append(str(reason)[:256])

    def after_prepare(self):
        observer = self.observer
        require(observer.enabled and observer.scalar.enabled,
                "original full-step scalar observer still valid")
        active = observer.events.active
        require(active is not None, "matching CUDA event pair is open")
        ordinal, start, end = active
        pending = observer.scalar._adapter._pending
        require(type(pending) is dict and pending["phase"] == "prepared" and
                pending["ordinal"] == ordinal, "actual prepared frame and event ordinal agree")
        frame = pending["frame"]
        offset = len(observer.frames)
        # Query only; no spin, stream wait or synchronize is introduced.
        ready = start.query()
        if offset not in self.selected_offsets:
            return
        require(frame.step_kind == "decode" and frame.active_decode == frame.batch > 0 and
                frame.prefill_tokens == 0, "selected step is actual original pure decode")
        require(ready, "selected GPU start event not complete at original prepared boundary")
        require(not any(row["step_offset"] == offset for row in self.actions),
                "selected native action must be observed exactly once")
        before = self.clock()
        result = None if self.action is None else self.action(offset, ordinal)
        after = self.clock()
        self.actions.append(dict(step_offset=offset, native_step_ordinal=ordinal,
            trigger_before_ns=before, trigger_after_ns=after,
            trigger_result=primitive(result), action_enabled=self.action is not None))

    def attach_prepare(self, runner):
        self.runner_ref = weakref.ref(runner)
        self.original_prepare = runner._prepare_inputs
        original = self.original_prepare
        def prepare(*args, **kwargs):
            # The original observer/prepare executes once with unchanged results.
            result = original(*args, **kwargs)
            if self.valid:
                try:
                    self.after_prepare()
                except Exception as exc:
                    self.fail(type(exc).__name__ + ": " + str(exc))
                except BaseException:
                    self.fail("observation control signal")
                    raise
            return result
        self.prepare_wrapper = prepare
        runner._prepare_inputs = prepare

    def export(self):
        require(not self._detached, "capture export precedes detach")
        observer = self.observer
        # Original sampling has completed. Query remaining diagnostic events once.
        observer.resolve_ready()
        frames = [asdict(frame) for frame in observer.frames]
        try:
            require(observer.enabled and observer.scalar.enabled and self.valid,
                    "all original scalar and event observations remain valid")
            require(observer.events.active is None and not observer.events.pending,
                    "natural model completion must resolve every event pair")
            require(len(frames) == 128, "exactly 128 complete actual original steps")
            ordinals = [frame["native_step_ordinal"] for frame in frames]
            require(ordinals == list(range(ordinals[0], ordinals[0] + 128)),
                    "complete contiguous original native ordinals")
            witnesses = self.factory.witnesses(ordinals)
            require(len(observer.events.diagnostics) == 128 and all(
                diagnostic.ordinal == witness["native_step_ordinal"] and
                diagnostic.raw_event_elapsed_ns == witness["gpu_elapsed_ns"]
                for diagnostic, witness in zip(observer.events.diagnostics, witnesses)),
                "new event bounds preserve the frozen G2 elapsed measurement")
            require(tuple(row["step_offset"] for row in self.actions) == self.selected_offsets,
                    "all and only frozen selected steps were observed")
        except Exception as exc:
            witnesses = []
            self.fail(type(exc).__name__ + ": " + str(exc))
        return dict(schema_version=1, scope="server11_full_step_native_capture_v1",
            run_id=self.run_id, origin=self.origin, valid=self.valid,
            frames=frames, event_witnesses=witnesses, actions=list(self.actions),
            failures=list(self.failures), event_source=self.event_source,
            selected_offsets=list(self.selected_offsets),
            pending_event_pairs=len(observer.events.pending),
            open_event_pair=observer.events.active is not None,
            gpu_duration_scope="execute_model_through_sample_tokens_current_stream",
            host_window_scope="original_execute_through_original_sample_host_calls",
            cross_clock_absolute_mapping=False, no_added_synchronization=True,
            production_qualified=False, performance_effect_verified=False)

    def detach(self):
        runner = self.runner_ref() if self.runner_ref is not None else None
        if runner is not None and vars(runner).get("_prepare_inputs") is self.prepare_wrapper:
            runner._prepare_inputs = self.original_prepare
        if self.observer is not None:
            self.observer.detach()
        self.action = self.original_prepare = self.prepare_wrapper = None
        self._detached = True


def install(worker, *, common, root, refs, run_id, event_class, selected_offsets=(16,),
            action=None, origin="native_gpu_recording"):
    """Called only inside the root's bounded GPU guard after source preflight.

    `common` is the existing frozen G2 module. The root retains responsibility
    for the actual device/source/SDK/ledger gate, native journal and shutdown.
    The callback only invokes the original handler.preload_async; this module
    never constructs or accesses a second execution/ownership queue.
    """
    root = Path(root)
    capture = FullStepCapture(run_id=run_id, origin=origin,
                              selected_offsets=selected_offsets, action=action)
    try:
        capture.event_source = event_class_source(event_class, origin)
        worker_module = common.load_ref(root, refs[common.WORKER])
        scalar = worker_module.load_pinned(common.safe(root, common.SCALAR),
                                           *worker_module.FROZEN["scalar"])
        binding = scalar.MethodBinding(tuple(scalar.SourceRef(
            str(common.safe(root, name)), refs[name]["bytes"], refs[name]["sha256"])
            for name in (common.CONTEXT, common.RUNNER)),
            tuple("GPUModelRunner." + name for name in scalar.METHODS),
            "native_candidate" if origin == "native_gpu_recording" else "cpu_fixture")
        capture.factory = BoundedEventFactory(event_class)
        capture.observer = worker_module.connect_worker_observation(worker,
            scalar_source=common.safe(root, common.SCALAR),
            frame_source=common.safe(root, common.FRAME), binding=binding,
            run_id=run_id,
            native_source_sha256=refs[
                "third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py"]["sha256"],
            event_factory=capture.factory, enabled=True, max_pending=128, max_steps=128)
        require(capture.observer.enabled,
                "frozen G2 native observer rejected this actual runtime")
        capture.attach_prepare(worker.model_runner)
        return capture
    except BaseException:
        capture.detach()
        raise
