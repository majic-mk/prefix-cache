"""Optional thin wiring to the original synchronous runner; stdlib at import.

This candidate does not implement an executor, sample a token, issue I/O, wait
on CUDA, release a native owner, or grant a cost/production qualification.
Actual CUDA events may be supplied only by the separately guarded GPU driver.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
import math
from pathlib import Path
import sys
import types
import weakref

FROZEN = {
    "scalar": (14795, "347fb842989e8e14c73968528396721b546a11d7312206aae2efbb3739b0f3cb"),
    "frame": (16067, "bea850fd60831010c2ac0bf88c1f5f121f3ffca50d63ab56759319ed6e512582"),
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def load_pinned(path, nbytes, sha256):
    """Bounded source-only import; caller pins its complete dependency source."""
    path = Path(path).absolute()
    require(not path.is_symlink() and path.is_file() and type(nbytes) is int and
            0 < nbytes <= 4 * 1024**2 and path.stat().st_size == nbytes,
            "bounded regular source bytes")
    with path.open("rb") as handle:
        raw = handle.read(nbytes + 1)
    require(len(raw) == nbytes and hashlib.sha256(raw).hexdigest() == sha256,
            "source bytes/SHA drift")
    name = "_g2_source_" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:20]
    if name in sys.modules:
        module = sys.modules[name]
        require(Path(module.__file__).resolve() == path.resolve(), "private module path")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@dataclass(frozen=True)
class StepEventDiagnostic:
    run_id: str
    ordinal: int
    origin: str
    raw_event_elapsed_ns: int
    # Event time has no proved mapping to NativeWindowJournal's host clock.
    scope: str = "execute_model_through_sample_tokens_current_stream"

    def __post_init__(self):
        require(type(self.run_id) is str and 0 < len(self.run_id) <= 128 and
                type(self.ordinal) is int and self.ordinal >= 0 and
                self.origin in ("cpu_fixture", "native_candidate") and
                type(self.raw_event_elapsed_ns) is int and self.raw_event_elapsed_ns > 0 and
                self.scope == "execute_model_through_sample_tokens_current_stream",
                "bounded unqualified event diagnostic scalars")

    @property
    def mapped_start_ns(self): return None
    @property
    def mapped_end_ns(self): return None
    @property
    def gpu_elapsed_ns(self): return None
    @property
    def cross_clock_mapping_verified(self): return False
    @property
    def GPU_collector_verified(self): return False
    @property
    def production_qualified(self): return False
    @property
    def performance_claim(self): return False


class QueryOnlyEventObserver:
    """Bounded observation Event handles; no synchronization or clock mapping."""
    def __init__(self, run_id, origin, *, max_pending=128, max_steps=4096):
        require(type(run_id) is str and 0 < len(run_id) <= 128, "bounded run identity")
        require(origin in ("cpu_fixture", "native_candidate"), "explicit event origin")
        require(type(max_pending) is int and 1 <= max_pending <= 128 and
                type(max_steps) is int and 1 <= max_steps <= 4096, "event capacity bounds")
        self.run_id, self.origin = run_id, origin
        self.max_pending, self.max_steps = max_pending, max_steps
        self.active = None
        self.pending = []
        self.diagnostics = []
        self.first_ordinal = None
        self.closed_steps = 0
        self.valid = True
        self.reason = "unqualified_event_diagnostics_only"

    def invalidate(self, reason):
        self.valid = False
        self.reason = str(reason)[:160]
        self.active = None
        self.pending.clear()

    def begin(self, ordinal, factory):
        require(self.valid and type(ordinal) is int and ordinal >= 0,
                "valid original ordinal")
        require(self.active is None and len(self.pending) < self.max_pending and
                self.closed_steps < self.max_steps, "event capacity or open step")
        if self.first_ordinal is None:
            self.first_ordinal = ordinal
        require(ordinal == self.first_ordinal + self.closed_steps, "contiguous event ordinal")
        start, end = factory(enable_timing=True), factory(enable_timing=True)
        require(start is not end, "distinct observation events")
        start.record()
        self.active = (ordinal, start, end)

    def finish(self, frame):
        require(self.valid and self.active is not None, "event frame still open")
        ordinal, start, end = self.active
        require(frame.native_step_ordinal == ordinal and frame.gpu_elapsed_ns is None and
                frame.prepared.context_basis == "pre_computed_tokens",
                "closed original scalar frame")
        end.record()
        self.pending.append((ordinal, start, end))
        self.active = None
        self.closed_steps += 1

    def resolve_ready(self):
        if not self.valid:
            return ()
        resolved, remaining = [], []
        for ordinal, start, end in self.pending:
            ready_start, ready_end = start.query(), end.query()
            require(type(ready_start) is bool and type(ready_end) is bool,
                    "event query exact bool")
            if not (ready_start and ready_end):
                remaining.append((ordinal, start, end))
                continue
            elapsed = start.elapsed_time(end)
            require(type(elapsed) in (int, float) and math.isfinite(elapsed) and
                    elapsed > 0, "positive finite event elapsed scalar")
            ns = round(elapsed * 1_000_000)
            require(ns > 0, "positive event nanoseconds")
            require(len(self.diagnostics) + len(resolved) < self.max_steps,
                    "bounded resolved scalar diagnostics")
            resolved.append(StepEventDiagnostic(self.run_id, ordinal, self.origin, ns))
        self.pending = remaining
        self.diagnostics.extend(resolved)
        return tuple(resolved)


class WorkerObservationConnection:
    runtime_hook_status = "not_installed"
    GPU_collector_verified = False
    production_qualified = False
    performance_claim = False

    def __init__(self):
        self.enabled = False
        self.status = "off"
        self.reason = "off"
        self.scalar = None
        self.events = None
        self._runner = None
        self._factory = None
        self._inner = ()
        self._outer = ()

    def invalidate(self, reason):
        self.enabled = False
        self.status, self.reason = "observation_invalid", str(reason)[:160]
        if self.scalar is not None:
            self.scalar._fail(self.reason)
        if self.events is not None:
            self.events.invalidate(self.reason)

    def _original_failed(self, reason):
        try:
            self.invalidate(reason)
        except BaseException:
            self.enabled = False

    def observe(self, action):
        if not self.enabled:
            return
        try:
            action()
        except Exception as exc:
            self.invalidate("observer:" + type(exc).__name__)

    def disable(self):
        # Installed wrappers immediately return through the original scalar path.
        self.invalidate("disabled")

    def detach(self):
        self.enabled = False
        runner = self._runner() if self._runner is not None else None
        if runner is not None:
            for name, inner, outer in zip(("execute_model", "sample_tokens"),
                                          self._inner, self._outer):
                if vars(runner).get(name) is outer:
                    setattr(runner, name, inner)
        if self.scalar is not None:
            self.scalar.detach()
        if self.events is not None:
            self.events.invalidate("detached")
        self.status = "detached"

    @property
    def frames(self):
        return () if self.scalar is None else self.scalar._adapter.frames

    def resolve_ready(self):
        resolved = []
        self.observe(lambda: resolved.extend(self.events.resolve_ready()))
        return tuple(resolved)


def connect_worker_observation(worker_wrapper, *, scalar_source, frame_source,
        binding, run_id, native_source_sha256, event_factory, enabled=False,
        clock=None, max_pending=128, max_steps=4096):
    """No owner retained. Off precedes every input/source/worker/factory read.

    Native binding is accepted only by the same frozen scalar connector; it is
    still unqualified observation, never a production table or effect result.
    The GPU caller, rather than this function, must first pass its scope guard.
    """
    connection = WorkerObservationConnection()
    if enabled is not True:
        return connection
    try:
        scalar = load_pinned(scalar_source, *FROZEN["scalar"])
        frame = scalar.load_frozen_adapter(frame_source)
        require(type(binding) is scalar.MethodBinding, "exact frozen binding")
        require(callable(event_factory), "ephemeral event factory required")
        factory_ref = (weakref.WeakMethod(event_factory) if type(event_factory) is types.MethodType
                       else weakref.ref(event_factory))
        runner = worker_wrapper.model_runner
        originals = scalar._check_original_methods(runner, binding)
        raw_path = Path(inspect.unwrap(originals[0]().__func__).__code__.co_filename).resolve()
        raw_ref = next(ref for ref in binding.refs if Path(ref.path).resolve() == raw_path)
        adapter = frame.FullStepFrameAdapter(run_id,
                raw_ref.sha256, native_source_sha256, enabled=True, max_steps=max_steps)
        arguments = {} if clock is None else {"clock": clock}
        inner = scalar.connect_runtime_scalar_observer(runner, adapter=adapter,
            adapter_source=frame_source, binding=binding, enabled=True, **arguments)
        connection.scalar = inner
        require(inner.enabled, "frozen scalar preflight rejected")
        connection.events = QueryOnlyEventObserver(run_id, binding.origin,
                max_pending=max_pending, max_steps=max_steps)
        connection._runner = weakref.ref(runner)
        connection._factory = factory_ref
        execute_inner, sample_inner = runner.execute_model, runner.sample_tokens
        connection._inner = (execute_inner, sample_inner)
        connection.enabled = True
        connection.status = "CPU_WIRING_READY_NATIVE_QUALIFICATION_PENDING"
        connection.reason = "normal synchronous original methods only"

        def execute(*args, **kwargs):
            if not connection.enabled:
                return execute_inner(*args, **kwargs)
            def begin():
                owner = connection._runner()
                factory = factory_ref()
                require(owner is not None and factory is not None, "live ephemeral runner/factory")
                connection.events.begin(owner._profile_step, factory)
            connection.observe(begin)
            try:
                result = execute_inner(*args, **kwargs)
            except BaseException:
                connection._original_failed("original_execute_failed")
                raise
            if connection.enabled and not inner.enabled:
                connection.invalidate("scalar_execute_invalid")
            return result

        def sample(*args, **kwargs):
            if not connection.enabled:
                return sample_inner(*args, **kwargs)
            before = len(adapter.frames)
            try:
                result = sample_inner(*args, **kwargs)
            except BaseException:
                connection._original_failed("original_sample_failed")
                raise
            def finish():
                require(inner.enabled and adapter.valid and len(adapter.frames) == before + 1,
                        "exactly one actual original sample frame")
                closed = adapter.frames[-1]
                require(type(closed) is frame.ClosedFrame, "frozen closed frame type")
                connection.events.finish(closed)
            connection.observe(finish)
            return result

        connection._outer = (execute, sample)
        runner.execute_model, runner.sample_tokens = execute, sample
        return connection
    except Exception as exc:
        connection.invalidate("preflight:" + type(exc).__name__)
        connection.detach()
        connection.status = "observation_invalid"
        return connection
