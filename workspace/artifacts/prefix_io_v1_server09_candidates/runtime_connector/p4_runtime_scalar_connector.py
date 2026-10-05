"""Optional CPU scalar wiring around original synchronous vLLM methods.

Candidate only: no live hook is installed by importing this module. Source
preflight performs CPU file reads before wiring; method calls perform no source
reads, CUDA operations, synchronization, sampling, I/O or resource release.
"""
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
import time
import types
import weakref
from dataclasses import dataclass

ADAPTER_BYTES = 16067
ADAPTER_SHA256 = "bea850fd60831010c2ac0bf88c1f5f121f3ffca50d63ab56759319ed6e512582"
METHODS = ("execute_model", "_prepare_inputs", "sample_tokens")
AUTHOR_RUNNER_BYTES = 329690
AUTHOR_RUNNER_SHA256 = "61828af13a5f523c9683df560374d75bc1df93254a9ac4a449c95447100e9075"
MAX_SOURCE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class SourceRef:
    path: str
    nbytes: int
    sha256: str

    def checked_path(self):
        if type(self.path) is not str or not Path(self.path).is_absolute():
            raise ValueError("absolute source path required")
        path = Path(self.path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("regular source file required")
        if type(self.nbytes) is not int or not 1 <= self.nbytes <= MAX_SOURCE_BYTES:
            raise ValueError("exact source byte count required")
        if (type(self.sha256) is not str or len(self.sha256) != 64 or
                any(c not in "0123456789abcdef" for c in self.sha256)):
            raise ValueError("exact source SHA required")
        if path.stat().st_size != self.nbytes:
            raise ValueError("source byte count drift")
        with path.open("rb") as source:
            data = source.read(self.nbytes + 1)
        if len(data) != self.nbytes or hashlib.sha256(data).hexdigest() != self.sha256:
            raise ValueError("source bytes/SHA drift")
        return path.resolve()


@dataclass(frozen=True)
class MethodBinding:
    """Every decorator code file and the underlying method file must be pinned."""
    refs: tuple
    qualnames: tuple
    origin: str


def load_frozen_adapter(path):
    """Import only the already frozen, SHA-verified stdlib adapter."""
    ref = SourceRef(str(Path(path).absolute()), ADAPTER_BYTES, ADAPTER_SHA256)
    source = ref.checked_path()
    name = "_p4_frozen_frame_" + hashlib.sha256(str(source).encode()).hexdigest()[:20]
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != source:
            raise ValueError("private adapter module identity differs")
        return module
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _check_original_methods(runner, binding):
    if type(binding) is not MethodBinding or binding.origin not in ("cpu_fixture", "native_candidate"):
        raise ValueError("explicit candidate source origin required")
    if type(binding.refs) is not tuple or not 1 <= len(binding.refs) <= 8:
        raise ValueError("bounded source refs required")
    sources = set()
    for ref in binding.refs:
        if type(ref) is not SourceRef:
            raise ValueError("exact source ref required")
        path = ref.checked_path()
        if path in sources:
            raise ValueError("duplicate source ref")
        sources.add(path)
    if type(binding.qualnames) is not tuple or len(binding.qualnames) != 3:
        raise ValueError("all original method qualnames required")
    if any(type(q) is not str or not q or len(q) > 200 for q in binding.qualnames):
        raise ValueError("bounded exact qualnames required")
    if binding.origin == "native_candidate" and binding.qualnames != tuple(
            "GPUModelRunner." + name for name in METHODS):
        raise ValueError("author runner method identities required")
    originals = []
    for name, qualname in zip(METHODS, binding.qualnames):
        if name in vars(runner):
            raise ValueError("existing instance method override cannot be replaced")
        original = getattr(runner, name)
        if type(original) is not types.MethodType or original.__self__ is not runner:
            raise ValueError("original bound Python method required")
        function = original.__func__
        seen = set()
        for _ in range(8):
            if type(function) is not types.FunctionType or id(function) in seen:
                raise ValueError("unsupported decorator chain")
            seen.add(id(function))
            if Path(function.__code__.co_filename).resolve() not in sources:
                raise ValueError("original/decorator code source is not pinned")
            if not hasattr(function, "__wrapped__"):
                break
            function = function.__wrapped__
        else:
            raise ValueError("decorator chain bound exceeded")
        if function.__qualname__ != qualname:
            raise ValueError("original method qualname differs")
        raw_path = Path(function.__code__.co_filename).resolve()
        raw_ref = next(ref for ref in binding.refs if Path(ref.path).resolve() == raw_path)
        if binding.origin == "native_candidate" and (
                raw_ref.nbytes != AUTHOR_RUNNER_BYTES or raw_ref.sha256 != AUTHOR_RUNNER_SHA256):
            raise ValueError("frozen author runner source required")
        originals.append(weakref.WeakMethod(original))
    return tuple(originals)


def _check_domain(runner):
    if (getattr(runner, "use_async_scheduling", None) is not False or
            getattr(runner, "speculative_config", object()) is not None or
            getattr(runner, "is_pooling_model", None) is not False):
        raise ValueError("explicit synchronous ordinary generation required")
    for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"):
        value = getattr(runner.parallel_config, name, None)
        if type(value) is not int or value != 1:
            raise ValueError("single TP/PP/DP domain required")


def _read_clock(clock_ref):
    if clock_ref is None:
        return time.monotonic_ns()
    clock = clock_ref()
    if clock is None:
        raise ValueError("CPU test clock no longer exists")
    return clock()


class RuntimeScalarConnection:
    """Only weak runner/method refs and the bounded scalar adapter are retained."""
    runtime_hook_status = "not_installed"
    GPU_collector_verified = False
    production_qualified = False
    gpu_verified = False

    def __init__(self):
        self.enabled = False
        self.status = "off"
        self.last_reason = "off"
        self.closed_run = None
        self._runner = None
        self._originals = ()
        self._wrappers = ()
        self._adapter = None
        self._clock = None

    def _fail(self, reason):
        self.enabled = False
        self.status = "observation_invalid"
        self.last_reason = str(reason)[:160]
        if self._adapter is not None:
            try:
                self._adapter.invalidate(self.last_reason)
            except Exception:
                pass

    def _observe(self, action):
        if self.enabled:
            try:
                action()
            except Exception as exc:
                # No exception/traceback/owner is retained; original work still runs.
                self._fail("observer:" + type(exc).__name__)

    def _original_failed(self, reason):
        # Cleanup diagnostics can never replace an already raised original error,
        # including when an observer invalidator raises a control signal.
        try:
            self._fail(reason)
        except BaseException:
            self.enabled = False

    def detach(self):
        """Remove only our instance wrappers; preserve a later foreign override."""
        self.enabled = False
        runner = self._runner() if self._runner is not None else None
        if runner is not None:
            for name, wrapper in zip(METHODS, self._wrappers):
                if vars(runner).get(name) is wrapper:
                    delattr(runner, name)
        if self.status == "connected_cpu_candidate":
            self.status = "detached"
        # No runner, output, scheduler, Future, tensor or owner is kept alive.

    def observe_shutdown(self, original_shutdown, *, snapshot_factory,
                         expected_outputs, enabled=True):
        """Ephemeral tail helper; only original shutdown releases native owners.

        This does not install a handler hook or infer drain from Future.done().
        Provider must return actual copied post-shutdown scalar counters. It is
        called only after original shutdown succeeds, and is never retained.
        """
        try:
            result = original_shutdown()
        except BaseException:
            if self.enabled and enabled is True:
                self._original_failed("original_shutdown_failed")
            raise
        if not self.enabled or enabled is not True:
            return result
        def observe():
            self.closed_run = self._adapter.close_run(expected_outputs, snapshot_factory())
            self.status = "CPU_FRAME_RECONCILIATION_ONLY"
            self.enabled = False
        self._observe(observe)
        return result


def connect_runtime_scalar_observer(runner, *, adapter, adapter_source,
                                    binding, enabled=False, clock=time.monotonic_ns):
    """Opt-in instance wiring candidate; off touches none of the input objects.

    No caller in the supplied project invokes this on a real runner. A native
    installation remains gated on independent source/ABI and GPU qualification.
    """
    connection = RuntimeScalarConnection()
    if enabled is not True:
        return connection
    try:
        module = load_frozen_adapter(adapter_source)
        if type(adapter) is not module.FullStepFrameAdapter or adapter.enabled is not True:
            raise ValueError("exact enabled frozen scalar adapter required")
        if not adapter.valid or adapter.frames or adapter._pending is not None or adapter._closed:
            raise ValueError("fresh adapter required")
        # Native clock is a builtin. CPU test functions are held only weakly;
        # even their defaults/globals cannot retain an owner through this handle.
        if clock is not time.monotonic_ns and (type(clock) is not types.FunctionType or clock.__closure__):
            raise ValueError("clock must be builtin monotonic_ns or a closure-free CPU function")
        if binding.origin == "native_candidate" and clock is not time.monotonic_ns:
            raise ValueError("native candidate host clock must be builtin monotonic_ns")
        clock_ref = None if clock is time.monotonic_ns else weakref.ref(clock)
        _check_domain(runner)
        originals = _check_original_methods(runner, binding)
        for original_ref in originals:
            raw_path = Path(inspect.unwrap(original_ref().__func__).__code__.co_filename).resolve()
            raw_ref = next(ref for ref in binding.refs if Path(ref.path).resolve() == raw_path)
            if adapter.source_sha256 != raw_ref.sha256:
                raise ValueError("adapter original source binding differs")
        runner_ref = weakref.ref(runner)
        connection._runner = runner_ref
        connection._originals = originals
        connection._adapter = adapter
        connection._clock = clock_ref
        connection.enabled = True
        connection.status = "connected_cpu_candidate"
        connection.last_reason = "CPU source-bound scalar wiring; native qualification pending"

        def execute(*args, **kwargs):
            original = originals[0]()
            if original is None:
                raise ReferenceError("original runner no longer exists")
            if not connection.enabled:
                return original(*args, **kwargs)
            if connection.enabled:
                connection._observe(lambda: adapter.begin(original.__self__._profile_step, _read_clock(clock_ref)))
            try:
                result = original(*args, **kwargs)
            except BaseException:
                if connection.enabled:
                    connection._original_failed("original_execute_failed")
                raise
            connection._observe(lambda: adapter.executed(returned_none=result is None))
            return result

        def prepare(*args, **kwargs):
            original = originals[1]()
            if original is None:
                raise ReferenceError("original runner no longer exists")
            if not connection.enabled:
                return original(*args, **kwargs)
            try:
                result = original(*args, **kwargs)
            except BaseException:
                if connection.enabled:
                    connection._original_failed("original_prepare_failed")
                raise
            if connection.enabled:
                def observe():
                    if args:
                        scheduler = args[0]
                    elif "scheduler_output" in kwargs:
                        scheduler = kwargs["scheduler_output"]
                    else:
                        raise ValueError("original scheduler argument absent")
                    adapter.prepared(original.__self__, scheduler)
                connection._observe(observe)
            return result

        def sample(*args, **kwargs):
            original = originals[2]()
            if original is None:
                raise ReferenceError("original runner no longer exists")
            if not connection.enabled:
                return original(*args, **kwargs)
            try:
                result = original(*args, **kwargs)
            except BaseException:
                if connection.enabled:
                    connection._original_failed("original_sample_failed")
                raise
            connection._observe(lambda: adapter.sampled(result, _read_clock(clock_ref)))
            return result

        connection._wrappers = (execute, prepare, sample)
        installed = []
        try:
            for name, wrapper in zip(METHODS, connection._wrappers):
                setattr(runner, name, wrapper)
                installed.append(name)
        except BaseException:
            for name in installed:
                if vars(runner).get(name) in connection._wrappers:
                    delattr(runner, name)
            raise
        return connection
    except Exception as exc:
        connection._fail("preflight:" + type(exc).__name__)
        connection.detach()
        return connection
