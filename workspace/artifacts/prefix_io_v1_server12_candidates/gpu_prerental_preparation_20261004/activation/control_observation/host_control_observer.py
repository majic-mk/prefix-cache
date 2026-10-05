"""Bounded host observation; no engine, GPU import, wait, or authority bypass.

Intervals are conservative host wall occupancy of explicitly named small method
boundaries. A boundary can contain an existing enqueue/await/implicit sync; its
duration is NOT asserted to be pure CPU compute. Never subtract a CUDA duration.
The observer itself never emits an actual-native-GPU qualification receipt.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import re
import threading
import time

CATEGORIES = ("scheduler", "sampling", "output", "controller")
BOUNDARIES = dict(scheduler="scheduler_metadata", sampling="sampling_metadata_preparation",
                  output="output_processing", controller="controller_bookkeeping")
FORBIDDEN_METHODS = {"step", "execute_model", "sample_tokens", "generate", "synchronize", "wait"}
SEMANTICS = "conservative_host_control_wall_interval_may_include_existing_enqueue_await_or_implicit_sync"


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, label, low=0, high=None):
    require(type(value) is int and value >= low and (high is None or value <= high), label)
    return value


def canonical_sha(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _unique_pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def closed_ref(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "regular source file")
    raw = path.read_bytes()
    require(0 < len(raw) <= 32 * 1024**2, "bounded nonempty source")
    return dict(path=path.as_posix(), bytes=len(raw), sha256=sha256(raw).hexdigest())


def read_ref(ref):
    require(type(ref) is dict and set(ref) == {"path", "bytes", "sha256"}, "exact closed reference")
    require(type(ref["path"]) is str and "\0" not in ref["path"] and Path(ref["path"]).is_absolute(), "absolute closed path")
    integer(ref["bytes"], "source bytes", 1, 32 * 1024**2)
    require(type(ref["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]), "source SHA-256")
    path = Path(ref["path"])
    require(not any(item.is_symlink() for item in (path, *path.parents)), "symlink source is not closed")
    require(closed_ref(path) == ref, "source byte drift")
    return path.read_bytes()


def json_ref(ref):
    return json.loads(read_ref(ref), object_pairs_hook=_unique_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def linux_clock_scope():
    require(os.name == "posix" and hasattr(time, "CLOCK_MONOTONIC") and hasattr(time, "clock_gettime_ns"),
            "production reserve requires actual Linux CLOCK_MONOTONIC")
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    require(re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", boot_id), "actual Linux boot id")
    return dict(host_id=os.uname().nodename, boot_id=boot_id, clock="CLOCK_MONOTONIC")


class HostClock:
    """Actual built-in clock only. Windows is diagnostic, never production."""
    fixture = False

    def __init__(self):
        self.production_scope = os.name == "posix" and hasattr(time, "CLOCK_MONOTONIC")
        self.scope = linux_clock_scope() if self.production_scope else dict(
            host_id=os.environ.get("COMPUTERNAME", "local-host"), boot_id=None, clock="LOCAL_PERF_COUNTER_DIAGNOSTIC")
        self.implementation = ("time.clock_gettime_ns(time.CLOCK_MONOTONIC)" if self.production_scope else "time.perf_counter_ns()")
        self.resolution_seconds = time.get_clock_info("monotonic" if self.production_scope else "perf_counter").resolution

    def now(self):
        result = time.clock_gettime_ns(time.CLOCK_MONOTONIC) if self.production_scope else time.perf_counter_ns()
        return integer(result, "actual host clock returned exact positive integer", 1)


class FixtureClock:
    """Explicit CPU mathematics fixture; never eligible for a native receipt."""
    fixture = True
    production_scope = False
    implementation = "explicit_CPU_fixture_counter"
    resolution_seconds = None

    def __init__(self, values, scope=None):
        self.values = iter(values)
        self.scope = scope or dict(host_id="CPU_FIXTURE", boot_id="00000000-0000-0000-0000-000000000001", clock="CPU_FIXTURE")

    def now(self):
        return integer(next(self.values), "fixture clock exact integer", 1)


@dataclass(frozen=True)
class _Token:
    sequence: int
    category: str
    ordinal: int
    boundary_id: str
    start_ns: int
    thread_id: int
    scope_sha256: str


class HostControlObserver:
    """Compact observational scalars, without retaining an engine/owner/tensor.

    The tiny lock protects these CPU records only; it never waits on a GPU event,
    stream, I/O completion, original owner, or model execution.
    """
    def __init__(self, run_id, *, max_steps=4096, max_intervals_per_step=32, _fixture_clock=None):
        require(type(run_id) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", run_id), "run id")
        self.run_id = run_id
        self.max_steps = integer(max_steps, "bounded original steps", 1, 4096)
        self.max_intervals = integer(max_intervals_per_step, "bounded interval count", 4, 64)
        require(_fixture_clock is None or type(_fixture_clock) is FixtureClock, "only explicit fixture clock override")
        self.clock = HostClock() if _fixture_clock is None else _fixture_clock
        self._lock = threading.Lock()
        self._boundaries, self._intervals, self._active = {}, {}, {}
        self._sequence = self._failure_count = 0
        self._failures = []
        self._sealed = False

    def _fail(self, reason):
        self._failure_count += 1
        if len(self._failures) < 32:
            self._failures.append(str(reason)[:300])

    def bind(self, boundary_id, category, original, *, source_ref, boundary_kind):
        require(not self._sealed, "observer already sealed")
        require(type(boundary_id) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", boundary_id), "boundary id")
        require(category in CATEGORIES and boundary_kind == BOUNDARIES[category], "explicit small host boundary kind")
        require(boundary_id not in self._boundaries and callable(original), "unique real original boundary")
        function = original.__func__ if inspect.ismethod(original) else original
        require(inspect.isfunction(function) and function.__name__ not in FORBIDDEN_METHODS, "whole GPU/executor method forbidden")
        require(not inspect.iscoroutinefunction(function), "async boundary requires original await-site hooks; no executor wrapper")
        read_ref(source_ref)
        require(Path(function.__code__.co_filename).resolve() == Path(source_ref["path"]).resolve(), "boundary actual Python source binding")
        self._boundaries[boundary_id] = dict(category=category, boundary_kind=boundary_kind,
            callable_qualname=function.__qualname__, callable_firstlineno=function.__code__.co_firstlineno,
            source_ref=dict(source_ref), semantics=SEMANTICS, pure_CPU_compute_claim=False)

    def begin(self, category, native_step_ordinal, boundary_id):
        """On diagnostic failure, return None; never disrupt an original call."""
        try:
            integer(native_step_ordinal, "actual native step ordinal")
            require(not self._sealed and category in CATEGORIES, "live known host category")
            require(self._boundaries.get(boundary_id, {}).get("category") == category, "unbound host boundary")
            with self._lock:
                require(native_step_ordinal in self._intervals or len(self._intervals) < self.max_steps, "observer step overflow")
                rows = self._intervals.setdefault(native_step_ordinal, [])
                pending = sum(t.ordinal == native_step_ordinal for t in self._active.values())
                require(len(rows) + pending < self.max_intervals, "observer interval overflow")
                start = self.clock.now()
                self._sequence += 1
                token = _Token(self._sequence, category, native_step_ordinal, boundary_id, start,
                               threading.get_ident(), canonical_sha(self.clock.scope))
                self._active[token.sequence] = token
                return token
        except Exception as exc:
            with self._lock:
                self._fail(type(exc).__name__ + ": " + str(exc))
            return None

    def end(self, token):
        if token is None:
            return
        try:
            with self._lock:
                require(type(token) is _Token and self._active.pop(token.sequence, None) is token, "exact live interval token")
                end = self.clock.now()
                require(threading.get_ident() == token.thread_id, "interval crossed original host thread")
                require(canonical_sha(self.clock.scope) == token.scope_sha256, "interval mixed host boot or clock")
                require(end > token.start_ns, "host interval absent or clock went backwards")
                self._intervals[token.ordinal].append(dict(category=token.category, boundary_id=token.boundary_id,
                    start_ns=token.start_ns, end_ns=end, thread_id=token.thread_id,
                    clock_scope=dict(self.clock.scope), semantics=SEMANTICS))
        except Exception as exc:
            with self._lock:
                self._fail(type(exc).__name__ + ": " + str(exc))

    @contextmanager
    def observe(self, category, native_step_ordinal, boundary_id):
        token = self.begin(category, native_step_ordinal, boundary_id)
        try:
            yield
        finally:
            self.end(token)

    def call(self, category, native_step_ordinal, boundary_id, original, *args, **kwargs):
        descriptor = self._boundaries.get(boundary_id)
        try:
            function = original.__func__ if inspect.ismethod(original) else original
            require(descriptor and inspect.isfunction(function) and
                    function.__qualname__ == descriptor["callable_qualname"] and
                    function.__code__.co_firstlineno == descriptor["callable_firstlineno"] and
                    Path(function.__code__.co_filename).resolve() == Path(descriptor["source_ref"]["path"]).resolve(),
                    "observed callable differs from original source-bound boundary")
        except Exception as exc:
            with self._lock:
                self._fail(str(exc))
        with self.observe(category, native_step_ordinal, boundary_id):
            return original(*args, **kwargs)

    def export(self):
        with self._lock:
            require(not self._sealed, "observer export only once")
            self._sealed = True
            if self._active:
                self._fail("unfinished original host intervals")
            rows = [dict(native_step_ordinal=ordinal, intervals=list(intervals))
                    for ordinal, intervals in sorted(self._intervals.items())]
            result = dict(schema="host_control_observation_v1", run_id=self.run_id,
                origin="actual_host_method_observation" if not self.clock.fixture else "explicit_CPU_fixture",
                synthetic_fixture=self.clock.fixture, clock_scope=dict(self.clock.scope),
                clock_implementation=self.clock.implementation, clock_resolution_seconds=self.clock.resolution_seconds,
                interval_semantics=SEMANTICS, GPU_delta_subtracted=False, pure_CPU_compute_claim=False,
                all_control_categories=list(CATEGORIES), boundaries=dict(self._boundaries), per_step=rows,
                observer_source_ref=closed_ref(__file__), active_intervals=len(self._active),
                failure_count=self._failure_count, failures=list(self._failures),
                valid=self._failure_count == 0 and bool(rows), max_original_steps=self.max_steps,
                max_intervals_per_step=self.max_intervals, actual_native_gpu_run=False,
                production_qualified=False, formal_goodput_allowed=False, no_added_GPU_synchronization=True,
                no_owner_or_tensor_retained=True, reserve_source_issued=False)
        return result


def checked_intervals(observation, *, require_production_clock=True):
    require(type(observation) is dict and observation.get("schema") == "host_control_observation_v1", "host observation schema")
    require(observation.get("valid") is True and observation.get("failure_count") == 0 and observation.get("failures") == []
            and observation.get("active_intervals") == 0, "host observation failure/unfinished interval")
    require(observation.get("GPU_delta_subtracted") is False and observation.get("interval_semantics") == SEMANTICS
            and observation.get("pure_CPU_compute_claim") is False, "host wall intervals; no mixed CUDA subtraction")
    require(observation.get("actual_native_gpu_run") is False and observation.get("production_qualified") is False
            and observation.get("formal_goodput_allowed") is False, "observer cannot self-qualify GPU")
    require(observation.get("all_control_categories") == list(CATEGORIES), "all four control categories")
    scope = observation.get("clock_scope")
    require(type(scope) is dict and set(scope) == {"host_id", "boot_id", "clock"}, "same host boot clock scope")
    if require_production_clock:
        require(observation.get("synthetic_fixture") is False and observation.get("origin") == "actual_host_method_observation"
                and scope["clock"] == "CLOCK_MONOTONIC" and
                re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", scope["boot_id"] or "")
                and type(scope["host_id"]) is str and scope["host_id"], "actual Linux host clock required")
        require(observation.get("clock_implementation") == "time.clock_gettime_ns(time.CLOCK_MONOTONIC)", "actual built-in Linux clock")
    source_refs = [observation.get("observer_source_ref")]
    read_ref(source_refs[0])
    boundaries = observation.get("boundaries")
    require(type(boundaries) is dict and boundaries, "actual small boundary bindings")
    for descriptor in boundaries.values():
        require(descriptor["category"] in CATEGORIES and descriptor["boundary_kind"] == BOUNDARIES[descriptor["category"]]
                and descriptor["semantics"] == SEMANTICS and descriptor["pure_CPU_compute_claim"] is False,
                "explicit source-bound conservative host scope")
        read_ref(descriptor["source_ref"])
        source_refs.append(descriptor["source_ref"])
    steps = observation.get("per_step")
    require(type(steps) is list and 1 <= len(steps) <= integer(observation.get("max_original_steps"), "step limit", 1, 4096), "bounded actual step list")
    ordinals, intervals = [], []
    for step in steps:
        require(type(step) is dict and set(step) == {"native_step_ordinal", "intervals"}, "native step and intervals")
        ordinal = integer(step["native_step_ordinal"], "native step ordinal")
        rows = step["intervals"]
        require(type(rows) is list and 4 <= len(rows) <= integer(observation.get("max_intervals_per_step"), "interval limit", 4, 64), "missing/overflowed host intervals")
        require({r.get("category") for r in rows} == set(CATEGORIES), "each step needs all four actual host categories; missing is unknown")
        union_input = []
        for row in rows:
            require(type(row) is dict and set(row) == {"category", "boundary_id", "start_ns", "end_ns", "thread_id", "clock_scope", "semantics"}, "exact host interval fields")
            require(row["clock_scope"] == scope and row["semantics"] == SEMANTICS, "mixed host boot or CUDA clock")
            descriptor = boundaries.get(row["boundary_id"])
            require(descriptor and descriptor["category"] == row["category"], "actual interval source binding")
            begin, end = integer(row["start_ns"], "actual host start", 1), integer(row["end_ns"], "actual host end", 1)
            integer(row["thread_id"], "actual original thread", 1)
            require(end > begin, "missing/backward host interval")
            union_input.append([begin, end])
        ordinals.append(ordinal)
        intervals.append(sorted(union_input))
    require(ordinals == list(range(ordinals[0], ordinals[0] + len(ordinals))), "complete contiguous native step association")
    return dict(clock_scope=scope, ordinals=ordinals, per_step_control_intervals=intervals,
                source_refs=list({r["path"]: r for r in source_refs}.values()))


def reserve_candidate(observation, *, declaration, protocol):
    """CPU-only prospectiveness/mathematics check, without a native witness."""
    require(declaration is not None, "independent deadline missing; ordinary effect remains closed")
    require(declaration.get("schema") == "independent_deadline_declaration_v1" and
            declaration.get("origin") == "independent_requirement_before_development", "independent prospective deadline")
    require(declaration.get("clock_scope") == observation.get("clock_scope"), "deadline mixed host boot or clock")
    authority = json_ref(declaration.get("authority_ref"))
    require(authority.get("schema") == "independent_deadline_authority_v1" and
            authority.get("origin") == "independent_requirement_before_development" and
            authority.get("full_control_window_deadline_ns") == declaration.get("full_control_window_deadline_ns") and
            authority.get("service_SLO") == declaration.get("service_SLO"), "independent authority byte closure")
    protocol.validate_service_SLO(declaration.get("service_SLO"))
    checked = checked_intervals(observation)
    first = min(i[0] for step in checked["per_step_control_intervals"] for i in step)
    require(integer(declaration.get("declared_monotonic_ns"), "prospective deadline time", 1) < first,
            "deadline was not declared before development")
    return dict(schema="host_control_reserve_candidate_v1", run_id=observation["run_id"],
        deadline_declaration_sha256=canonical_sha(declaration), clock_scope=checked["clock_scope"],
        per_step_control_intervals=checked["per_step_control_intervals"], native_step_ordinals=checked["ordinals"],
        first_development_monotonic_ns=first, source_refs=checked["source_refs"],
        actual_native_gpu_run=False, synthetic_fixture=False, ordinary_effect_allowed=False,
        formal_goodput_allowed=False, awaiting_completed_original_guard_and_actual_CUDA_native_witness=True)
