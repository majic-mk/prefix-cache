"""CPU-only full-step event/journal contracts; no live runtime hook or executor.

This module never imports torch/vLLM/native extensions. Candidate event pairs
belong only to observation; query/elapsed reads never synchronize or wait. A
normal-model GPU runner remains blocked until its independent source/scope and
real timing, worker, model, I/O and drain provenance are frozen and qualified.
"""
from dataclasses import dataclass
from math import isfinite

STAGES = ("ssd_read", "ssd_write", "h2d", "d2h")


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "explicit scalar integer")
    return value


def scalar_text(value):
    require(type(value) is str and 0 < len(value) <= 128, "bounded scalar text")
    return value


def digest(value):
    require(type(value) is str and len(value) == 64 and
        all(c in "0123456789abcdef" for c in value), "exact source SHA")
    return value


def event_ns(milliseconds):
    require(type(milliseconds) in (int, float) and isfinite(milliseconds) and
        milliseconds >= 0, "finite actual event elapsed scalar")
    return round(milliseconds * 1_000_000)


@dataclass(frozen=True)
class CPUClockReference:
    event: object
    anchor_ns: int
    run_id: str
    cross_clock_proved_in_fixture: bool = False
    fallback_used: bool = False
    origin: str = "cpu_fixture"


@dataclass(frozen=True)
class CompleteStepEventWindow:
    run_id: str
    ordinal: int
    event_elapsed_ns: int
    mapped_start_ns: object
    mapped_end_ns: object
    mapping_known_in_fixture: bool
    native_source_sha256: str
    scope: str = "full_decode_step"
    origin: str = "cpu_fixture"
    status: str = "CPU_EVENT_LIFECYCLE_CONTRACT_ONLY"
    gpu_elapsed_ns: object = None

    def __post_init__(self):
        scalar_text(self.run_id); digest(self.native_source_sha256)
        integer(self.ordinal); integer(self.event_elapsed_ns, 1)
        require(type(self.mapping_known_in_fixture) is bool and self.origin == "cpu_fixture" and
            self.gpu_elapsed_ns is None, "CPU event contract cannot assert native GPU timing")
        if self.mapping_known_in_fixture:
            integer(self.mapped_start_ns, 1); integer(self.mapped_end_ns, 1)
            require(self.mapped_end_ns - self.mapped_start_ns == self.event_elapsed_ns,
                "CPU mapped event interval differs")
        else:
            require(self.mapped_start_ns is None and self.mapped_end_ns is None,
                "unknown cross-clock mapping cannot carry wall endpoints")

    @property
    def gpu_verified(self): return False
    @property
    def production_qualified(self): return False


class FullStepEventContract:
    """A fake-event lifecycle candidate, deliberately incapable of GPU claims.

    A future caller begins immediately before original execute_model and ends
    only after original sample_tokens returned and frozen scalar frame closed.
    This class itself does not call, replace or own the model executor.
    """
    runtime_hook_status = "not_installed"
    GPU_collector_verified = False
    production_qualified = False

    def __init__(self, run_id, native_source_sha256, *, frame_type,
                 enabled=False, max_pending=128, max_steps=4096):
        self.run_id = scalar_text(run_id)
        self.native_source_sha256 = digest(native_source_sha256)
        require(type(enabled) is bool and type(max_pending) is int and
            1 <= max_pending <= 128 and type(max_steps) is int and
            1 <= max_steps <= 4096, "bounded optional event state")
        self.enabled = enabled
        self.valid = True
        self.reason = "not_installed"
        if enabled:
            require(type(frame_type) is type, "closed scalar frame class required; no owner instance")
        self.frame_type = frame_type if enabled else None
        self.max_pending, self.max_steps = max_pending, max_steps
        self.pending = []
        self.active = None
        self.first_ordinal = None
        self.closed_steps = 0

    def invalidate(self, reason):
        self.enabled = False
        self.valid = False
        self.reason = str(reason)[:160]
        self.pending.clear()
        self.active = None

    def before_execute(self, ordinal, event_factory):
        if not self.enabled:
            return
        try:
            integer(ordinal)
            require(self.active is None and len(self.pending) < self.max_pending and
                self.closed_steps < self.max_steps, "event pending/step bound or open step")
            if self.first_ordinal is None:
                self.first_ordinal = ordinal
            require(ordinal == self.first_ordinal + self.closed_steps,
                "full-step event ordinal missing/duplicate")
            start, end = event_factory(enable_timing=True), event_factory(enable_timing=True)
            require(start is not end, "distinct event pair required")
            start.record()
            self.active = (ordinal, start, end)
        except Exception as exc:
            self.invalidate("event_begin:" + type(exc).__name__)

    def after_sample(self, frame):
        if not self.enabled:
            return
        try:
            require(self.active is not None and type(frame) is self.frame_type,
                "original sample plus exact closed scalar frame required")
            ordinal, start, end = self.active
            require(frame.native_step_ordinal == ordinal and frame.gpu_elapsed_ns is None and
                frame.prepared.context_basis == "pre_computed_tokens" and
                type(frame.outputs) is tuple and 0 < frame.start_ns < frame.end_ns,
                "closed original frame provenance differs")
            end.record()
            # Keep observation events and copied scalars only, no output/runner.
            self.pending.append((ordinal, start, end))
            self.active = None
            self.closed_steps += 1
        except Exception as exc:
            self.invalidate("event_end:" + type(exc).__name__)

    def resolve_ready(self, reference):
        if not self.enabled:
            return ()
        out, remaining = [], []
        try:
            require(type(reference) is CPUClockReference and reference.origin == "cpu_fixture" and
                reference.run_id == self.run_id and type(reference.cross_clock_proved_in_fixture) is bool and
                type(reference.fallback_used) is bool, "explicit CPU reference identity")
            integer(reference.anchor_ns, 1)
            for ordinal, start, end in self.pending:
                start_ready, end_ready = start.query(), end.query()
                require(type(start_ready) is bool and type(end_ready) is bool,
                    "actual event query must return exact bool")
                if not (start_ready and end_ready):
                    remaining.append((ordinal, start, end))
                    continue
                duration = event_ns(start.elapsed_time(end))
                require(duration > 0, "positive event scope")
                mapped = reference.cross_clock_proved_in_fixture and not reference.fallback_used
                if mapped:
                    ready = reference.event.query()
                    require(type(ready) is bool, "reference query must return exact bool")
                    mapped = ready
                first = reference.anchor_ns + event_ns(reference.event.elapsed_time(start)) if mapped else None
                out.append(CompleteStepEventWindow(self.run_id, ordinal, duration, first,
                    first + duration if first is not None else None, bool(mapped),
                    self.native_source_sha256))
            self.pending = remaining
            return tuple(out)
        except Exception as exc:
            self.invalidate("event_resolve:" + type(exc).__name__)
            return ()


@dataclass(frozen=True)
class PhysicalAmount:
    ops: int
    nbytes: int


@dataclass(frozen=True)
class CompleteStepIOAttribution:
    status: str
    reason: str
    existing_io: object = None
    new_io: object = None
    completed_new_io: object = None
    origin: str = "cpu_fixture"

    @property
    def gpu_verified(self): return False
    @property
    def production_qualified(self): return False


def attribute_published_full_step(window, published, *, run_id, native_source_sha256):
    """Read only NativeWindowJournal.published() immutable CPU copies at tail.

    Caller obtains the original published copy; no native stats/owner is read
    here. New stage bytes count backend acceptance, not bandwidth or release.
    Old journal.attribute(model_forward, single stage) is untouched.
    """
    def unknown(message):
        return CompleteStepIOAttribution("UNKNOWN", message)
    try:
        require(type(window) is CompleteStepEventWindow and window.origin == "cpu_fixture" and
            window.scope == "full_decode_step" and window.mapping_known_in_fixture is True and
            window.run_id == run_id and window.native_source_sha256 == native_source_sha256,
            "full-step mapped source/run scope unknown")
        begin, end = window.mapped_start_ns, window.mapped_end_ns
        integer(begin, 1); integer(end, 1)
        require(begin < end, "event interval invalid")
        require(type(published) is tuple and len(published) == 3, "original published immutable tuple")
        events, frames, valid = published
        require(valid is True and type(events) is tuple and type(frames) is tuple and
            len(events) <= 4096 and 0 < len(frames) <= 4096,
            "complete bounded native journal required")
        previous, last_ns = 0, -1
        accepted, completed = {}, {}
        accepted_ops, accepted_bytes = [0] * 4, [0] * 4
        completed_ops, completed_bytes = [0] * 4, [0] * 4
        def state_vector(ops, nbytes):
            return tuple((ops[i], nbytes[i]) for i in range(4))
        def prefix_state():
            return (state_vector(accepted_ops, accepted_bytes),
                    state_vector(completed_ops, completed_bytes),
                    tuple((accepted_ops[i] - completed_ops[i],
                           accepted_bytes[i] - completed_bytes[i]) for i in range(4)))
        prefixes = [prefix_state()]
        for event in events:
            integer(event.sequence, 1); integer(event.operation_sequence, 1)
            integer(event.at_ns); integer(event.physical_bytes, 1)
            require(event.sequence == previous + 1 and event.stage in STAGES and
                event.kind in ("accepted", "completed"), "native event sequence/stage invalid")
            require(event.at_ns >= last_ns, "native event clock moved backwards")
            stage_index = STAGES.index(event.stage)
            if event.kind == "accepted":
                require(event.operation_sequence == event.sequence, "native accepted identity differs")
                accepted[event.operation_sequence] = event
                accepted_ops[stage_index] += 1
                accepted_bytes[stage_index] += event.physical_bytes
            else:
                prior_event = accepted.get(event.operation_sequence)
                require(prior_event is not None and prior_event.stage == event.stage and
                    prior_event.physical_bytes == event.physical_bytes,
                    "native completion without matching prior acceptance")
                require(event.operation_sequence not in completed and type(event.result) is int and
                    event.result == prior_event.physical_bytes,
                    "duplicate, partial or failed actual completion anywhere in tail")
                completed[event.operation_sequence] = event
                completed_ops[stage_index] += 1
                completed_bytes[stage_index] += event.physical_bytes
            previous = event.sequence
            last_ns = event.at_ns
            prefixes.append(prefix_state())
            require(event.at_ns not in (begin, end), "ambiguous native/GPU boundary")
        require(set(accepted) == set(completed), "tail contains unfinished existing/outside native I/O")
        frame_ns, frame_sequence = -1, 0
        def amount_vector(value):
            require(type(value) is tuple and len(value) == 4, "four immutable owner stage amounts")
            result = []
            for amount in value:
                ops, nbytes = integer(amount.ops), integer(amount.nbytes)
                require((ops == 0) == (nbytes == 0), "owner stage geometry invalid")
                result.append((ops, nbytes))
            return tuple(result)
        for frame in frames:
            require(frame.run_id == run_id and frame.source_sha256 == native_source_sha256 and
                frame.clock_domain == "monotonic_ns" and frame.valid is True and
                frame.journal_complete is True, "native owner frame identity/validity unknown")
            require(type(frame.failed_ops) is tuple and len(frame.failed_ops) == 4 and
                all(type(v) is int and v == 0 for v in frame.failed_ops), "actual native failures present")
            integer(frame.captured_ns)
            require(frame.captured_ns >= frame_ns, "owner frame clock moved backwards")
            sequence = integer(frame.sequence)
            require(frame_sequence <= sequence <= len(events), "owner frame sequence outside event prefix")
            require(sequence == 0 or events[sequence - 1].at_ns <= frame.captured_ns,
                "owner prefix event follows capture")
            require(sequence == len(events) or events[sequence].at_ns >= frame.captured_ns,
                "owner capture omits prior observed event")
            actual_vectors = (amount_vector(frame.accepted), amount_vector(frame.completed),
                              amount_vector(frame.inflight))
            require(actual_vectors == prefixes[sequence],
                "owner accepted/completed/inflight differs from complete actual event prefix")
            frame_ns = frame.captured_ns
            frame_sequence = sequence
        require(frames[-1].sequence == len(events) and
            amount_vector(frames[-1].inflight) == ((0, 0),) * 4,
            "complete native tail owner frame absent")
        prior = [frame for frame in frames if frame.captured_ns <= begin]
        after = [frame for frame in frames if frame.captured_ns >= end]
        require(prior and after, "actual native owner coverage absent")
        first = prior[-1]
        require(type(first.inflight) is tuple and len(first.inflight) == 4, "four physical inflight stages")
        existing = []
        for amount in first.inflight:
            ops, nbytes = integer(amount.ops), integer(amount.nbytes)
            require((ops == 0) == (nbytes == 0), "inflight stage geometry invalid")
            existing.append(PhysicalAmount(ops, nbytes))
        added = [event for event in events if event.kind == "accepted" and begin < event.at_ns < end]
        amounts = []
        for stage in STAGES:
            selected = [event for event in added if event.stage == stage]
            for event in selected:
                result = completed.get(event.operation_sequence)
                require(result is not None and result.stage == stage and
                    result.physical_bytes == event.physical_bytes and type(result.result) is int and
                    result.result == event.physical_bytes, "actual added native I/O partial/unfinished")
            amounts.append(PhysicalAmount(len(selected), sum(event.physical_bytes for event in selected)))
        return CompleteStepIOAttribution("CPU_ATTRIBUTION_CONTRACT_ONLY", "no native GPU/cost qualification",
            tuple(existing), tuple(amounts), tuple(amounts))
    except Exception as exc:
        return unknown(type(exc).__name__ + ": " + str(exc)[:140])
