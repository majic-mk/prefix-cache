"""Opt-in P3 simple four-stage allowances, with no native resource ownership.

The reactor owns queues, staging, events, completion and parent admission.
An immediate attempt contains values only. Unknown is never measured zero.
Performance allowances may be exceeded for progress; native safety may not.
P4 dependency/interference/joint policies are deliberately absent.
"""
from collections import deque
from dataclasses import dataclass
from threading import get_ident

from .dispatch_budget import Amount, DispatchBudget, Permit, STAGES, ZERO, integer, vector
from .dispatch_shadow import ShadowState

EPOCH_CANDIDATES_NS = (1_000_000, 10_000_000, 100_000_000, 1_000_000_000)
PROGRESS = frozenset(("mandatory", "mandatory_support", "continuation", "age", "shutdown"))
MODES = frozenset(("off", "shadow", "fixed", "pressure"))


def _run_id(value):
    if type(value) is not str or not value.strip() or len(value) > 128:
        raise ValueError("bounded nonempty run identity required")


def _work_id(value):
    # These are native-issued identity VALUES, never native jobs or Futures.
    def component(item):
        return ((type(item) is int and item >= 0) or
                (type(item) is str and bool(item) and len(item) <= 128))
    if component(value):
        return
    if type(value) is tuple and 1 <= len(value) <= 4 and all(component(x) for x in value):
        return
    raise ValueError("work_id must be a bounded scalar or tuple of scalar values")


@dataclass(frozen=True)
class SimpleStageConfig:
    mode: str
    epoch_ns: int
    byte_quantum: int
    cumulative: tuple[Amount, ...]
    inflight: tuple[Amount, ...]
    shared_ssd_cumulative: Amount
    shared_ssd_inflight: Amount
    shared_copy_cumulative_bytes: int
    shared_copy_inflight_bytes: int
    reserve_staging_bytes: int
    # Publication metadata only. Common native intake owns/enforces this bound.
    max_accepted_parents: int
    sample_max_age_ns: int
    max_wait_ns: int
    max_waiting_keys: int = 64
    max_records: int = 96

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("only P3 off/shadow/fixed/pressure are supported")
        if type(self.epoch_ns) is not int or self.epoch_ns not in EPOCH_CANDIDATES_NS:
            raise ValueError("epoch must use the frozen finite candidates")
        integer(self.byte_quantum, "byte_quantum", 1)
        vector(self.cumulative)
        vector(self.inflight)
        if type(self.shared_ssd_cumulative) is not Amount or type(self.shared_ssd_inflight) is not Amount:
            raise ValueError("shared SSD allowances require Amount")
        for name in ("shared_copy_cumulative_bytes", "shared_copy_inflight_bytes", "reserve_staging_bytes"):
            integer(getattr(self, name), name)
        for name in ("max_accepted_parents", "sample_max_age_ns", "max_wait_ns"):
            integer(getattr(self, name), name, 1)
        for name, bound in (("max_waiting_keys", 64), ("max_records", 96)):
            integer(getattr(self, name), name, 1)
            if getattr(self, name) > bound:
                raise ValueError(name + " exceeds the bounded metadata window")
        byte_caps = [a.nbytes for a in self.cumulative + self.inflight]
        byte_caps += [self.shared_ssd_cumulative.nbytes, self.shared_ssd_inflight.nbytes,
                      self.shared_copy_cumulative_bytes, self.shared_copy_inflight_bytes,
                      self.reserve_staging_bytes]
        if any(value % self.byte_quantum for value in byte_caps):
            raise ValueError("byte allowances must be quantized to byte_quantum")
        if self.mode == "fixed" and self.reserve_staging_bytes:
            raise ValueError("fixed mode has no staging pressure reserve")


@dataclass(frozen=True)
class StageAttempt:
    stage: str
    nbytes: int
    started_ns: int
    work_id: object
    permit: Permit | None
    action: str
    performance_reasons: tuple[str, ...]
    native_reasons: tuple[str, ...]
    unknown_fields: tuple[str, ...]
    progress: str | None

    @property
    def epoch(self):
        return self.permit.epoch if self.permit is not None else None


@dataclass(frozen=True)
class StageDecision:
    action: str  # issue / defer / native_fallback
    attempt: StageAttempt | None
    reasons: tuple[str, ...]
    unknown_fields: tuple[str, ...]


def make_dispatch_controller(run_id, config):
    """Off creates no controller, owner, grant or metadata."""
    if type(config) is not SimpleStageConfig:
        raise TypeError("frozen SimpleStageConfig required")
    if config.mode == "off":
        return None
    return DispatchController(run_id, config)


class DispatchController:
    """Single reactor owner; no deferred work queue or resource reference.

    Epochs follow the reactor monotonic clock independently of worker grants.
    A same-epoch tick never replenishes used allowance. State must be captured
    at the native issue boundary. accepted_parents is audit-only here: common
    native intake must independently provide and enforce its count.

    A native_fallback decision still has an immediate audit attempt. Successful
    API return charges the current epoch even when the snapshot was unknown.
    CUDA uncertainty is sticky; it cannot establish zero execution or drain.
    """
    def __init__(self, run_id, config):
        _run_id(run_id)
        if type(config) is not SimpleStageConfig:
            raise TypeError("frozen SimpleStageConfig required")
        if config.mode == "off":
            raise ValueError("off must use the factory and create no object")
        self.run_id = run_id
        self.config = config
        self.bound = False
        self.owner = None
        self.last_now = None
        self.grant = None
        self.used = list(ZERO)
        self.pending = None
        self.waiting = {}
        self.records = deque(maxlen=config.max_records)
        self.overwritten_records = 0
        self.epoch_refreshes = 0
        self.denied_performance = 0
        self.denied_native = 0
        self.metadata_fallbacks = 0
        self.native_fallbacks = 0
        self.accepted_totals = {stage: Amount() for stage in STAGES}
        self.accepted_without_epoch_ops = 0
        self.accepted_without_epoch_bytes = 0
        self.rejected_ops = 0
        self.uncertain_ops = 0
        self.uncertain_requested_bytes = 0
        self.completion_unknown_ops = 0
        self.completion_unknown_stages = {}
        self.performance_override_ops = 0
        self.shadow_overruns = 0
        self.progress_totals = {name: 0 for name in PROGRESS}
        self.faulted = False
        self.error = None

    def bind(self):
        if self.bound:
            raise ValueError("controller already belongs to a reactor")
        self.bound = True

    def _owner(self):
        if not self.bound:
            raise RuntimeError("controller must be explicitly bound")
        ident = get_ident()
        if self.owner is None:
            self.owner = ident
        if self.owner != ident:
            raise RuntimeError("controller must stay on its reactor owner")

    def _time(self, now_ns):
        self._owner()
        integer(now_ns, "clock")
        if self.last_now is not None and now_ns < self.last_now:
            raise ValueError("monotonic clock regressed")
        self.last_now = now_ns

    def _refresh(self, now_ns):
        if self.faulted:
            return False
        c = self.config
        epoch = now_ns // c.epoch_ns
        if self.grant is not None and epoch == self.grant.epoch:
            return False
        if self.pending is not None:
            raise RuntimeError("settle the immediate backend attempt before changing epoch")
        issued_ns = epoch * c.epoch_ns
        if self.grant is not None:
            if epoch <= self.grant.epoch or issued_ns < self.grant.expires_ns:
                raise ValueError("epochs must advance without overlapping grants")
        self.grant = DispatchBudget(
            self.run_id, epoch, issued_ns, issued_ns + c.epoch_ns,
            c.cumulative, c.inflight, c.shared_ssd_cumulative, c.shared_ssd_inflight,
            c.shared_copy_cumulative_bytes, c.shared_copy_inflight_bytes,
            c.reserve_staging_bytes, c.max_accepted_parents, c.sample_max_age_ns)
        self.used = list(ZERO)
        self.epoch_refreshes += 1
        return True

    def refresh_epoch(self, now_ns):
        self._time(now_ns)
        return self._refresh(now_ns)

    def _preview(self, stage, nbytes, state, now_ns, staging_bytes_needed):
        g = self.grant
        current = g is not None and g.issued_ns <= now_ns < g.expires_ns
        fresh = (current and state is not None and
                 0 <= now_ns - state.captured_ns <= g.sample_max_age_ns)
        unknown, required_unknown, perf, native = [], [], [], []

        def missing(name, required=True):
            unknown.append(name)
            if required:
                required_unknown.append(name)

        if not current:
            missing("current_grant")
        if state is None:
            missing("native_state")
        elif not fresh:
            missing("fresh_native_state")
        inflight = state.inflight if fresh else None
        free = state.free_staging_bytes if fresh else None
        safe = state.native_issue_safe if fresh else None
        parents = state.accepted_parents if fresh else None
        if inflight is None:
            missing("inflight")
        if safe is None:
            missing("native_issue_safe")
        if free is None:
            missing("free_staging_bytes",
                    self.config.mode == "pressure" and staging_bytes_needed > 0)
        if parents is None:
            missing("accepted_parents", False)
        if self.faulted:
            missing("controller_fault")
        if self.uncertain_ops:
            missing("prior_acceptance_uncertain")
        if self.completion_unknown_ops:
            missing("prior_completion_unknown")
        if safe is False or (free is not None and staging_bytes_needed > free):
            native.append("native_capacity_or_dependency")
        if current:
            i = STAGES.index(stage)
            used, cap = self.used[i], g.cumulative[i]
            for name, value, bound in (
                ("stage_epoch_ops", used.ops + 1, cap.ops),
                ("stage_epoch_bytes", used.nbytes + nbytes, cap.nbytes),
            ):
                if value > bound:
                    perf.append(name)
            if inflight is not None:
                active, cap = inflight[i], g.inflight[i]
                for name, value, bound in (
                    ("stage_inflight_ops", active.ops + 1, cap.ops),
                    ("stage_inflight_bytes", active.nbytes + nbytes, cap.nbytes),
                ):
                    if value > bound:
                        perf.append(name)
            if i < 2:
                for prefix, amounts, cap in (
                    ("shared_ssd_epoch", self.used[:2], g.shared_ssd_cumulative),
                    ("shared_ssd_inflight", inflight[:2] if inflight is not None else None,
                     g.shared_ssd_inflight),
                ):
                    if amounts is not None:
                        if sum(a.ops for a in amounts) + 1 > cap.ops:
                            perf.append(prefix + "_ops")
                        if sum(a.nbytes for a in amounts) + nbytes > cap.nbytes:
                            perf.append(prefix + "_bytes")
            else:
                if sum(a.nbytes for a in self.used[2:]) + nbytes > g.shared_copy_cumulative_bytes:
                    perf.append("shared_copy_epoch_bytes")
                if (inflight is not None and
                        sum(a.nbytes for a in inflight[2:]) + nbytes > g.shared_copy_inflight_bytes):
                    perf.append("shared_copy_inflight_bytes")
            if (self.config.mode in ("pressure", "shadow") and staging_bytes_needed and
                    free is not None and free - staging_bytes_needed < g.reserve_staging_bytes):
                perf.append("staging_reserve")
        return current, tuple(perf), tuple(native), tuple(unknown), tuple(required_unknown)

    def decide(self, stage, nbytes, state=None, *, now_ns, work_id,
               staging_bytes_needed=0, progress=None):
        if stage not in STAGES:
            raise ValueError("unknown physical stage")
        integer(nbytes, "actual physical bytes", 1)
        integer(staging_bytes_needed, "required staging")
        _work_id(work_id)
        if progress is not None and progress not in PROGRESS:
            raise ValueError("unknown progress reason")
        if state is not None and (type(state) is not ShadowState or state.run_id != self.run_id):
            raise ValueError("wrong native state identity")
        self._time(now_ns)
        if self.pending is not None:
            raise RuntimeError("one immediate backend attempt at a time")
        self._refresh(now_ns)
        current, perf, native, unknown, required_unknown = self._preview(
            stage, nbytes, state, now_ns, staging_bytes_needed)
        active = self.config.mode in ("fixed", "pressure")
        effective_progress = progress
        key = (stage, work_id)
        action = "issue"
        if active and native:
            self.denied_native += 1
            return StageDecision("defer", None, native + perf, unknown)
        if not active or required_unknown:
            action = "native_fallback"
        elif effective_progress is None:
            if key not in self.waiting:
                if len(self.waiting) >= self.config.max_waiting_keys:
                    self.metadata_fallbacks += 1
                    unknown += ("waiting_metadata_capacity",)
                    action = "native_fallback"
                else:
                    self.waiting[key] = now_ns
            if action != "native_fallback" and now_ns - self.waiting[key] >= self.config.max_wait_ns:
                effective_progress = "age"
            if action != "native_fallback" and perf and effective_progress is None:
                self.denied_performance += 1
                return StageDecision("defer", None, perf, unknown)
        if action == "native_fallback":
            self.native_fallbacks += 1
        permit = Permit(self.grant.epoch, stage, nbytes, effective_progress or "ordinary") if current else None
        attempt = StageAttempt(stage, nbytes, now_ns, work_id, permit, action,
                               perf, native, unknown, effective_progress)
        self.pending = attempt
        return StageDecision(action, attempt, native + perf, unknown)

    def _settle(self, attempt, outcome, reason):
        self._owner()
        if self.pending is None or self.pending is not attempt:
            raise ValueError("stale, foreign or duplicate immediate attempt")
        if type(reason) is not str:
            raise ValueError("bounded text reason required")
        self.pending = None
        self.waiting.pop((attempt.stage, attempt.work_id), None)
        if outcome == "accepted":
            old = self.accepted_totals[attempt.stage]
            self.accepted_totals[attempt.stage] = Amount(old.ops + 1, old.nbytes + attempt.nbytes)
            if attempt.permit is not None:
                i = STAGES.index(attempt.stage)
                old = self.used[i]
                self.used[i] = Amount(old.ops + 1, old.nbytes + attempt.nbytes)
            else:
                self.accepted_without_epoch_ops += 1
                self.accepted_without_epoch_bytes += attempt.nbytes
            if attempt.progress is not None:
                self.progress_totals[attempt.progress] += 1
            if attempt.performance_reasons:
                if self.config.mode == "shadow":
                    self.shadow_overruns += 1
                elif attempt.action == "issue" and attempt.progress is not None:
                    self.performance_override_ops += 1
        elif outcome == "rejected":
            self.rejected_ops += 1
        else:
            self.uncertain_ops += 1
            self.uncertain_requested_bytes += attempt.nbytes
            self.faulted = True
            self.error = ("acceptance uncertain: " + reason)[:160]
        if len(self.records) == self.records.maxlen:
            self.overwritten_records += 1
        self.records.append(dict(
            stage=attempt.stage, requested_bytes=attempt.nbytes,
            started_ns=attempt.started_ns, epoch=attempt.epoch, action=attempt.action,
            performance_reasons=attempt.performance_reasons,
            native_reasons=attempt.native_reasons, unknown_fields=attempt.unknown_fields,
            progress=attempt.progress, outcome=outcome, reason=reason[:160]))
        return True

    def accepted(self, attempt):
        return self._settle(attempt, "accepted", "")

    def rejected(self, attempt, reason="", *, proved=False):
        if proved is not True:
            raise ValueError("rejected requires explicit proof of no backend acceptance")
        return self._settle(attempt, "rejected", reason)

    def uncertain(self, attempt, reason=""):
        return self._settle(attempt, "uncertain", reason)

    def mark_completion_unknown(self, stage, reason=""):
        self._owner()
        if stage not in STAGES or type(reason) is not str:
            raise ValueError("known stage and bounded text reason required")
        self.completion_unknown_ops += 1
        self.completion_unknown_stages[stage] = reason[:160]
        self.faulted = True
        self.error = ("completion unknown: " + reason)[:160]

    def fail(self, reason):
        """Sticky diagnostic, permitted even when detecting a foreign owner.

        This is not owner permission: it never changes pending/used/time/grant
        or proves physical drain. Like DispatchShadow.fail, optional diagnostic
        failure must not raise a second owner error into native resource cleanup.
        """
        self.faulted = True
        self.error = str(reason)[:160]
        # Do not classify or clear a pending attempt and do not claim resources drained.

    def retire_work(self, stage, work_id):
        """Optional native-terminal notification for value-only age metadata."""
        self._owner()
        if stage not in STAGES:
            raise ValueError("unknown physical stage")
        _work_id(work_id)
        if self.pending is not None and (self.pending.stage, self.pending.work_id) == (stage, work_id):
            raise RuntimeError("settle the immediate attempt before retiring metadata")
        self.waiting.pop((stage, work_id), None)

    def snapshot(self, *, native_shutdown=False):
        """Complete flags classify attempts only, never DMA completion or drain.

        native_shutdown is a caller precondition that the reactor cannot mutate
        this object. No controller field establishes physical resource release.
        """
        if type(native_shutdown) is not bool:
            raise ValueError("explicit native shutdown precondition required")
        if not native_shutdown:
            self._owner()
        return dict(
            run_id=self.run_id, mode=self.config.mode, bound=self.bound, owner=self.owner,
            epoch=self.grant.epoch if self.grant is not None else None,
            epoch_refreshes=self.epoch_refreshes,
            used={s: dict(ops=a.ops, bytes=a.nbytes) for s, a in zip(STAGES, self.used)},
            observed_api_accepted={s: dict(ops=a.ops, bytes=a.nbytes)
                                   for s, a in self.accepted_totals.items()},
            accepted_without_epoch_ops=self.accepted_without_epoch_ops,
            accepted_without_epoch_bytes=self.accepted_without_epoch_bytes,
            rejected_ops=self.rejected_ops, uncertain_ops=self.uncertain_ops,
            uncertain_requested_bytes=self.uncertain_requested_bytes,
            uncertain_bytes_are_requested_not_executed=True,
            epoch_totals_are_lower_bounds=self.uncertain_ops > 0 or self.faulted,
            completion_unknown_ops=self.completion_unknown_ops,
            completion_unknown_stages=dict(self.completion_unknown_stages),
            acceptance_accounting_complete=(not self.faulted and not self.uncertain_ops and self.pending is None),
            completion_accounting_complete=(not self.faulted and not self.uncertain_ops
                                            and not self.completion_unknown_ops and self.pending is None),
            complete_flags_scope="audit classifications only; not DMA completion or drain",
            faulted=self.faulted, error=self.error, pending_attempt=self.pending is not None,
            waiting_keys=len(self.waiting), max_waiting_keys=self.config.max_waiting_keys,
            denied_performance=self.denied_performance, denied_native=self.denied_native,
            metadata_fallbacks=self.metadata_fallbacks, native_fallbacks=self.native_fallbacks,
            performance_override_ops=self.performance_override_ops,
            shadow_overruns=self.shadow_overruns, progress_totals=dict(self.progress_totals),
            records=list(self.records), overwritten_records=self.overwritten_records,
            native_shutdown_read=native_shutdown, parent_cap_enforced=False,
            dependency_policy_enabled=False, interference_policy_enabled=False,
            joint_policy_enabled=False, owns_native_queue=False,
            physical_drain_inferred=False, resource_release_inferred=False, gpu_qualified=False)
