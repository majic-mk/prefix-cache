"""Resource-free shadow audit of API acceptance; this module installs no adapter.

An audit attempt is one synchronous observation, never a work permit.  Native
dispatch always proceeds.  API-returned acceptance consumes epoch allowance
even when the preview would defer.  A CUDA exception may have issued work:
uncertain is neither rejected nor proof of zero execution or physical drain.
"""
from collections import deque
from dataclasses import dataclass
from threading import get_ident

from .dispatch_budget import (
    Amount, DispatchBudget, Permit, PROGRESS, STAGES, ZERO, integer, vector,
)


def _run_id(value):
    if type(value) is not str or not value.strip() or len(value) > 128:
        raise ValueError("bounded nonempty run identity required")


@dataclass(frozen=True)
class ShadowState:
    """Owner-observed fields; None means unavailable, never zero capacity.

    accepted_parents includes native accepted incoming/active/draining jobs if
    actually supported.  This audit NEVER evaluates or enforces parent intake.
    native_issue_safe is a provenance-bearing observation of original checks,
    not permission from this object to perform an operation.
    """
    run_id: str
    captured_ns: int
    inflight: tuple[Amount, ...] | None = None
    free_staging_bytes: int | None = None
    accepted_parents: int | None = None
    native_issue_safe: bool | None = None

    def __post_init__(self):
        _run_id(self.run_id)
        integer(self.captured_ns, "captured_ns")
        if self.inflight is not None:
            vector(self.inflight)
        for name in ("free_staging_bytes", "accepted_parents"):
            value = getattr(self, name)
            if value is not None:
                integer(value, name)
        if self.native_issue_safe is not None and type(self.native_issue_safe) is not bool:
            raise ValueError("native safety must be explicit boolean or unknown")


@dataclass(frozen=True)
class ShadowAttempt:
    """Bounded values only: no Future, job, FD, Event or staging ownership."""
    stage: str
    nbytes: int
    started_ns: int
    permit: Permit | None
    verdict: str
    performance_reasons: tuple[str, ...]
    native_reasons: tuple[str, ...]
    unknown_fields: tuple[str, ...]
    progress: str | None

    @property
    def epoch(self):
        return self.permit.epoch if self.permit is not None else None


class DispatchShadow:
    """Single-owner, off/shadow acceptance audit.  It never calls a backend.

    Current-epoch attempts always receive an AUDIT ticket, including hypothetical
    defer/unknown.  Backend-returned acceptance charges it once.  Completion has
    no refund API.  Without a current epoch, the global acceptance totals remain
    observable but cannot be attributed to an epoch.

    Rejected requires a caller-proven pre-acceptance failure.  An uncertain
    exception permanently marks this audit incomplete for the run; a new epoch
    does not establish drain or recover unknown physical usage.
    """
    def __init__(self, run_id, *, mode="off", max_records=96):
        _run_id(run_id)
        if mode not in ("off", "shadow"):
            raise ValueError("only off and shadow are qualified")
        if type(max_records) is not int or not 1 <= max_records <= 96:
            raise ValueError("bounded audit history must be 1..96")
        self.run_id = run_id
        self.mode = mode
        self.bound = False
        self.owner = None
        self.faulted = False
        self.error = None
        self.completion_unknown_ops = 0
        self.completion_unknown_stages = {}
        self.last_now = None
        self.grant = None
        self.used = list(ZERO)
        self.pending = None
        self.records = deque(maxlen=max_records)
        self.overwritten_records = 0
        self.accepted_totals = {stage: Amount() for stage in STAGES}
        self.rejected_ops = 0
        self.uncertain_ops = 0
        self.uncertain_requested_bytes = 0
        self.shadow_overruns = 0
        self.accepted_without_epoch_ops = 0
        self.accepted_without_epoch_bytes = 0

    def bind(self):
        """Exclusive reactor binding; no owner thread is claimed here."""
        if self.bound:
            raise ValueError("shadow audit already belongs to a reactor")
        self.bound = True

    def fail(self, reason):
        """Optional audit faults never alter native dispatch or ownership."""
        self.faulted = True
        self.error = str(reason)[:160]

    def _owner(self):
        if not self.bound:
            raise RuntimeError("shadow audit must be explicitly bound")
        ident = get_ident()
        if self.owner is None:
            self.owner = ident
        if self.owner != ident:
            raise RuntimeError("shadow audit must stay on its owner thread")

    def _time(self, now_ns):
        self._owner()
        integer(now_ns, "clock")
        if self.last_now is not None and now_ns < self.last_now:
            raise ValueError("monotonic clock regressed")
        self.last_now = now_ns

    def publish(self, grant, *, now_ns):
        if self.mode == "off" or self.faulted:
            return False
        self._time(now_ns)
        if type(grant) is not DispatchBudget or grant.run_id != self.run_id:
            raise ValueError("wrong shadow grant identity")
        if not grant.issued_ns <= now_ns < grant.expires_ns:
            raise ValueError("new grant must be current")
        old = self.grant
        if old is not None and grant.epoch == old.epoch:
            if grant != old:
                raise ValueError("same epoch cannot change or replenish allowance")
            return False
        if self.pending is not None:
            raise RuntimeError("settle the immediate audit attempt before a new epoch")
        if old is not None and (grant.epoch <= old.epoch or grant.issued_ns < old.expires_ns):
            raise ValueError("epochs must advance without overlap")
        self.grant = grant
        self.used = list(ZERO)
        return True

    def _preview(self, stage, nbytes, sample, now_ns, staging_bytes_needed, progress):
        g = self.grant
        current = g is not None and g.issued_ns <= now_ns < g.expires_ns
        unknown = []
        perf = []
        native = []
        if not current:
            unknown.append("current_grant")
        fresh = (sample is not None and current and
                 0 <= now_ns - sample.captured_ns <= g.sample_max_age_ns)
        if sample is None:
            unknown.append("native_state")
        elif not fresh:
            unknown.append("fresh_native_state")
        inflight = sample.inflight if fresh else None
        free = sample.free_staging_bytes if fresh else None
        parents = sample.accepted_parents if fresh else None
        safe = sample.native_issue_safe if fresh else None
        for name, value in (("inflight", inflight), ("free_staging_bytes", free),
                            ("accepted_parents", parents), ("native_issue_safe", safe)):
            if value is None:
                unknown.append(name)
        if self.uncertain_ops:
            unknown.append("prior_acceptance_uncertain")
        if self.completion_unknown_ops:
            unknown.append("prior_completion_unknown")
        if safe is False or (free is not None and staging_bytes_needed > free):
            native.append("native_capacity_or_dependency")
        if current:
            i = STAGES.index(stage)
            used = self.used[i]
            cap = g.cumulative[i]
            for name, value, bound in (
                    ("stage_epoch_ops", used.ops + 1, cap.ops),
                    ("stage_epoch_bytes", used.nbytes + nbytes, cap.nbytes)):
                if value > bound:
                    perf.append(name)
            if inflight is not None:
                active = inflight[i]
                simultaneous = g.inflight[i]
                for name, value, bound in (
                        ("stage_inflight_ops", active.ops + 1, simultaneous.ops),
                        ("stage_inflight_bytes", active.nbytes + nbytes, simultaneous.nbytes)):
                    if value > bound:
                        perf.append(name)
            if i < 2:
                for prefix, amounts, bound in (
                        ("shared_ssd_epoch", self.used[:2], g.shared_ssd_cumulative),
                        ("shared_ssd_inflight", inflight[:2] if inflight is not None else None,
                         g.shared_ssd_inflight)):
                    if amounts is not None:
                        if sum(a.ops for a in amounts) + 1 > bound.ops:
                            perf.append(prefix + "_ops")
                        if sum(a.nbytes for a in amounts) + nbytes > bound.nbytes:
                            perf.append(prefix + "_bytes")
            else:
                if sum(a.nbytes for a in self.used[2:]) + nbytes > g.shared_copy_cumulative_bytes:
                    perf.append("shared_copy_epoch_bytes")
                if (inflight is not None and
                        sum(a.nbytes for a in inflight[2:]) + nbytes > g.shared_copy_inflight_bytes):
                    perf.append("shared_copy_inflight_bytes")
            if (staging_bytes_needed and free is not None and
                    free - staging_bytes_needed < g.reserve_staging_bytes):
                perf.append("staging_reserve")
        # Factual mandatory/continuation tags never implement an override.
        verdict = ("unknown" if unknown else "would_defer" if native or perf else "would_issue")
        return current, verdict, tuple(perf), tuple(native), tuple(unknown)

    def begin(self, stage, nbytes, sample=None, *, now_ns,
              staging_bytes_needed=0, progress=None):
        if self.mode == "off" or self.faulted:
            return None
        self._time(now_ns)
        if stage not in STAGES:
            raise ValueError("unknown physical stage")
        integer(nbytes, "physical bytes", 1)
        integer(staging_bytes_needed, "required staging")
        if progress is not None and progress not in PROGRESS:
            raise ValueError("unknown factual progress label")
        if sample is not None and (type(sample) is not ShadowState or sample.run_id != self.run_id):
            raise ValueError("wrong shadow sample identity")
        if self.pending is not None:
            raise RuntimeError("one immediate native audit attempt at a time")
        current, verdict, perf, native, unknown = self._preview(
            stage, nbytes, sample, now_ns, staging_bytes_needed, progress)
        # An audit ticket does not authorize native work and does not enforce caps.
        permit = Permit(self.grant.epoch, stage, nbytes, "shadow_audit") if current else None
        attempt = ShadowAttempt(stage, nbytes, now_ns, permit, verdict,
                                perf, native, unknown, progress)
        self.pending = attempt
        return attempt

    def settle(self, attempt, *, outcome, reason=""):
        """Immediately after the native call. Never wait for completion here.

        accepted: API returned successfully (may only have queued work).
        rejected: caller proved failure occurred before THIS work was accepted.
        uncertain: an exception might have partially launched/accepted THIS work.
        """
        if self.mode == "off":
            if attempt is not None:
                raise ValueError("off mode cannot settle a shadow attempt")
            return False
        self._owner()
        if self.pending is None or self.pending is not attempt:
            raise ValueError("stale, foreign or duplicate audit attempt")
        if outcome not in ("accepted", "rejected", "uncertain"):
            raise ValueError("explicit accepted/rejected/uncertain outcome required")
        if type(reason) is not str:
            raise ValueError("bounded text reason required")
        self.pending = None
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
            self.shadow_overruns += int(bool(attempt.performance_reasons))
        elif outcome == "rejected":
            self.rejected_ops += 1
        else:
            self.uncertain_ops += 1
            # Requested payload of ambiguous calls, NOT actual executed bytes.
            self.uncertain_requested_bytes += attempt.nbytes
        if len(self.records) == self.records.maxlen:
            self.overwritten_records += 1
        self.records.append(dict(
            stage=attempt.stage, requested_bytes=attempt.nbytes,
            started_ns=attempt.started_ns, epoch=attempt.epoch,
            verdict=attempt.verdict, performance_reasons=attempt.performance_reasons,
            native_reasons=attempt.native_reasons, unknown_fields=attempt.unknown_fields,
            progress=attempt.progress, outcome=outcome, reason=reason[:160]))
        return True

    def accepted(self, attempt):
        return False if attempt is None else self.settle(attempt, outcome="accepted")

    def rejected(self, attempt, reason=""):
        return False if attempt is None else self.settle(attempt, outcome="rejected", reason=reason)

    def uncertain(self, attempt, reason=""):
        return False if attempt is None else self.settle(attempt, outcome="uncertain", reason=reason)

    def mark_completion_unknown(self, stage, reason=""):
        """API returned, but an end-event/physical completion witness failed.

        This cannot refund accepted bytes or mean physical work has drained.
        The caller retains all native error/resource handling.
        """
        if self.mode == "off":
            return False
        self._owner()
        if stage not in STAGES or type(reason) is not str:
            raise ValueError("known stage and bounded text reason required")
        self.completion_unknown_ops += 1
        self.completion_unknown_stages[stage] = reason[:160]
        return True

    def snapshot(self, *, native_shutdown=False):
        """Owner read, or caller-proven native shutdown read.

        native_shutdown is an explicit CALLER precondition that no owner can
        mutate this object. It does not prove DMA drain or release any resource.
        This resource-free audit cannot independently inspect a native worker.
        Complete flags describe audit attempt CLASSIFICATION completeness only;
        they never establish native DMA completion, physical drain or reuse.
        """
        if type(native_shutdown) is not bool:
            raise ValueError("explicit native shutdown precondition required")
        if self.mode != "off" and not native_shutdown:
            self._owner()
        return dict(
            run_id=self.run_id, mode=self.mode, bound=self.bound,
            faulted=self.faulted, error=self.error,
            completion_unknown_ops=self.completion_unknown_ops,
            completion_unknown_stages=dict(self.completion_unknown_stages),
            completion_accounting_complete=(self.completion_unknown_ops == 0 and
                self.uncertain_ops == 0 and not self.faulted and self.pending is None),
            native_shutdown_read=native_shutdown,
            epoch=self.grant.epoch if self.grant is not None else None,
            used={s: dict(ops=a.ops, bytes=a.nbytes) for s, a in zip(STAGES, self.used)},
            observed_api_accepted={s: dict(ops=a.ops, bytes=a.nbytes) for s, a in self.accepted_totals.items()},
            rejected_ops=self.rejected_ops, uncertain_ops=self.uncertain_ops,
            uncertain_requested_bytes=self.uncertain_requested_bytes,
            uncertain_bytes_are_requested_not_executed=True,
            acceptance_accounting_complete=(self.uncertain_ops == 0 and
                not self.faulted and self.pending is None),
            complete_flags_scope="audit classifications only; not DMA completion or drain",
            epoch_totals_are_lower_bounds=self.uncertain_ops > 0 or self.faulted,
            shadow_overruns=self.shadow_overruns,
            accepted_without_epoch_ops=self.accepted_without_epoch_ops,
            accepted_without_epoch_bytes=self.accepted_without_epoch_bytes,
            pending_attempt=self.pending is not None,
            records=list(self.records), overwritten_records=self.overwritten_records,
            native_adapter_installed_by_this_module=False, byte_caps_enforced=False,
            parent_cap_enforced=False, progress_override_enforced=False,
            physical_drain_inferred=False, resource_release_inferred=False,
            gpu_qualified=False)
