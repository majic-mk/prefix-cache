"""CPU-qualified dispatch contract; not wired into a native submission path.

This ledger owns only cumulative allowance and one ephemeral issue permit.
Native queues, staging, CUDA dependencies and completion retain ownership.
In-flight values must be freshly read on the reactor owner after each native
submission/completion. They are not derived from cumulative issued bytes.
"""
from dataclasses import dataclass
from threading import get_ident

STAGES = ("ssd_read", "ssd_write", "h2d", "d2h")
PROGRESS = frozenset(("mandatory", "continuation", "age", "shutdown"))

def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))

@dataclass(frozen=True)
class Amount:
    ops: int = 0
    nbytes: int = 0
    def __post_init__(self):
        integer(self.ops, "ops")
        integer(self.nbytes, "bytes")

ZERO = (Amount(),) * 4

def vector(value):
    if type(value) is not tuple or len(value) != 4 or any(type(x) is not Amount for x in value):
        raise ValueError("exactly four immutable stage amounts required")

@dataclass(frozen=True)
class DispatchBudget:
    run_id: str
    epoch: int
    issued_ns: int
    expires_ns: int
    cumulative: tuple[Amount, ...]
    inflight: tuple[Amount, ...]
    shared_ssd_cumulative: Amount
    shared_ssd_inflight: Amount
    shared_copy_cumulative_bytes: int
    shared_copy_inflight_bytes: int
    reserve_staging_bytes: int
    max_accepted_parents: int
    sample_max_age_ns: int

    def __post_init__(self):
        if type(self.run_id) is not str or not self.run_id.strip() or len(self.run_id) > 128:
            raise ValueError("bounded run identity required")
        for name in ("epoch", "issued_ns", "expires_ns", "shared_copy_cumulative_bytes",
                     "shared_copy_inflight_bytes", "reserve_staging_bytes"):
            integer(getattr(self, name), name)
        for name in ("max_accepted_parents", "sample_max_age_ns"):
            integer(getattr(self, name), name, 1)
        if self.expires_ns <= self.issued_ns:
            raise ValueError("grant must have a finite nonempty interval")
        vector(self.cumulative)
        vector(self.inflight)
        if type(self.shared_ssd_cumulative) is not Amount or type(self.shared_ssd_inflight) is not Amount:
            raise ValueError("shared SSD limits require immutable Amount values")

@dataclass(frozen=True)
class NativeState:
    run_id: str
    captured_ns: int
    inflight: tuple[Amount, ...]
    free_staging_bytes: int
    # Native intake owns this count, including accepted jobs not yet in _active.
    accepted_parents: int
    # True only after original resource/event safety checks at the issue boundary.
    native_issue_safe: bool

    def __post_init__(self):
        if type(self.run_id) is not str or not self.run_id.strip():
            raise ValueError("native run identity required")
        vector(self.inflight)
        for name in ("captured_ns", "free_staging_bytes", "accepted_parents"):
            integer(getattr(self, name), name)
        if type(self.native_issue_safe) is not bool:
            raise ValueError("native safety must be explicit")

@dataclass(frozen=True)
class Permit:
    epoch: int
    stage: str
    nbytes: int
    reason: str

@dataclass(frozen=True)
class Decision:
    action: str  # issue / defer / native_fallback
    reasons: tuple[str, ...]
    permit: Permit | None = None

class DispatchLedger:
    """Single owner; no Future, resource reference, deferred work or completion.

    All limits in DispatchBudget except max_accepted_parents are performance
    allowances. Progress may exceed them, explicitly counted, but cannot
    override native_issue_safe. The parent bound belongs before native intake;
    stage dispatch NEVER rejects an already accepted parent to enforce it.

    A fused launch is one operation with the sum of actual copy sizes. A shared
    disk read is charged once, each physical destination copy is still charged.
    Accepted API calls consume allowance even if their later completion fails.
    """
    def __init__(self, run_id):
        if type(run_id) is not str or not run_id.strip() or len(run_id) > 128:
            raise ValueError("bounded run identity required")
        self.run_id = run_id
        self.owner = None
        self.last_now = None
        self.grant = None
        self.used = list(ZERO)
        self.pending = None
        self.accepted_ops = 0
        self.rejected_ops = 0
        self.overrides = 0

    def _owner(self):
        ident = get_ident()
        if self.owner is None:
            self.owner = ident
        if self.owner != ident:
            raise RuntimeError("dispatch contract must stay on its owner thread")

    def _time(self, now_ns):
        self._owner()
        integer(now_ns, "clock")
        if self.last_now is not None and now_ns < self.last_now:
            raise ValueError("monotonic clock regressed")
        self.last_now = now_ns

    def publish(self, grant, *, now_ns):
        self._time(now_ns)
        if type(grant) is not DispatchBudget or grant.run_id != self.run_id:
            raise ValueError("wrong dispatch grant identity")
        if not grant.issued_ns <= now_ns < grant.expires_ns:
            raise ValueError("new grant must be current")
        old = self.grant
        if old is not None and grant.epoch == old.epoch:
            if grant != old:
                raise ValueError("same epoch cannot change or replenish allowance")
            return False
        if self.pending is not None:
            raise RuntimeError("settle the immediate backend issue before replacing its grant")
        if old is not None and (grant.epoch <= old.epoch or grant.issued_ns < old.expires_ns):
            raise ValueError("epochs must advance without overlapping grants")
        self.grant = grant
        self.used = list(ZERO)
        return True

    def _sample(self, sample, now_ns):
        if type(sample) is not NativeState or sample.run_id != self.run_id:
            raise ValueError("wrong native sample identity")
        g = self.grant
        return (g is not None and g.issued_ns <= now_ns < g.expires_ns
                and 0 <= now_ns - sample.captured_ns <= g.sample_max_age_ns)

    def parent_admission(self, sample, *, now_ns):
        """Read-only pre-intake check, NOT a replacement native admission API.

        None means unknown: the caller must use its independently bounded common
        intake path. No current native intake adapter is installed by this module.
        """
        self._time(now_ns)
        if not self._sample(sample, now_ns):
            return None
        return sample.accepted_parents < self.grant.max_accepted_parents

    def reserve(self, stage, nbytes, sample, *, now_ns, staging_bytes_needed=0, progress=None):
        self._time(now_ns)
        if stage not in STAGES:
            raise ValueError("unknown physical stage")
        integer(nbytes, "physical bytes", 1)
        integer(staging_bytes_needed, "required staging")
        if progress is not None and progress not in PROGRESS:
            raise ValueError("unknown progress override")
        if self.pending is not None:
            raise RuntimeError("one immediate native issue at a time")
        if not self._sample(sample, now_ns):
            # A stale snapshot is not permission to bypass native resource checks.
            return Decision("native_fallback", ("expired_or_missing_grant_or_sample",))
        if not sample.native_issue_safe or staging_bytes_needed > sample.free_staging_bytes:
            return Decision("defer", ("native_capacity_or_dependency",))
        g = self.grant
        i = STAGES.index(stage)
        used, active = self.used[i], sample.inflight[i]
        cap, simultaneous = g.cumulative[i], g.inflight[i]
        reasons = []
        for name, value, bound in (
                ("stage_epoch_ops", used.ops + 1, cap.ops),
                ("stage_epoch_bytes", used.nbytes + nbytes, cap.nbytes),
                ("stage_inflight_ops", active.ops + 1, simultaneous.ops),
                ("stage_inflight_bytes", active.nbytes + nbytes, simultaneous.nbytes)):
            if value > bound:
                reasons.append(name)
        if i < 2:
            for prefix, amounts, bound in (
                    ("shared_ssd_epoch", self.used[:2], g.shared_ssd_cumulative),
                    ("shared_ssd_inflight", sample.inflight[:2], g.shared_ssd_inflight)):
                if sum(a.ops for a in amounts) + 1 > bound.ops:
                    reasons.append(prefix + "_ops")
                if sum(a.nbytes for a in amounts) + nbytes > bound.nbytes:
                    reasons.append(prefix + "_bytes")
        else:
            if sum(a.nbytes for a in self.used[2:]) + nbytes > g.shared_copy_cumulative_bytes:
                reasons.append("shared_copy_epoch_bytes")
            if sum(a.nbytes for a in sample.inflight[2:]) + nbytes > g.shared_copy_inflight_bytes:
                reasons.append("shared_copy_inflight_bytes")
        if staging_bytes_needed and sample.free_staging_bytes - staging_bytes_needed < g.reserve_staging_bytes:
            reasons.append("staging_reserve")
        if reasons and progress is None:
            return Decision("defer", tuple(reasons))
        ticket = Permit(g.epoch, stage, nbytes, progress or "ordinary")
        self.pending = (ticket, bool(reasons))
        return Decision("issue", tuple(reasons), ticket)

    def settle(self, permit, *, accepted):
        """Immediately after backend return; rejection does not consume allowance.

        No grant replacement is allowed while this token is outstanding. A call
        that returns after expiry is charged to the issuing epoch, not refunded.
        Later DMA/CQE failure also does not refund bandwidth already submitted.
        """
        self._owner()
        if type(accepted) is not bool:
            raise ValueError("explicit backend acceptance required")
        if self.pending is None or self.pending[0] is not permit:
            raise ValueError("stale, foreign or duplicate issue token")
        _, exceeded = self.pending
        self.pending = None
        if accepted:
            i = STAGES.index(permit.stage)
            old = self.used[i]
            self.used[i] = Amount(old.ops + 1, old.nbytes + permit.nbytes)
            self.accepted_ops += 1
            self.overrides += int(exceeded)
        else:
            self.rejected_ops += 1

    def snapshot(self):
        self._owner()
        return dict(run_id=self.run_id, epoch=self.grant.epoch if self.grant else None,
                    used={s: dict(ops=a.ops, bytes=a.nbytes) for s, a in zip(STAGES, self.used)},
                    pending_issue=self.pending is not None, accepted_ops=self.accepted_ops,
                    rejected_ops=self.rejected_ops, progress_overrides=self.overrides,
                    native_adapter_installed=False, gpu_qualified=False)
