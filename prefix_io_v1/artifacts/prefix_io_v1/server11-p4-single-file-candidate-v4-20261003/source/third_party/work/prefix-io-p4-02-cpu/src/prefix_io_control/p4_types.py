"""P4 immutable scalar observations; these objects confer no resource ownership.

All live values must be collected by existing owners. CPU fixtures are not GPU
qualification. None is unknown, never a measured zero or a permission to issue.
"""
from dataclasses import dataclass
from .dependencies import Parent, Resource, ResourceId
from .dispatch_budget import STAGES, integer
from .dispatch_shadow import ShadowState
from .simple_stage_policy import PROGRESS, _run_id, _work_id

MODES = frozenset(("off", "shadow", "dependency_only", "interference", "joint"))


def text(value, name):
    if type(value) is not str or not value.strip() or len(value) > 128:
        raise ValueError(name + " must be bounded nonempty text")


def optional_integer(value, name):
    if value is not None:
        integer(value, name)


def parent_values(parents):
    if type(parents) is not tuple or len(parents) > 32 or any(type(p) is not Parent for p in parents):
        raise ValueError("at most 32 immutable native parent values required")
    for p in parents:
        _run_id(p.run_id)
        for name in ("job_id", "total_files", "done_files", "inflight_files"):
            integer(getattr(p, name), name)
        if type(p.future_done) is not bool or type(p.failed) is not bool:
            raise ValueError("explicit parent terminal flags required")
        optional_integer(p.remaining_stages, "remaining_stages")
        optional_integer(p.completion_estimate_ns, "completion_estimate_ns")


def resource_id(value):
    if type(value) is not ResourceId:
        raise TypeError("native identity values required")
    _run_id(value.run_id)
    text(value.pool, "pool")
    for name in ("group", "block", "generation"):
        integer(getattr(value, name), name)


@dataclass(frozen=True)
class WaitingTarget:
    target_id: object
    native_order: int
    arrived_ns: int
    need_gpu_bytes: int = 0
    need_cpu_bytes: int = 0
    restore_parent_id: int | None = None

    def __post_init__(self):
        _work_id(self.target_id)
        for name in ("native_order", "arrived_ns", "need_gpu_bytes", "need_cpu_bytes"):
            integer(getattr(self, name), name)
        optional_integer(self.restore_parent_id, "restore_parent_id")


@dataclass(frozen=True)
class SystemSnapshot:
    run_id: str
    snapshot_epoch: int
    monotonic_ns: int
    capabilities: frozenset[str]
    parents: tuple[Parent, ...] = ()
    native_state: ShadowState | None = None
    targets: tuple[WaitingTarget, ...] = ()
    generations: tuple[tuple[ResourceId, int], ...] = ()
    gpu_immediately_reusable_bytes: int | None = None
    cpu_clean_reclaimable_bytes: int | None = None
    load_signature: tuple = ()

    def __post_init__(self):
        _run_id(self.run_id)
        integer(self.snapshot_epoch, "snapshot_epoch")
        integer(self.monotonic_ns, "monotonic_ns")
        if type(self.capabilities) is not frozenset or len(self.capabilities) > 32:
            raise ValueError("bounded frozen capability mask required")
        for cap in self.capabilities:
            text(cap, "capability")
        parent_values(self.parents)
        if any(p.run_id != self.run_id for p in self.parents):
            raise ValueError("snapshot parent identity must be same run")
        if self.native_state is not None and (type(self.native_state) is not ShadowState or
                                               self.native_state.run_id != self.run_id):
            raise ValueError("same-run native state required")
        if type(self.targets) is not tuple or len(self.targets) > 32 or any(
                type(t) is not WaitingTarget for t in self.targets):
            raise ValueError("bounded waiting-target values required")
        if len({t.target_id for t in self.targets}) != len(self.targets):
            raise ValueError("ambiguous target identities")
        if type(self.generations) is not tuple or len(self.generations) > 64:
            raise ValueError("bounded generation values required")
        seen = set()
        for entry in self.generations:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError("generation identity pairs required")
            identity, current = entry
            resource_id(identity)
            if identity.run_id != self.run_id:
                raise ValueError("snapshot resource identity must be same run")
            integer(current, "current generation")
            key = (identity.run_id, identity.pool, identity.group, identity.block)
            if key in seen:
                raise ValueError("ambiguous generation observation")
            seen.add(key)
        optional_integer(self.gpu_immediately_reusable_bytes, "GPU immediate bytes")
        optional_integer(self.cpu_clean_reclaimable_bytes, "CPU clean reclaim bytes")
        if type(self.load_signature) is not tuple or len(self.load_signature) > 16:
            raise ValueError("bounded exact load signature required")
        for item in self.load_signature:
            if type(item) not in (str, int) or (type(item) is str and len(item) > 128):
                raise ValueError("load signature contains values only")

    def fresh(self, *, run_id, now_ns, expected_epoch, max_age_ns):
        integer(now_ns, "now_ns")
        integer(expected_epoch, "expected_epoch")
        integer(max_age_ns, "max_age_ns", 1)
        return (self.run_id == run_id and self.snapshot_epoch == expected_epoch and
                0 <= now_ns - self.monotonic_ns <= max_age_ns)


@dataclass(frozen=True)
class WorkDescriptor:
    run_id: str
    snapshot_epoch: int
    parent_id: int
    child_id: object
    stage: str
    nbytes: int
    native_order: int
    created_ns: int
    submitted: bool
    accepted: bool = True
    staging_bytes_needed: int = 0
    generation: int | None = None
    progress: str | None = None
    minimum_unit_bytes: int = 1
    resource_identity: ResourceId | None = None

    def __post_init__(self):
        _run_id(self.run_id)
        _work_id(self.child_id)
        if type(self.child_id) not in (str, int):
            raise ValueError("native child identity must be scalar")
        for name in ("snapshot_epoch", "parent_id", "native_order", "created_ns",
                     "staging_bytes_needed"):
            integer(getattr(self, name), name)
        integer(self.nbytes, "legal work bytes", 1)
        integer(self.minimum_unit_bytes, "minimum unit bytes", 1)
        if self.nbytes % self.minimum_unit_bytes:
            raise ValueError("work must use legal backend units")
        if self.stage not in STAGES:
            raise ValueError("unknown native physical stage")
        for name in ("submitted", "accepted"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("explicit work lifecycle flags required")
        optional_integer(self.generation, "work generation")
        if self.progress is not None and self.progress not in PROGRESS:
            raise ValueError("unknown native progress reason")
        if self.resource_identity is not None:
            resource_id(self.resource_identity)
            if self.resource_identity.run_id != self.run_id:
                raise ValueError("work resource identity must be same run")
            if self.generation != self.resource_identity.generation:
                raise ValueError("work identity and generation must agree")

    @property
    def work_id(self):
        return (self.parent_id, self.child_id, self.stage)


@dataclass(frozen=True)
class ReleaseWitness:
    target_id: object
    resource: Resource
    parents: tuple[Parent, ...]
    resource_kind: str
    work_ids: tuple
    owner_source: str
    estimated_unblock_ns: int | None
    error_ns: int = 0
    interference_ns: int | None = None
    completed_restore_parent_id: int | None = None
    release_scope: str = "parent_job"

    def __post_init__(self):
        _work_id(self.target_id)
        if type(self.resource) is not Resource:
            raise TypeError("reuse the existing immutable Resource contract")
        r = self.resource
        resource_id(r.identity)
        integer(r.bytes, "physical resource bytes", 1)
        optional_integer(r.active_refs, "active references")
        if type(r.protecting_jobs) is not frozenset or len(r.protecting_jobs) > 32:
            raise ValueError("bounded protecting parent identities required")
        for pid in r.protecting_jobs:
            integer(pid, "protecting parent")
        if r.evidence not in ("observed_blocking", "candidate", "unknown"):
            raise ValueError("unknown evidence level")
        if type(r.clean_evictable) is not bool or (r.native_reusable is not None and
                                                  type(r.native_reusable) is not bool):
            raise ValueError("explicit native reclaim observations required")
        parent_values(self.parents)
        if any(p.run_id != r.identity.run_id for p in self.parents):
            raise ValueError("release parents and resource must be same run")
        if self.resource_kind not in ("gpu", "cpu", "restore"):
            raise ValueError("resource units must remain separate")
        if type(self.work_ids) is not tuple or len(self.work_ids) > 64:
            raise ValueError("bounded existing work identities required")
        for wid in self.work_ids:
            _work_id(wid)
        if len(set(self.work_ids)) != len(self.work_ids):
            raise ValueError("duplicate physical work identity")
        text(self.owner_source, "original owner source")
        optional_integer(self.estimated_unblock_ns, "estimated unblock time")
        integer(self.error_ns, "measurement error")
        optional_integer(self.interference_ns, "interference estimate")
        optional_integer(self.completed_restore_parent_id, "restore parent")
        if self.release_scope not in ("per_copy", "per_file", "parent_job", "multi_protector"):
            raise ValueError("explicit native release scope required")


@dataclass(frozen=True)
class P4Config:
    mode: str
    sample_max_age_ns: int
    max_wait_ns: int
    internal_step_budget_ns: int | None = None
    candidate_batches: tuple[int, ...] = (1, 2, 4, 8)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("only bounded V1 P4 modes are supported")
        integer(self.sample_max_age_ns, "sample age", 1)
        integer(self.max_wait_ns, "max wait", 1)
        optional_integer(self.internal_step_budget_ns, "internal step budget")
        if self.internal_step_budget_ns == 0:
            raise ValueError("internal budget must be positive or unknown")
        if (type(self.candidate_batches) is not tuple or not self.candidate_batches or
                len(self.candidate_batches) > 4 or any(type(x) is not int or
                x not in (1, 2, 4, 8) for x in self.candidate_batches) or
                tuple(sorted(set(self.candidate_batches))) != self.candidate_batches):
            raise ValueError("finite legal batch candidates required")


@dataclass(frozen=True)
class PolicyChoice:
    action: str
    selected_work_ids: tuple
    proposed_work_ids: tuple
    closure_resource_ids: tuple
    reason: str
    target_id: object | None = None
    gpu_release_credit_bytes: int | None = None
    cpu_release_credit_bytes: int | None = None
    observed_work_count: int = 0
    window_truncated: bool = False
    tail_unchanged: bool = True
    gpu_qualified: bool = False


@dataclass(frozen=True)
class IssuePreview:
    action: str
    reason: str
    progress_override: bool = False
    predicted_total_ns: int | None = None
    production_qualified: bool = False


@dataclass(frozen=True)
class BatchChoice:
    action: str
    work_ids: tuple
    stage: str | None
    physical_bytes: int
    storage_units: int
    predicted_total_ns: int | None
    reason: str
    mock_only: bool = False
    production_qualified: bool = False
