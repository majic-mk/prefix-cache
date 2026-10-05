"""Native P4 value bridge. No job/resource queue, backend, completion, or grant.

Publication is one immutable value view. The native owner must rebind every
publication to its current ready/parent/generation facts before using advice.
CPU qualification cannot qualify an interference table or GPU release.
"""
from dataclasses import dataclass, replace, asdict
from threading import get_ident
from .p4_policy import P4Policy
from .p4_types import SystemSnapshot, WorkDescriptor, ReleaseWitness

GPU_CAPS = frozenset(("gpu_owner_generation", "gpu_active_refs", "gpu_protectors"))

@dataclass(frozen=True)
class NativePublication:
    snapshot: SystemSnapshot
    works: tuple[WorkDescriptor, ...]
    witnesses: tuple[ReleaseWitness, ...] = ()

    def __post_init__(self):
        if type(self.snapshot) is not SystemSnapshot:
            raise TypeError("immutable owner snapshot required")
        if type(self.works) is not tuple or len(self.works) > 64 or any(
                type(w) is not WorkDescriptor for w in self.works):
            raise ValueError("at most 64 native value candidates required")
        if type(self.witnesses) is not tuple or len(self.witnesses) > 8 or any(
                type(w) is not ReleaseWitness for w in self.witnesses):
            raise ValueError("at most 8 native value witnesses required")
        if len({w.work_id for w in self.works}) != len(self.works):
            raise ValueError("ambiguous physical work identity")
        if self.snapshot.gpu_immediately_reusable_bytes is not None or (
                GPU_CAPS & self.snapshot.capabilities) or any(
                w.resource_kind == "gpu" for w in self.witnesses):
            raise ValueError("this native adapter has no GPU-owner release capability")
        if any(w.run_id != self.snapshot.run_id or
               w.snapshot_epoch != self.snapshot.snapshot_epoch for w in self.works):
            raise ValueError("same-run/epoch work values required")

def _parent_facts(parent):
    return replace(parent, completion_estimate_ns=None)

def _witness_facts(witness):
    return replace(witness, parents=tuple(_parent_facts(p) for p in witness.parents),
                   estimated_unblock_ns=None, error_ns=0, interference_ns=None)

def _state_facts(state):
    return replace(state, captured_ns=0) if state is not None else None

class NativeP4Bridge:
    """One owner, scalar generations, one replaceable publication; no owners held."""
    def __init__(self, policy):
        if type(policy) is not P4Policy:
            raise TypeError("exact P4 policy required")
        if policy.table is not None:
            # No table with CPU/mock evidence may enter the production adapter.
            if not policy.table.production_qualified:
                raise ValueError("production interference table is not GPU qualified")
        self.policy = policy
        self.run_id = policy.run_id
        self._bound = False
        self._owner_ident = None
        self._epoch = 0
        self._epoch_started_ns = None
        self._last_clock_ns = None
        self._signature = None
        self._publication = None
        self._generation_serial = 0
        self._generations = {}  # original slot indices -> scalar allocation lifetime
        self.fault = None
        self.last_reason = "not_observed"
        self.choices = self.publications = self.rejected_publications = 0
        self.last_choice = None
        self._choice_epoch = None
        self.last_preview = None
        self.generation_witness_omissions = 0
        from .p4_eta import NativeClosureHistory
        self.eta_history = NativeClosureHistory(self.run_id,
            max_age_ns=policy.config.max_wait_ns)
        self._eta_diagnostics = {}

    @property
    def mode(self):
        return self.policy.config.mode

    def bind(self):
        if self._bound:
            raise RuntimeError("P4 bridge cannot be shared")
        self._bound = True

    def _owner(self):
        if not self._bound:
            raise RuntimeError("P4 bridge requires explicit native binding")
        ident = get_ident()
        if self._owner_ident is None:
            self._owner_ident = ident
        elif self._owner_ident != ident:
            raise RuntimeError("P4 values must be observed by the native owner")

    def fail(self, reason):
        if self.fault is None:
            self.fault = str(reason)[:160]
        self._publication = None
        self.eta_history.abort_all()
        self._eta_diagnostics.clear()
        self.last_reason = "optional_bridge_fault"

    def observe_native_closure(self, parent_id, geometry, context, *, now_ns):
        """Observe scalar original-owner closure facts; never publish a forecast."""
        self._owner()
        self.eta_history.observe(run_id=self.run_id, parent_id=parent_id,
            geometry=geometry, context=context, now_ns=now_ns)
        estimate = self.eta_history.estimate(run_id=self.run_id, parent_id=parent_id,
            geometry=geometry, context=context, now_ns=now_ns,
            execution="shadow_diagnostic")
        if estimate is not None:
            self._eta_diagnostics[parent_id] = dict(parent_id=parent_id,
                estimate=asdict(estimate),
                source="actual_owner_complete_ready_history",
                production_completion_estimate_ns=None)
        else:
            self._eta_diagnostics.pop(parent_id, None)
        return None  # no CPU/fake or unqualified history enters native publication

    def complete_native_closure(self, parent_id, *, now_ns, successful, drain_known):
        self._owner()
        self._eta_diagnostics.pop(parent_id, None)
        return self.eta_history.finish(run_id=self.run_id, parent_id=parent_id,
            now_ns=now_ns, successful=successful, drain_known=drain_known)

    def note_reservation(self, index):
        self._owner()
        if type(index) is not int or index < 0:
            raise ValueError("native scalar slot index required")
        self._generation_serial += 1
        self._generations[index] = self._generation_serial

    def generation(self, index):
        self._owner()
        if type(index) is not int or index < 0:
            raise ValueError("native scalar slot index required")
        return self._generations.get(index)

    def epoch(self, signature, now_ns):
        self._owner()
        if type(signature) is not tuple or type(now_ns) is not int or now_ns < 0:
            raise ValueError("bounded owner signature and monotonic time required")
        nodes = [signature]
        seen = 0
        while nodes:
            value = nodes.pop()
            seen += 1
            if seen > 2048:
                raise ValueError("bounded owner signature metadata exceeded")
            if type(value) is tuple:
                if len(value) > 64:
                    raise ValueError("bounded signature tuple exceeded")
                nodes.extend(value)
            elif value is not None and type(value) not in (int,bool,str):
                raise TypeError("owner signature contains non-value metadata")
            elif type(value) is str and len(value) > 128:
                raise ValueError("bounded signature string exceeded")
        if self._last_clock_ns is not None and now_ns < self._last_clock_ns:
            raise ValueError("owner clock moved backwards")
        self._last_clock_ns = now_ns
        expired = (self._epoch_started_ns is not None and
                   now_ns - self._epoch_started_ns > self.policy.config.sample_max_age_ns)
        if signature != self._signature or expired:
            self._signature = signature
            self._epoch += 1
            self._epoch_started_ns = now_ns
            self._publication = None
            self._choice_epoch = None
        if self._epoch_started_ns is not None and now_ns < self._epoch_started_ns:
            raise ValueError("owner clock moved backwards")
        return self._epoch

    def _validate_binding(self, publication, current, now_ns):
        if type(publication) is not NativePublication or type(current) is not NativePublication:
            raise TypeError("strict native publication ABI required")
        s, actual = publication.snapshot, current.snapshot
        if not s.fresh(run_id=self.run_id, now_ns=now_ns,
                       expected_epoch=actual.snapshot_epoch,
                       max_age_ns=self.policy.config.sample_max_age_ns):
            raise ValueError("stale run/epoch/time publication")
        for state in (s.native_state, actual.native_state):
            if state is not None and (state.run_id != self.run_id or
                    state.captured_ns > now_ns or
                    now_ns-state.captured_ns > self.policy.config.sample_max_age_ns):
                raise ValueError("stale native stage-state publication")
        if (s.capabilities != actual.capabilities or
            s.generations != actual.generations or
            s.targets != actual.targets or
            s.cpu_clean_reclaimable_bytes != actual.cpu_clean_reclaimable_bytes or
            _state_facts(s.native_state) != _state_facts(actual.native_state) or
            s.load_signature != actual.load_signature or
            s.parents != actual.parents):
            raise ValueError("publication differs from actual owner facts")
        if publication.works != current.works:
            raise ValueError("publication work differs from actual native ready work")
        if len(publication.witnesses) != len(current.witnesses) or any(
                w != actual_w for w, actual_w in
                zip(publication.witnesses, current.witnesses)):
            raise ValueError("publication closure differs from actual owner protocol")
        known = {p.job_id: p for p in s.parents}
        if any(any(known.get(p.job_id) != p for p in w.parents) for w in publication.witnesses):
            raise ValueError("witness parent forecasts differ from published parent values")

    def publish(self, publication, current, *, now_ns):
        self._owner()
        try:
            self._validate_binding(publication, current, now_ns)
            if self._publication == publication:
                return False  # duplicate notification is metadata-idempotent
            if self._choice_epoch == current.snapshot.snapshot_epoch:
                raise ValueError("owner epoch already selected; publish before dispatch boundary")
            if self._publication is not None:
                raise ValueError("one frozen publication per owner epoch")
            self._publication = publication
            self.publications += 1
            return True
        except Exception:
            self.rejected_publications += 1
            raise  # native intake returns this diagnostic; no parent state changes

    def order(self, current, works, *, now_ns):
        self._owner()
        if not works:
            return ()  # no advice or epoch choice on an empty native window
        try:
            self._validate_binding(current,current,now_ns)
        except (TypeError,ValueError):
            self.last_reason = "stale_native_owner_view"
            return tuple(w.work_id for w in works)
        if self.fault is not None:
            self.last_reason = "optional_bridge_fault"
            return tuple(w.work_id for w in works)
        publication = current
        if self._publication is not None:
            try:
                self._validate_binding(self._publication, current, now_ns)
                publication = self._publication
            except (TypeError, ValueError):
                self.last_reason = "stale_or_changed_native_publication"
                self._publication = None
                return tuple(w.work_id for w in works)
        epoch = current.snapshot.snapshot_epoch
        if self._choice_epoch != epoch:
            self.last_choice = self.policy.choose(publication.snapshot, current.works,
                publication.witnesses, now_ns=now_ns, expected_epoch=epoch)
            self._choice_epoch = epoch
            self.choices += 1
            self.last_reason = self.last_choice.reason
            if current.works:
                self.last_preview = self.policy.issue_preview(current.works[0],
                    publication.snapshot, now_ns=now_ns, expected_epoch=epoch,
                    execution="production")
        choice = self.last_choice
        native_all = tuple(w.work_id for w in current.works)
        if set(choice.selected_work_ids) != set(native_all) or len(choice.selected_work_ids) != len(native_all):
            raise ValueError("advice is not a permutation of the original bounded window")
        if choice.gpu_qualified or choice.gpu_release_credit_bytes is not None:
            raise ValueError("unsupported GPU release/qualification credit")
        native = tuple(w.work_id for w in works)
        selected = tuple(wid for wid in choice.selected_work_ids if wid in set(native))
        if set(selected) != set(native):
            raise ValueError("native candidate subset changed")
        return selected

    def preview_issue(self, work, snapshot, *, now_ns):
        self._owner()
        if self.fault is not None:
            from .p4_types import IssuePreview
            return IssuePreview("native_fallback","optional_bridge_fault")
        preview = self.policy.issue_preview(work,snapshot,now_ns=now_ns,
            expected_epoch=snapshot.snapshot_epoch,execution="production")
        self.last_preview = preview
        return preview

    def snapshot(self, *, native_shutdown=False):
        if type(native_shutdown) is not bool:
            raise ValueError("explicit native shutdown precondition required")
        if not native_shutdown:
            self._owner()
        return dict(schema_version=1, run_id=self.run_id, mode=self.mode,
            owner_epoch=self._epoch, valid=self.fault is None, fault=self.fault,
            reason=self.last_reason, choices=self.choices, publications=self.publications,
            rejected_publications=self.rejected_publications,
            generation_witness_omissions=self.generation_witness_omissions,
            last_choice=asdict(self.last_choice) if self.last_choice is not None else None,
            last_preview=asdict(self.last_preview) if self.last_preview is not None else None,
            eta_history=self.eta_history.snapshot(),
            eta_diagnostics=tuple(self._eta_diagnostics.values()),
            production_eta_qualified=False,
            held_job_or_resource_owners=False, new_work_queues=0,
            gpu_release_credit=None, gpu_qualified=False,
            interference_production_qualified=self.policy.production_interference_qualified)

def make_native_bridge(run_id, config, *, table=None):
    from .p4_policy import make_p4_policy
    policy = make_p4_policy(run_id, config, table=table)
    return NativeP4Bridge(policy) if policy is not None else None
