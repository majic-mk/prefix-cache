"""Native P4 value bridge. No job/resource queue, backend, completion, or grant.

Publication is one immutable value view. The native owner must rebind every
publication to its current ready/parent/generation facts before using advice.
CPU qualification cannot qualify an interference table or GPU release.
"""
from dataclasses import dataclass, replace, asdict
from threading import get_ident
import weakref
import hashlib
import inspect
from pathlib import Path
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
            if not policy.table.production_qualified and not (
                    policy.config.mode == "interference" and policy.table.development_input is not None):
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
        self.restore_calibration = None
        self.restore_order_journal = None
        self.restore_forecasts_published = 0
        self.restore_forecast_fallbacks = 0
        self.last_batch = None
        self._batch_key = None
        self.prepared_cost_summary = (("status","NO_PREPARED_TABLE"),)
        self.declared_common_fixed_base = False
        self._single_file_capture = None
        self._single_file_runtime = None
        self.single_file_shadow = True
        self.single_file_previews = self.single_file_deferrals = 0
        self.single_file_blocked_attempts = 0
        self.single_file_observation_reuses = 0
        self.single_file_reasons = {}
        self._last_single_file_step = None
        self.single_file_decisions = []
        self._single_file_enriched_snapshot = None

    def attach_single_file_capture(self, capture, *, runtime_identity, runtime_refs, shadow=True):
        """Setup-only weak binding to the source-pinned native full-step probe.

        The surrounding guarded experiment verifies actual device/model and
        common source refs, and records this candidate overlay separately.
        A Python private/type boundary prevents accidental promotion; it is
        not a cryptographic security boundary against arbitrary process code.
        """
        receipt = self.policy.single_file
        from single_file_runtime_binding import RuntimeSingleFileIdentity
        if (type(runtime_identity) is not RuntimeSingleFileIdentity or
                not runtime_identity.matches_runner(capture.runner_ref()) or receipt is None or
                runtime_identity.signature_prefix != receipt.signature[:4]):
            raise ValueError("actual runner/model/layout/GPU/kernel identity must match calibrated condition")
        if (receipt is None or type(shadow) is not bool or self._single_file_capture is not None or
                getattr(capture, "origin", None) != "native_gpu_recording" or
                getattr(capture, "valid", None) is not True or
                capture.event_source.get("sha256") != receipt.cuda_event_source_sha256 or
                not callable(getattr(capture, "current_single_file_step", None))):
            raise ValueError("verified single-file receipt and actual source-bound full-step capture required")
        binding = receipt.binding_ref
        if (type(runtime_refs) is not dict or runtime_refs.get(binding.path) !=
                dict(path=binding.path, bytes=binding.bytes, sha256=binding.sha256)):
            raise ValueError("independently frozen runtime closure must include receipt binding")
        root = Path(capture.project_root).resolve()
        for kind in (type(capture), type(runtime_identity)):
            source = Path(inspect.getsourcefile(kind)).resolve()
            data = source.read_bytes()
            if not any(source == (root / ref.path).resolve() and len(data) == ref.bytes and
                       hashlib.sha256(data).hexdigest() == ref.sha256
                       for ref in receipt.runtime_overlay_refs):
                raise ValueError("actual live collector/identity helper must match the declared runtime overlay")
        self._single_file_capture = weakref.ref(capture)
        self._single_file_runtime = runtime_identity
        self.single_file_shadow = shadow

    def _single_file_state(self):
        capture = self._single_file_capture() if self._single_file_capture is not None else None
        if (capture is None or self._single_file_runtime is None or
                not self._single_file_runtime.matches_runner(capture.runner_ref())):
            return None
        return capture.current_single_file_step()

    def record_single_file_deferral(self, work_id, *, at_ns):
        """Compact real owner decision evidence, not hypothetical issue credit."""
        self._owner()
        state = self._last_single_file_step
        if (self.single_file_shadow or state is None or self._single_file_state() != state or
                type(at_ns) is not int or at_ns < state[2]):
            self._single_file_enriched_snapshot = None
            return False
        key = (state[0], str(work_id))
        row = next((r for r in self.single_file_decisions if r["key"] == key), None)
        if row is None:
            if len(self.single_file_decisions) >= 8:
                self.fail("single-file decision evidence bound exceeded")
                return False
            row = dict(key=key, native_step_ordinal=state[0],
                start_record_before_ns=state[1], start_completed_query_ns=state[2],
                first_defer_ns=at_ns, last_defer_ns=at_ns, count=0,
                resource_release_credit=False, production_qualified=False)
            self.single_file_decisions.append(row)
        row["last_defer_ns"] = at_ns
        row["count"] += 1
        self.single_file_blocked_attempts += 1
        return True

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
        self._single_file_enriched_snapshot = None
        self.eta_history.abort_all()
        self._eta_diagnostics.clear()
        self.last_reason = "optional_bridge_fault"

    def configure_restore_development(self, calibration=None, *, history_max_age_ns, order_journal=None):
        """Setup-only restore forecast input; no release or cost-table authority."""
        from .p4_restore_forecast import RestoreClosureCalibration
        from .p4_restore_order_journal import RestoreOrderJournal
        from .p4_eta import NativeClosureHistory
        if self._bound or self.mode != "dependency_only" or self.policy.table is not None:
            raise ValueError("restore-only setup before native bind is required")
        if calibration is not None and type(calibration) is not RestoreClosureCalibration:
            raise TypeError("actual guard-closed restore calibration input required")
        if (type(history_max_age_ns) is not int or not 0 < history_max_age_ns <= 600_000_000_000 or
                calibration is not None and calibration.max_age_ns != history_max_age_ns):
            raise ValueError("one bounded predeclared calibration lifetime required")
        if order_journal is not None and (type(order_journal) is not RestoreOrderJournal or
                order_journal.run_id != self.run_id):
            raise ValueError("same-run common original order journal required")
        self.policy.restore_development = True
        self.restore_calibration = calibration
        self.restore_order_journal = order_journal
        self.eta_history = NativeClosureHistory(self.run_id, max_age_ns=history_max_age_ns)

    def observe_native_closure(self, parent_id, geometry, context, *, now_ns):
        """Original production stays diagnostic; explicit D_R may use empirical ETA."""
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
        if self.policy.restore_development and self.restore_calibration is not None:
            forecast = self.restore_calibration.predict(self.eta_history, parent_id, geometry, context, now_ns=now_ns)
            self.restore_forecasts_published += forecast is not None
            self.restore_forecast_fallbacks += forecast is None
            return forecast
        if self.policy.restore_development:
            self.restore_forecast_fallbacks += 1
        return None  # original production/CPU paths remain diagnostic only

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
            self._batch_key = None
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
            if self.restore_order_journal is not None:
                try:
                    self.restore_order_journal.record_choice(current, self.last_choice, at_ns=now_ns)
                except Exception as exc:
                    self.restore_order_journal.invalidate(type(exc).__name__ + ": " + str(exc))
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


    def batch_prefix(self, current, works, *, now_ns):
        """Advisory prefix only. No allowance is granted, consumed or retained."""
        self._owner()
        if self.fault is not None or not works:
            return None
        self._validate_binding(current,current,now_ns)
        ids=tuple(w.work_id for w in works)
        if len(ids)>64 or len(set(ids))!=len(ids) or any(
                w not in current.works for w in works):
            raise ValueError("existing bounded native batch values required")
        if any(w.progress is not None for w in works):
            return None  # never cut an accepted downstream continuation
        key=(current.snapshot.snapshot_epoch,ids)
        if key != self._batch_key:
            self.last_batch=self.policy.choose_batch(current.snapshot,works,
                now_ns=now_ns,expected_epoch=current.snapshot.snapshot_epoch,
                execution="production")
            self._batch_key=key
        advice=self.last_batch
        if (advice.action!="selected" or advice.mock_only or
                not advice.production_qualified or
                self.mode not in ("interference","joint") or
                not self.policy.production_interference_qualified or
                "production_gpu_load_state" not in current.snapshot.capabilities):
            return None
        if (advice.stage!="ssd_read" or advice.work_ids!=ids[:len(advice.work_ids)] or
                not advice.work_ids or any(w.stage!="ssd_read" for w in works)):
            raise ValueError("batch advice must be an existing ordinary SSD-read prefix")
        chosen=works[:len(advice.work_ids)]
        if (advice.physical_bytes != sum(w.nbytes for w in chosen) or
                advice.storage_units != sum(w.nbytes//w.minimum_unit_bytes for w in chosen)):
            raise ValueError("batch advice differs from actual physical bytes/quantum")
        return advice.work_ids

    def preview_issue(self, work, snapshot, *, now_ns):
        self._owner()
        if self.fault is not None:
            self._single_file_enriched_snapshot = None
            from .p4_types import IssuePreview
            return IssuePreview("native_fallback","optional_bridge_fault")
        state = None
        if self.policy.single_file is not None:
            state = self._single_file_state()
            if state is not None:
                receipt = self.policy.single_file
                # attach checked this actual runtime prefix, not merely labels
                # copied from the historical receipt.
                prefix = self._single_file_runtime.signature_prefix
                signature = prefix + state[3:] + (917504,)
                cached = self._single_file_enriched_snapshot
                if (cached is not None and cached[0] is snapshot and
                        cached[1] == state and cached[2] == prefix):
                    snapshot = cached[3]
                else:
                    original = snapshot
                    snapshot = replace(snapshot, load_signature=signature,
                        capabilities=snapshot.capabilities | frozenset(("conditional_single_file_live_step",)))
                    # One immutable value pair only. Original timestamps remain
                    # unchanged; every call still runs policy freshness and the
                    # real state check below. No capture/runner/ready is held.
                    self._single_file_enriched_snapshot = (original, state, prefix, snapshot)
            else:
                self._single_file_enriched_snapshot = None
        else:
            self._single_file_enriched_snapshot = None
        preview = self.policy.issue_preview(work,snapshot,now_ns=now_ns,
            expected_epoch=snapshot.snapshot_epoch,execution=("gpu_development" if
                self.policy.table is not None and self.policy.table.development_input is not None else "production"),
            _single_file_state=state)
        if self.policy.single_file is not None:
            from .p4_types import IssuePreview
            if state is not None and self._single_file_state() != state:
                preview = IssuePreview("native_fallback", "single_file_step_ended_during_preview")
            if preview.reason == "verified_single_file_experimental_condition":
                self.single_file_previews += 1
                self.single_file_deferrals += preview.action == "defer"
            reason = preview.reason
            if reason not in self.single_file_reasons and len(self.single_file_reasons) >= 16:
                reason = "other_bounded_reason"
            self.single_file_reasons[reason] = self.single_file_reasons.get(reason, 0) + 1
        self.last_preview = preview
        self._last_single_file_step = (state if preview.reason ==
            "verified_single_file_experimental_condition" and preview.action == "defer" else None)
        if self._last_single_file_step is None or self.single_file_shadow:
            self._single_file_enriched_snapshot = None
        if self.policy.single_file is not None and self.single_file_shadow and preview.action == "defer":
            return IssuePreview("native_fallback", "single_file_shadow_preserves_native_issue")
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
            restore_development=dict(enabled=self.policy.restore_development,
                calibration_present=self.restore_calibration is not None,
                no_calibration_fallback_reason="no_actionable_calibration_input"
                    if self.restore_calibration is None else None,
                forecast_calls=self.restore_forecasts_published,
                forecast_fallbacks=self.restore_forecast_fallbacks,
                production_qualified=False, gpu_release_credit=None,
                empirical_ordering_only=True, earliest_completion_guaranteed=False),
            last_batch=asdict(self.last_batch) if self.last_batch is not None else None,
            prepared_cost_summary=dict(self.prepared_cost_summary),
            declared_common_fixed_base=self.declared_common_fixed_base,
            scheduled_load_is_active_gpu_state=False,
            conditional_single_file=dict(enabled=self.policy.single_file is not None,
                shadow=self.single_file_shadow, previews=self.single_file_previews,
                proposed_deferrals=self.single_file_deferrals,
                actual_blocked_attempts=self.single_file_blocked_attempts,
                observation_reuses=self.single_file_observation_reuses,
                reason_counts=tuple(sorted(self.single_file_reasons.items())),
                actual_decisions=tuple(dict(row) for row in self.single_file_decisions),
                evidence_binding_sha256=(self.policy.single_file.binding_ref.sha256
                    if self.policy.single_file is not None else None),
                generic_production_qualified=False),
            held_job_or_resource_owners=False, new_work_queues=0,
            gpu_release_credit=None, gpu_qualified=False,
            interference_production_qualified=self.policy.production_interference_qualified)

def make_native_bridge(run_id, config, *, table=None, single_file=None):
    from .p4_policy import make_p4_policy
    policy = make_p4_policy(run_id, config, table=table, single_file=single_file)
    return NativeP4Bridge(policy) if policy is not None else None
