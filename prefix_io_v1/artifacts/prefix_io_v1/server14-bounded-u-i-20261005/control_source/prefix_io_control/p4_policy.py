"""Finite P4 advice over existing ready work, without native ownership.

No allocator, backend, queue, Future, completion, or cumulative budget lives
here. The original DispatchController is the sole allowance consumer. A choice
is a bounded hint; work outside the window remains in its native queue.
"""
from dataclasses import dataclass
from itertools import islice
from .dependencies import analyze, ResourceId
from .dispatch_budget import integer
from .p4_types import (
    SystemSnapshot, WorkDescriptor, ReleaseWitness, WaitingTarget, P4Config,
    PolicyChoice, IssuePreview, BatchChoice,
)
from .p4_cost_table import CostTable
from .simple_stage_policy import PROGRESS, _run_id, _work_id


@dataclass(frozen=True)
class _Closure:
    resources: tuple
    work_ids: tuple
    gpu_bytes: int
    cpu_bytes: int
    restore_parents: frozenset
    unblock_ns: int
    error_ns: int
    interference_ns: int | None
    new_bytes: int
    native_order: int


def make_p4_policy(run_id, config, *, table=None, single_file=None):
    _run_id(run_id)
    if type(config) is not P4Config:
        raise TypeError("frozen P4 configuration required")
    if config.mode == "off":
        return None
    return P4Policy(run_id, config, table=table, single_file=single_file)


class P4Policy:
    def __init__(self, run_id, config, *, table=None, single_file=None):
        _run_id(run_id)
        if type(config) is not P4Config or config.mode == "off":
            raise ValueError("off must create no policy object")
        if table is not None and type(table) is not CostTable:
            raise TypeError("exact finite cost-table interface required")
        self.run_id = run_id
        self.config = config
        self.table = table
        if single_file is not None:
            from .p4_single_file_receipt import ExactSingleFileReceipt
            if (type(single_file) is not ExactSingleFileReceipt or
                    config.mode != "interference" or table is not None or
                    config.internal_step_budget_ns != single_file.step_budget_ns):
                raise ValueError("verified exact single-file experiment and calibration-only budget required")
        self.single_file = single_file

    @property
    def production_interference_qualified(self):
        return self.table is not None and self.table.production_qualified

    def _fresh(self, snapshot, now_ns, expected_epoch):
        return (type(snapshot) is SystemSnapshot and snapshot.fresh(
            run_id=self.run_id, now_ns=now_ns, expected_epoch=expected_epoch,
            max_age_ns=self.config.sample_max_age_ns))

    @staticmethod
    def _generation(snapshot, identity):
        key = (identity.run_id, identity.pool, identity.group, identity.block)
        for observed, current in snapshot.generations:
            if (observed.run_id, observed.pool, observed.group, observed.block) == key:
                return current
        return None

    def _witness(self, witness, snapshot, works, target, now_ns):
        if type(witness) is not ReleaseWitness or witness.target_id != target.target_id:
            return None, "wrong_target_or_invalid_witness"
        r = witness.resource
        required = {"owner_release_protocol"}
        if witness.resource_kind == "gpu":
            required |= {"gpu_owner_generation", "gpu_active_refs", "gpu_protectors"}
        elif witness.resource_kind == "cpu":
            required |= {"cpu_owner_generation", "cpu_active_refs", "cpu_protectors"}
        else:
            required |= {"restore_parent_completion"}
        if not required.issubset(snapshot.capabilities):
            return None, "missing_release_capability"
        generation = self._generation(snapshot, r.identity)
        if generation is None:
            return None, "unknown_generation"
        observed = {p.job_id: p for p in snapshot.parents if p.run_id == self.run_id}
        if len(observed) != len(snapshot.parents):
            return None, "ambiguous_or_wrong_run_parent"
        if any(observed.get(p.job_id) != p for p in witness.parents):
            return None, "parent_observation_mismatch"
        release = analyze(r, witness.parents, run_id=self.run_id, current_generation=generation)
        if release.reason not in ("observed_closure", "native_release_confirmed"):
            return None, release.reason
        if r.evidence != "observed_blocking":
            return None, "unselected_victim_is_not_actual_blocker"
        if release.reason == "native_release_confirmed":
            return None, "native_zero_io_reclaim_first"
        if (witness.estimated_unblock_ns is None or release.completion_estimate_ns is None or
                witness.estimated_unblock_ns < release.completion_estimate_ns or
                witness.estimated_unblock_ns < now_ns):
            return None, "unknown_or_premature_legal_unblock_time"
        # Native parent source fences cannot be retired by a per-file/per-copy estimate.
        if r.protecting_jobs and witness.release_scope not in ("parent_job", "multi_protector"):
            return None, "release_scope_weaker_than_parent_fence"
        if len(r.protecting_jobs) > 1 and witness.release_scope != "multi_protector":
            return None, "multi_protector_scope_required"
        by_id = {w.work_id: w for w in works}
        if not witness.work_ids or any(wid not in by_id for wid in witness.work_ids):
            return None, "incomplete_ready_work_closure"
        selected = tuple(by_id[wid] for wid in witness.work_ids)
        if any(w.submitted or not w.accepted or w.progress is not None for w in selected):
            return None, "already_submitted_or_native_progress"
        if any(w.parent_id not in release.pending_parents for w in selected):
            return None, "closure_work_not_native_parent"
        # Every still-blocking parent needs a ready selectable stage; submitted-only
        # blockers are observations, not a manufactured independent ready queue.
        if not set(release.pending_parents).issubset({w.parent_id for w in selected}):
            return None, "not_all_protectors_have_selectable_work"
        if any(w.resource_identity is not None and
                self._generation(snapshot, w.resource_identity) != w.generation for w in selected):
            return None, "stale_work_generation"
        if witness.resource_kind == "restore":
            restore_id = witness.completed_restore_parent_id
            if (restore_id is None or restore_id != target.restore_parent_id or
                    set(release.pending_parents) != {restore_id} or
                    any(w.parent_id != restore_id or w.stage not in ("ssd_read", "h2d")
                        for w in selected)):
                return None, "restore_parent_or_direction_mismatch"
        credit = release.potential_bytes
        return _Closure(
            (r.identity,), witness.work_ids,
            credit if witness.resource_kind == "gpu" else 0,
            credit if witness.resource_kind == "cpu" else 0,
            frozenset((witness.completed_restore_parent_id,))
                if witness.resource_kind == "restore" and witness.completed_restore_parent_id is not None
                else frozenset(),
            witness.estimated_unblock_ns, witness.error_ns, witness.interference_ns,
            sum(w.nbytes for w in selected), min(w.native_order for w in selected)), "verified_closure"

    @staticmethod
    def _combine(closures, works):
        resources = {}
        work_ids = set()
        restores = set()
        for c in closures:
            # Repeated resource identity must never produce repeated release credit.
            for identity in c.resources:
                if identity not in resources:
                    resources[identity] = c
            work_ids.update(c.work_ids)
            restores.update(c.restore_parents)
        if not resources:
            return None
        chosen = tuple(resources.values())
        selected = tuple(w for w in works if w.work_id in work_ids)
        return _Closure(tuple(resources), tuple(w.work_id for w in selected),
                        sum(c.gpu_bytes for c in chosen), sum(c.cpu_bytes for c in chosen),
                        frozenset(restores), max(c.unblock_ns for c in chosen),
                        max(c.error_ns for c in chosen),
                        chosen[0].interference_ns if len(chosen) == 1 else None,
                        sum(w.nbytes for w in selected), min(w.native_order for w in selected))

    @staticmethod
    def _satisfies(c, target):
        return (c.gpu_bytes >= target.need_gpu_bytes and c.cpu_bytes >= target.need_cpu_bytes and
                (target.restore_parent_id is None or target.restore_parent_id in c.restore_parents))

    def choose(self, snapshot, works, witnesses=(), *, now_ns, expected_epoch):
        integer(now_ns, "now_ns")
        integer(expected_epoch, "expected_epoch")
        observed = tuple(islice(works, 65))
        if any(type(w) is not WorkDescriptor for w in observed):
            raise TypeError("existing native WorkDescriptor values required")
        truncated = len(observed) > 64
        window = observed[:64]
        native = tuple(w.work_id for w in window)
        if len(set(native)) != len(native):
            raise ValueError("ambiguous native work identity")
        gpu_known = (self._fresh(snapshot, now_ns, expected_epoch) and
                     "gpu_owner_generation" in snapshot.capabilities and
                     "gpu_active_refs" in snapshot.capabilities and
                     "gpu_protectors" in snapshot.capabilities and
                     "owner_release_protocol" in snapshot.capabilities)
        def choice(action, reason, proposed=native, closure=None, target=None):
            actual = proposed if action == "selected" else native
            return PolicyChoice(action, actual, proposed,
                closure.resources if closure is not None else (), reason,
                target.target_id if target is not None else None,
                closure.gpu_bytes if closure is not None and gpu_known else (0 if gpu_known else None),
                closure.cpu_bytes if closure is not None else None,
                len(window), truncated)
        if truncated:
            return choice("native_fallback", "work_window_exceeded")
        if not self._fresh(snapshot, now_ns, expected_epoch):
            return choice("native_fallback", "stale_or_incompatible_snapshot")
        if "native_ready_work" not in snapshot.capabilities:
            return choice("native_fallback", "missing_ready_work_capability")
        if any(w.run_id != self.run_id or w.snapshot_epoch != expected_epoch for w in window):
            return choice("native_fallback", "stale_work_identity")
        if any(w.created_ns > now_ns for w in window):
            return choice("native_fallback", "future_work_timestamp")
        if any(w.submitted for w in window):
            return choice("native_fallback", "submitted_work_cannot_be_reordered")
        if any(not w.accepted for w in window):
            return choice("native_fallback", "unaccepted_work_not_strategy_owned")
        if any(w.progress is not None for w in window):
            return choice("native", "native_progress_before_policy")
        if any(w.resource_identity is not None and
               self._generation(snapshot, w.resource_identity) != w.generation for w in window):
            return choice("native_fallback", "stale_work_generation")
        if self.config.mode in ("interference", "joint") and not self.production_interference_qualified:
            return choice("native_fallback", "production_interference_gpu_gate_blocked")
        if self.config.mode == "interference":
            return choice("native", "interference_preserves_native_target_order")
        targets = [t for t in snapshot.targets if
                   t.need_gpu_bytes or t.need_cpu_bytes or t.restore_parent_id is not None]
        if not targets:
            return choice("native", "no_observed_blocked_target")
        if any(t.arrived_ns > now_ns for t in targets):
            return choice("native_fallback", "future_target_timestamp")
        # Age prevents endless yielding without inventing a global request scheduler.
        target = min(targets, key=lambda t: (
            0 if now_ns - t.arrived_ns >= self.config.max_wait_ns else 1, t.native_order))
        if (target.need_gpu_bytes and snapshot.gpu_immediately_reusable_bytes is not None and
                snapshot.gpu_immediately_reusable_bytes >= target.need_gpu_bytes):
            return choice("native", "native_immediate_gpu_reuse_first", target=target)
        if (target.need_cpu_bytes and snapshot.cpu_clean_reclaimable_bytes is not None and
                snapshot.cpu_clean_reclaimable_bytes >= target.need_cpu_bytes):
            return choice("native", "native_clean_cpu_reclaim_first", target=target)
        selected_witnesses = tuple(islice(witnesses, 9))
        if len(selected_witnesses) > 8:
            return choice("native_fallback", "release_window_exceeded", target=target)
        closures, reasons, identities = [], [], set()
        for w in selected_witnesses:
            c, reason = self._witness(w, snapshot, window, target, now_ns)
            reasons.append(reason)
            if c is not None and c.resources[0] not in identities:
                identities.add(c.resources[0])
                closures.append(c)
        # At most seven individual closures plus one bounded multi-resource union.
        candidates = closures[:7]
        if len(closures) > 1:
            union = self._combine(closures, window)
            if union is not None:
                candidates = candidates + [union]
        feasible = [c for c in candidates if self._satisfies(c, target)]
        if not feasible:
            return choice("native_fallback", "no_proved_legal_closure:" +
                          (reasons[0] if reasons else "no_witness"), target=target)
        earliest = min(feasible, key=lambda c: (c.unblock_ns, c.native_order))
        equivalent = [c for c in feasible if
                      abs(c.unblock_ns - earliest.unblock_ns) <= c.error_ns + earliest.error_ns]
        if self.config.mode == "joint" and self.production_interference_qualified:
            best = min(equivalent, key=lambda c: (
                c.interference_ns is None,
                c.interference_ns if c.interference_ns is not None else 0,
                c.new_bytes, c.native_order))
        else:
            # D and unqualified shadow cannot quietly become an interference arm.
            best = min(equivalent, key=lambda c: (c.new_bytes, c.native_order))
        chosen = set(best.work_ids)
        proposed = tuple(w.work_id for w in window if w.work_id in chosen) + tuple(
            w.work_id for w in window if w.work_id not in chosen)
        action = "observed" if self.config.mode == "shadow" else "selected"
        return choice(action, "verified_bounded_native_release_closure",
                      proposed=proposed, closure=best, target=target)

    def issue_preview(self, work, snapshot, *, now_ns, expected_epoch, progress=None,
                      execution="production", _single_file_state=None):
        """Additional cost constraint only; never grants or charges any allowance."""
        if type(work) is not WorkDescriptor:
            raise TypeError("existing native work values required")
        if progress is None:
            progress = work.progress
        if progress is not None and progress not in PROGRESS:
            raise ValueError("known native progress reason required")
        if not self._fresh(snapshot, now_ns, expected_epoch):
            return IssuePreview("native_fallback", "stale_or_incompatible_snapshot")
        if execution not in ("production", "cpu_mock", "gpu_development"):
            raise ValueError("explicit production/CPU mock scope required")
        if execution == "gpu_development" and (self.config.mode != "interference" or
                self.single_file is not None or self.table is None or self.table.development_input is None):
            raise ValueError("one actual development I table only; no single-file/other policy promotion")
        if "native_ready_work" not in snapshot.capabilities:
            return IssuePreview("native_fallback", "missing_ready_work_capability")
        if (work.run_id != self.run_id or work.snapshot_epoch != expected_epoch or
                work.submitted or not work.accepted or work.created_ns > now_ns):
            return IssuePreview("native_fallback", "stale_or_nonselectable_work")
        if (work.resource_identity is not None and
                self._generation(snapshot, work.resource_identity) != work.generation):
            return IssuePreview("native_fallback", "stale_work_generation")
        if progress is not None or now_ns - work.created_ns >= self.config.max_wait_ns:
            return IssuePreview("issue", "native_progress_override", True)
        if self.single_file is not None:
            # This capability does not open the generic production table, batch
            # selection, release ETA, or any other load/stage/byte condition.
            from .dispatch_budget import ZERO
            receipt = self.single_file
            state = _single_file_state
            if (execution != "production" or type(state) is not tuple or len(state) != 7 or
                    any(type(v) is not int for v in state) or
                    not 0 < state[1] <= state[2] <= now_ns or
                    now_ns - state[2] > self.config.sample_max_age_ns or
                    state[3:] != (1, 1, 0, 144) or
                    "conditional_single_file_live_step" not in snapshot.capabilities or
                    snapshot.load_signature != receipt.signature or
                    snapshot.native_state is None or snapshot.native_state.inflight != ZERO or
                    not 0 <= now_ns - snapshot.native_state.captured_ns <= self.config.sample_max_age_ns or
                    work.stage != "ssd_read" or work.nbytes != 917504 or
                    work.minimum_unit_bytes != 917504 or work.parent_id != 0 or
                    type(work.child_id) is not str or not work.child_id.startswith("preload:")):
                return IssuePreview("native_fallback", "outside_verified_single_file_condition")
            action = "issue" if receipt.cost_upper_ns <= receipt.step_budget_ns else "defer"
            return IssuePreview(action, "verified_single_file_experimental_condition",
                                False, receipt.cost_upper_ns, False)
        if self.config.mode in ("shadow", "dependency_only"):
            return IssuePreview("issue", "existing_common_budget_only")
        if self.table is None or snapshot.native_state is None or snapshot.native_state.inflight is None:
            return IssuePreview("native_fallback", "unsupported_cost_state")
        if execution == "production" and not self.production_interference_qualified:
            return IssuePreview("native_fallback", "production_interference_gpu_gate_blocked")
        estimate = self.table.lookup(snapshot.load_signature, snapshot.native_state.inflight,
                                     work.stage, work.nbytes, execution=execution)
        if estimate is None or self.config.internal_step_budget_ns is None:
            return IssuePreview("native_fallback", "unsupported_exact_cost_cell_or_step_budget")
        if not 0 <= now_ns - snapshot.native_state.captured_ns <= self.config.sample_max_age_ns:
            return IssuePreview("native_fallback", "stale_existing_io_baseline")
        action = "issue" if estimate.total_ns <= self.config.internal_step_budget_ns else "defer"
        return IssuePreview(action, "mock_only_cost_preview" if execution == "cpu_mock"
                            else "observed_gpu_development_cost_preview" if execution == "gpu_development"
                            else "qualified_cost_preview", False, estimate.total_ns, False)


    def choose_batch(self, snapshot, works, *, now_ns, expected_epoch,
                     target_work_ids=(), execution="production"):
        """Finite legal native work prefixes; no splitting, fusing or submission.

        The result is only an advisory bound on an existing native batch. Native
        hard limits, existing fusion and DispatchController settlement remain
        authoritative. Unsupported state returns the baseline path, not zero
        cost. CPU mock forecasts cannot authorize a production batch.
        """
        integer(now_ns, "now_ns")
        integer(expected_epoch, "expected_epoch")
        observed = tuple(islice(works, 65))
        if any(type(w) is not WorkDescriptor for w in observed):
            raise TypeError("existing native work values required")
        native = tuple(w.work_id for w in observed[:64])
        def fallback(reason):
            return BatchChoice("native_fallback", native, None, 0, 0, None, reason,
                               execution == "cpu_mock", False)
        if len(observed) > 64 or len(set(native)) != len(native):
            return fallback("bounded_or_ambiguous_batch_window")
        if not self._fresh(snapshot, now_ns, expected_epoch):
            return fallback("stale_or_incompatible_snapshot")
        if "native_ready_work" not in snapshot.capabilities:
            return fallback("missing_ready_work_capability")
        if not observed:
            return fallback("no_existing_ready_work")
        if any(w.run_id != self.run_id or w.snapshot_epoch != expected_epoch or w.submitted
               or not w.accepted or w.created_ns > now_ns for w in observed):
            return fallback("nonselectable_native_work")
        if any(w.resource_identity is not None and
               self._generation(snapshot, w.resource_identity) != w.generation for w in observed):
            return fallback("stale_work_generation")
        if any(w.progress is not None or now_ns - w.created_ns >= self.config.max_wait_ns
               for w in observed):
            return fallback("native_progress_batch_before_policy")
        if execution not in ("production", "cpu_mock"):
            raise ValueError("explicit production/CPU mock scope required")
        if self.config.mode in ("interference", "joint"):
            if execution == "production" and not self.production_interference_qualified:
                return fallback("production_interference_gpu_gate_blocked")
            if (self.table is None or snapshot.native_state is None or
                    snapshot.native_state.inflight is None or
                    self.config.internal_step_budget_ns is None or
                    not 0 <= now_ns - snapshot.native_state.captured_ns <= self.config.sample_max_age_ns):
                return fallback("unsupported_existing_io_baseline")
        if type(target_work_ids) is not tuple or len(target_work_ids) > 64:
            raise ValueError("bounded target existing work IDs required")
        for wid in target_work_ids:
            _work_id(wid)
        if len(set(target_work_ids)) != len(target_work_ids):
            raise ValueError("target work identities must be unique")
        if any(wid not in native for wid in target_work_ids):
            return fallback("target_outside_existing_ready_window")
        stage = observed[0].stage
        quantum = observed[0].minimum_unit_bytes
        if any(w.stage == stage and w.minimum_unit_bytes != quantum for w in observed):
            return fallback("mixed_frozen_storage_unit_geometry")
        units, physical, candidates = 0, 0, []
        prefix = []
        for work in observed:
            if work.stage != stage:
                break
            # Never split a storage unit or a native descriptor.
            units += work.nbytes // work.minimum_unit_bytes
            physical += work.nbytes
            prefix.append(work.work_id)
            if units > self.config.candidate_batches[-1]:
                break
            if units not in self.config.candidate_batches:
                continue
            estimate = None
            if self.config.mode in ("interference", "joint"):
                estimate = self.table.lookup(snapshot.load_signature, snapshot.native_state.inflight,
                                             stage, physical, execution=execution)
                if estimate is None or estimate.total_ns > self.config.internal_step_budget_ns:
                    continue
            candidates.append((tuple(prefix), physical, units,
                               estimate.total_ns if estimate is not None else None))
        if not candidates:
            return fallback("no_supported_legal_batch")
        needs = set(target_work_ids)
        covering = [c for c in candidates if needs and needs.issubset(c[0])]
        # Advance enough to unblock the selected finite closure, otherwise use
        # the largest supported native batch. This is not a global optimum.
        chosen = min(covering, key=lambda c: c[2]) if covering else max(candidates, key=lambda c: c[2])
        action = "observed" if self.config.mode == "shadow" else "selected"
        return BatchChoice(action, chosen[0], stage, chosen[1], chosen[2], chosen[3],
                           "finite_supported_native_prefix", execution == "cpu_mock", False)
