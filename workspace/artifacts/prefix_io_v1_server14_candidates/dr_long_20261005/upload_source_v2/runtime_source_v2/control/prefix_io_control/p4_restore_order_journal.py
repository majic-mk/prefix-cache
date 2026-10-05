"""Common bounded scalar taps at original order and accepted read boundaries.

This records real work identities, never grants I/O or changes native owners.
Overflow makes the evidence incomplete; it never drops a native operation.
"""
from collections import Counter
from threading import get_ident


class RestoreOrderJournal:
    def __init__(self, run_id, *, submission_limit=16384, change_limit=1024):
        if type(run_id) is not str or not run_id or len(run_id) > 128:
            raise ValueError("actual run identity required")
        if (type(submission_limit) is not int or not 1 <= submission_limit <= 16384 or
                type(change_limit) is not int or not 1 <= change_limit <= 1024):
            raise ValueError("bounded journal limits required")
        self.run_id = run_id
        self.submission_limit = submission_limit
        self.change_limit = change_limit
        self._owner_ident = None
        self._last_ns = None
        self._submissions = []
        self._changes = []
        self._diagnostics = []
        self._diagnostic_keys = set()
        self._native_windows = []
        self._native_window_keys = set()
        self._reasons = Counter()
        self.calls = 0
        self.native_ready_calls = 0
        self.h2d_windows = self.pure_ssd_windows = self.unknown_restore_eta_windows = 0
        self.overflow = False
        self.fault = None

    def invalidate(self, reason):
        self.fault = str(reason)[:128]

    def _owner(self, at_ns):
        owner = get_ident()
        if self._owner_ident is None:
            self._owner_ident = owner
        if owner != self._owner_ident or type(at_ns) is not int or at_ns < 0:
            raise ValueError("original native owner and monotonic timestamp required")
        if self._last_ns is not None and at_ns < self._last_ns:
            raise ValueError("order journal owner clock moved backwards")
        self._last_ns = at_ns

    @staticmethod
    def _work_id(work_id):
        if (type(work_id) is not tuple or len(work_id) != 3 or
                type(work_id[0]) is not int or work_id[0] < 1 or
                type(work_id[1]) is not int or work_id[1] < 0 or
                work_id[2] not in ("ssd_read", "h2d", "d2h")):
            raise ValueError("actual accepted parent/file/stage identity required")
        return work_id

    def record_choice(self, publication, choice, *, at_ns):
        self._owner(at_ns)
        if publication.snapshot.run_id != self.run_id:
            raise ValueError("owner publication run mismatch")
        before = tuple(self._work_id(w.work_id) for w in publication.works)
        after = tuple(self._work_id(w) for w in choice.selected_work_ids)
        if len(before) > 64 or len(set(before)) != len(before) or set(before) != set(after) or len(after) != len(before):
            raise ValueError("native advice must preserve the complete work-ID permutation")
        self.calls += 1
        reason = choice.reason
        if reason not in self._reasons and len(self._reasons) >= 64:
            self.overflow = True
            return
        self._reasons[reason] += 1
        targets = tuple(sorted(t.restore_parent_id for t in publication.snapshot.targets
            if t.restore_parent_id is not None))
        parents = tuple(p for p in publication.snapshot.parents if p.job_id in targets)
        complete = tuple(sorted(p.job_id for p in parents if p.remaining_stages is not None and
            not p.failed and not p.future_done and p.done_files+p.inflight_files == p.total_files))
        known_eta = tuple(sorted(p.job_id for p in parents if p.completion_estimate_ns is not None))
        self.h2d_windows += any(w.stage == "h2d" for w in publication.works)
        self.pure_ssd_windows += bool(before) and all(w.stage == "ssd_read" for w in publication.works)
        self.unknown_restore_eta_windows += any(p.completion_estimate_ns is None for p in parents)
        diagnostic_key = (targets, reason, complete, known_eta)
        if diagnostic_key not in self._diagnostic_keys:
            if len(self._diagnostics) >= self.change_limit:
                self.overflow = True
            else:
                self._diagnostic_keys.add(diagnostic_key)
                self._diagnostics.append(dict(sequence=len(self._diagnostics)+1, at_ns=at_ns,
                    owner_epoch=publication.snapshot.snapshot_epoch, action=choice.action,
                    reason=reason, dependency_parent_ids=targets,
                    complete_ready_parent_ids=complete, eta_known_parent_ids=known_eta,
                    before_work_ids=before, after_work_ids=after,
                    gpu_release_credit=None, production_qualified=False))
        if after == before:
            return
        if len(self._changes) >= self.change_limit:
            self.overflow = True
            return
        identities = set(choice.closure_resource_ids)
        witnesses = tuple(w for w in publication.witnesses if w.resource.identity in identities)
        parents = tuple(sorted({p.job_id for w in witnesses for p in w.parents}))
        if (choice.action != "selected" or not parents or
                any(w.resource_kind != "restore" for w in witnesses)):
            raise ValueError("D_R changed order needs the actual selected restore-parent closure")
        self._changes.append(dict(sequence=len(self._changes)+1, at_ns=at_ns,
            owner_epoch=publication.snapshot.snapshot_epoch, action=choice.action,
            reason=choice.reason, target_id=choice.target_id,
            dependency_parent_ids=parents, before_work_ids=before, after_work_ids=after,
            actual_native_submission_proved_here=False))

    def record_native_submission(self, work_id, *, user_data, fd, slot_index, at_ns):
        self._owner(at_ns)
        work_id = self._work_id(work_id)
        if work_id[2] != "ssd_read" or any(type(v) is not int or v < 0 for v in (user_data, fd, slot_index)):
            raise ValueError("original accepted SSD-read operation identity required")
        if len(self._submissions) >= self.submission_limit:
            self.overflow = True
            return
        self._submissions.append(dict(sequence=len(self._submissions)+1, at_ns=at_ns,
            work_id=work_id, user_data=user_data, fd=fd, slot_index=slot_index,
            boundary="original_queue_read_returned_then_original_stage_accepted"))

    def record_native_ready_order(self, before, after, parent_facts, *, h2d_ready_count, at_ns):
        """Both arms: actual original SSD ready queue around its order call."""
        self._owner(at_ns)
        before = tuple(self._work_id(w) for w in before)
        after = tuple(self._work_id(w) for w in after)
        if len(before) > 64 or set(before) != set(after) or len(set(before)) != len(after):
            self.overflow = True
            return
        if (type(parent_facts) is not tuple or len(parent_facts) > 32 or
                any(type(r) is not tuple or len(r) != 7 for r in parent_facts) or
                type(h2d_ready_count) is not int or h2d_ready_count < 0):
            raise ValueError("original bounded parent/file facts required")
        parent_ids = tuple(r[0] for r in parent_facts)
        if set(parent_ids) != {w[0] for w in before} or len(set(parent_ids)) != len(parent_ids):
            raise ValueError("ready file identities must belong to the same observed native parents")
        by_parent = {}
        for pid,total,done,inflight,next_file,future_done,failed in parent_facts:
            if (any(type(v) is not int or v < 0 for v in (pid,total,done,inflight,next_file)) or
                    pid < 1 or total < 1 or done+inflight > total or next_file > total or
                    type(future_done) is not bool or type(failed) is not bool):
                raise ValueError("actual native parent file bounds required")
            by_parent[pid] = total
        if any(w[1] >= by_parent[w[0]] for w in before):
            raise ValueError("ready file identity outside its actual native parent")
        self.native_ready_calls += 1
        key = (parent_ids, bool(h2d_ready_count), before != after)
        if key in self._native_window_keys:
            return
        if len(self._native_windows) >= self.change_limit:
            self.overflow = True
            return
        self._native_window_keys.add(key)
        self._native_windows.append(dict(sequence=len(self._native_windows)+1, at_ns=at_ns,
            before_work_ids=before, after_work_ids=after,
            parent_facts=parent_facts, h2d_ready_count=h2d_ready_count,
            boundary="original_ready_load_fds_around_original_order_call"))

    def snapshot(self):
        return dict(schema="bounded_original_restore_order_journal_v1", run_id=self.run_id,
            valid=self.fault is None and not self.overflow, fault=self.fault,
            overflow=self.overflow, choices=self.calls,
            reason_counts=dict(sorted(self._reasons.items())),
            first_parent_reason_windows=tuple(self._diagnostics),
            common_native_ready_windows=tuple(self._native_windows), native_ready_calls=self.native_ready_calls,
            h2d_windows=self.h2d_windows, pure_ssd_windows=self.pure_ssd_windows,
            unknown_restore_eta_windows=self.unknown_restore_eta_windows,
            actual_order_changes=tuple(self._changes),
            actual_accepted_read_submission_order=tuple(self._submissions),
            production_qualified=False, gpu_release_credit=None,
            new_work_queues=0, held_native_owners=False)
