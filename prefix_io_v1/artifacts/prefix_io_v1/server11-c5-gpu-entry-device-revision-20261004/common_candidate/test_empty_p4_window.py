"""Source-extracted CPU regressions for empty advice windows; no backend import."""
from __future__ import annotations

import ast
from collections import deque
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
REL = Path('source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py')
OLD = Path(os.environ.get('SERVER11_PREVIOUS_CANDIDATE_DIR',
                          str(HERE.with_name('p4_single_file_candidate_v1')))) / REL
NEW = HERE / REL
CHANGED = {'_prefix_p4_order', '_prefix_p4_read_batch_ids'}


def tree(path):
    module = ast.parse(path.read_text(encoding='utf-8-sig'))
    owner = next(x for x in module.body if isinstance(x, ast.ClassDef) and x.name == 'IoReactor')
    return module, {x.name: x for x in owner.body if isinstance(x, ast.FunctionDef)}


def methods(path):
    _, values = tree(path)
    names = CHANGED | {'_prefix_p4_parent_terminal', '_has_work'}
    unit = ast.Module(body=[values[name] for name in sorted(names)], type_ignores=[])
    ns = {'time': SimpleNamespace(monotonic_ns=lambda: 1000),
          'threading': SimpleNamespace(get_ident=lambda: 11)}
    exec(compile(ast.fix_missing_locations(unit), str(path), 'exec'), ns)
    return ns


class Bridge:
    def __init__(self, reverse=False):
        self.calls, self.failures, self.reverse = [], [], reverse

    def order(self, view, works, *, now_ns):
        ids = tuple(w.work_id for w in works)
        self.calls.append(('order', ids, now_ns))
        return ids[::-1] if self.reverse else ids

    def batch_prefix(self, view, works, *, now_ns):
        ids = tuple(w.work_id for w in works)
        self.calls.append(('batch', ids, now_ns))
        return ids

    def fail(self, reason):
        self.failures.append(reason)

    def complete_native_closure(self, parent_id, **kwargs):
        self.calls.append(('complete', parent_id, kwargs))


class Owner:
    def __init__(self, path=NEW, *, bridge=True, reverse=False):
        self.methods = methods(path)
        self._prefix_p4_bridge = Bridge(reverse) if bridge else None
        self._active, self._ready_fds_load = [], deque()
        self._inflight = self._pending_copies = self._copy_ready = []
        self._preload_pending = self._preload_slots = self._ready_fds_preload = []
        self._shared_cached = {b'preserved-native-cache': object()}
        self._worker = SimpleNamespace(ident=11)
        self._native_drain_unknown = False
        self.collects = 0
        self.reject = False

    def call(self, name, *args, **kwargs):
        return self.methods[name](self, *args, **kwargs)

    def _prefix_p4_collect(self):
        self.collects += 1
        if self.reject:
            return None
        works = [SimpleNamespace(work_id=(j.accepted_parent_sequence, j.next_file_index, 'd2h'),
                    progress='mandatory' if j.mandatory else None)
                 for j in self._active if j.is_store and j.failed is None and j.next_file_index < j.total_files]
        works += [SimpleNamespace(work_id=(r.job.accepted_parent_sequence, r.file_index, 'ssd_read'),
                    progress='mandatory' if r.job.mandatory else None)
                  for r in self._ready_fds_load if r.job is not None and r.job.failed is None]
        return SimpleNamespace(works=tuple(works))


def job(pid, **changes):
    values = dict(accepted_parent_sequence=pid, is_store=True, failed=None,
                  next_file_index=0, total_files=1, mandatory=False,
                  done_files=1, inflight_files=0, accepted_parent_retired=False)
    values.update(changes)
    return SimpleNamespace(**values)


def three_entries(owner):
    a = owner.call('_prefix_p4_order', owner._ready_fds_load, 'reads')
    b = owner.call('_prefix_p4_read_batch_ids')
    c = owner.call('_prefix_p4_order', owner._active, 'stores')
    return a, b, c


class EmptyWindowTests(unittest.TestCase):
    def test_only_documented_advice_function_bodies_changed(self):
        old, old_methods = tree(OLD)
        new, new_methods = tree(NEW)
        changed = {name for name in old_methods if ast.dump(old_methods[name]) != ast.dump(new_methods[name])}
        self.assertEqual(changed, CHANGED | {'_prefix_stage_decide', '_run'})
        # test_retained_idle_wait separately proves the exact common idle edit.
        allowed = CHANGED | {'_prefix_stage_decide', '_prefix_single_file_retry_key',
                             '_run', '_has_poll_work'}
        for module in (old, new):
            owner = next(x for x in module.body if isinstance(x, ast.ClassDef) and x.name == 'IoReactor')
            owner.body = [x for x in owner.body if not isinstance(x, ast.FunctionDef) or x.name not in allowed]
        self.assertEqual(ast.dump(old), ast.dump(new))

    def test_retained_staging_still_keeps_original_work_condition(self):
        owner = Owner()
        self.assertTrue(owner.call('_has_work'))
        for _ in range(1000):
            reads, batch, stores = three_entries(owner)
            self.assertIs(reads, owner._ready_fds_load)
            self.assertIs(stores, owner._active)
            self.assertIsNone(batch)
        self.assertEqual(owner.collects, 0)
        self.assertEqual(len(owner._shared_cached), 1)

    def test_original_same_empty_fixture_collects_three_times(self):
        owner = Owner(OLD)
        for _ in range(100):
            three_entries(owner)
        self.assertEqual(owner.collects, 300)
        self.assertEqual(owner._prefix_p4_bridge.calls, [])

    def test_nonempty_but_ineligible_returns_without_collect(self):
        owner = Owner()
        owner._active = [job(1, is_store=False), job(2, failed='failed'), job(3, next_file_index=1)]
        owner._ready_fds_load.extend([SimpleNamespace(job=None), SimpleNamespace(job=job(4, failed='failed'))])
        three_entries(owner)
        self.assertEqual(owner.collects, 0)
        self.assertEqual(owner._prefix_p4_bridge.failures, [])

    def test_populated_native_order_and_mandatory_candidate_match_v1(self):
        for reverse in (False, True):
            observed = []
            for path in (OLD, NEW):
                owner = Owner(path, reverse=reverse)
                owner._active = [job(1), job(2, mandatory=True), job(3, is_store=False)]
                owner._ready_fds_load.extend([SimpleNamespace(job=job(4, mandatory=True), file_index=0),
                                              SimpleNamespace(job=job(5), file_index=0)])
                reads, batch, stores = three_entries(owner)
                observed.append((tuple(r.job.accepted_parent_sequence for r in reads), batch,
                    tuple(j.accepted_parent_sequence for j in stores), owner.collects,
                    owner._prefix_p4_bridge.calls, owner._prefix_p4_bridge.failures))
                self.assertEqual(owner.collects, 3)
                self.assertIn(2, observed[-1][2])
                self.assertIn(4, observed[-1][0])
            self.assertEqual(observed[0], observed[1])

    def test_oversized_views_retain_original_rejection_not_scan(self):
        owner = Owner(); owner.reject = True
        # Attribute reads would fail if the bounded precheck inspected these.
        owner._active = [object() for _ in range(33)]
        owner._ready_fds_load.extend(object() for _ in range(65))
        three_entries(owner)
        self.assertEqual(owner.collects, 3)
        self.assertEqual(owner._prefix_p4_bridge.failures, [])

    def test_off_precedes_len_and_iteration(self):
        class Unreadable:
            def __len__(self): raise AssertionError('off inspected candidates')
            def __iter__(self): raise AssertionError('off iterated candidates')
        owner = Owner(bridge=False)
        owner._active = owner._ready_fds_load = Unreadable()
        three_entries(owner)
        self.assertEqual(owner.collects, 0)

    def test_parent_completion_hook_remains_independent_of_collection(self):
        owner = Owner()
        for successful in (True, False):
            terminal = job(9, next_file_index=1)
            owner.call('_prefix_p4_parent_terminal', terminal, successful=successful)
        self.assertEqual(owner.collects, 0)
        self.assertEqual(owner._prefix_p4_bridge.calls,
            [('complete', 9, dict(now_ns=1000, successful=True, drain_known=True)),
             ('complete', 9, dict(now_ns=1000, successful=False, drain_known=True))])


if __name__ == '__main__':
    unittest.main()
