"""CPU-only notification semantics. All raw events below are SYNTHETIC.

The real collector scalar accessor and source-extracted native reserve/preview/
drain/wait code execute. This does not qualify CUDA events or native I/O.
Full _run/_pump_once overhead is measured separately by the root harness.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace as NS
from typing import Any
import unittest
import weakref

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_wait_retry_support', HERE/'test_single_file_retry_observation.py')
retry = importlib.util.module_from_spec(spec); spec.loader.exec_module(retry)
collector = retry.support.collector
parsed = ast.parse(retry.SOURCE.read_text(encoding='utf-8-sig'))
native = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'IoReactor')
WAIT_METHODS = {'install_p4_single_file_wait', '_prefix_wake_single_file_locked',
    'notify_p4_single_file_wait', '_prefix_single_file_wait_deadline',
    '_prefix_wait_single_file_retry', 'publish_p4_scheduled_load', '_intake'}
nodes = [n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == '_SingleFileWake']
nodes += [n for n in native.body if isinstance(n, ast.FunctionDef) and n.name in WAIT_METHODS]
G = dict(retry.NSOURCE, Any=Any, weakref=weakref, queue=queue)
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(retry.SOURCE), 'exec'), G)
Wake = G['_SingleFileWake']


class SyntheticRawEvent:
    """No GPU API; an explicitly synthetic already-ready event."""
    def __init__(self): self.records = self.queries = 0
    def record(self): self.records += 1; return 'synthetic-record'
    def query(self): self.queries += 1; return True


class RecordingQueue(queue.Queue):
    def __init__(self):
        super().__init__(); self.before_get = None; self.timeouts = []
        self.entered = threading.Event()
    def get(self, block=True, timeout=None):
        if block:
            self.timeouts.append(timeout)
            if self.before_get is not None:
                callback, self.before_get = self.before_get, None
                callback()
            self.entered.set()
        return super().get(block=block, timeout=timeout)


class WaitOwner(retry.Owner):
    def __init__(self, *, clock=None, max_wait=500000, sample_age=1000000,
                 off=False, shadow=False):
        super().__init__(off=off, shadow=shadow, max_wait=max_wait)
        self.clock = clock or (lambda: retry.CLOCK[0])
        retry.NSOURCE['time'].monotonic_ns = self.clock
        self.bridge.policy.config = retry.support.P4Config('interference', sample_age, max_wait, 100)
        self._incoming = RecordingQueue(); self._closed = False; self.seen = []
        if clock is not None:
            self.ready.open_start_ns = self.clock()
    def _intake(self, item):
        if type(item) is Wake:
            return G['_intake'](self, item)
        self.seen.append(item)  # synthetic metadata; full native intake tested separately

for _name in WAIT_METHODS - {'_intake'}:
    setattr(WaitOwner, _name, G[_name])


def make_capture(owner, *, attach=True):
    """Borrow actual collector paths, with manually built SYNTHETIC event metadata.

    native_gpu_recording selects the existing scalar accessor branch only. No
    install/source verification/event-class qualification is invoked or claimed.
    """
    capture = collector.FullStepCapture(run_id='r', origin='native_gpu_recording',
        selected_offsets=(1,), action=None)
    start = collector.EventProxy(SyntheticRawEvent(), owner.clock)
    end = collector.EventProxy(SyntheticRawEvent(), owner.clock)
    now = owner.clock(); start.record_before_ns = max(1, now - 30)
    start.record_after_ns = max(1, now - 20); start.completed_query_ns = max(1, now - 10)
    frame = NS(step_kind='decode', batch=1, active_decode=1, prefill_tokens=0, context_length=144)
    pending = {'phase':'prepared', 'ordinal':5, 'frame':frame}
    capture.observer = NS(enabled=True, scalar=NS(enabled=True, _adapter=NS(_pending=pending)),
                          events=NS(active=(5, start, end)), frames=[], detach=lambda:None)
    owner.bridge._single_file_state = capture.current_single_file_step
    if attach:
        capture.attach_single_file_wait(owner)
    capture.after_prepare()
    return capture, start, end


class SingleFileWaitTests(unittest.TestCase):
    def prime(self, **kwargs):
        owner = WaitOwner(**kwargs); capture, start, end = make_capture(owner)
        self.assertFalse(owner._drain_ready_preload_fds())
        self.assertIsNotNone(owner._prefix_single_file_retry)
        return owner, capture, start, end

    def test_end_before_arm_cannot_park(self):
        owner, capture, start, end = self.prime(); end.record()
        self.assertFalse(owner._prefix_wait_single_file_retry())
        self.assertEqual(owner._incoming.timeouts, [])
        owner._drain_ready_preload_fds(); self.assertEqual(owner.submitted, [(1, 3)])

    def test_end_between_arm_and_final_check_cannot_park(self):
        owner, capture, start, end = self.prime()
        original = capture.arm_single_file_wait
        def arm(*args):
            result = original(*args); end.record(); return result
        capture.arm_single_file_wait = arm
        self.assertFalse(owner._prefix_wait_single_file_retry())
        self.assertEqual(owner._incoming.timeouts, [])
        self.assertEqual(owner._incoming.qsize(), 1)

    def test_end_after_check_before_get_is_retained_by_original_queue(self):
        owner, capture, start, end = self.prime()
        previews = []
        original_preview = owner.bridge.preview_issue
        def preview(*args, **kwargs):
            previews.append(True); return original_preview(*args, **kwargs)
        owner.bridge.preview_issue = preview
        owner._incoming.before_get = end.record
        self.assertTrue(owner._prefix_wait_single_file_retry())
        self.assertIsNone(owner._prefix_single_file_wait_armed)
        self.assertIsNone(capture._wait_registration)
        self.assertEqual((end.raw.records, end.raw.queries), (1, 0))
        owner._drain_ready_preload_fds()
        self.assertEqual(owner.submitted, [(1, 3)])
        self.assertEqual(previews, [True])

    def test_actual_queue_independent_producer_wakes_after_get(self):
        owner, capture, start, end = self.prime(clock=time.monotonic_ns,
            max_wait=100000000, sample_age=100000000)
        def producer():
            self.assertTrue(owner._incoming.entered.wait(1)); end.record()
        worker = threading.Thread(target=producer); worker.start()
        self.assertTrue(owner._prefix_wait_single_file_retry()); worker.join(1)
        self.assertFalse(worker.is_alive()); self.assertEqual(end.raw.records, 1)

    def test_old_duplicate_wakes_only_return_to_dispatch(self):
        owner, capture, start, end = self.prime()
        old = ('r', 'old-capture', 4, 1)
        owner._incoming.put(Wake(old)); owner._incoming.put(Wake(old))
        self.assertTrue(owner._prefix_wait_single_file_retry())
        self.assertIsNone(owner._prefix_single_file_wait_armed)
        self.assertEqual(owner.bridge.single_file_previews, 1)
        owner._drain_ready_preload_fds()
        self.assertEqual(owner.bridge.single_file_previews, 2)
        self.assertEqual(owner.submitted, [])

    def test_wake_coalescing_is_bounded_and_old_queued_token_cannot_hide_end(self):
        owner, capture, start, end = self.prime()
        old = ('r', 'old', 4, 1)
        owner._prefix_single_file_wake_pending = old; owner._incoming.put(Wake(old))
        def flood():
            for _ in range(100): owner.notify_p4_single_file_wait()
            end.record()
            self.assertEqual(owner._incoming.qsize(), 1)
        owner._incoming.before_get = flood
        self.assertTrue(owner._prefix_wait_single_file_retry())
        self.assertIsNone(owner._prefix_single_file_wake_pending)
        owner._drain_ready_preload_fds(); self.assertEqual(owner.submitted, [(1, 3)])

    def test_capture_fail_detach_and_bridge_fault_wake(self):
        for kind in ('fail', 'detach', 'bridge'):
            with self.subTest(kind=kind):
                owner, capture, start, end = self.prime()
                callback = {'fail':lambda:capture.fail('synthetic fault'),
                    'detach':capture.detach, 'bridge':lambda:owner.bridge.fail('synthetic fault')}[kind]
                owner._incoming.before_get = callback
                self.assertTrue(owner._prefix_wait_single_file_retry())
                self.assertFalse(owner._prefix_wait_single_file_retry())
                self.assertIsNone(capture._wait_registration)

    def test_bridge_fail_exactly_once_and_restored_on_detach(self):
        owner = WaitOwner(); original = owner.bridge.fail.__func__
        capture, start, end = make_capture(owner)
        owner.bridge.fail('first'); self.assertEqual(owner.bridge.fault, 'first')
        capture.detach(); self.assertIs(owner.bridge.fail.__func__, original)

    def test_detach_does_not_overwrite_later_installer(self):
        owner, capture, start, end = self.prime()
        later = lambda reason:None; owner.bridge.fail = later
        capture.detach(); self.assertIs(owner.bridge.fail, later)

    def test_original_native_message_returns_to_intake_and_pump(self):
        for item in ('synthetic STOP', 'synthetic mandatory', 'synthetic new job'):
            owner, capture, start, end = self.prime()
            owner._incoming.before_get = lambda:owner._incoming.put(item)
            self.assertTrue(owner._prefix_wait_single_file_retry())
            self.assertEqual(owner.seen, [item]); self.assertIsNone(capture._wait_registration)

    def test_intake_exception_is_not_swallowed(self):
        owner, capture, start, end = self.prime()
        def bad_intake(item): raise RuntimeError('native intake failure')
        owner._intake = bad_intake; owner._incoming.put('synthetic job')
        with self.assertRaisesRegex(RuntimeError, 'native intake failure'):
            owner._prefix_wait_single_file_retry()
        self.assertIsNone(owner._prefix_single_file_wait_armed)

    def test_only_release_delta_allowed_no_capacity_credit(self):
        owner, capture, start, end = self.prime()
        cached = owner._prefix_single_file_retry; capacity = cached[0][3]
        self.assertEqual(owner._prefix_capacity_components()[2], capacity[2] + 1)
        self.assertEqual(owner._prefix_capacity_components()[-1], capacity[-1] + 1)
        self.assertIsNotNone(owner._prefix_single_file_wait_deadline(cached))
        owner.staging_pool.free_count -= 1
        self.assertIsNone(owner._prefix_single_file_wait_deadline(cached))
        self.assertEqual((owner.staging_pool.reserves, owner.staging_pool.releases), (1, 1))

    def test_other_native_work_never_parks(self):
        for field, value in (('_pending_copies',[object()]), ('_inflight',{1:object()}),
                ('_copy_ready',[object()]), ('_preload_pending',[object()]),
                ('_ready_fds_load',[object()]), ('_open_inflight',1), ('_data_inflight',1),
                ('parents',1), ('_stop',True)):
            owner, capture, start, end = self.prime(); setattr(owner, field, value)
            self.assertFalse(owner._prefix_wait_single_file_retry(), field)
            self.assertEqual(owner._incoming.timeouts, [])

    def test_live_accounting_waiter_epoch_and_capacity_changes_refuse(self):
        for kind in ('journal', 'ops', 'bytes', 'waiter', 'epoch', 'capacity'):
            owner, capture, start, end = self.prime()
            if kind == 'journal': owner._prefix_stage_accounting.event_sequence += 1
            if kind == 'ops': owner._prefix_stage_accounting.stats['ssd_read']['inflight_ops'] = 1
            if kind == 'bytes': owner._prefix_stage_accounting.stats['ssd_read']['inflight_bytes'] = 1
            if kind == 'waiter': owner._preload_waiters[retry.HASH] = [(NS(future='mandatory'),0)]
            if kind == 'epoch': owner.bridge._epoch += 1
            if kind == 'capacity': owner.staging_pool.free_count -= 1
            self.assertFalse(owner._prefix_wait_single_file_retry(), kind)

    def test_deadline_is_original_earliest_bound_and_never_extended(self):
        owner, capture, start, end = self.prime(max_wait=2000000, sample_age=1000000)
        cached = owner._prefix_single_file_retry
        expected = min(owner.ready.open_start_ns + 2000000, cached[1].monotonic_ns + 1000000,
            cached[1].native_state.captured_ns + 1000000, cached[0][0][2] + 1000000)
        self.assertEqual(owner._prefix_single_file_wait_deadline(cached), expected)
        retry.CLOCK[0] += 100
        self.assertEqual(owner._prefix_single_file_wait_deadline(cached), expected)
        retry.CLOCK[0] = expected
        self.assertIsNone(owner._prefix_single_file_wait_deadline(cached))

    def test_expired_timeout_is_original_fallback_without_fake_completion(self):
        owner, capture, start, end = self.prime()
        class DeadlineQueue(RecordingQueue):
            def get(self, block=True, timeout=None):
                self.timeouts.append(timeout)
                retry.CLOCK[0] = owner.ready.open_start_ns + owner.bridge.policy.config.max_wait_ns
                raise queue.Empty
        owner._incoming = DeadlineQueue()
        self.assertTrue(owner._prefix_wait_single_file_retry())
        self.assertEqual(end.raw.records, 0)
        owner._drain_ready_preload_fds(); self.assertEqual(owner.submitted, [(1, 3)])
        self.assertEqual(owner.bridge.last_preview.reason, 'native_progress_override')

    def test_off_shadow_and_uninstalled_never_wait(self):
        for kwargs in ({'off':True}, {'shadow':True}):
            owner = WaitOwner(**kwargs)
            with self.assertRaises(ValueError): make_capture(owner)
            self.assertFalse(owner._prefix_wait_single_file_retry())
        owner = WaitOwner(); self.assertFalse(owner._prefix_wait_single_file_retry())

    def test_event_proxy_still_delegates_record_exactly_once(self):
        owner, capture, start, end = self.prime()
        self.assertEqual(end.record(), 'synthetic-record')
        self.assertEqual(end.raw.records, 1)

    def test_original_event_record_exception_preserved_and_wait_woken(self):
        owner, capture, start, end = self.prime()
        def bad_record():
            end.raw.records += 1
            raise RuntimeError('original synthetic record failed')
        end.raw.record = bad_record
        def record():
            with self.assertRaisesRegex(RuntimeError, 'original synthetic record failed'):
                end.record()
        owner._incoming.before_get = record
        self.assertTrue(owner._prefix_wait_single_file_retry())
        self.assertEqual(end.raw.records, 1)

    def test_notification_is_after_original_record_and_timestamp_bracket(self):
        owner, capture, start, end = self.prime()
        original = owner.notify_p4_single_file_wait
        order = []
        def notify(*args):
            order.append((end.raw.records, end.record_after_ns is not None))
            return original(*args)
        owner.notify_p4_single_file_wait = notify
        owner._incoming.before_get = end.record
        self.assertTrue(owner._prefix_wait_single_file_retry())
        self.assertEqual(order, [(1, True)])
        with self.assertRaises(ValueError): end.record()
        self.assertEqual(end.raw.records, 1)

    def test_messages_are_frozen_scalar_values(self):
        owner, capture, start, end = self.prime()
        token = ('r', capture.wait_nonce, 5, 1)
        owner._prefix_single_file_wait_armed = token
        owner.notify_p4_single_file_wait(token)
        msg = owner._incoming.get_nowait()
        self.assertTrue(msg.__dataclass_params__.frozen)
        self.assertEqual(tuple(type(v) for v in msg.token), (str,str,int,int))
        self.assertIsInstance(capture._wait_reactor_ref, weakref.ReferenceType)
        self.assertIsInstance(owner._prefix_single_file_wait_capture, weakref.ReferenceType)

    def test_scheduled_load_publication_after_final_check_wakes(self):
        from prefix_io_control.p4_load_observation import SchedulerLoadUnavailable
        owner, capture, start, end = self.prime()
        observation = SchedulerLoadUnavailable('r', 1, owner.clock(), 'synthetic unavailable')
        owner._incoming.before_get = lambda:owner.publish_p4_scheduled_load(observation)
        self.assertTrue(owner._prefix_wait_single_file_retry())
        self.assertFalse(owner._prefix_wait_single_file_retry())
        self.assertEqual(owner._prefix_p4_load_sequence, 1)
        self.assertEqual(end.raw.records, 0)

    def test_conflicting_scheduled_load_clears_and_wakes_without_lock_recursion(self):
        from prefix_io_control.p4_load_observation import SchedulerLoadUnavailable
        owner, capture, start, end = self.prime()
        first = SchedulerLoadUnavailable('r', 1, owner.clock(), 'one')
        second = SchedulerLoadUnavailable('r', 1, owner.clock(), 'two')
        token = ('r', capture.wait_nonce, 5, 1)
        # Once a prior publication exists, normal qualification prohibits park.
        # Exercise this original mutation producer directly with an armed token.
        owner._prefix_single_file_wait_armed = token
        owner._prefix_p4_load_frame, owner._prefix_p4_load_sequence = first, 1
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            owner.publish_p4_scheduled_load(second)
        self.assertIsNone(owner._prefix_p4_load_frame)
        self.assertEqual(owner._incoming.get_nowait(), Wake(token))

    def test_notification_failure_cannot_skip_original_event_record_or_repark(self):
        owner, capture, start, end = self.prime()
        token = ('r', capture.wait_nonce, 5, 1)
        owner._prefix_single_file_wait_armed = token
        self.assertTrue(capture.arm_single_file_wait(token, owner.bridge._single_file_state()))
        def bad_notify(*args): raise RuntimeError('synthetic queue failure')
        owner.notify_p4_single_file_wait = bad_notify
        self.assertEqual(end.record(), 'synthetic-record')
        self.assertEqual(end.raw.records, 1)
        self.assertTrue(capture._wait_failed)
        self.assertFalse(owner._prefix_wait_single_file_retry())
        self.assertEqual(len(capture.wait_failures), 1)

    def test_bound_observer_invalidation_wrapper_calls_original_once_and_wakes(self):
        class SyntheticInvalidator:
            def __init__(self): self.calls = 0
            def invalidate(self, reason): self.calls += 1; return reason
        owner, capture, start, end = self.prime(); target = SyntheticInvalidator()
        original = target.invalidate.__func__
        capture._attach_wait_invalidator(target, 'invalidate')
        owner._incoming.before_get = lambda:target.invalidate('synthetic observer fault')
        self.assertTrue(owner._prefix_wait_single_file_retry())
        self.assertEqual(target.calls, 1)
        capture.detach(); self.assertIs(target.invalidate.__func__, original)

    def test_source_scope_original_pump_policy_borrower_and_queue_are_preserved(self):
        previous = Path(os.environ.get('SERVER11_PREVIOUS_CANDIDATE_ROOT',
                        str(HERE.with_name('p4_single_file_candidate_v4'))))
        source = Path('source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py')
        old_tree = ast.parse((previous/source).read_text(encoding='utf-8-sig'))
        old_owner = next(n for n in old_tree.body if isinstance(n, ast.ClassDef) and n.name == 'IoReactor')
        old = {n.name:n for n in old_owner.body if isinstance(n, ast.FunctionDef)}
        new = {n.name:n for n in native.body if isinstance(n, ast.FunctionDef)}
        changed = {name for name in old if ast.dump(old[name]) != ast.dump(new[name])}
        self.assertEqual(changed, {'_run', '_intake', '_prefix_single_file_retry_key',
                                   'publish_p4_scheduled_load'})
        self.assertEqual(set(new)-set(old), WAIT_METHODS - {'_intake','publish_p4_scheduled_load'})
        self.assertEqual(ast.dump(ast.Module(body=old['_intake'].body, type_ignores=[])),
                         ast.dump(ast.Module(body=new['_intake'].body[1:], type_ignores=[])))
        for name in ('_pump_once','_drain_ready_preload_fds','_prefix_stage_decide',
                     '_drain_incoming','_schedule_work','_drain_cuda_copies',
                     '_poll_ring_completions','_flush_copy_batch','_finish_jobs',
                     'shutdown','_has_work','_has_poll_work'):
            self.assertEqual(ast.dump(old[name]), ast.dump(new[name]), name)
        policy = Path('source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_policy.py')
        self.assertEqual((previous/policy).read_bytes(), (HERE/policy).read_bytes())
        old_c = ast.parse((previous/'native_full_step_collector.py').read_text(encoding='utf-8-sig'))
        new_c = ast.parse((HERE/'native_full_step_collector.py').read_text(encoding='utf-8-sig'))
        def borrower(tree):
            cl = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FullStepCapture')
            return next(n for n in cl.body if isinstance(n, ast.FunctionDef) and n.name == 'current_single_file_step')
        self.assertEqual(ast.dump(borrower(old_c)), ast.dump(borrower(new_c)))


if __name__ == '__main__': unittest.main()
