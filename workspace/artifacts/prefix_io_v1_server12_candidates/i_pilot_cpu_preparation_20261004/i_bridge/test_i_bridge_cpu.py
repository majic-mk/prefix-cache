"""Real frozen I-only functions with explicit CPU lifecycle fixtures.

This suite never imports the reactor module, Torch, a model, a native SDK or a
real backend. Reactor functions are compiled from original, unedited AST nodes.
Full pure-CPU policy/bridge and collector modules execute unchanged. Memory
receipt, resource pool, live scalar/event data and submit boundary are TEST ONLY;
they cannot qualify source binding, CUDA cost coverage, real I/O or performance.
The fixed historical failed cost cell is preserved, not recalibrated.
"""
from __future__ import annotations
import argparse
import ast
from collections import deque
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import queue
import sys
import threading
import time
from types import ModuleType, SimpleNamespace as NS
from typing import Any
import unittest
from unittest.mock import Mock
import weakref

HERE = Path(__file__).resolve().parent
FROZEN = HERE / 'frozen'
SOURCE_MANIFEST_SHA256 = '4a2e4a2ffa7a16eefbbf7eda462b0856403a037adc7d7664c7a0d7b17f99efc1'
FORBIDDEN = ('torch', 'vllm', 'py_kvcache', 'cupy', 'pynvml', 'paramiko', 'ctypes')
FROZEN_COST = 16_238_752
FROZEN_BUDGET = 13_171_328
IO = 917_504
HASH = b'x' * 32
CLOCK = [1000]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check_sources():
    manifest_bytes = (HERE / 'FROZEN_CPU_SOURCE_MANIFEST.json').read_bytes()
    if digest(manifest_bytes) != SOURCE_MANIFEST_SHA256:
        raise ValueError('source manifest does not match frozen GPU-delivery provenance')
    manifest = json.loads(manifest_bytes.decode('utf-8'))
    if manifest['status'] != 'PASS_MATCHES_ACTUAL_GPU_DELIVERY_PAYLOAD':
        raise ValueError('source bundle provenance did not pass')
    rows = []
    for row in manifest['files']:
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts or not row['path'].startswith('frozen/'):
            raise ValueError('invalid bounded source path')
        path = HERE / rel
        if path.is_symlink() or not path.is_file():
            raise ValueError('source must be an actual regular file')
        data = path.read_bytes()
        if len(data) != row['bytes'] or digest(data) != row['sha256']:
            raise ValueError('frozen source mismatch: ' + row['path'])
        rows.append(dict(path=row['path'], bytes=len(data), sha256=digest(data)))
    return rows


SOURCE_BEFORE = check_sources()


class RejectGPUImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise ImportError('CPU test forbids import: ' + fullname)


sys.meta_path.insert(0, RejectGPUImports())
package = ModuleType('prefix_io_control')
package.__path__ = [str(FROZEN / 'control')]
sys.modules['prefix_io_control'] = package
from prefix_io_control.dispatch_budget import STAGES
from prefix_io_control.p4_bridge import NativeP4Bridge
from prefix_io_control.p4_policy import P4Policy, make_p4_policy
from prefix_io_control.p4_types import P4Config
from prefix_io_control.p4_single_file_receipt import ExactSingleFileReceipt, FileRef
spec = importlib.util.spec_from_file_location('_i_cpu_frozen_collector', FROZEN / 'native_full_step_collector.py')
collector = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = collector
spec.loader.exec_module(collector)

TREE = ast.parse((FROZEN / 'reactor.py').read_text(encoding='utf-8-sig'))
OWNER_NODE = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'IoReactor')
FUNCTION_NAMES = {
    '_prefix_stage_decide', '_prefix_ready_read_decision', '_prefix_single_file_retry_key',
    '_drain_ready_preload_fds', '_reserve_preload_slot', '_reserve_foreground_slot',
    '_prefix_p4_note_reservation', '_prefix_p4_collect', '_prefix_p4_load_values',
    '_prefix_p4_existing_io_state', '_prefix_capacity_components', '_prefix_clean_reclaimable_bytes',
    '_prefix_invalidate_capacity', '_has_real_load_pressure', '_can_issue_speculative_preload',
    'install_p4_single_file_wait', '_prefix_wake_single_file_locked',
    'notify_p4_single_file_wait', '_prefix_single_file_wait_deadline',
    '_prefix_wait_single_file_retry', 'publish_p4_scheduled_load', '_intake',
    '_pump_once', '_run', '_has_work', '_has_poll_work', '_cleanup_retained_cache_after_stop',
    '_drain_incoming',
}
CLASS_NAMES = {'_ReadyFd', '_PreloadInfo', '_StageDispatch', '_P4Deferred', '_SingleFileWake',
    '_P4Inspect', '_P4Publish', '_AdmissionDrain', '_OwnerSnapshot', '_MandatoryWait', '_PreloadRequest'}
NODES = [n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name in CLASS_NAMES]
NODES += [n for n in OWNER_NODE.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTION_NAMES]
if {n.name for n in NODES if isinstance(n, ast.FunctionDef)} != FUNCTION_NAMES:
    raise ValueError('missing expected original reactor function')
G = dict(dataclass=dataclass, threading=threading, queue=queue, weakref=weakref,
    Any=Any, time=NS(monotonic_ns=lambda: CLOCK[0]), logger=NS(exception=Mock()),
    add_event=Mock(side_effect=AssertionError('unexpected profiler side effect')),
    now_ns=lambda: CLOCK[0])
unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + NODES,
                  type_ignores=[])
exec(compile(ast.fix_missing_locations(unit), str(FROZEN / 'reactor.py'), 'exec'), G)
FUNCTION_PROVENANCE = [dict(name=n.name, line=n.lineno,
    ast_sha256=digest(ast.dump(n, include_attributes=False).encode('utf-8')))
    for n in NODES if isinstance(n, ast.FunctionDef)]


def cpu_receipt(*, cost=FROZEN_COST, budget=FROZEN_BUDGET):
    """TEST ONLY memory fixture; never serialized or installed into a real owner.

    Object construction is deliberately conspicuous: factory/source/event
    validation cannot run from CPU fixture data. The test invokes existing
    typed policy branches, rather than weakening the production factory.
    """
    result = object.__new__(ExactSingleFileReceipt)
    values = dict(signature=('CPU-FIXTURE-MODEL', 'CPU-FIXTURE-GPU', 'layout', 'kernel', 1, 1, 0, 144, IO),
        cost_upper_ns=cost, step_budget_ns=budget, calibration_source_lock_sha256='0'*64,
        common_source_refs=(), runtime_common_refs=(), runtime_overlay_refs=(),
        cuda_event_source_sha256='1'*64, calibration_native_source_sha256='2'*64,
        binding_ref=FileRef('CPU_ONLY_NOT_A_NATIVE_RECEIPT.json', 1, '3'*64))
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


class ResourcePoolFixture:
    """Memory accounting only: not the original pool and not a release witness."""
    def __init__(self):
        self.slot_count = self.free_count = 16
        self._closed = False
        self.reserves = self.releases = 0
    def try_reserve(self):
        self.free_count -= 1
        self.reserves += 1
        return NS(index=3)
    def release(self, index):
        if index != 3:
            raise AssertionError('unexpected fixture slot')
        self.free_count += 1
        self.releases += 1


class RecordingQueue(queue.Queue):
    def __init__(self):
        super().__init__()
        self.before_get = None
        self.timeouts = []
    def get(self, block=True, timeout=None):
        if block:
            self.timeouts.append(timeout)
            if self.before_get:
                callback, self.before_get = self.before_get, None
                callback()
        return super().get(block=block, timeout=timeout)


class OwnerFixture:
    _STOP = object()
    def __init__(self, *, mode='on', cost=FROZEN_COST, budget=FROZEN_BUDGET, max_wait=500_000):
        CLOCK[0] = 1000
        G['time'].monotonic_ns = lambda: CLOCK[0]
        receipt = cpu_receipt(cost=cost, budget=budget)
        self.bridge = NativeP4Bridge(P4Policy('cpu-fixture',
            P4Config('interference', 1_000_000, max_wait, budget), single_file=receipt))
        self.bridge.bind()
        self.bridge.single_file_shadow = mode == 'shadow'
        self.bridge._single_file_runtime = NS(signature_prefix=receipt.signature[:4])
        self.live = (5, 10, 20, 1, 1, 0, 144)
        self.bridge._single_file_state = lambda: self.live
        self._prefix_p4_bridge = None if mode == 'off' else self.bridge
        self._prefix_dispatch_controller = None
        self._prefix_stage_accounting = NS(valid=True, journal_valid=True, event_sequence=0, _owner=lambda: None,
            stats={s: {'inflight_ops': 0, 'inflight_bytes': 0} for s in STAGES})
        self._active = []
        self._inflight = {}
        self._pending_copies = []
        self._copy_ready = []
        self._ready_fds_load = deque()
        self._preload_pending = deque()
        self.ready = G['_ReadyFd'](fd=7, job=None, file_index=-1, preload_hash=HASH,
            open_start_ns=100, preload_info=G['_PreloadInfo'](HASH, 'preload', 'request', 'CPU_FIXTURE', 0, 1), sequence=1)
        self._ready_fds_preload = deque([self.ready])
        self._ready_sequence = 1
        self._preload_waiters = {}
        self._open_inflight = self._data_inflight = 0
        self.iodepth = 4
        self._preload_inflight_total = 1
        self._preload_inflight_hashes = {HASH: 1}
        self._shared_cached = {}
        self._preload_slots = {}
        self._preload_cached_total = 0
        self._prefix_shared_cached_pinned_slots = self._prefix_shared_cached_observed_count = 0
        self._prefix_capacity_valid = True
        self._prefix_capacity_error = None
        self._staging_cache = None
        self.staging_pool = ResourcePoolFixture()
        self.file_store = NS(io_size=IO)
        self.layout = NS(storage_block_bytes=IO, bytes_per_kernel_block=(IO,))
        self._stop = self._native_drain_unknown = False
        self._native_fatal_reason = None
        self._worker = NS(ident=threading.get_ident())
        self._submit_lock = threading.Lock()
        self._prefix_p4_load_frame = None
        self._prefix_p4_load_sequence = 0
        self._incoming = RecordingQueue()
        self._preload_refcount = {}
        self._preload_pending_count = {}
        self._preload_pending_cancel = {}
        self._preload_owned = {}
        self._preload_blocked_on_write = {}
        self._known_missing = {}
        self._prefix_start_budget = None
        self._observation_sink = None
        self.submitted = []
        self.parents = 0
        self.native_calls = []
        self.failed = None
    def accepted_parent_count(self):
        return self.parents
    def is_mandatory(self, future):
        return future == 'mandatory'
    def _has_preload_waiter(self, key):
        return bool(self._preload_waiters.get(key))
    def _evict_one_cache_slot(self, reason):
        raise AssertionError('unexpected cache eviction')
    def _evict_one_preload_slot(self):
        raise AssertionError('unexpected preload eviction')
    def _submit_read_from_ready(self, ready, index, **kwargs):
        # Boundary fixture, not native backend acceptance/completion evidence.
        self.submitted.append((ready.sequence, index, kwargs.get('stage_dispatch')))
    def _file_terminal(self, job, **kwargs):
        raise AssertionError('unexpected terminal fixture')
    def _fail_everything(self, exc):
        self.failed = exc


for name in FUNCTION_NAMES:
    setattr(OwnerFixture, name, G[name])


class SyntheticRawEvent:
    def __init__(self):
        self.records = 0
        self.queries = 0
    def record(self):
        self.records += 1
        return 'CPU_SYNTHETIC_EVENT'
    def query(self):
        self.queries += 1
        return True


def attach_capture(owner):
    """Invoke real collector paths over declared SYNTHETIC event metadata.

    The origin string selects its frozen accessor branch; no install, source
    qualification, real Event or export of native evidence occurs in this test.
    """
    capture = collector.FullStepCapture(run_id='cpu-fixture', origin='native_gpu_recording',
        selected_offsets=(16,), action=None)
    capture.clock = lambda: CLOCK[0]
    start = collector.EventProxy(SyntheticRawEvent(), capture.clock)
    end = collector.EventProxy(SyntheticRawEvent(), capture.clock)
    start.record_before_ns, start.record_after_ns, start.completed_query_ns = 10, 15, 20
    frame = NS(step_kind='decode', batch=1, active_decode=1, prefill_tokens=0, context_length=144)
    capture.observer = NS(enabled=True, events=NS(active=(5, start, end)),
        scalar=NS(enabled=True, _adapter=NS(_pending=dict(phase='prepared', ordinal=5, frame=frame))),
        frames=[None]*16, detach=lambda: None)
    owner.bridge._single_file_state = capture.current_single_file_step
    capture.attach_single_file_wait(owner)
    capture.after_prepare()
    return capture, start, end


class InterferenceCPUIntegrationTests(unittest.TestCase):
    def prime(self, **kwargs):
        owner = OwnerFixture(**kwargs)
        capture, start, end = attach_capture(owner)
        self.assertFalse(owner._drain_ready_preload_fds())
        self.assertEqual(owner.bridge.last_preview.action, 'defer')
        self.assertIsNotNone(owner._prefix_single_file_retry)
        self.assertEqual(owner.staging_pool.free_count, 16)
        return owner, capture, start, end

    def test_off_no_optional_clock_collection_or_metadata_mutation(self):
        owner = OwnerFixture(mode='off')
        owner.ready.sequence = 0
        owner._prefix_p4_collect = Mock(side_effect=AssertionError('off collection'))
        G['time'].monotonic_ns = Mock(side_effect=AssertionError('off clock'))
        self.assertTrue(owner._drain_ready_preload_fds())
        self.assertEqual(owner.ready.sequence, 0)
        self.assertEqual(len(owner.submitted), 1)
        self.assertEqual(owner.submitted[0][2], None)
        self.assertFalse(owner._prefix_wait_single_file_retry())
        self.assertIsNone(make_p4_policy('cpu-fixture', P4Config('off', 100, 100)))

    def test_shadow_prediction_never_blocks_original_queue(self):
        owner = OwnerFixture(mode='shadow')
        self.assertTrue(owner._drain_ready_preload_fds())
        self.assertEqual(len(owner.submitted), 1)
        self.assertEqual(owner.bridge.single_file_deferrals, 1)
        self.assertEqual(owner.bridge.single_file_blocked_attempts, 0)
        self.assertIsNone(getattr(owner, '_prefix_single_file_retry', None))
        self.assertFalse(owner._prefix_wait_single_file_retry())

    def test_frozen_failed_cost_cell_blocks_and_releases_only_temporary_slot(self):
        owner, capture, start, end = self.prime()
        self.assertEqual(owner.bridge.policy.single_file.cost_upper_ns, FROZEN_COST)
        self.assertEqual(owner.bridge.policy.single_file.step_budget_ns, FROZEN_BUDGET)
        self.assertIsNone(owner._prefix_dispatch_controller)
        self.assertIs(owner._ready_fds_preload[0], owner.ready)
        self.assertEqual((owner.ready.fd, owner.ready.open_start_ns), (7, 100))
        self.assertEqual(owner._preload_inflight_total, 1)
        self.assertEqual((owner.staging_pool.reserves, owner.staging_pool.releases), (1, 1))
        self.assertEqual(owner.submitted, [])

    def test_retry_retains_original_snapshot_age_and_single_work_identity(self):
        owner, capture, start, end = self.prime()
        cached = owner._prefix_single_file_retry
        for _ in range(128):
            CLOCK[0] += 100
            self.assertFalse(owner._drain_ready_preload_fds())
            self.assertIs(owner._prefix_single_file_retry[1], cached[1])
            self.assertIs(owner._prefix_single_file_retry[2], cached[2])
        self.assertEqual(owner.bridge.single_file_observation_reuses, 128)
        self.assertEqual(owner.staging_pool.reserves, owner.staging_pool.releases)
        self.assertEqual(owner.staging_pool.free_count, 16)
        self.assertEqual(owner.ready.open_start_ns, 100)
        self.assertEqual(owner._preload_inflight_total, 1)

    def test_legitimate_test_only_cost_issue_preserves_original_single_submission(self):
        # Deliberately separate CPU fixture; not a replacement historical receipt.
        owner = OwnerFixture(cost=90, budget=100)
        self.assertTrue(owner._drain_ready_preload_fds())
        self.assertEqual(owner.bridge.last_preview.action, 'issue')
        self.assertEqual(len(owner.submitted), 1)
        self.assertEqual(owner.submitted[0][2], None)
        self.assertEqual(owner.staging_pool.releases, 0)
        self.assertEqual(owner.bridge.single_file_blocked_attempts, 0)

    def test_mandatory_support_after_defer_bypasses_only_performance_gate(self):
        owner, capture, start, end = self.prime()
        owner._preload_waiters[HASH] = [(NS(future='mandatory'), 0)]
        self.assertTrue(owner._drain_ready_preload_fds())
        self.assertEqual(owner.bridge.last_preview.reason, 'native_progress_override')
        self.assertEqual(len(owner.submitted), 1)
        self.assertEqual(owner.ready.open_start_ns, 100)
        self.assertIsNone(owner._prefix_single_file_retry)

    def test_mandatory_cannot_bypass_original_device_or_slot_capacity(self):
        for capacity in ('device', 'slot'):
            owner, capture, start, end = self.prime()
            owner._preload_waiters[HASH] = [(NS(future='mandatory'), 0)]
            if capacity == 'device':
                owner._data_inflight = owner.iodepth
            else:
                owner.staging_pool.try_reserve = lambda: None
                # Original foreground admission may attempt to reclaim clean
                # cache entries. Both caches are empty in this CPU fixture.
                owner._evict_one_cache_slot = lambda reason: False
                owner._evict_one_preload_slot = lambda: False
            self.assertFalse(owner._drain_ready_preload_fds())
            self.assertEqual(owner.submitted, [])
            self.assertIs(owner._ready_fds_preload[0], owner.ready)

    def test_original_arrival_deadline_not_reset_by_deferred_retries(self):
        owner, capture, start, end = self.prime()
        deadline = owner._prefix_single_file_wait_deadline(owner._prefix_single_file_retry)
        self.assertEqual(deadline, 500100)
        CLOCK[0] = deadline
        self.assertFalse(owner._prefix_wait_single_file_retry())
        self.assertTrue(owner._drain_ready_preload_fds())
        self.assertEqual(owner.bridge.last_preview.reason, 'native_progress_override')
        self.assertEqual(owner.ready.open_start_ns, 100)

    def test_original_stop_message_wakes_queue_then_intake_drains_without_new_epoch(self):
        owner, capture, start, end = self.prime()
        epoch = owner.bridge._epoch
        owner._incoming.before_get = lambda: owner._incoming.put(owner._STOP)
        self.assertTrue(owner._prefix_wait_single_file_retry())
        self.assertTrue(owner._stop)
        self.assertIsNone(owner._prefix_single_file_wait_armed)
        self.assertIsNone(capture._wait_registration)
        self.assertEqual(owner.bridge._epoch, epoch)
        self.assertTrue(owner._drain_ready_preload_fds())
        self.assertEqual(owner.bridge.last_preview.reason, 'native_progress_override')
        self.assertEqual(len(owner.submitted), 1)

    def test_actual_end_notification_returns_to_original_queue_without_poll_or_sync(self):
        owner, capture, start, end = self.prime()
        owner._incoming.before_get = end.record
        self.assertTrue(owner._prefix_wait_single_file_retry())
        self.assertEqual((end.raw.records, end.raw.queries), (1, 0))
        self.assertIsNone(owner._prefix_single_file_wake_pending)
        self.assertTrue(owner._drain_ready_preload_fds())
        self.assertEqual(len(owner.submitted), 1)
        # Closing the live step declines the narrow cost-preview branch before
        # incrementing its counter, while restoring the original native path.
        self.assertEqual(owner.bridge.single_file_previews, 1)

    def test_end_between_policy_preview_and_deferral_restores_native_path(self):
        owner = OwnerFixture()
        capture, start, end = attach_capture(owner)
        original = owner.bridge.preview_issue
        def end_after_preview(*args, **kwargs):
            result = original(*args, **kwargs)
            end.record()
            return result
        owner.bridge.preview_issue = end_after_preview
        self.assertTrue(owner._drain_ready_preload_fds())
        self.assertEqual(owner.bridge.single_file_blocked_attempts, 0)
        self.assertEqual(len(owner.submitted), 1)

    def test_expired_or_changed_context_returns_native_without_fabricated_completion(self):
        for kind in ('end', 'context', 'stale'):
            owner, capture, start, end = self.prime(max_wait=2_000_000)
            if kind == 'end':
                end.record()
            elif kind == 'context':
                capture.observer.scalar._adapter._pending['frame'].context_length = 145
            else:
                CLOCK[0] = 1_000_101
            self.assertFalse(owner._prefix_wait_single_file_retry())
            self.assertTrue(owner._drain_ready_preload_fds())
            self.assertEqual(len(owner.submitted), 1)
            self.assertIsNone(owner._prefix_single_file_retry)

    def test_bridge_or_collector_fault_wakes_and_falls_back_without_moving_fd(self):
        for kind in ('bridge', 'collector', 'detach'):
            owner, capture, start, end = self.prime()
            callback = {'bridge': lambda: owner.bridge.fail('CPU fixture fault'),
                'collector': lambda: capture.fail('CPU fixture fault'), 'detach': capture.detach}[kind]
            owner._incoming.before_get = callback
            self.assertTrue(owner._prefix_wait_single_file_retry())
            self.assertFalse(owner._prefix_wait_single_file_retry())
            self.assertTrue(owner._drain_ready_preload_fds())
            self.assertEqual(len(owner.submitted), 1)
            self.assertEqual(owner.ready.fd, 7)

    def test_new_native_work_or_unknown_accounting_never_parks(self):
        cases = [('_pending_copies', [object()]), ('_inflight', {1: object()}),
            ('_open_inflight', 1), ('_data_inflight', 1), ('parents', 1), ('_native_drain_unknown', True)]
        for name, value in cases:
            owner, capture, start, end = self.prime()
            setattr(owner, name, value)
            self.assertFalse(owner._prefix_wait_single_file_retry(), name)
            self.assertEqual(owner._incoming.timeouts, [])
        owner, capture, start, end = self.prime()
        owner._prefix_stage_accounting.valid = False
        self.assertFalse(owner._prefix_wait_single_file_retry())

    def test_intake_failure_not_swallowed_by_optional_wait(self):
        owner, capture, start, end = self.prime()
        owner._intake = Mock(side_effect=RuntimeError('native fixture intake failure'))
        owner._incoming.put(object())
        with self.assertRaisesRegex(RuntimeError, 'native fixture intake failure'):
            owner._prefix_wait_single_file_retry()
        self.assertIsNone(owner._prefix_single_file_wait_armed)

    def test_deferred_metadata_is_bounded_immutable_without_resource_owners(self):
        owner, capture, start, end = self.prime()
        def visit(value):
            if value is None or type(value) in (str, int, bool, bytes):
                return
            if type(value) in (tuple, frozenset):
                self.assertLessEqual(len(value), 64)
                for item in value:
                    visit(item)
                return
            self.assertTrue(is_dataclass(value))
            self.assertTrue(value.__dataclass_params__.frozen)
            for field in fields(value):
                visit(getattr(value, field.name))
        visit(owner._prefix_single_file_retry)
        self.assertIsInstance(owner._prefix_single_file_wait_capture, weakref.ReferenceType)

    def test_original_pump_preserves_native_stage_order(self):
        owner = OwnerFixture(mode='off')
        owner._ready_fds_preload.clear()
        expected = ['cuda', 'cqe', 'schedule', 'fusion', 'submit', 'finish']
        for name, label in [('_drain_cuda_copies', 'cuda'), ('_poll_ring_completions', 'cqe'),
            ('_schedule_work', 'schedule'), ('_flush_copy_batch', 'fusion'), ('_finish_jobs', 'finish')]:
            setattr(owner, name, lambda label=label: owner.native_calls.append(label) or False)
        owner.ring = NS(submit_pending=lambda: owner.native_calls.append('submit'))
        self.assertFalse(owner._pump_once())
        self.assertEqual(owner.native_calls, expected)

    def test_original_run_stop_wait_intake_pump_and_cleanup_path_is_finite(self):
        owner, capture, start, end = self.prime()
        owner._incoming.before_get = lambda: owner._incoming.put(owner._STOP)
        for name, label in [('_drain_cuda_copies', 'cuda'), ('_poll_ring_completions', 'cqe'),
            ('_flush_copy_batch', 'fusion'), ('_finish_jobs', 'finish')]:
            setattr(owner, name, lambda label=label: owner.native_calls.append(label) or False)
        def native_schedule_fixture():
            owner.native_calls.append('schedule')
            # The original ready-queue method, not a replacement scheduler or
            # backend, executes its shutdown-progress decision here.
            return owner._drain_ready_preload_fds()
        owner._schedule_work = native_schedule_fixture
        owner.ring = NS(submit_pending=lambda: owner.native_calls.append('submit'))
        owner._run()
        self.assertIsNone(owner.failed)
        self.assertTrue(owner._stop)
        self.assertFalse(owner._has_work())
        self.assertEqual(len(owner.submitted), 1)
        self.assertEqual(owner.bridge.last_preview.reason, 'native_progress_override')
        self.assertIsNone(owner._prefix_single_file_wait_armed)
        self.assertIsNone(capture._wait_registration)
        self.assertEqual(owner.native_calls, ['cuda', 'cqe', 'schedule', 'fusion', 'submit', 'finish'])
        self.assertEqual(len(owner._incoming.timeouts), 1)

    def test_test_loader_rejects_forbidden_import_without_loading_gpu_module(self):
        for name in FORBIDDEN:
            with self.assertRaisesRegex(ImportError, 'CPU test forbids import'):
                importlib.import_module(name)

    def test_source_bundle_exact_and_forbidden_modules_absent(self):
        self.assertEqual(check_sources(), SOURCE_BEFORE)
        self.assertFalse([n for n in sys.modules if n.split('.')[0] in FORBIDDEN])
        # Source-extracted methods were never rewritten or replaced.
        for name in FUNCTION_NAMES:
            self.assertIs(getattr(OwnerFixture, name), G[name])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stream = io.StringIO()
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(InterferenceCPUIntegrationTests)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    after = check_sources()
    forbidden = [n for n in sys.modules if n.split('.')[0] in FORBIDDEN]
    record = dict(schema='i-only-real-function-cpu-integration-v1',
        status='PASS_CPU_I_ONLY_INTEGRATION' if result.wasSuccessful() and after == SOURCE_BEFORE and not forbidden else 'FAIL',
        cpu_fixture_only=True, gpu_operations=0, native_backend_submissions=0,
        production_source_changes=0, cost_gate_changed=False,
        frozen_cost_upper_ns=FROZEN_COST, frozen_A_only_budget_ns=FROZEN_BUDGET,
        ordinary_frozen_candidate_admissible=FROZEN_COST <= FROZEN_BUDGET,
        cost_coverage_gpu_qualified=False, live_gpu_on_qualified=False, performance_effect_verified=False,
        source_before=SOURCE_BEFORE, source_after=after,
        original_function_provenance=FUNCTION_PROVENANCE,
        tests=dict(run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped)),
        elapsed_seconds=time.monotonic()-started, forbidden_imports=forbidden, test_log=stream.getvalue(),
        missing_gpu_evidence=['real on physical I/O and CUDA fence completion',
            'real original shutdown and OS session drain', 'prospective exact cost coverage and independent development budget',
            'same-executor off/on causal request-stream effect comparison'],
        limitations=['Receipt/live-state/event/resource/submit values are declared CPU fixtures.',
            'A release in this pool fixture is not physical native resource reuse.',
            'Function extraction preserves original AST but does not initialize the real reactor/backend.',
            'No GPU, runtime source qualification, normal-load problem witness or method benefit is established.'])
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(record, out, ensure_ascii=False, indent=2)
        out.write('\n')
    sys.stdout.write(stream.getvalue())
    print(json.dumps({k: record[k] for k in ('status', 'tests', 'gpu_operations', 'ordinary_frozen_candidate_admissible')}))
    raise SystemExit(0 if record['status'] == 'PASS_CPU_I_ONLY_INTEGRATION' else 1)


if __name__ == '__main__':
    main()
