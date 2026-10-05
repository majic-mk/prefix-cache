"""Real CPU collector API and strict negative native integration boundaries.

Every worker/connector/event scalar is explicitly synthetic; no typed native
receipt, capture relabeling or bridge weakref mutation is used in these tests.
"""
import ast
import collections
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import importlib.util
import json
import logging
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
import weakref

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
STAGE_A = Path(os.environ.get('C5_COMBINED_STAGE_A_ROOT', str(BASE / 'notification_v5_runtime_preparation')))
NATIVE = Path(os.environ.get('C5_COMBINED_NATIVE_ROOT', str(BASE / 'notification_v5_native_cost_preparation_cpu')))
CANDIDATE = NATIVE / 'common_candidate'
CONTROL = 'source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control'
REACTOR = CANDIDATE / 'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'
COLLECTOR = CANDIDATE / 'native_full_step_collector.py'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


A = load(HERE / 'notification_runtime_adapter.py', '_combined_exact_notification_adapter')
T = load(HERE / 'combined_runtime_contract.py', '_combined_runtime_contract_tests')
L = load(HERE / 'run_p4_single_file_experiment.py', '_combined_original_runtime_launcher')
J = load(HERE / 'control_p4_single_file.py', '_combined_original_runtime_controller')
V = load(HERE / 'verify_p4_single_file.py', '_combined_original_runtime_verifier')
C = load(COLLECTOR, '_combined_actual_C5_collector')


def native_classes():
    """Compile original class AST with synthetic interfaces; no engine import."""
    source = REACTOR.read_bytes()
    classes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)]
    module = ModuleType('_combined_synthetic_original_reactor')
    sys.modules[module.__name__] = module
    module.__dict__.update(dataclass=dataclass, field=field, collections=collections, Future=Future,
        queue=queue, threading=threading, time=time, weakref=weakref, Any=object,
        logger=logging.getLogger('COMBINED_CPU_ONLY'), now_ns=time.monotonic_ns,
        add_event=lambda *args, **kwargs: None, emit_transfer_events=lambda *args, **kwargs: None)
    unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + classes,
        type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(REACTOR), 'exec'), module.__dict__)
    return module.__dict__


G = native_classes()
PACKAGE = str(CANDIDATE / 'source/third_party/work/prefix-io-p4-02-cpu/src')
sys.path.insert(0, PACKAGE)
sys.path.insert(0, str(CANDIDATE))  # Exactly the normal launcher's identity-helper search path.
import prefix_io_control.p4_policy as policy
import prefix_io_control.p4_bridge as bridge
import prefix_io_control.p4_single_file_receipt as receipt
import prefix_io_control.p4_types as types


class SyntheticEvent:
    def __init__(self, **kwargs): self.records = self.queries = 0
    def record(self): self.records += 1; return 'synthetic-record'
    def query(self): self.queries += 1; return True
    def elapsed_time(self, other): return 0.001


class SyntheticScalar:
    METHODS = ('execute_model', '_prepare_inputs', 'sample_tokens')
    def __init__(self): self.enabled = True; self.failures = []; self.detaches = 0; self._adapter = NS(_pending=None)
    def _fail(self, reason): self.failures.append(reason); self.enabled = False
    def detach(self): self.detaches += 1; self.enabled = False
    @staticmethod
    def SourceRef(*values): return tuple(values)
    @staticmethod
    def MethodBinding(*values): return NS(refs=values[0], methods=values[1], origin=values[2])


class SyntheticObserver:
    def __init__(self):
        self.enabled = True; self.scalar = SyntheticScalar(); self.frames = []
        self.events = NS(active=None, pending=[], diagnostics=[]); self.detaches = 0; self.invalidations = []
    def invalidate(self, reason): self.invalidations.append(reason); self.enabled = False
    def detach(self): self.detaches += 1; self.enabled = False; self.scalar.detach()
    def resolve_ready(self): pass


class SyntheticRunner:
    def __init__(self, observer, factory): self.observer = observer; self.factory = factory; self.calls = 0
    def _prepare_inputs(self, value):
        self.calls += 1
        start, end = self.factory(enable_timing=True), self.factory(enable_timing=True)
        start.record()
        self.observer.events.active = (self.calls, start, end)
        self.observer.scalar._adapter._pending = dict(phase='prepared', ordinal=self.calls,
            frame=NS(step_kind='decode', batch=1, active_decode=1, prefill_tokens=0, context_length=144))
        return ('synthetic-original-prepare', value)


class SyntheticWorkerModule:
    FROZEN = {'scalar': ('synthetic-bytes', 'synthetic-sha')}
    @staticmethod
    def load_pinned(*args): return SyntheticScalar
    @staticmethod
    def connect_worker_observation(worker, **kwargs):
        worker.connect_arguments = kwargs
        worker.model_runner.factory = kwargs['event_factory']
        return worker.model_runner.observer


class SyntheticCommon:
    WORKER = 'synthetic-worker.py'; SCALAR = 'synthetic-scalar.py'; FRAME = 'synthetic-frame.py'
    CONTEXT = 'synthetic-context.py'; RUNNER = 'synthetic-runner.py'
    @staticmethod
    def safe(root, name): return Path(root) / name
    @staticmethod
    def load_ref(root, ref): return SyntheticWorkerModule


def fixture(mode='off'):
    observer = SyntheticObserver()
    worker = NS(model_runner=SyntheticRunner(observer, None))
    refs = {name: dict(path=name, bytes=1, sha256='a' * 64) for name in
        (SyntheticCommon.WORKER, SyntheticCommon.CONTEXT, SyntheticCommon.RUNNER)}
    actual = L.file_ref(NATIVE, 'common_candidate/' + T.REACTOR)
    refs[L.LEGACY_COLLECTOR_REACTOR_KEY] = actual
    capture = C.install(worker, common=SyntheticCommon, root=NATIVE, refs=refs, run_id='r',
        event_class=SyntheticEvent, selected_offsets=(16,), origin='cpu_fixture')
    native = G['IoReactor']; owner = object.__new__(native)
    owner._incoming = queue.Queue(); owner._submit_lock = threading.Lock()
    owner._prefix_single_file_retry = None; owner._prefix_progress = None
    owner._stop = False; owner._preload_cached_total = 0; owner._staging_cache = None
    for name in ('_preload_slots', '_shared_cached', '_preload_refcount', '_preload_pending_count',
                 '_preload_pending_cancel', '_preload_owned', '_preload_blocked_on_write', '_known_missing', '_preload_waiters'):
        setattr(owner, name, {})
    owner._preload_pending = []; owner._active = []
    actual_bridge = None
    if mode != 'off':
        actual_bridge = bridge.make_native_bridge('r', types.P4Config('interference', 100000000, 100000000), single_file=None)
        actual_bridge.single_file_shadow = mode == 'shadow'
    owner._prefix_p4_bridge = actual_bridge
    return owner, worker, capture, actual_bridge


def attach(mode, owner, capture, actual_bridge):
    return L.notification_adapter().prepare_notification_capture(mode, owner, actual_bridge, capture,
        run_id='r', request_id='r-p0-B', reactor_source=REACTOR, collector_source=COLLECTOR,
        wake_type=G['_SingleFileWake'])


class CombinedIntegrationTests(unittest.TestCase):
    def test_actual_cpu_collector_installs_same_helper_parameters(self):
        owner, worker, capture, actual_bridge = fixture()
        try:
            self.assertEqual(worker.connect_arguments['run_id'], 'r')
            self.assertEqual(worker.connect_arguments['native_source_sha256'], T.C5_REACTOR_SHA256)
            self.assertEqual(worker.connect_arguments['max_steps'], 128)
            self.assertEqual(worker.connect_arguments['max_pending'], 128)
            self.assertEqual(worker.connect_arguments['binding'].origin, 'cpu_fixture')
            self.assertEqual(capture.origin, 'cpu_fixture')
            self.assertFalse(capture.event_source['native_class_qualified'])
            self.assertIsNone(capture.current_single_file_step())
        finally: capture.detach()

    def test_original_prepare_delegates_once_cpu_live_state_unknown(self):
        owner, worker, capture, actual_bridge = fixture()
        try:
            self.assertEqual(worker.model_runner._prepare_inputs(7), ('synthetic-original-prepare', 7))
            self.assertEqual(worker.model_runner.calls, 1)
            self.assertEqual(capture.observer.events.active[1].raw.queries, 1)
            self.assertIsNone(capture.current_single_file_step())
            state = (1, 1, 2, 1, 1, 0, 144)
            self.assertFalse(capture.arm_single_file_wait(('r', 'uninstalled', 1, 1), state))
        finally: capture.detach()

    def test_four_actual_bound_fixture_hooks_are_available(self):
        owner, worker, capture, actual_bridge = fixture()
        try:
            import inspect
            for target, name in ((capture.observer, 'invalidate'), (capture.observer, 'detach'),
                                 (capture.observer.scalar, '_fail'), (capture.observer.scalar, 'detach')):
                self.assertTrue(inspect.ismethod(getattr(target, name)))
                self.assertIs(getattr(target, name).__self__, target)
            self.assertEqual(capture._wait_observer_bindings, [])
        finally: capture.detach()

    def test_off_original_queue_and_helper_install_detach(self):
        self.control_lifecycle('off')

    def test_shadow_original_queue_and_helper_install_detach(self):
        self.control_lifecycle('shadow')

    def control_lifecycle(self, mode):
        owner, worker, capture, actual_bridge = fixture(mode)
        original_queue = owner._incoming
        audit = attach(mode, owner, capture, actual_bridge)
        self.assertFalse(audit.installed)
        self.assertIs(type(owner._incoming), queue.Queue)
        self.assertNotIn('get', vars(owner._incoming))
        self.assertFalse(owner._prefix_wait_single_file_retry())
        self.assertEqual(worker.model_runner._prepare_inputs(7), ('synthetic-original-prepare', 7))
        owner._incoming.put(owner._STOP)
        owner._drain_incoming(block=False)
        self.assertTrue(owner._stop); self.assertTrue(owner._incoming.empty())
        capture.detach(); audit.close()
        doc = audit.export()
        result = A.validate_notification_evidence(doc, mode=mode, run_id='r', request_id='r-p0-B')
        self.assertEqual(result['status'], 'UNINSTALLED_CONTROL_PATH')
        self.assertFalse(result['notification_exercised'])
        self.assertIs(owner._incoming, original_queue)
        self.assertIs(worker.model_runner._prepare_inputs.__func__, SyntheticRunner._prepare_inputs)
        self.assertEqual(capture.observer.detaches, 1)

    def test_original_wake_intake_uses_exact_queue_without_work_credit(self):
        owner, worker, capture, actual_bridge = fixture()
        try:
            token = ('r', 'cpu-explicit-old-wake', 1, 1)
            owner._prefix_single_file_wake_pending = token
            owner._incoming.put(G['_SingleFileWake'](token))
            owner._drain_incoming(block=False)
            self.assertIsNone(owner._prefix_single_file_wake_pending)
            self.assertTrue(owner._incoming.empty())
            self.assertIsNone(capture._wait_registration)
        finally: capture.detach()

    def test_on_helper_refuses_without_real_bridge_capture_receipt(self):
        owner, worker, capture, actual_bridge = fixture('on')
        original_queue = owner._incoming
        try:
            with self.assertRaisesRegex(ValueError, 'same capture already bound'): attach('on', owner, capture, actual_bridge)
            self.assertIs(owner._incoming, original_queue)
            self.assertNotIn('get', vars(original_queue))
            self.assertIsNone(actual_bridge._single_file_capture)
            self.assertIsNone(actual_bridge.policy.single_file)
            self.assertIsNone(capture._wait_reactor_ref)
        finally: capture.detach()

    def test_original_on_installation_refuses_absent_native_receipt(self):
        owner, worker, capture, actual_bridge = fixture('on')
        try:
            with self.assertRaisesRegex(ValueError, 'explicit active bridge'): owner.install_p4_single_file_wait(capture)
            self.assertIsNone(getattr(owner, '_prefix_single_file_wait_capture', None))
            self.assertIsNone(capture._wait_registration)
        finally: capture.detach()

    def test_actual_bridge_attachment_rejects_cpu_or_unverified_identity(self):
        owner, worker, capture, actual_bridge = fixture('on')
        try:
            with self.assertRaisesRegex(ValueError, 'runner/model/layout/GPU/kernel'):
                actual_bridge.attach_single_file_capture(capture, runtime_identity=None, runtime_refs={}, shadow=False)
            self.assertIsNone(actual_bridge._single_file_capture)
            self.assertIsNone(actual_bridge._single_file_runtime)
        finally: capture.detach()

    def test_real_canonical_package_and_exact_policy_type(self):
        self.assertEqual(Path(receipt.__file__).resolve(), (CANDIDATE / CONTROL / 'p4_single_file_receipt.py').resolve())
        self.assertIs(bridge.P4Policy, policy.P4Policy)
        self.assertEqual(Path(policy.__file__).resolve(), (CANDIDATE / CONTROL / 'p4_policy.py').resolve())
        config = types.P4Config('interference', 100000000, 100000000, 100)
        for invalid in ({'step_budget_ns': 100}, NS(step_budget_ns=100)):
            with self.assertRaisesRegex(ValueError, 'verified exact'): policy.P4Policy('r', config, single_file=invalid)
        with self.assertRaisesRegex(ValueError, 'source-bound'): receipt.ExactSingleFileReceipt()
        with self.assertRaisesRegex(RuntimeError, 'GPU_BLOCKED'): receipt.load_verified_single_file(None, None)

    def test_native_event_impersonation_is_refused_by_real_collector(self):
        with self.assertRaisesRegex(ValueError, 'original installed torch'): C.event_class_source(SyntheticEvent, 'native_gpu_recording')

    def test_capture_run_frontend_namespaces_remain_distinct(self):
        A.validate_capture_identity(dict(request_id='r-p0-B', native_request_id='native-17'),
            dict(run_id='r'), run_id='r', request_id='r-p0-B')
        with self.assertRaises(ValueError):
            A.validate_capture_identity(dict(request_id='r-p0-B', native_request_id='native-17'),
                dict(run_id='r-p0-B'), run_id='r', request_id='r-p0-B')

    def test_detach_preserves_later_prepare_wrapper(self):
        owner, worker, capture, actual_bridge = fixture()
        later = lambda value: ('later', value)
        worker.model_runner._prepare_inputs = later
        capture.detach()
        self.assertIs(worker.model_runner._prepare_inputs, later)

    def test_no_native_origin_or_receipt_metadata_mutations_in_fixture(self):
        owner, worker, capture, actual_bridge = fixture('shadow')
        try:
            self.assertEqual(capture.origin, 'cpu_fixture')
            self.assertIsNone(actual_bridge.policy.single_file)
            self.assertIsNone(actual_bridge._single_file_capture)
            self.assertIsNone(actual_bridge._single_file_runtime)
            self.assertIsNone(capture._wait_registration)
        finally: capture.detach()


class SourceAndBlockingTests(unittest.TestCase):
    def test_source_groups_actual_stage_c_collector_and_runtime_identity(self):
        result = T.metadata_source_groups(NATIVE, stage_a=STAGE_A)
        self.assertEqual(result['source_refs']['runtime_identity']['path'], str((CANDIDATE / 'single_file_runtime_binding.py').resolve()))
        self.assertEqual(sum(row == result['source_refs']['collector'] for row in result['runtime_overlay_refs']), 1)
        self.assertFalse(result['source_groups_are_native_binding'])
        self.assertFalse(result['actual_on_installed'])
        self.assertIsNone(result['effective_cost_upper_ns'])

    def test_adapter_is_byte_exact_stage_a(self):
        self.assertEqual((HERE / 'notification_runtime_adapter.py').read_bytes(), (STAGE_A / 'notification_runtime_adapter.py').read_bytes())
        self.assertEqual(sha256((HERE / 'notification_runtime_adapter.py').read_bytes()).hexdigest(), T.ADAPTER_SHA256)

    def test_normal_model_and_numerical_ast_unchanged(self):
        result = T.assert_original_runtime_ast(STAGE_A)
        self.assertEqual(len(result['functions']), 10)

    def test_positive_on_callsite_only_source_audited_with_new_overlay(self):
        self.assertEqual(L.CANDIDATE, T.CANDIDATE)
        self.assertEqual(J.RECEIPT, T.CANDIDATE + '/' + CONTROL + '/p4_single_file_receipt.py')
        source = (HERE / 'run_p4_single_file_experiment.py').read_bytes()
        fn = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'execute_window')
        calls = [node for node in ast.walk(fn) if isinstance(node, ast.Call)]
        prepare = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'prepare_notification_capture']
        self.assertEqual(len(prepare), 1)
        call = prepare[0]
        self.assertEqual([key.value.id for key in call.keywords if key.arg == 'run_id'], ['LABEL'])
        self.assertEqual([key.value.id for key in call.keywords if key.arg == 'request_id'], ['rid'])
        self.assertIn('collector_source_view(refs)', source.decode())
        self.assertIn('len(scalar_adapter.frames)==128', source.decode())

    def test_runtime_accounting_contract_points_exact_stage_c_code(self):
        raw = (NATIVE / 'native_conditional_cost.py').read_bytes()
        self.assertEqual(V.V6, T.NATIVE + '/native_conditional_cost.py')
        self.assertEqual((V.V6_BYTES, V.V6_SHA), (len(raw), sha256(raw).hexdigest()))

    def test_unresolved_effective_authorities_and_no_cost(self):
        self.assertIsNone(L.GPU_UUID); self.assertIsNone(L.STORAGE); self.assertIsNone(J.GPU)
        self.assertIsNone(J.PERMISSION); self.assertIsNone(J.BINDING); self.assertIsNone(J.LOCK)
        result = T.no_grant()
        self.assertFalse(result['full_runtime_cost_qualified'])
        self.assertIsNone(result['valid_native_receipt'])

    def test_direct_gpu_scope_launch_and_native_qualification_block_first(self):
        callbacks = [lambda: L.execute_window(None, None, None, None), lambda: L.load_configuration(None),
            lambda: L.verify_guard(None), J.load_receipt, lambda: J.freeze_common(None), lambda: J.prepare(None),
            lambda: J.launch(None), lambda: J.after(None), lambda: J.check_sources(None, None),
            lambda: V.verify_runtime(None, config_ref=None, result_ref=None, guard_ref=None, before_ref=None, after_ref=None)]
        for callback in callbacks:
            with self.assertRaisesRegex(RuntimeError, A.BLOCKED): callback()

    def test_blocked_clis_do_not_read_invalid_config_or_budget(self):
        for name in ('run_p4_single_file_experiment.py', 'control_p4_single_file.py', 'verify_p4_single_file.py'):
            result = subprocess.run([sys.executable, '-B', '-I', '-S', str(HERE / name), '--config', 'missing'],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stderr, '')
            self.assertEqual(json.loads(result.stdout)['status'], A.BLOCKED)

    def test_no_torch_vllm_or_native_backend_import(self):
        self.assertFalse(any(name == 'torch' or name == 'vllm' or name.startswith(('torch.', 'vllm.', 'py_kvcache.')) for name in sys.modules))


if __name__ == '__main__':
    unittest.main(verbosity=2)
