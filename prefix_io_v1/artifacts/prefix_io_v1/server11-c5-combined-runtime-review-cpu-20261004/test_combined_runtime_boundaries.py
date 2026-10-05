"""Independent combined entry boundaries. All worker/Event objects are CPU fixtures.

There is deliberately no fabricated native receipt, no private bridge mutation,
no positive native on attachment, and no timing benchmark in these tests.
"""
from __future__ import annotations
import ast
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import queue
import sys
import threading
import unittest
from unittest.mock import patch
import weakref

STAGE = Path(os.environ['C5_COMBINED_REVIEW_STAGE_ROOT'])
CANDIDATE = Path(os.environ['C5_COMBINED_REVIEW_CANDIDATE_ROOT'])
PREVIOUS = Path(os.environ['C5_COMBINED_REVIEW_PREVIOUS_ROOT'])
CONTROL = Path('source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control')
REACTOR = Path('source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py')
PREP = 'artifacts/prefix_io_v1/server11-c5-combined-runtime-cpu-20261004'
NATIVE = 'artifacts/prefix_io_v1/server11-c5-native-cost-preparation-cpu-20261004'
COMMON = NATIVE+'/common_candidate'
C5_R = 'a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47'
C5_C = 'bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def functions(path):
    return {node.name:node for node in ast.parse(path.read_bytes()).body if isinstance(node,ast.FunctionDef)}


class Poison:
    def __getattribute__(self, name): raise AssertionError('input touched before block: '+name)
    def __str__(self): raise AssertionError('input converted before block')
    def __fspath__(self): raise AssertionError('path converted before block')


class SyntheticEvent:
    def __init__(self, **kwargs): self.records = self.queries = self.elapsed = 0
    def record(self): self.records += 1; return 'synthetic-record-result'
    def query(self): self.queries += 1; return True
    def elapsed_time(self, other): self.elapsed += 1; return 0.001


class SyntheticRunner:
    def __init__(self): self.calls = 0
    def _prepare_inputs(self, value): self.calls += 1; return ('original-synthetic-prepare',value)


class SyntheticFrame:
    step_kind = 'prefill'; active_decode = 0; batch = 1; prefill_tokens = 1


class SyntheticAdapter:
    def __init__(self):
        self._pending = dict(phase='prepared',ordinal=1,frame=SyntheticFrame())


class SyntheticScalarState:
    def __init__(self): self.enabled = True; self._adapter = SyntheticAdapter()
    def _fail(self, reason): self.enabled = False
    def detach(self): self.enabled = False


class SyntheticEvents:
    def __init__(self, collector):
        self.active = (1,collector.EventProxy(SyntheticEvent(),__import__('time').monotonic_ns),
                       collector.EventProxy(SyntheticEvent(),__import__('time').monotonic_ns))
        self.active[1].record()  # Explicit CPU event: original query requires prior record bounds.


class SyntheticObserver:
    def __init__(self, collector):
        self.enabled = True; self.scalar = SyntheticScalarState()
        self.events = SyntheticEvents(collector); self.frames = []; self.detaches = 0
    def invalidate(self, reason): self.enabled = False
    def detach(self): self.detaches += 1; self.enabled = False; self.scalar.detach()


class SyntheticBinding:
    def __init__(self, refs, methods, origin): self.refs,self.methods,self.origin = refs,methods,origin


class SyntheticScalarModule:
    METHODS = ('execute_model','_prepare_inputs','sample_tokens')
    @staticmethod
    def SourceRef(*values): return tuple(values)
    MethodBinding = SyntheticBinding


class SyntheticWorker:
    def __init__(self): self.model_runner = SyntheticRunner()


class CombinedRuntimeReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load('_independent_combined_adapter',STAGE/'notification_runtime_adapter.py')
        cls.launcher = load('_independent_combined_launcher',STAGE/'run_p4_single_file_experiment.py')
        cls.controller = load('_independent_combined_controller',STAGE/'control_p4_single_file.py')
        cls.verifier = load('_independent_combined_verifier',STAGE/'verify_p4_single_file.py')
        cls.contract = load('_independent_combined_contract',STAGE/'combined_runtime_contract.py')
        cls.launcher.notification_adapter()  # Resolve the actual new shared helper once, before poison tests.
        for name in tuple(sys.modules):
            if name == 'prefix_io_control' or name.startswith('prefix_io_control.') or name == 'single_file_runtime_binding':
                del sys.modules[name]
        sys.path.insert(0,str(CANDIDATE/CONTROL.parent))
        sys.path.insert(0,str(CANDIDATE))
        cls.receipt = importlib.import_module('prefix_io_control.p4_single_file_receipt')
        cls.policy = importlib.import_module('prefix_io_control.p4_policy')
        cls.bridge = importlib.import_module('prefix_io_control.p4_bridge')
        cls.types = importlib.import_module('prefix_io_control.p4_types')
        cls.identity = importlib.import_module('single_file_runtime_binding')
        cls.collector = load('_independent_combined_actual_collector',CANDIDATE/'native_full_step_collector.py')
        tree = ast.parse((CANDIDATE/REACTOR).read_bytes())
        native = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name == 'IoReactor')
        method_names = {'install_p4_single_file_wait','_prefix_wait_single_file_retry','_prefix_wake_single_file_locked'}
        nodes = [ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
        nodes += [deepcopy(node) for node in tree.body if isinstance(node,ast.ClassDef) and node.name == '_SingleFileWake']
        nodes += [deepcopy(node) for node in native.body if isinstance(node,ast.FunctionDef) and node.name in method_names]
        namespace = dict(dataclass=dataclass,weakref=weakref,queue=queue,threading=threading)
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(CANDIDATE/REACTOR),'exec'),namespace)
        cls.wake_type = namespace['_SingleFileWake']
        cls.owner_type = type('SyntheticInterfaceOwner',(),{name:namespace[name] for name in method_names})

    def actual_cpu_capture(self):
        collector = self.collector; observer = SyntheticObserver(collector); calls = {}
        class SyntheticWorkerModule:
            FROZEN = dict(scalar=('synthetic-bytes','synthetic-sha'))
            @staticmethod
            def load_pinned(*args): calls['scalar_load'] = args; return SyntheticScalarModule
            @staticmethod
            def connect_worker_observation(worker,**kwargs):
                calls['worker'] = worker; calls['connect'] = kwargs; return observer
        class SyntheticCommon:
            WORKER='synthetic-worker.py';SCALAR='synthetic-scalar.py';FRAME='synthetic-frame.py'
            CONTEXT='synthetic-context.py';RUNNER='synthetic-runner.py'
            @staticmethod
            def safe(root,name): return Path(root)/name
            @staticmethod
            def load_ref(root,ref): calls['worker_ref']=ref; return SyntheticWorkerModule
        worker = SyntheticWorker()
        refs = {name:dict(path=name,bytes=1,sha256='a'*64) for name in
                (SyntheticCommon.WORKER,SyntheticCommon.CONTEXT,SyntheticCommon.RUNNER)}
        refs['third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'] = dict(
            path=COMMON+'/'+REACTOR.as_posix(),bytes=(CANDIDATE/REACTOR).stat().st_size,sha256=C5_R)
        capture = collector.install(worker,common=SyntheticCommon,root=STAGE,refs=refs,run_id='cpu-review',
            event_class=SyntheticEvent,selected_offsets=(16,),origin='cpu_fixture')
        return worker,capture,observer,calls

    def owner(self,mode):
        owner = self.owner_type(); owner._incoming = queue.Queue(); owner._submit_lock = threading.Lock()
        bridge = None
        if mode != 'off':
            bridge = self.bridge.make_native_bridge('cpu-review',self.types.P4Config('interference',100000000,100000000))
            # Public mode field only; no receipt or private capture/runtime fields are assigned.
            bridge.single_file_shadow = mode == 'shadow'
        owner._prefix_p4_bridge = bridge
        return owner,bridge

    def helper(self,mode,owner,bridge,capture):
        return self.adapter.prepare_notification_capture(mode,owner,bridge,capture,run_id='cpu-review',
            request_id='cpu-review-p0-B',reactor_source=CANDIDATE/REACTOR,
            collector_source=CANDIDATE/'native_full_step_collector.py',wake_type=self.wake_type)

    def assert_blocked(self,fn,*args,**kwargs):
        with ExitStack() as stack:
            for target in ('builtins.open','pathlib.Path.read_bytes','pathlib.Path.read_text',
                           'pathlib.Path.write_bytes','pathlib.Path.write_text','pathlib.Path.mkdir',
                           'subprocess.Popen','subprocess.run'):
                stack.enter_context(patch(target,side_effect=AssertionError('side effect before block: '+target)))
            with self.assertRaisesRegex(RuntimeError,self.adapter.BLOCKED): fn(*args,**kwargs)

    def test_01_new_launcher_and_controller_resolve_same_stage_c_overlay(self):
        self.assertEqual(self.launcher.DELIVERY,PREP)
        self.assertEqual(self.controller.D,PREP)
        self.assertEqual(self.launcher.CANDIDATE,COMMON)
        self.assertEqual(self.controller.C,COMMON)
        self.assertEqual(self.launcher.OVERLAY,COMMON+'/source')
        self.assertEqual(self.launcher.REACTOR,COMMON+'/'+REACTOR.as_posix())
        self.assertEqual(self.controller.RECEIPT,COMMON+'/'+(CONTROL/'p4_single_file_receipt.py').as_posix())
        self.assertTrue(self.verifier.V6.startswith(NATIVE+'/'))

    def test_02_old_operation_identifiers_are_inert_none(self):
        for value in (self.launcher.GPU_UUID,self.launcher.STORAGE,self.controller.GPU,
                      self.controller.PERMISSION,self.controller.LOCK,self.controller.BINDING):
            self.assertIsNone(value)

    def test_03_shared_adapter_is_exact_stage_a_bytes_and_actual_new_module(self):
        self.assertEqual((STAGE/'notification_runtime_adapter.py').read_bytes(),
                         (PREVIOUS/'notification_runtime_adapter.py').read_bytes())
        helper = self.launcher.notification_adapter()
        self.assertIs(helper,self.controller.notification_adapter())
        self.assertIs(helper,self.verifier.notification_adapter())
        self.assertEqual(Path(helper.__file__).resolve(),(STAGE/'notification_runtime_adapter.py').resolve())

    def test_04_stage_c_canonical_receipt_policy_bridge_and_identity_are_same_module_domain(self):
        self.assertIs(self.bridge.P4Policy,self.policy.P4Policy)
        self.assertIs(sys.modules['prefix_io_control.p4_single_file_receipt'],self.receipt)
        self.assertIs(sys.modules['single_file_runtime_binding'],self.identity)
        for module in (self.receipt,self.policy,self.bridge,self.identity):
            self.assertTrue(Path(module.__file__).resolve().is_relative_to(CANDIDATE.resolve()))

    def test_05_dict_same_named_class_and_direct_constructor_are_not_native_receipts(self):
        class ExactSingleFileReceipt: pass
        config = self.types.P4Config('interference',100000000,100000000)
        for value in (dict(status='PASS',origin='native_gpu_recording'),ExactSingleFileReceipt()):
            with self.subTest(value_type=type(value).__name__),self.assertRaises(ValueError):
                self.policy.make_p4_policy('cpu-review',config,single_file=value)
        with self.assertRaisesRegex(ValueError,'source-bound'):
            self.receipt.ExactSingleFileReceipt(origin='synthetic_cpu_contract')

    def test_06_native_receipt_issuance_and_all_normal_entrypoints_block_before_access(self):
        p = Poison()
        self.assert_blocked(self.launcher.execute_window,p,p,p,p)
        self.assert_blocked(self.launcher.load_configuration,p)
        self.assert_blocked(self.launcher.verify_guard,p)
        for name,values in (('load_receipt',()),('freeze_common',(p,)),('prepare',(p,)),('launch',(p,)),('after',(p,))):
            with self.subTest(entry=name): self.assert_blocked(getattr(self.controller,name),*values)
        self.assert_blocked(self.verifier.verify_runtime,p,config_ref=p,result_ref=p,guard_ref=p,before_ref=p,after_ref=p)
        with self.assertRaisesRegex(RuntimeError,'GPU_BLOCKED_C5_NATIVE_COST_CODE_PREPARATION_ONLY'):
            self.receipt.load_verified_single_file(p,p)

    def test_07_cli_main_never_reads_config_or_legacy_authorization(self):
        for module in (self.launcher,self.controller,self.verifier):
            with self.subTest(module=module.__name__),redirect_stdout(io.StringIO()) as stream:
                with patch.object(sys,'argv',['entry.py','--config','old-GPU-scope.json']):
                    self.assertEqual(module.main(),2)
                document=json.loads(stream.getvalue())
                self.assertEqual(document['status'],self.adapter.BLOCKED)
                self.assertIs(document['gpu_started'],False)
                self.assertIsNone(document['native_cost_receipt'])

    def test_08_original_normal_runtime_and_validation_ast_preserved(self):
        proof=self.contract.assert_original_runtime_ast(PREVIOUS)
        self.assertGreaterEqual(len(proof['functions']),10)
        self.assertIs(proof['native_execution_verified'],False)
        self.assertIs(proof['full_runtime_cost_qualified'],False)

    def test_09_source_groups_contain_actual_collector_and_runtime_identity_overlay(self):
        proof=self.contract.metadata_source_groups(CANDIDATE.parent,stage_a=PREVIOUS)
        common,overlay=proof['runtime_common_refs'],proof['runtime_overlay_refs']
        collector=str((CANDIDATE/'native_full_step_collector.py').resolve())
        helper=str((CANDIDATE/'single_file_runtime_binding.py').resolve())
        reactor=str((CANDIDATE/REACTOR).resolve())
        self.assertEqual(sum(row['path']==collector for row in overlay),1)
        self.assertFalse(any(row['path']==collector for row in common))
        self.assertTrue(any(row['path']==reactor for row in common))
        self.assertTrue(any(row['path']==helper for row in overlay))
        self.assertIs(proof['source_groups_are_native_binding'],False)
        self.assertIs(proof['receipt_issued'],False)
        self.assertIs(proof['positive_on_callsite_source_audited_only'],True)
        self.assertEqual(proof['timings_measured'],0)
        for key in ('gpu_launch_allowed','native_execution_verified','native_cost_qualified',
                    'full_runtime_cost_qualified','on_observation_cost_measured','performance_claim'):
            self.assertIs(proof[key],False)
        for key in ('gpu_uuid','valid_native_receipt','effective_cost_upper_ns','effective_step_budget_ns'):
            self.assertIsNone(proof[key])

    def test_10_c5_source_hashes_and_new_source_dependent_verifier_pins_match(self):
        self.assertEqual(sha256((CANDIDATE/REACTOR).read_bytes()).hexdigest(),C5_R)
        self.assertEqual(sha256((CANDIDATE/'native_full_step_collector.py').read_bytes()).hexdigest(),C5_C)
        verifier=CANDIDATE.parent/'native_conditional_cost.py'
        self.assertEqual(self.verifier.V6_BYTES,verifier.stat().st_size)
        self.assertEqual(self.verifier.V6_SHA,sha256(verifier.read_bytes()).hexdigest())

    def test_11_actual_cpu_collector_install_retains_correct_origin_source_and_capacity(self):
        worker,capture,observer,calls=self.actual_cpu_capture()
        try:
            self.assertEqual(capture.origin,'cpu_fixture')
            self.assertEqual(capture.run_id,'cpu-review')
            self.assertIs(capture.event_source['native_class_qualified'],False)
            self.assertEqual(calls['connect']['native_source_sha256'],C5_R)
            self.assertEqual((calls['connect']['max_pending'],calls['connect']['max_steps']),(128,128))
            self.assertEqual(calls['connect']['binding'].origin,'cpu_fixture')
            self.assertIsNone(capture._wait_registration)
            self.assertIsNone(capture._wait_reactor_ref)
            self.assertIsNone(capture.current_single_file_step())
        finally: capture.detach()
        self.assertEqual(observer.detaches,1)

    def test_12_actual_prepare_delegates_once_and_detach_restores_without_native_claim(self):
        worker,capture,observer,calls=self.actual_cpu_capture()
        self.assertEqual(worker.model_runner._prepare_inputs(7),('original-synthetic-prepare',7))
        self.assertEqual(worker.model_runner.calls,1)
        self.assertEqual(observer.events.active[1].raw.queries,1)
        self.assertTrue(capture.valid)
        capture.detach()
        self.assertIs(worker.model_runner._prepare_inputs.__func__,SyntheticRunner._prepare_inputs)
        self.assertEqual(worker.model_runner._prepare_inputs(8),('original-synthetic-prepare',8))
        self.assertEqual(worker.model_runner.calls,2)
        self.assertIsNone(capture.current_single_file_step())

    def test_13_actual_helper_off_and_shadow_keep_exact_original_queue_and_no_hook_install(self):
        for mode in ('off','shadow'):
            worker,capture,observer,calls=self.actual_cpu_capture();owner,bridge=self.owner(mode)
            original_queue=owner._incoming
            audit=self.helper(mode,owner,bridge,capture)
            self.assertIs(owner._incoming,original_queue)
            self.assertIs(type(original_queue),queue.Queue)
            self.assertNotIn('get',vars(original_queue))
            self.assertFalse(audit.installed)
            self.assertEqual(capture._wait_observer_bindings,[])
            self.assertIsNone(capture._wait_registration)
            self.assertFalse(owner._prefix_wait_single_file_retry())
            capture.detach();audit.close()
            evidence=audit.export()
            result=self.adapter.validate_notification_evidence(evidence,mode=mode,run_id='cpu-review',request_id='cpu-review-p0-B')
            self.assertEqual(result['status'],'UNINSTALLED_CONTROL_PATH')
            self.assertIs(result['notification_exercised'],False)
            self.assertIs(evidence['queue_get_restored'],True)
            self.assertIs(evidence['native_cost_qualified'],False)

    def test_14_combined_on_helper_rejects_unbound_cpu_capture_without_private_mutation(self):
        worker,capture,observer,calls=self.actual_cpu_capture();owner,bridge=self.owner('on')
        original_queue=owner._incoming
        with self.assertRaisesRegex(ValueError,'same capture already bound'):
            self.helper('on',owner,bridge,capture)
        self.assertIsNone(bridge.policy.single_file)
        self.assertIsNone(bridge._single_file_capture)
        self.assertIsNone(bridge._single_file_runtime)
        self.assertIsNone(capture._wait_reactor_ref)
        self.assertEqual(capture._wait_observer_bindings,[])
        self.assertIs(owner._incoming,original_queue)
        self.assertNotIn('get',vars(original_queue))
        capture.detach()

    def test_15_actual_reactor_installer_rejects_missing_native_receipt(self):
        worker,capture,observer,calls=self.actual_cpu_capture();owner,bridge=self.owner('on')
        with self.assertRaisesRegex(ValueError,'explicit active bridge'):
            capture.attach_single_file_wait(owner)
        self.assertFalse(hasattr(owner,'_prefix_single_file_wait_capture'))
        self.assertIsNone(capture._wait_reactor_ref)
        self.assertNotIn('get',vars(owner._incoming))
        capture.detach()

    def test_16_unchanged_native_bridge_and_identity_factory_refuse_cpu_attachment(self):
        worker,capture,observer,calls=self.actual_cpu_capture();owner,bridge=self.owner('on')
        with self.assertRaisesRegex(ValueError,'actual runner/model/layout/GPU/kernel'):
            bridge.attach_single_file_capture(capture,runtime_identity=None,runtime_refs={},shadow=False)
        with self.assertRaisesRegex(ValueError,'actual native verification'):
            self.identity.RuntimeSingleFileIdentity((),(),'',{},worker.model_runner)
        self.assertIsNone(bridge._single_file_capture)
        self.assertIsNone(bridge._single_file_runtime)
        capture.detach()

    def test_17_cpu_event_class_cannot_be_qualified_as_torch_cuda_event(self):
        with self.assertRaisesRegex(ValueError,'original installed torch'):
            self.collector.event_class_source(SyntheticEvent,'native_gpu_recording')
        source=self.collector.event_class_source(SyntheticEvent,'cpu_fixture')
        self.assertIs(source['native_class_qualified'],False)

    def test_18_native_attachment_origin_and_exact_runtime_identity_checks_preserved_in_source(self):
        source=(CANDIDATE/CONTROL/'p4_bridge.py').read_text(encoding='utf-8')
        self.assertIn('type(runtime_identity) is not RuntimeSingleFileIdentity',source)
        self.assertIn('getattr(capture, "origin", None) != "native_gpu_recording"',source)
        self.assertIn('runtime_identity.matches_runner(capture.runner_ref())',source)
        self.assertIn('actual live collector/identity helper must match the declared runtime overlay',source)

    def test_19_normal_source_callsite_keeps_owner_and_frontend_identity_distinct(self):
        source=(STAGE/'run_p4_single_file_experiment.py').read_bytes();tree=ast.parse(source)
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
        installs=[node for node in calls if isinstance(node.func,ast.Attribute) and node.func.attr=='install'
                  and isinstance(node.func.value,ast.Name) and node.func.value.id=='collector']
        self.assertEqual(len(installs),1)
        self.assertEqual(next(kw.value.id for kw in installs[0].keywords if kw.arg=='run_id'),'LABEL')
        helpers=[node for node in calls if isinstance(node.func,ast.Attribute) and node.func.attr=='prepare_notification_capture']
        self.assertEqual(len(helpers),1)
        self.assertEqual(next(kw.value.id for kw in helpers[0].keywords if kw.arg=='run_id'),'LABEL')
        self.assertEqual(next(kw.value.id for kw in helpers[0].keywords if kw.arg=='request_id'),'rid')
        self.assertEqual(sum(isinstance(node.func,ast.Attribute) and node.func.attr=='attach_single_file_capture' for node in calls),1)

    def test_20_four_hook_and_queue_observer_source_unchanged_without_positive_on_claim(self):
        source=(STAGE/'notification_runtime_adapter.py').read_text(encoding='utf-8')
        self.assertIn('type(queue_obj) is queue.Queue',source)
        self.assertIn('original_function = queue.Queue.get',source)
        self.assertIn('item = original_function(target, block=block, timeout=timeout)',source)
        self.assertIn('if mode != "on":',source)
        for token in ('"invalidate"','"detach"','"_fail"','invalidation_hook_count=4'):
            self.assertIn(token,source)
        self.assertEqual((STAGE/'notification_runtime_adapter.py').read_bytes(),
                         (PREVIOUS/'notification_runtime_adapter.py').read_bytes())

    def test_21_recorded_queue_subclass_cannot_claim_actual_on_observer_compatibility(self):
        class LegacyRecordedQueue(queue.Queue): pass
        worker,capture,observer,calls=self.actual_cpu_capture();owner,bridge=self.owner('off')
        owner._incoming=LegacyRecordedQueue()
        audit=self.adapter.NotificationAudit('on',owner,bridge,capture,'cpu-review','cpu-review-p0-B',{})
        with self.assertRaisesRegex(ValueError,'unmodified original Queue.get'):
            audit.install_queue_observation(self.wake_type)
        self.assertNotIn('get',vars(owner._incoming))
        capture.detach()

    def test_22_cpu_metadata_and_cli_cannot_grant_qualification_or_effective_cost(self):
        fields=self.contract.no_grant()
        for key in ('gpu_launch_allowed','native_execution_verified','native_cost_qualified','full_runtime_cost_qualified',
                    'on_observation_cost_measured','actual_on_installed','performance_claim','production_qualified'):
            self.assertIs(fields[key],False)
        for key in ('gpu_uuid','valid_native_receipt','effective_cost_upper_ns','effective_step_budget_ns'):
            self.assertIsNone(fields[key])
        self.assertEqual(fields['actual_model_processes_started'],0)

    def test_23_no_model_modules_loaded(self):
        self.assertFalse(any(name=='torch' or name.startswith('torch.') or name=='vllm' or name.startswith('vllm.')
                             for name in sys.modules))


if __name__=='__main__':
    unittest.main(verbosity=2)
