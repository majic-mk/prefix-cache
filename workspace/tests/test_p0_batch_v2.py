from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from probekv.p0_batch_v2 import validate_p0_batch, run_p0_native_batch, _ActionWallDeadline, _ActionDeadlineExceeded
from probekv.p0_evidence_v2 import read_p0_events
from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema10_storage import file_digest
from tests import test_native_p0_request_v2 as fixtures
from tests.test_native_consumption_v2 import CPUTransport


def sign(manifest):
    manifest['manifest_sha256']=digest_json({k:v for k,v in manifest.items() if k!='manifest_sha256'})
    return manifest


class P0BatchTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.NativeP0RequestTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        store,c,snap,profile,actions=self.f.setup_request()
        store.end_request(snap)
        self.store,self.c=store,c
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.output=Path(temp.name)/'batch'
        c.request['max_new_tokens']=2
        c.request['segments']=[dict(c.segments['C'])]
        now=time.time()
        job=dict(action_id='a1',request=deepcopy(c.request),request_sha256=digest_json(c.request),
            input_origin='controlled_provenance_diagnostic',upper_seconds=30.,
            sources_by_segment={'C':[dict(source_id=actions[0].source_ids[0])]},comparison_profile=asdict(profile))
        binding=dict(code_commit='c'*40,runtime_digest=profile.runtime_digest,patch_sha256='a'*64,
            model_signature='model',tokenizer_hash='tokenizer',instance_id='CPU-TEST-NOT-A-SERVER',
            gpu_uuid='CPU-TEST-NO-GPU',input_manifest_sha256=digest_json([dict(action_id='a1',
            request_sha256=job['request_sha256'],input_origin=job['input_origin'])]),
            initial_pool_sha256=file_digest(store.root/'catalog.json'),
            initial_registry_sha256=file_digest(store.registry._root/'registry.json'))
        self.manifest=sign(dict(kind='bounded_native_p0_batch_v2',phase='P0',locked_test_accessed=False,
            automatic_rental_allowed=False,native_manifest_sha256='f'*64,binding=binding,
            authority=dict(approval_reference='CPU test fixture, no GPU permission',
                instance_id=binding['instance_id'],gpu_uuid=binding['gpu_uuid'],starts_at_unix=now-60,
                expires_at_unix=now+7200,unit_price_per_hour=2.,maximum_cost=2.,maximum_gpu_hours=1.),
            limits=dict(initialization_upper_seconds=20.,cleanup_seconds=10.,maximum_actions=1,
                host_capture_bytes=100000,host_comparison_bytes=100000,cuda_comparison_bytes=100000),
            registry_budget=dict(max_bytes=1000000,max_manifest_bytes=100000),jobs=[job]))
        self.binding=deepcopy(binding)
        c.adapter.active=None;c.adapter.deadline=float('inf');c.adapter.reset=Mock()
        @contextmanager
        def open_request(request,*,arrival_ns):
            c.arrival_ns=arrival_ns
            try:yield c
            finally:c.close()
        c.adapter.open_request=open_request

    def validate(self,m=None):
        return validate_p0_batch(m or self.manifest,actual_binding=self.binding,now_unix=time.time())

    def run_batch(self,session_started_ns=None):
        c=self.c
        with patch('probekv.p0_batch_v2._native_origin',return_value='cpu_fixture'), \
             patch.object(c,'configure_cuda_comparison_v2',side_effect=lambda **kw:self.f.f.cpu_workspace(c,**kw)), \
             patch('probekv.native_consumption_v2.PhysicalLayerwiseSourceLoader',CPUTransport):
            return run_p0_native_batch(c.adapter,self.store,self.manifest,actual_binding=self.binding,
                output=self.output,session_started_ns=session_started_ns or time.perf_counter_ns())

    def test_valid_input_does_not_authorize_GPU_or_qualify_numerics(self):
        result=self.validate()
        self.assertEqual(result['status'],'INPUTS_VALIDATED')
        self.assertFalse(result['P1_execution_allowed'])
        self.assertEqual(result['native_numerics'],'NOT_RUN')

    def test_post_model_publication_overrun_cannot_complete_action(self):
        self.manifest['jobs'][0]['upper_seconds']=0.001
        sign(self.manifest)
        result=self.run_batch()
        self.assertEqual(result['status'],'FAILED')
        self.assertEqual(result['completed_action_ids'],[])
        self.assertEqual(result['failed_action_ids'],['a1'])
        self.assertTrue(self.c.closed)


    def test_missing_authority_expired_wrong_instance_and_budget_reject(self):
        for changes in ({'approval_reference':None},{'expires_at_unix':time.time()-1},
                        {'instance_id':'old-server'},{'maximum_cost':.0001},{'unit_price_per_hour':0.}):
            m=deepcopy(self.manifest);m['authority'].update(changes);sign(m)
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.validate(m)

    def time_only_authority(self):
        self.manifest['authority'].update(billing_mode='operator_managed_time_cap',
            operator_managed_billing=True,unit_price_per_hour=None,maximum_cost=None)
        sign(self.manifest)

    def test_explicit_operator_time_cap_keeps_unknown_bill_null(self):
        self.time_only_authority()
        self.assertEqual(self.validate()['available_seconds'],3600.)
        result=self.run_batch()
        self.assertEqual(result['status'],'COMPLETED')
        self.assertIsNone(result['estimated_usage_cost'])
        self.assertEqual(result['billing_mode'],'operator_managed_time_cap')

    def test_time_only_needs_explicit_mode_and_operator_billing(self):
        self.time_only_authority()
        for changes in ({'billing_mode':'metered_caps'},{'billing_mode':'unknown'},
                        {'operator_managed_billing':False},{'maximum_cost':0},
                        {'unit_price_per_hour':0},{'maximum_gpu_hours':float('inf')},
                        {'maximum_gpu_hours':0},{'approval_reference':None}):
            m=deepcopy(self.manifest);m['authority'].update(changes)
            # Non-finite input must fail even before canonical signing.
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                sign(m);self.validate(m)

    def test_time_only_insufficient_cap_and_expired_window_reject(self):
        self.time_only_authority()
        for changes in ({'maximum_gpu_hours':.001},{'expires_at_unix':time.time()-1}):
            m=deepcopy(self.manifest);m['authority'].update(changes);sign(m)
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.validate(m)

    def test_time_only_initialization_still_consumes_one_hour_cap(self):
        self.time_only_authority()
        with patch('probekv.p0_batch_v2.time.perf_counter_ns',return_value=4000_000_000_000):
            result=self.run_batch(1)
        self.assertEqual(result['status'],'BUDGET_STOP')
        self.assertIsNone(result['estimated_usage_cost'])
        self.c.adapter.reset.assert_not_called()

    def test_bad_signature_identity_and_future_Source_references_reject(self):
        m=deepcopy(self.manifest);m['jobs'][0]['upper_seconds']=31.
        with self.assertRaises(ValueError):self.validate(m)
        m=deepcopy(self.manifest);m['binding']['gpu_uuid']='other';sign(m)
        with self.assertRaises(ValueError):self.validate(m)
        m=deepcopy(self.manifest)
        m['jobs'][0]['sources_by_segment']['C']=[dict(birth_action_id='a1',segment_id='C')];sign(m)
        with self.assertRaises(ValueError):self.validate(m)

    def test_no_P1_locked_or_unbounded_actions(self):
        for changes in ({'phase':'P1-E'},{'locked_test_accessed':True},{'automatic_rental_allowed':True}):
            m=deepcopy(self.manifest);m.update(changes);sign(m)
            with self.assertRaises(ValueError):self.validate(m)
        m=deepcopy(self.manifest);m['limits']['maximum_actions']=2;sign(m)
        with self.assertRaises(ValueError):self.validate(m)

    def test_raw_request_change_or_illegal_full_capture_reject(self):
        m=deepcopy(self.manifest);m['jobs'][0]['request']['token_ids'][0]=999;sign(m)
        with self.assertRaises(ValueError):self.validate(m)
        m=deepcopy(self.manifest);m['jobs'][0]['request']['capture_original_full_prefill']=True
        m['jobs'][0]['request_sha256']=digest_json(m['jobs'][0]['request']);sign(m)
        with self.assertRaises(ValueError):self.validate(m)

    def test_real_driver_CPU_transport_writes_answer_without_numeric_PASS(self):
        result=self.run_batch()
        self.assertEqual(result['status'],'COMPLETED')
        self.assertEqual(result['completed_action_ids'],['a1'])
        self.assertEqual(result['evidence_origin'],'cpu_fixture')
        self.assertEqual(result['numerical_verdict'],'NOT_EVALUATED')
        self.assertFalse(result['gpu_runtime_qualified'])
        raw=json.loads((self.output/'a1/request.json').read_text())
        self.assertEqual(raw['answer']['token_ids'],[1,2])
        self.assertEqual(raw['final_admission']['status'],'UNSUPPORTED')
        self.assertTrue(raw['cleanup']['passed'])
        self.assertEqual(read_p0_events(self.output/'actions.jsonl',binding=self.binding)[-1]['kind'],'batch_stopped')

    def test_initialization_elapsed_budget_stops_before_native_reset(self):
        # Windows Python 3.8's counter can start at process initialization;
        # simulate elapsed time explicitly instead of inventing a negative start.
        with patch('probekv.p0_batch_v2.time.perf_counter_ns',return_value=4000_000_000_000):
            result=self.run_batch(1)
        self.assertEqual(result['status'],'BUDGET_STOP')
        self.assertEqual(result['pending_action_ids'],['a1'])
        self.c.adapter.reset.assert_not_called()

    def test_model_failure_stops_batch_preserves_partial_audit_and_no_retry(self):
        self.c.finish=Mock(side_effect=RuntimeError('fake model error'))
        result=self.run_batch()
        self.assertEqual(result['status'],'FAILED')
        self.assertEqual(result['failed_action_ids'],['a1'])
        events=read_p0_events(self.output/'actions.jsonl',binding=self.binding)
        failure=next(r for r in events if r['kind']=='action_failed')
        self.assertIn('C',failure['payload']['partial_request_audit']['comparison_receipts'])
        self.assertFalse(failure['payload']['resume_allowed'])
        self.c.finish.assert_called_once()

    def test_unavailable_previous_publication_is_not_implicit_source_build(self):
        # Validator disallows nonexistent birth actions before model execution.
        m=deepcopy(self.manifest)
        m['jobs'][0]['sources_by_segment']['C']=[dict(birth_action_id='missing',segment_id='C')]
        sign(m)
        with self.assertRaises(ValueError):self.validate(m)
        self.c.adapter.reset.assert_not_called()


class ActionWallDeadlineTests(unittest.TestCase):
    def test_cpu_elapsed_guard_covers_work_after_model_return(self):
        with patch('probekv.p0_batch_v2.time.perf_counter',side_effect=[10.,12.]):
            timer=_ActionWallDeadline(1.,native=False).start()
            with self.assertRaisesRegex(_ActionDeadlineExceeded,'including publication'):timer.check()
        timer.close()

    def test_native_timeout_disarms_before_raising_and_restores_handler(self):
        fake=Mock();fake.getitimer.return_value=(0.,0.)
        fake.getsignal.return_value='previous-handler'
        with patch('probekv.p0_batch_v2.signal',fake):
            timer=_ActionWallDeadline(1.,native=True).start()
            with self.assertRaises(_ActionDeadlineExceeded):timer._expired(None,None)
            self.assertFalse(timer.armed)
            fake.setitimer.assert_called_with(fake.ITIMER_REAL,0)
            fake.signal.assert_called_with(fake.SIGALRM,'previous-handler')
            timer.close()

    def test_native_guard_does_not_steal_existing_timer(self):
        fake=Mock();fake.getitimer.return_value=(5.,0.)
        with patch('probekv.p0_batch_v2.signal',fake):
            with self.assertRaisesRegex(RuntimeError,'another active'):
                _ActionWallDeadline(1.,native=True).start()
            fake.signal.assert_not_called()

    def test_publication_candidate_exception_handler_cannot_swallow_cancellation(self):
        self.assertFalse(issubclass(_ActionDeadlineExceeded,Exception))
        with self.assertRaises(_ActionDeadlineExceeded):
            try:
                raise _ActionDeadlineExceeded('stop entire action')
            except Exception:
                self.fail('publication failure handling swallowed cancellation')


if __name__=='__main__':unittest.main()
