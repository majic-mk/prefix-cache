"""Mixed-reference batch validation/routing only; mocked driver is not GPU evidence."""
from copy import deepcopy
import json
import time
import unittest
from unittest.mock import Mock, patch

import torch

from probekv.native_p0_request_v2 import P0RequestFailure
from probekv.p0_batch_v2 import run_p0_native_batch, validate_p0_batch
from probekv.p0_evidence_v2 import read_p0_events
from probekv.source_comparison_v2 import runtime_binding_digest
from probekv.v8_schema10_execution import digest_json
from tests import test_p0_batch_v2 as fixtures


class P0MixedBatchTests(unittest.TestCase):
    def setUp(self):
        base=fixtures.P0BatchTests(); base.setUp(); self.addCleanup(base.doCleanups)
        self.c,self.store,self.output=base.c,base.store,base.output
        self.manifest=deepcopy(base.manifest)
        self.source_job=deepcopy(self.manifest['jobs'][0])
        source_id=self.source_job['sources_by_segment']['C'][0]['source_id']
        row=self.store._catalog['rows'][source_id]
        self.c.request.update(capture_logits=True,teacher_token_ids=[7],
            segments=[dict(segment_id='C',positions=[2,3],token_ids=[12,13]),
                      dict(segment_id='T',positions=[4],token_ids=[14])])
        self.manifest['jobs']=[dict(operation='explicit_mixed_reference',
            action_id='mixed-ref',request=deepcopy(self.c.request),upper_seconds=30.,
            input_origin='controlled_provenance_diagnostic',target_id='T',upstream_segment_id='C',
            source=dict(source_id=source_id,artifact_digest=row['artifact_digest']),
            first_reuse_layer=2,repair_positions_by_layer={'2':[2],'3':[2]})]
        self.manifest['limits'].update(mixed_reference_host_bytes=100000,
                                     mixed_reference_cuda_bytes=200000)
        self.manifest['exact_capture_pairs']=[]
        self.sign()

    def sign(self):
        for job in self.manifest['jobs']:
            job['request_sha256']=digest_json(job['request'])
        self.manifest['limits']['maximum_actions']=len(self.manifest['jobs'])
        self.manifest['binding']['runtime_digest']=runtime_binding_digest()
        self.manifest['binding']['input_manifest_sha256']=digest_json([
            dict(action_id=j['action_id'],request_sha256=j['request_sha256'],input_origin=j['input_origin'])
            for j in self.manifest['jobs']])
        fixtures.sign(self.manifest)
        self.binding=deepcopy(self.manifest['binding'])

    def validate(self):
        return validate_p0_batch(self.manifest,actual_binding=self.binding,now_unix=time.time())

    def run_batch(self,session_started_ns=None):
        with patch('probekv.p0_batch_v2._native_origin',return_value='cpu_fixture'):
            return run_p0_native_batch(self.c.adapter,self.store,self.manifest,
                actual_binding=self.binding,output=self.output,
                session_started_ns=session_started_ns or time.perf_counter_ns())

    def completed_driver(self,context,store,snapshot,job,**kwargs):
        self.assertIs(context,self.c); self.assertIs(store,self.store)
        self.assertIn(snapshot.snapshot_id,store._snapshots)
        self.assertEqual(job,self.manifest['jobs'][0])
        self.assertEqual(kwargs,dict(host_reference_bytes=100000,cuda_reference_bytes=200000))
        context.logit_trace=[torch.tensor([1.,2.,3.]),torch.tensor([3.,2.,1.])]
        context.p0_reference_target_layers_v2=tuple(
            (torch.full((1,2,2),float(i),dtype=torch.bfloat16),
             torch.full((1,2,2),float(i+1),dtype=torch.bfloat16)) for i in range(3))
        context.close()
        return dict(kind='p0_explicit_mixed_reference',status='COMPLETED',
            request_id=context.request['request_id'],answer=dict(token_ids=[2,0]),
            cleanup=dict(passed=True,failures=[]),publication=dict(status='DIAGNOSTIC_NOT_PUBLISHED'),
            native_runtime_qualified=False,P1_execution_allowed=False)

    def test_reference_validates_without_selector_profile_or_source_decision_fields(self):
        self.assertNotIn('comparison_profile',self.manifest['jobs'][0])
        self.assertNotIn('sources_by_segment',self.manifest['jobs'][0])
        result=self.validate()
        self.assertEqual(result['status'],'INPUTS_VALIDATED')
        self.assertEqual(result['native_numerics'],'NOT_RUN')
        self.assertFalse(result['P1_execution_allowed'])
        self.c.adapter.reset.assert_not_called()

    def test_missing_or_invalid_reference_caps_rejected_before_execution(self):
        original=deepcopy(self.manifest['limits'])
        for name in ('mixed_reference_host_bytes','mixed_reference_cuda_bytes'):
            for value in (None,0,-1,True,1.5):
                with self.subTest(name=name,value=value):
                    self.manifest['limits']=deepcopy(original)
                    if value is None: self.manifest['limits'].pop(name)
                    else: self.manifest['limits'][name]=value
                    self.sign()
                    with self.assertRaisesRegex(ValueError,'mixed reference.*bounds'):
                        self.validate()
        self.c.adapter.reset.assert_not_called()

    def test_literal_immutable_source_required_not_birth_reference(self):
        original=deepcopy(self.manifest['jobs'][0])
        for source in ({'birth_action_id':'previous','segment_id':'C'},
                       {'source_id':'a'}, {'source_id':'a','artifact_digest':'not-a-digest'},
                       {'source_id':'a','artifact_digest':'a'*64,'birth_action_id':'previous'}):
            with self.subTest(source=source):
                self.manifest['jobs'][0]=deepcopy(original)
                self.manifest['jobs'][0]['source']=source; self.sign()
                with self.assertRaisesRegex(ValueError,'immutable already-published'):
                    self.validate()

    def test_diagnostic_cannot_supply_later_publication_reference(self):
        later=deepcopy(self.source_job)
        later['action_id']='later'; later['request']['request_id']='later-request'
        later['sources_by_segment']={'C':[dict(birth_action_id='mixed-ref',segment_id='T')]}
        self.manifest['jobs'].append(later); self.sign()
        with self.assertRaisesRegex(ValueError,'publication reference'):
            self.validate()

    def test_selector_fields_and_unbound_teacher_decode_rejected(self):
        original=deepcopy(self.manifest['jobs'][0])
        for field,value in (('sources_by_segment',{}),('comparison_profile',{})):
            with self.subTest(field=field):
                self.manifest['jobs'][0]=deepcopy(original)
                self.manifest['jobs'][0][field]=value; self.sign()
                with self.assertRaises(ValueError): self.validate()
        for field,value in (('capture_logits',False),('teacher_token_ids',[]),
                            ('teacher_token_ids',[True]),('native_dense_continuation',True)):
            with self.subTest(field=field,value=value):
                self.manifest['jobs'][0]=deepcopy(original)
                self.manifest['jobs'][0]['request'][field]=value; self.sign()
                with self.assertRaises(ValueError): self.validate()
        self.c.adapter.reset.assert_not_called()

    def test_repair_recipe_cannot_be_changed_to_full_dense_or_reentry(self):
        for masks in ({'2':[2,3],'3':[2,3]},{'2':[2],'3':[3]}):
            with self.subTest(masks=masks):
                self.manifest['jobs'][0]['repair_positions_by_layer']=masks; self.sign()
                with self.assertRaises(ValueError): self.validate()

    def test_successful_reference_releases_snapshot_and_writes_raw_target_layers(self):
        catalog_before=(self.store.root/'catalog.json').read_bytes()
        registry_before=(self.store.registry._root/'registry.json').read_bytes()
        with patch('probekv.p0_batch_v2.execute_explicit_mixed_reference',side_effect=self.completed_driver) as driver, \
             patch('probekv.p0_batch_v2.execute_p0_request',side_effect=AssertionError('no Source request')), \
             patch.object(self.store,'commit_publication',side_effect=AssertionError('no diagnostic publication')):
            result=self.run_batch()
        driver.assert_called_once()
        self.assertEqual(result['status'],'COMPLETED')
        self.assertEqual(result['completed_action_ids'],['mixed-ref'])
        self.assertEqual(result['evidence_origin'],'cpu_fixture')
        self.assertEqual(result['numerical_verdict'],'NOT_EVALUATED')
        self.assertFalse(result['gpu_runtime_qualified']); self.assertFalse(result['P1_execution_allowed'])
        self.assertFalse(self.store._snapshots)
        self.assertEqual(self.c.p0_reference_target_layers_v2,())
        self.assertEqual((self.store.root/'catalog.json').read_bytes(),catalog_before)
        self.assertEqual((self.store.registry._root/'registry.json').read_bytes(),registry_before)
        record=json.loads((self.output/'mixed-ref/record.json').read_text())
        self.assertEqual(record['target_kv']['layer_count'],3)
        self.assertEqual(record['target_kv']['shape'],[1,2,2])
        self.assertEqual(record['target_kv']['encoding'],'bfloat16_bits_int16')
        self.assertTrue((self.output/'mixed-ref/target-kv/layer-0003-V.npy').is_file())

    def test_completed_reference_requires_actual_target_evidence(self):
        def missing(*args,**kwargs):
            result=self.completed_driver(*args,**kwargs)
            self.c.p0_reference_target_layers_v2=None
            return result
        with patch('probekv.p0_batch_v2.execute_explicit_mixed_reference',side_effect=missing):
            result=self.run_batch()
        self.assertEqual(result['status'],'FAILED')
        self.assertFalse(result['completed_action_ids'])
        self.assertFalse(self.store._snapshots)

    def test_completed_reference_requires_every_target_layer(self):
        def partial(*args,**kwargs):
            result=self.completed_driver(*args,**kwargs)
            self.c.p0_reference_target_layers_v2=self.c.p0_reference_target_layers_v2[:2]
            return result
        with patch('probekv.p0_batch_v2.execute_explicit_mixed_reference',side_effect=partial):
            result=self.run_batch()
        self.assertEqual(result['status'],'FAILED')
        self.assertFalse(result['completed_action_ids'])
        self.assertFalse(self.store._snapshots)

    def test_completed_reference_target_rows_match_declared_target(self):
        def wrong_rows(*args,**kwargs):
            result=self.completed_driver(*args,**kwargs)
            self.c.p0_reference_target_layers_v2=tuple(
                (torch.ones((2,2,2),dtype=torch.bfloat16),torch.ones((2,2,2),dtype=torch.bfloat16))
                for _ in range(3))
            return result
        with patch('probekv.p0_batch_v2.execute_explicit_mixed_reference',side_effect=wrong_rows):
            result=self.run_batch()
        self.assertEqual(result['status'],'FAILED')
        self.assertFalse(result['completed_action_ids'])
        self.assertFalse(self.store._snapshots)

    def test_target_writer_failure_clears_host_references_and_keeps_failure(self):
        with patch('probekv.p0_batch_v2.execute_explicit_mixed_reference',side_effect=self.completed_driver), \
             patch('probekv.p0_batch_v2.P0EvidenceWriter.write_action',side_effect=OSError('evidence disk failure')):
            result=self.run_batch()
        self.assertEqual(result['status'],'FAILED')
        self.assertEqual(result['failed_action_ids'],['mixed-ref'])
        self.assertFalse(self.store._snapshots)
        self.assertEqual(self.c.p0_reference_target_layers_v2,())
        events=read_p0_events(self.output/'actions.jsonl',binding=self.binding)
        failure=next(e['payload'] for e in events if e['kind']=='action_failed')
        self.assertEqual(failure['error_type'],'OSError')
        self.assertFalse(failure['snapshot_retained'])

    def test_failed_fence_retains_snapshot_and_original_audit(self):
        original_close=self.c.close
        self.c.close=Mock(side_effect=RuntimeError('device fence failed'))
        partial=dict(kind='p0_explicit_mixed_reference',status='FAILED',
            cleanup=dict(passed=False,failures=['device fence failed']),answer=None)
        def failed(*args,**kwargs):
            raise P0RequestFailure(partial,RuntimeError('first device error'))
        try:
            with patch('probekv.p0_batch_v2.execute_explicit_mixed_reference',side_effect=failed):
                result=self.run_batch()
            self.assertEqual(result['status'],'FAILED')
            self.assertFalse(self.c.closed); self.assertEqual(len(self.store._snapshots),1)
            events=read_p0_events(self.output/'actions.jsonl',binding=self.binding)
            payload=next(e['payload'] for e in events if e['kind']=='action_failed')
            self.assertTrue(payload['snapshot_retained'])
            self.assertEqual(payload['partial_request_audit'],partial)
            self.assertFalse(payload['resume_allowed'])
        finally:
            self.c.close=original_close; self.c.close()
            for snapshot in tuple(self.store._snapshots.values()): self.store.end_request(snapshot)

    def test_budget_stop_before_source_snapshot_or_reset(self):
        with patch('probekv.p0_batch_v2.time.perf_counter_ns',return_value=4000_000_000_000), \
             patch.object(self.store,'begin_request',side_effect=AssertionError('budget already exhausted')), \
             patch('probekv.p0_batch_v2.execute_explicit_mixed_reference',side_effect=AssertionError('must not run')):
            result=self.run_batch(1)
        self.assertEqual(result['status'],'BUDGET_STOP')
        self.assertEqual(result['pending_action_ids'],['mixed-ref'])
        self.c.adapter.reset.assert_not_called()


if __name__=='__main__':
    unittest.main()
