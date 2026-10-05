"""Temporary synthetic files test validators, never GPU or QA evidence."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json,unittest
from tests import test_p1_build_dispatch_v2 as fixtures
from probekv.p1_action_runner_v2 import (runner_binding,validate_p1_action,
    verify_compiled_operation,verify_build_evidence,validate_comparison_capacity)
from probekv.p1_build_dispatch_v2 import _seal,dispatch_binding
from probekv.p0_evidence_v2 import P0EvidenceWriter,write_new_json
from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema10_storage import file_digest


class ActionRunnerTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.BuildDispatchTests();self.f.setUp();self.o=self.f.compile()
        self.binding={'instance_id':'fixture','gpu_uuid':'fixture-device'}
        self.m=dict(kind='bounded_P1_single_action_v2',binding=self.binding,runner_binding=runner_binding(),
            maximum_actions=1,phase='P1-E',locked_test_accessed=False,automatic_rental_allowed=False,
            operation_sha256=self.o['operation_sha256'],limits=dict(initialization_upper_seconds=30,cleanup_seconds=20),
            authority=dict(phase='P1-E',approval_reference='CPU validator fixture only',instance_id='fixture',gpu_uuid='fixture-device',
                starts_at_unix=100,expires_at_unix=1000,maximum_gpu_hours=.25,billing_mode='operator_managed_time_cap',
                operator_managed_billing=True,unit_price_per_hour=None,maximum_cost=None))
        _seal(self.m,'manifest_sha256')

    def run_validation(self):
        return validate_p1_action(self.m,self.o,actual_binding=self.binding,now_unix=200)

    def reseal(self):
        self.m.pop('manifest_sha256',None);_seal(self.m,'manifest_sha256')

    def test_validation_is_not_execution_permission(self):
        r=self.run_validation();self.assertFalse(r['GPU_execution_allowed']);self.assertEqual(r['required_seconds'],230)

    def capacity_fixture(self):
        return (dict(kind='P1_fixed_source_QA_operation_v2',arm='historical_1',target_id='C',
            request=dict(token_ids=[1]*2661,segments=[dict(segment_id='C',positions=list(range(512)))]),
            resources=dict(cuda_comparison_bytes=1207959552)),
            dict(hidden_size=4096,num_attention_heads=32,num_key_value_heads=8))

    def test_long_request_rejected_before_gpu_without_budget_mutation(self):
        o,c=self.capacity_fixture();before=deepcopy(o)
        self.assertRaisesRegex(ValueError,'required_bytes=1794277376',validate_comparison_capacity,o,c)
        self.assertEqual(before,o)

    def test_explicit_sufficient_capacity_reports_estimate_not_execution(self):
        o,c=self.capacity_fixture();o['resources']['cuda_comparison_bytes']=2*1024**3
        r=validate_comparison_capacity(o,c)
        self.assertEqual(r['required_bytes'],1794277376);self.assertFalse(r['comparison_executed'])

    def test_dense_and_exact_birth_do_not_require_comparison(self):
        o,c=self.capacity_fixture();o['arm']='dense'
        self.assertEqual(validate_comparison_capacity(o,{})['status'],'NOT_APPLICABLE')
        self.assertEqual(validate_comparison_capacity(self.o,{})['status'],'NOT_APPLICABLE')

    def test_invalid_gqa_geometry_rejected(self):
        o,c=self.capacity_fixture();c['num_key_value_heads']=7
        self.assertRaisesRegex(ValueError,'GQA geometry',validate_comparison_capacity,o,c)

    def test_workspace_that_fits_alone_can_fail_joint_hbm(self):
        o,c=self.capacity_fixture();o['resources']['cuda_comparison_bytes']=2*1024**3
        c['num_hidden_layers']=32
        with self.assertRaisesRegex(ValueError,'joint HBM preflight'):
            validate_comparison_capacity(o,c,allocator_capacity_bytes=6*1024**3)

    def test_joint_preflight_keeps_safety_and_winner_reservation(self):
        o,c=self.capacity_fixture();o['resources']['cuda_comparison_bytes']=2*1024**3
        c['num_hidden_layers']=32
        r=validate_comparison_capacity(o,c,allocator_capacity_bytes=7381975040)
        self.assertEqual(r['working_kv_bytes'],348782592)
        self.assertEqual(r['winner_kv_bytes'],67108864)
        self.assertEqual(r['safety_bytes'],4*1024**3)
        self.assertEqual(r['joint_required_bytes'],6858342400)

    def test_worker_preserves_audited_import_priority_and_literal_arguments(self):
        from scripts.server.run_decoupled_v2_p1_action import worker_command
        import sys,runpy
        command=worker_command('/fixture/entry.py',['--output',"quote' folder"],['/audited/src','/audited/patch'])
        with patch.object(sys,'path',list(sys.path)),patch.object(sys,'argv',[]),patch.object(runpy,'run_path') as run:
            exec(command[-1],{})
            self.assertEqual(sys.path,['/audited/src','/audited/patch'])
            self.assertEqual(sys.argv,['/fixture/entry.py','--output',"quote' folder"])
            run.assert_called_once_with('/fixture/entry.py',run_name='__main__')

    def test_p0_authorization_not_inherited(self):
        self.m['authority']['phase']='P0';self.reseal();self.assertRaisesRegex(ValueError,'authorization',self.run_validation)

    def test_expired_or_short_window_rejected(self):
        self.m['authority']['expires_at_unix']=199;self.reseal();self.assertRaises(ValueError,self.run_validation)
        self.m['authority']['expires_at_unix']=400;self.reseal();self.assertRaisesRegex(ValueError,'remaining',self.run_validation)

    def test_changed_entry_hash_rejected(self):
        self.m['runner_binding']['entry_sha256']='0'*64;self.reseal();self.assertRaises(ValueError,self.run_validation)

    def test_more_than_one_action_rejected(self):
        self.m['maximum_actions']=2;self.reseal();self.assertRaises(ValueError,self.run_validation)

    def test_actual_frozen_recipe_recompiled(self):
        args=dict(self.f.f.identity,root='fixture',index_sha256='a'*64)
        with patch('probekv.p1_input_consumer_v2.load_frozen_documents',return_value=self.f.f.docs):
            verify_compiled_operation(self.o,args)
            self.o['request']['token_ids'][0]=99;self.o.pop('operation_sha256');_seal(self.o,'operation_sha256')
            self.assertRaisesRegex(ValueError,'frozen recipe',verify_compiled_operation,self.o,args)

    def build_files(self,root,*,origin='real_cuda_execution',status='COMPLETED'):
        # Deliberately synthetic writer input: validates integrity checks only.
        w=P0EvidenceWriter(root,binding=self.binding,manifest=self.m)
        write_new_json(w.root/'operation.json',self.o)
        audit={'status':'COMPLETED','cleanup':{'passed':True},'CPU_fixture_only':True}
        receipt=dict(operation_sha256=self.o['operation_sha256'],publication_audit_sha256=digest_json(audit))
        _seal(receipt,'receipt_sha256')
        ref=write_new_json(w.root/'source_build_receipt.json',receipt);w.append('P1_build_receipt','action',ref)
        w.write_action('action',audit=audit,logits=(),origin=origin)
        w.finalize(dict(status=status,evidence_origin=origin))
        return receipt

    def test_raw_chain_and_completed_build_required(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'fixture';r=self.build_files(p)
            result=verify_build_evidence(p/'result.json',result_sha256=file_digest(p/'result.json'),receipt=r)
            self.assertTrue(result['raw_build_verified']);self.assertFalse(result['paper_evidence'])

    def test_failed_batch_cannot_launder_successful_receipt(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'fixture';r=self.build_files(p,status='FAILED')
            self.assertRaisesRegex(ValueError,'completed',verify_build_evidence,p/'result.json',result_sha256=file_digest(p/'result.json'),receipt=r)

    def test_cpu_record_cannot_become_native_build(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'fixture';r=self.build_files(p,origin='cpu_fixture')
            self.assertRaises(ValueError,verify_build_evidence,p/'result.json',result_sha256=file_digest(p/'result.json'),receipt=r)

    def test_corrupt_audit_rejected(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'fixture';r=self.build_files(p);(p/'action/request.json').write_text('{}')
            self.assertRaisesRegex(ValueError,'digest',verify_build_evidence,p/'result.json',result_sha256=file_digest(p/'result.json'),receipt=r)

    def test_torn_event_rejected(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'fixture';r=self.build_files(p);f=p/'actions.jsonl';f.write_bytes(f.read_bytes()[:-1])
            self.assertRaises(ValueError,verify_build_evidence,p/'result.json',result_sha256=file_digest(p/'result.json'),receipt=r)
