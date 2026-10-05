"""CPU-only dispatch contracts. Mock forwards are never GPU evidence."""
from copy import deepcopy
from contextlib import nullcontext
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch, Mock

from probekv.p1_build_dispatch_v2 import compile_p1_build, validate_build_context, execute_p1_source_build, _seal, bind_request_content_keys
from probekv.source_comparison_v2 import runtime_binding_digest, scorer_digest
from probekv.native_p0_request_v2 import P0RequestFailure
from probekv.v8_schema10_execution import digest_json
from tests import test_p1_input_consumer_v2 as fixtures


class BuildDispatchTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.InputConsumerTests();self.f.setUp()
        self.p=dict(model_signature='a'*64,provenance_policy='ALLOW_MIXED_G1',completed_depth=2,
            trim_ratio=.15,tau_reuse=.25,tau_add=.25,repair_policy_digest='e'*64,
            runtime_digest=runtime_binding_digest(),scoring_function_digest=scorer_digest(),diagnostic_only=True)
        self.resources=dict(host_capture_bytes=1000000,host_comparison_bytes=1000000,cuda_comparison_bytes=1000000)

    def compile(self, bid='b0'):
        return compile_p1_build(self.f.docs,identity=self.f.identity,build_id=bid,
                                comparison_profile=self.p,resources=self.resources)

    def test_original_prompt_unchanged_separate_one_token_birth(self):
        before=deepcopy(self.f.docs);o=self.compile()
        self.assertEqual(self.f.docs,before)
        q=before['group-00.json']['requests'][0]['request']
        self.assertEqual(q['token_ids'],o['request']['token_ids'])
        self.assertEqual(q['segments'],o['request']['segments'])
        self.assertEqual(o['request']['max_new_tokens'],1)
        self.assertNotEqual(q['request_id'],o['request']['request_id'])
        self.assertFalse(o['GPU_execution_allowed']);self.assertFalse(o['QA_evaluated'])
        self.assertEqual(o['expected_generation'],0)
        self.assertIsNone(o['parent_build_id'])

    def test_independent_s0_is_not_fifth_online_variant(self):
        o=self.compile('b4')
        self.assertEqual(o['role'],'independent_S0');self.assertTrue(o['isolated_store_required'])
        self.assertNotIn('answers',o['request'])

    def test_future_consumer_is_not_build(self):
        with self.assertRaisesRegex(ValueError,'build ID'):self.compile('q4')

    def test_old_runtime_rejected(self):
        self.p['runtime_digest']='0'*64
        self.assertRaises(ValueError,self.compile)

    def test_wrong_depth_and_missing_resource_bound_rejected(self):
        self.p['completed_depth']=8;self.assertRaises(ValueError,self.compile)
        self.p['completed_depth']=2;self.resources.pop('host_capture_bytes')
        self.assertRaises(ValueError,self.compile)

    def test_no_teacher_source_even_with_rehashed_input(self):
        q=self.f.docs['group-00.json']['requests'][0]
        q['request']['teacher_token_ids']=[1];q['request_sha256']=digest_json(q['request'])
        self.f.docs['planned_P1_E_actions.json']['builds'][0]['request_sha256']=q['request_sha256']
        self.f.rehash();self.assertRaisesRegex(ValueError,'override',self.compile)

    def context(self,o,rows=None):
        c=NS(request=o['request'],current_completed_depth=0,adapter=NS(costs=None,spec=NS(num_layers=32)))
        s=NS(lock=nullcontext(),_snapshot=Mock(),_snapshots={'one':1},_plans={},_lease_counts={},
             _catalog={'rows':rows or {}},_verify_row=Mock(),content_key=lambda tokens:digest_json(tokens))
        c.request=bind_request_content_keys(o['request'],s)
        snap=NS(snapshot_id='one')
        return c,s,snap

    def test_runtime_content_key_binding_preserves_frozen_request(self):
        o=self.compile();before=deepcopy(o);c,s,snap=self.context(o)
        self.assertEqual(o,before);self.assertIn('content_key',c.request['segments'][0])
        self.assertEqual(c.request['token_ids'],o['request']['token_ids'])
        with patch('probekv.native_p0_operation_v2.validate_p0_comparison_action'):
            validate_build_context(c,s,snap,o,expected_operation_sha256=o['operation_sha256'])

    def test_wrong_runtime_namespace_refused_before_native(self):
        o=self.compile();c,s,snap=self.context(o);c.request['segments'][0]['content_key']='foreign'
        with patch('probekv.native_p0_operation_v2.validate_p0_comparison_action') as native:
            self.assertRaises(ValueError,validate_build_context,c,s,snap,o,expected_operation_sha256=o['operation_sha256'])
            native.assert_not_called()

    def test_nonempty_exact_pool_rejected_before_native(self):
        o=self.compile();c,s,snap=self.context(o,{'old':{}})
        with patch('probekv.native_p0_operation_v2.validate_p0_comparison_action') as native:
            with self.assertRaisesRegex(ValueError,'empty independent'):
                validate_build_context(c,s,snap,o,expected_operation_sha256=o['operation_sha256'])
            native.assert_not_called()

    def test_mutated_operation_refused_before_store_touch(self):
        o=self.compile();c,s,snap=self.context(o);sha=o['operation_sha256'];o['request']['token_ids'][0]=42
        with self.assertRaisesRegex(ValueError,'differs'):
            validate_build_context(c,s,snap,o,expected_operation_sha256=sha)
        s._snapshot.assert_not_called()

    def test_valid_exact_dispatch_has_only_empty_target_comparison(self):
        o=self.compile();c,s,snap=self.context(o)
        with patch('probekv.native_p0_operation_v2.validate_p0_comparison_action'):
            p,actions,job=validate_build_context(c,s,snap,o,expected_operation_sha256=o['operation_sha256'])
        self.assertEqual(len(actions),1);self.assertEqual(actions[0].source_ids,())
        self.assertEqual(job['operation'],'source_request')
        self.assertEqual(actions[0].completed_depth,2)

    def test_missing_mixed_parent_is_not_fallback_to_exact(self):
        o=self.compile();o.pop('operation_sha256');o.update(role='mixed_M1',expected_generation=1)
        _seal(o,'operation_sha256');c,s,snap=self.context(o)
        with self.assertRaisesRegex(ValueError,'parent build receipt'):
            validate_build_context(c,s,snap,o,expected_operation_sha256=o['operation_sha256'])

    def test_self_hashed_parent_receipt_cannot_hide_wrong_birth_prefix(self):
        o=self.compile();o.pop('operation_sha256')
        pq=dict(request_id='parent',token_ids=[7,8,9,10],
                segments=[dict(segment_id='U',token_ids=[8,9],positions=[1,2])])
        o.update(role='mixed_M1',expected_generation=1,parent_build_id='parent-build',
                 parent_original_request_sha256='b'*64,parent_birth_request=pq)
        _seal(o,'operation_sha256')
        ref=dict(manifest_id='m',manifest_digest='e'*64,target_occurrence='U',prefix_end=1,authorization_domain='test')
        row=dict(artifact_digest='c'*64,generation=0,origin='EXACT_CONTEXT',manifest_reference=ref,
                 birth_request_id='parent',token_ids=[8,9],birth_target_positions=[1,2])
        receipt=dict(build_id='parent-build',role='G0_parent',input_graph_sha256=o['input_graph_sha256'],
            original_request_sha256='b'*64,dispatch_binding=o['dispatch_binding'],generation=0,cleanup_passed=True,
            source_id='source',artifact_digest='c'*64,manifest_reference=ref,birth_request_id='parent')
        _seal(receipt,'receipt_sha256');c,s,snap=self.context(o,{'source':row})
        s._visible_row=Mock(return_value=row)
        s._manifest_record=Mock(return_value=(dict(token_ids=[999,8,9,10],absolute_positions=[0,1,2,3]),{}))
        with self.assertRaisesRegex(ValueError,'historical context'):
            validate_build_context(c,s,snap,o,expected_operation_sha256=o['operation_sha256'],parent_receipt=receipt)
        s._verify_row.assert_not_called()

    def test_non_target_overlap_in_original_inventory_refused(self):
        e=self.f.docs['group-00.json']['requests'][0]
        e['request']['segments'].append(dict(segment_id='overlap',token_ids=[0],positions=[1]))
        e['request_sha256']=digest_json(e['request'])
        self.f.docs['planned_P1_E_actions.json']['builds'][0]['request_sha256']=e['request_sha256']
        self.f.rehash()
        with self.assertRaisesRegex(ValueError,'nonoverlapping'):self.compile()

    def test_publication_failure_preserves_audit_no_second_forward(self):
        o=self.compile();c,s,snap=self.context(o)
        audit=dict(status='COMPLETED',cleanup={'passed':True},extra_forward_count=0,
                   publication={'targets':{'C':{'publication_performed':False}}})
        with patch('probekv.p1_build_dispatch_v2.validate_build_context',return_value=(None,(),{})), \
             patch('probekv.native_p0_request_v2.execute_p0_request',return_value=audit) as forward:
            with self.assertRaises(P0RequestFailure) as exc:
                execute_p1_source_build(c,s,snap,o,expected_operation_sha256=o['operation_sha256'])
            self.assertIs(exc.exception.audit,audit);forward.assert_called_once()

    def test_success_requires_actual_row_and_keeps_qa_unknown(self):
        o=self.compile();target=o['request']['segments'][0]
        row=dict(source_id='source',artifact_digest='b'*64,token_ids=target['token_ids'],
            birth_target_positions=target['positions'],birth_request_id=o['request']['request_id'],
            generation=0,origin='EXACT_CONTEXT',layer_count=32,parent_owned_kv_bytes=0,prefix_shadow_bytes=0,
            manifest_reference={'id':'CPU-fixture-not-GPU'})
        c,s,snap=self.context(o,{'source':row})
        audit=dict(status='COMPLETED',cleanup={'passed':True},extra_forward_count=0,
            publication={'targets':{'C':dict(publication_performed=True,source_id='source')}})
        with patch('probekv.p1_build_dispatch_v2.validate_build_context',return_value=(None,(),{})), \
             patch('probekv.native_p0_request_v2.execute_p0_request',return_value=audit):
            result=execute_p1_source_build(c,s,snap,o,expected_operation_sha256=o['operation_sha256'])
        s._verify_row.assert_called_once_with(row,full=True)
        self.assertFalse(result['receipt']['QA_evaluated']);self.assertFalse(result['receipt']['paper_evidence'])


if __name__=='__main__':unittest.main()
