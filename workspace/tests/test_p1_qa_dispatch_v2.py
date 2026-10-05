"""CPU contracts only; no simulated forward is model-quality evidence."""
from copy import deepcopy
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import unittest
from tests import test_p1_build_dispatch_v2 as fixtures
from probekv.p1_qa_dispatch_v2 import compile_p1_qa,execute_p1_qa,P1FixedSourceSession
from probekv.p1_build_dispatch_v2 import _seal,bind_request_content_keys
from probekv.v8_schema10_execution import digest_json


class QADispatchTests(unittest.TestCase):
    def setUp(self):
        self.b=fixtures.BuildDispatchTests();self.b.setUp();self.f=self.b.f
        e=self.f.docs['group-00.json']['requests'][4]
        e['request']['max_new_tokens']=32;e['request_sha256']=digest_json(e['request'])
        for a in self.f.docs['planned_P1_E_actions.json']['actions']:a['request_sha256']=e['request_sha256']
        self.f.rehash();self.p=dict(self.b.p,completed_depth=8)
        self.resources={k:v for k,v in self.b.resources.items() if k!='host_capture_bytes'}

    def compile(self,arm='historical_1'):
        return compile_p1_qa(self.f.docs,identity=self.f.identity,action_key=['P1-E','g','q4',arm],
                              comparison_profile=self.p,resources=self.resources)

    def test_fixed_arm_preserves_prompt_and_disables_publication(self):
        before=deepcopy(self.f.docs);o=self.compile()
        self.assertEqual(before,self.f.docs);self.assertEqual(o['expected_build_id'],'b0')
        self.assertEqual(o['request']['token_ids'],before['group-00.json']['requests'][4]['request']['token_ids'])
        self.assertEqual(o['first_reuse_layer'],9);self.assertEqual(o['repair_ratio'],.15)
        self.assertEqual(o['repair_metric'],'legacy_normalized_kv')
        self.assertFalse(o['publication_allowed']);self.assertFalse(o['source_selection_evaluated'])

    def test_dense_has_no_source_dependency(self):
        o=self.compile('dense');self.assertIsNone(o['expected_build_id']);self.assertIsNone(o['first_reuse_layer'])

    def test_s0_is_fixed_independent_source_not_candidate_reranking(self):
        self.assertEqual(self.compile('S0')['expected_build_id'],'b4')

    def test_wrong_depth_and_resources_fail(self):
        self.p['completed_depth']=2;self.assertRaises(ValueError,self.compile)
        self.p['completed_depth']=8;self.resources['host_capture_bytes']=100
        self.assertRaises(ValueError,self.compile)

    def test_teacher_override_refused_even_when_rehashed(self):
        e=self.f.docs['group-00.json']['requests'][4];e['request']['teacher_token_ids']=[1]
        e['request_sha256']=digest_json(e['request'])
        for a in self.f.docs['planned_P1_E_actions.json']['actions']:a['request_sha256']=e['request_sha256']
        self.f.rehash();self.assertRaisesRegex(ValueError,'free-generation',self.compile)

    def context(self,o):
        c=NS(request=o['request'],current_completed_depth=0,cached_prefix_tokens=0,prepared={},committed={},frozen={},
             finished=False,closed=False,repair_ratio=.15,source_capture_v2=None,source_consumption_v2=None,
             arrival_ns=1,adapter=NS(native_repair_metric='normalized_kv_deviation',path='legacy_multicheckpoint',
             depths=(1,2,4,5,8),costs=None,provenance={'model_signature':'a'*64},check_deadline=Mock()))
        s=NS(lock=nullcontext(),_snapshot=Mock(),config={'model_signature':'a'*64},end_request=Mock(),
             content_key=lambda tokens:digest_json(tokens))
        c.request=bind_request_content_keys(o['request'],s)
        snap=NS(request_id=c.request['request_id']);return c,s,snap

    def test_missing_source_is_failure_not_dense_fallback(self):
        o=self.compile();c,s,snap=self.context(o);c.finish=Mock()
        with self.assertRaisesRegex(ValueError,'silently fall back'):
            execute_p1_qa(c,s,snap,o,expected_operation_sha256=o['operation_sha256'])
        c.finish.assert_not_called()

    def test_dense_finishes_once_and_closes_no_publication(self):
        o=self.compile('dense');c,s,snap=self.context(o)
        def finish(cb):c.finished=True;cb();return {'fixture_only':True}
        c.finish=Mock(side_effect=finish);c.close=Mock()
        audit=execute_p1_qa(c,s,snap,o,expected_operation_sha256=o['operation_sha256'])
        c.finish.assert_called_once();c.close.assert_called_once();s.end_request.assert_called_once_with(snap)
        self.assertEqual(audit['status'],'COMPLETED');self.assertFalse(audit['production_reuse_commit_observed'])

    def test_dense_model_mismatch_refused(self):
        o=self.compile('dense');c,s,snap=self.context(o);c.adapter.provenance['model_signature']='b'*64
        self.assertRaises(ValueError,execute_p1_qa,c,s,snap,o,expected_operation_sha256=o['operation_sha256'])

    def test_fixed_preparation_has_explicit_nonproduction_marker(self):
        o=self.compile();c,s,snap=self.context(o);c.current_completed_depth=8
        from probekv.source_comparison_v2 import ComparisonProfileBindingV2
        bridge=P1FixedSourceSession.__new__(P1FixedSourceSession);bridge.context=c
        bridge.store=s
        bridge.comparison=NS(profile=ComparisonProfileBindingV2(**o['comparison_profile']));bridge._prepare_bound=Mock()
        r=NS(segment_id='C',winner_source_id='s',eligible_ids=('s',),available_ids=('s',),compared_ids=('s',))
        bridge.prepare_prescribed(r,o,'s')
        marker=bridge._prepare_bound.call_args.kwargs['controlled_recipe']
        self.assertFalse(marker['production_execution_allowed'])
        r.compared_ids=('s','other');self.assertRaises(ValueError,bridge.prepare_prescribed,r,o,'s')

    def test_dense_failure_preserves_audit_and_releases_snapshot(self):
        from probekv.native_p0_request_v2 import P0RequestFailure
        o=self.compile('dense');c,s,snap=self.context(o);c.close=Mock();c.finish=Mock(side_effect=RuntimeError('fixture failure'))
        with self.assertRaises(P0RequestFailure) as raised:
            execute_p1_qa(c,s,snap,o,expected_operation_sha256=o['operation_sha256'])
        self.assertEqual(raised.exception.audit['status'],'FAILED');c.close.assert_called_once();s.end_request.assert_called_once()

    def source_flow(self,ready,audit_failure=False):
        """Mock state plumbing only, deliberately not a model execution."""
        from dataclasses import dataclass
        @dataclass
        class Receipt: source_id:str='s'
        o=self.compile();c,s,snap=self.context(o);snap.snapshot_id='snap'
        c.supports={'C':{9:[1]}};c.engine=NS(session=NS(commits={}))
        c.advance_to_depth=Mock(side_effect=lambda d:setattr(c,'current_completed_depth',d))
        c.configure_cuda_comparison_v2=Mock();c.finish_selection=Mock()
        c.ready_for_final_commit=Mock(return_value=(ready,'mask'));c.close=Mock()
        def finish(cb):c.finished=True;cb();return {'fixture_only':True}
        c.finish=Mock(side_effect=finish)
        issuer=Mock();issuer.compare_native.return_value=Receipt()
        bridge=Mock();bridge.audit.return_value={'CPU_fixture_only':True}
        if audit_failure:bridge.audit.side_effect=RuntimeError('audit fixture failure')
        def commit(ctx,sid):ctx.committed[sid]=9;ctx.engine.session.commits[sid]=NS(source_id='s')
        with patch('probekv.p1_qa_dispatch_v2.verify_qa_source',return_value={'source_id':'s'}), \
             patch('probekv.native_p0_operation_v2.validate_p0_comparison_action'), \
             patch('probekv.p1_qa_dispatch_v2.ComparisonSessionV2',return_value=issuer), \
             patch('probekv.p1_qa_dispatch_v2.P1FixedSourceSession',return_value=bridge), \
             patch('probekv.p0_lineage_control_v2._commit_recipe',side_effect=commit):
            try:return execute_p1_qa(c,s,snap,o,expected_operation_sha256=o['operation_sha256'],source_receipt={}),c
            finally:
                c.close.assert_called_once();issuer.close.assert_called_once();s.end_request.assert_called_once()

    def test_source_flow_preserves_fixed_boundary_and_single_finish(self):
        audit,c=self.source_flow({'C':9});c.advance_to_depth.assert_called_once_with(8);c.finish.assert_called_once()
        self.assertEqual(audit['committed_boundaries'],{'C':9});self.assertFalse(audit['source_publication_allowed'])

    def test_moved_boundary_cannot_produce_dense_qa_result(self):
        from probekv.native_p0_request_v2 import P0RequestFailure
        with self.assertRaises(P0RequestFailure) as raised:self.source_flow({'C':10})
        self.assertIsNone(raised.exception.audit['answer']);self.assertEqual(raised.exception.audit['status'],'FAILED')

    def test_consumption_audit_failure_still_closes_and_ends_snapshot(self):
        from probekv.native_p0_request_v2 import P0RequestFailure
        with self.assertRaises(P0RequestFailure) as raised:self.source_flow({'C':9},audit_failure=True)
        self.assertFalse(raised.exception.audit['cleanup']['snapshot_retained'])
