"""CPU-only boundary contracts; no mocked result is GPU qualification."""
from copy import deepcopy
from unittest.mock import Mock
import unittest

from tests import test_p1_qa_dispatch_v2 as fixtures
from probekv.p1_boundary_diagnostic_v2 import (compile_boundary_operation,
    verify_boundary_operation, validate_layer_recipe, BoundarySourceSession, commit_boundary_diagnostic)
from probekv.p1_build_dispatch_v2 import _seal


class BoundaryDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.QADispatchTests(); self.f.setUp()

    def operation(self, mode='source_qa', arm='historical_1'):
        return compile_boundary_operation(self.f.compile(arm), mode=mode)

    def test_new_operation_preserves_legacy_and_prompt(self):
        old = self.f.compile(); before = deepcopy(old)
        o = compile_boundary_operation(old, mode='source_qa')
        self.assertEqual(old, before); self.assertEqual(old['first_reuse_layer'], 9)
        self.assertEqual(o['first_reuse_layer'], 3); self.assertEqual(o['completed_depth'], 2)
        self.assertEqual(o['request']['token_ids'], old['request']['token_ids'])
        self.assertEqual(o['expected_build_id'], old['expected_build_id'])
        self.assertEqual(o['repair_ratio'], .15); self.assertFalse(o['source_selection_evaluated'])
        verify_boundary_operation(o, old)

    def test_teacher_is_fixed_prompt_tail_not_answers(self):
        a = self.operation('dense_teacher', 'dense'); b = self.operation('source_teacher_r1')
        self.assertEqual(a['request']['teacher_token_ids'], b['request']['teacher_token_ids'])
        self.assertEqual(b['request']['teacher_token_ids'], b['request']['token_ids'][-31:])
        self.assertEqual(b['request']['correctness_repair_ratio'], 1.)
        self.assertTrue(a['request']['capture_logits']); self.assertFalse(a['publication_allowed'])

    def test_teacher_and_qa_stop_contracts_do_not_mix(self):
        from probekv.v8_schema10_native_adapter import validate_native_sampling_request
        old=self.f.compile(); old['request']['answer_boundary_contract']='qa_next_question_boundary_v1'
        old.pop('operation_sha256'); _seal(old,'operation_sha256')
        teacher=compile_boundary_operation(old,mode='source_teacher_r1')
        qa=compile_boundary_operation(old,mode='source_qa')
        self.assertNotIn('answer_boundary_contract',teacher['request'])
        self.assertEqual(qa['request']['answer_boundary_contract'],'qa_next_question_boundary_v1')
        validate_native_sampling_request(teacher['request']);validate_native_sampling_request(qa['request'])

    def test_r1_requires_first_source_not_result_selected(self):
        with self.assertRaisesRegex(ValueError, 'first historical'):
            self.operation('source_teacher_r1', 'S0')

    def test_dense_and_source_roles_cannot_be_swapped(self):
        with self.assertRaises(ValueError): self.operation('dense_qa')
        with self.assertRaises(ValueError): self.operation('source_qa', 'dense')

    def test_only_fit_request_and_supported_mode(self):
        old = self.f.compile(); old['request']['partition_role'] = 'validation'
        old.pop('operation_sha256'); _seal(old, 'operation_sha256')
        with self.assertRaises(ValueError): compile_boundary_operation(old, mode='source_qa')
        with self.assertRaises(ValueError): self.operation('source_qa_lowratio')

    def test_resealed_lower_ratio_or_other_depth_still_rejected(self):
        old = self.f.compile()
        for field, value in [('first_reuse_layer', 4), ('repair_ratio', .1)]:
            o = compile_boundary_operation(old, mode='source_qa'); o[field] = value
            o.pop('operation_sha256'); _seal(o, 'operation_sha256')
            with self.assertRaises(ValueError): verify_boundary_operation(o, old)

    def layers(self, full=False):
        allrows = list(range(10)); reduced = allrows if full else [0,1,4,8,9]
        return [dict(layer=l, active_before=allrows if l<=3 else reduced,
                     active_after=allrows if l<3 else reduced) for l in range(1,6)]

    def recipe(self, layers, support=(4,)):
        return validate_layer_recipe(layers, prompt_count=10, target_positions=range(2,8),
                                     support=support, total_layers=5, source=True)

    def test_third_layer_projection_full_then_selective(self):
        self.assertTrue(self.recipe(self.layers())['block3_full_projection_accounted'])

    def test_missing_duplicate_or_late_layers_rejected(self):
        with self.assertRaises(ValueError): self.recipe(self.layers()[1:])
        rows=self.layers(); rows[2]['active_after']=list(range(10))
        with self.assertRaises(ValueError): self.recipe(rows)
        rows=self.layers(); rows[2]['layer']=4
        with self.assertRaises(ValueError): self.recipe(rows)

    def test_suffix_and_non_target_rows_cannot_drop(self):
        rows=self.layers(); rows[3]['active_after']=[0,1,4,8]
        with self.assertRaises(ValueError): self.recipe(rows)

    def test_r1_full_rows_all_layers(self):
        self.assertTrue(self.recipe(self.layers(full=True), range(2,8))['layers_once'])

    def test_diagnostic_bridge_cannot_commit_as_ordinary_production(self):
        bridge=BoundarySourceSession.__new__(BoundarySourceSession)
        bridge.controlled_recipe_parents={'C': {'production_execution_allowed':False}}
        from types import SimpleNamespace as NS
        from contextlib import nullcontext
        bridge.store=NS(lock=nullcontext())
        with self.assertRaisesRegex(ValueError, 'production'):
            bridge.assert_can_commit('C')

    def test_commit_rejects_layer9_or_unrelated_bridge(self):
        c=Mock(current_completed_depth=8, prepared={'C':object()}, committed={})
        with self.assertRaises(ValueError): commit_boundary_diagnostic(c,self.operation())


if __name__ == '__main__': unittest.main()
