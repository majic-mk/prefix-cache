"""CPU contract tests, never real-model qualification."""
from copy import deepcopy
from unittest.mock import patch
import unittest

from tests import test_p1_qa_dispatch_v2 as fixtures
from probekv.p1_build_dispatch_v2 import _seal
from probekv.p1_boundary_diagnostic_v2 import compile_boundary_operation
from probekv.p1_layer3_matrix_v2 import (compile_layer3_operation, verify_layer3_operation,
    layer3_binding, require_layer3_gate, validate_layer_recipe)


class Layer3MatrixTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.QADispatchTests(); self.f.setUp()

    def old(self, arm='historical_1', role='fit', stage='P1-E'):
        old = self.f.compile('historical_1' if arm in ('M1', 'E') else arm)
        old['request']['partition_role'] = role
        old['stage'] = stage; old['arm'] = arm
        old['expected_source_generation'] = None if arm == 'dense' else int(arm == 'M1')
        old.pop('operation_sha256'); _seal(old, 'operation_sha256')
        return old

    def test_validation_matrix_is_new_entry_not_relaxed_old_diagnostic(self):
        old = self.old(role='validation'); before = deepcopy(old)
        with self.assertRaises(ValueError): compile_boundary_operation(old, mode='source_qa')
        op = compile_layer3_operation(old, mode='source_qa')
        self.assertEqual(before, old)
        self.assertEqual(op['kind'], 'P1_layer3_operation_v2')
        self.assertEqual(op['first_reuse_layer'], 3)
        self.assertEqual(op['comparison_profile']['completed_depth'], 2)
        self.assertEqual(op['original_request_sha256'], old['original_request_sha256'])
        self.assertEqual(op['expected_build_id'], old['expected_build_id'])
        self.assertEqual(op['request']['token_ids'], old['request']['token_ids'])
        self.assertEqual(op['request']['segments'], old['request']['segments'])
        verify_layer3_operation(op, old)

    def test_mixed_and_exact_preserve_provenance_and_recipe(self):
        for arm in ('E', 'M1'):
            old = self.old(arm, 'validation', 'P1-M')
            op = compile_layer3_operation(old, mode='source_qa')
            self.assertEqual(op['stage'], 'P1-M')
            self.assertEqual(op['expected_source_generation'], int(arm == 'M1'))
            self.assertEqual(op['expected_birth_request'], old['expected_birth_request'])
            self.assertEqual(op['repair_ratio'], .15)
            self.assertFalse(op['publication_allowed'])
            self.assertFalse(op['source_selection_evaluated'])
            self.assertFalse(op['production_reuse_commit_observed'])

    def test_validation_never_becomes_numeric_calibration(self):
        for mode, arm in (('dense_teacher','dense'), ('source_teacher_r1','historical_1'),
                          ('source_greedy_r1','historical_1')):
            with self.assertRaisesRegex(ValueError, 'fit only'):
                compile_layer3_operation(self.old(arm, 'validation'), mode=mode)

    def test_prescribed_mixed_r1_does_not_substitute_exact(self):
        for mode in ('source_teacher_r1', 'source_greedy_r1'):
            op = compile_layer3_operation(self.old('M1', stage='P1-M'), mode=mode)
            self.assertEqual(op['repair_ratio'], 1.)
            with self.assertRaises(ValueError):
                compile_layer3_operation(self.old('E', stage='P1-M'), mode=mode)

    def test_resealed_overrides_do_not_verify(self):
        old = self.old()
        for field, value in [('repair_ratio', .1), ('first_reuse_layer', 9), ('stage', 'P1-M'),
                             ('publication_allowed', True), ('expected_source_generation', 1)]:
            op = compile_layer3_operation(old, mode='source_qa'); op[field] = value
            op.pop('operation_sha256'); _seal(op, 'operation_sha256')
            with self.assertRaises(ValueError): verify_layer3_operation(op, old)

    def test_qa_retains_original_sampling_and_answers(self):
        old = self.old(); op = compile_layer3_operation(old, mode='source_qa')
        for key, value in old['request'].items():
            if key != 'request_id': self.assertEqual(op['request'][key], value)
        self.assertNotIn('teacher_token_ids', op['request'])

    def test_teacher_uses_prompt_tail_not_reference(self):
        op = compile_layer3_operation(self.old(), mode='source_teacher_r1')
        self.assertEqual(op['request']['teacher_token_ids'], op['request']['token_ids'][-31:])
        self.assertNotIn('answer_boundary_contract', op['request'])

    def gate(self, stage='P1-E', complete=True):
        return dict(status='PASSED', stage=stage, layer3_binding=layer3_binding(),
            fixed15_boundary_QA_allowed=complete, evidence={'dense_teacher': {'directory':'CPU_fixture'}})

    def test_gate_cannot_cross_unlock_exact_and_mixed(self):
        op = compile_layer3_operation(self.old('M1', stage='P1-M'), mode='source_qa')
        gate = self.gate()
        with patch('probekv.p1_layer3_matrix_v2.evaluate_layer3_gate', return_value=gate):
            with self.assertRaisesRegex(ValueError, 'phase-specific'):
                require_layer3_gate(op, gate, actual_binding={})

    def test_teacher_only_gate_cannot_unlock_fixed15(self):
        op = compile_layer3_operation(self.old(), mode='source_qa'); gate = self.gate(complete=False)
        with patch('probekv.p1_layer3_matrix_v2.evaluate_layer3_gate', return_value=gate):
            with self.assertRaisesRegex(ValueError, 'greedy'):
                require_layer3_gate(op, gate, actual_binding={})

    def test_gate_rechecks_real_environment_not_passed_boolean(self):
        op = compile_layer3_operation(self.old(), mode='source_qa'); gate = self.gate()
        keys = ('code_commit','runtime_digest','patch_sha256','model_signature','tokenizer_hash','gpu_uuid','instance_id')
        binding = {k:'actual' for k in keys}
        with patch('probekv.p1_layer3_matrix_v2.evaluate_layer3_gate', return_value=gate), \
             patch('probekv.p1_layer3_matrix_v2.read_layer3_evidence', return_value={'manifest':{'binding':binding}}):
            require_layer3_gate(op, gate, actual_binding=binding)
            for key in keys:
                with self.assertRaisesRegex(ValueError, 'environment'):
                    require_layer3_gate(op, gate, actual_binding=dict(binding, **{key:'changed'}))

    def test_unknown_stage_role_or_mode_rejected(self):
        for stage, role, mode in [('P2','fit','source_qa'), ('P1-E','locked','source_qa'),
                                 ('P1-E','fit','source_lowratio')]:
            with self.assertRaises(ValueError):
                compile_layer3_operation(self.old(role=role, stage=stage), mode=mode)

    def test_layers_first_two_full_and_suffix_preserved(self):
        layers = [dict(layer=i, active_before=list(range(6)) if i<=3 else [0,2,5],
                       active_after=list(range(6)) if i<3 else [0,2,5]) for i in range(1,5)]
        self.assertTrue(validate_layer_recipe(layers, prompt_count=6, target_positions=[1,2,3,4],
            support=[2], total_layers=4, source=True)['block3_full_projection_accounted'])
        layers[-1]['active_after'] = [0,2]
        with self.assertRaises(ValueError):
            validate_layer_recipe(layers, prompt_count=6, target_positions=[1,2,3,4],
                support=[2], total_layers=4, source=True)


if __name__ == '__main__': unittest.main()
