"""CPU contract tests; no GPU numerical claims."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import test_p1_input_consumer_v2 as input_fixtures
from probekv.p0_natural_input_v2 import compile_natural_p0_controls, bind_natural_p0_manifest
from probekv.source_comparison_v2 import runtime_binding_digest, scorer_digest
from probekv.v8_schema10_execution import digest_json


class NaturalP0Tests(unittest.TestCase):
    def setUp(self):
        fixture = input_fixtures.InputConsumerTests(); fixture.setUp()
        self.docs, self.identity = fixture.docs, fixture.identity
        self.args = dict(identity=self.identity, build_id='b0', teacher_token_ids=[9]*31,
            comparison_profile=dict(model_signature=self.identity['model_signature'],
                provenance_policy='ALLOW_MIXED_G1', completed_depth=1, trim_ratio=.15,
                tau_reuse=0., tau_add=0., repair_policy_digest='e'*64,
                runtime_digest=runtime_binding_digest(), scoring_function_digest=scorer_digest()),
            upper_seconds_by_arm=dict(off=180,on=180,birth=180))

    def compile(self):
        return compile_natural_p0_controls(self.docs, **self.args)

    def test_prompt_unchanged_reference_not_published_and_budget_explicit(self):
        before = deepcopy(self.docs)
        controls = self.compile()
        self.assertEqual(self.docs, before)
        q = before['group-00.json']['requests'][0]['request']
        diag = controls['exact_controls'][0]['request']
        birth = controls['birth_controls'][0]['request']
        for req in (diag,birth):
            self.assertEqual(req['token_ids'], q['token_ids'])
            self.assertEqual(req['segments'], q['segments'])
        self.assertEqual(diag['max_new_tokens'],32)
        self.assertEqual(birth['max_new_tokens'],1)
        self.assertNotIn('teacher_token_ids',birth)
        self.assertNotIn('capture_logits',birth)
        self.assertEqual(controls['maximum_action_seconds'],540)
        self.assertEqual(controls['maximum_actions'],3)
        for key in ('P1_execution_allowed','gpu_execution_allowed','paper_evidence',
                    'automatic_rental_allowed','metadata_reranking_enabled'):
            self.assertFalse(controls[key])

    def test_s0_and_history_separate_ids_no_answer_conditioning(self):
        first = self.compile()
        self.args['build_id']='b4'
        second = self.compile()
        self.assertNotEqual(first['controls_sha256'], second['controls_sha256'])
        self.assertNotIn('question', second['birth_controls'][0]['request'])

    def test_legacy_namespace_and_prefetch_are_explicit_p0_overrides(self):
        q = self.docs['group-00.json']['requests'][0]['request']
        q['prefetch_window'] = 1
        q['segments'][0]['content_key'] = 'old-plan-namespace'
        self.docs['group-00.json']['requests'][0]['request_sha256'] = digest_json(q)
        plan = self.docs['planned_P1_E_actions.json']
        plan['builds'][0]['request_sha256'] = digest_json(q)
        plan['builds_sha256'] = digest_json(plan['builds'])
        controls = self.compile()
        self.assertEqual(controls['execution_overrides']['original_prefetch_window'],1)
        self.assertEqual(controls['execution_overrides']['original_content_keys']['C'],'old-plan-namespace')
        self.assertEqual(controls['birth_controls'][0]['request']['prefetch_window'],0)
        self.assertNotIn('content_key',controls['birth_controls'][0]['request']['segments'][0])

    def test_bad_or_unfrozen_inputs_fail_before_model(self):
        self.docs['group-00.json']['requests'][0]['request']['token_ids'][5] = 800
        with self.assertRaisesRegex(ValueError,'digest'): self.compile()

    def test_teacher_diagnostic_drops_qa_stop_only_birth_preserves_it(self):
        from probekv.p0_launch_recipe_v2 import _request
        q = self.docs['group-00.json']['requests'][0]['request']
        q.update(answer_boundary_contract='qa_next_question_boundary_v1',
                 answers=['reference'], quality_contract={'max_answer_f1_drop': .02})
        self.docs['group-00.json']['requests'][0]['request_sha256'] = digest_json(q)
        plan = self.docs['planned_P1_E_actions.json']
        plan['builds'][0]['request_sha256'] = digest_json(q)
        plan['builds_sha256'] = digest_json(plan['builds'])
        before = deepcopy(self.docs)
        controls = self.compile()
        diagnostic = controls['exact_controls'][0]['request']
        birth = controls['birth_controls'][0]['request']
        _request(diagnostic)
        _request(birth)
        self.assertEqual(self.docs, before)
        for field in ('answer_boundary_contract', 'answers', 'quality_contract'):
            self.assertNotIn(field, diagnostic)
            self.assertEqual(birth[field], q[field])
            self.assertEqual(controls['execution_overrides']['diagnostic_removed_qa_fields'][field], q[field])
        self.assertEqual(diagnostic['token_ids'], q['token_ids'])
        self.assertEqual(diagnostic['segments'], q['segments'])

    def test_no_implicit_teacher_budget_or_runtime(self):
        for key,value in (('teacher_token_ids',[9]*30), ('build_id','unknown'),
                          ('upper_seconds_by_arm',dict(off=181,on=180,birth=180))):
            with self.subTest(key=key):
                args = deepcopy(self.args); args[key] = value
                with self.assertRaises(ValueError): compile_natural_p0_controls(self.docs, **args)
        self.args['comparison_profile']['runtime_digest'] = '0'*64
        with self.assertRaisesRegex(ValueError,'current code'): self.compile()

    def test_bind_delegates_actual_preflight_and_keeps_input_graph(self):
        controls = self.compile()
        args = dict(limits=dict(maximum_actions=3),numerical_policy=dict(exact=dict(
            relative_l2_limit=1e-4,minimum_positions=32,require_predicted_token_ids_equal=True)))
        # Only test wiring; real assets still checked by the launch builder.
        with patch('probekv.p0_launch_recipe_v2.build_controlled_p0_manifest',
                   return_value=dict(manifest=dict(manifest_sha256='old'),preflight={})) as call:
            result = bind_natural_p0_manifest(controls, **args)
            self.assertEqual(call.call_args.kwargs['birth_controls'],controls['birth_controls'])
            self.assertEqual(result['manifest']['natural_input_binding']['input_graph_sha256'],
                             controls['input_graph_sha256'])
            manifest = result['manifest']
            self.assertEqual(manifest['manifest_sha256'],digest_json({k:v for k,v in manifest.items() if k!='manifest_sha256'}))

    def test_modified_controls_and_weakened_policy_rejected(self):
        controls = self.compile()
        args = dict(limits=dict(maximum_actions=3),numerical_policy=dict(exact=dict(
            relative_l2_limit=.1,minimum_positions=32,require_predicted_token_ids_equal=True)))
        with self.assertRaisesRegex(ValueError,'strict'): bind_natural_p0_manifest(controls,**args)
        controls['birth_controls'][0]['request']['token_ids'][0]=8
        with self.assertRaisesRegex(ValueError,'unaltered'): bind_natural_p0_manifest(controls,**args)


if __name__ == '__main__': unittest.main()
