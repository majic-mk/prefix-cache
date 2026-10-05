"""CPU raw-evidence fixtures only; these tests never execute a native model."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from probekv.p0_decode_evidence_v2 import DecodeInputRecorderV2
from probekv.p0_evidence_v2 import P0EvidenceWriter
from probekv.p0_exact_pair_v2 import validate_exact_pairs, evaluate_exact_pairs
from probekv.source_manifest_v2 import request_input_digest
from probekv.v8_schema10_execution import digest_json


class ExactCapturePairTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / 'batch'
        self.binding = {k: 'cpu-fixture-only' for k in ('code_commit', 'runtime_digest', 'patch_sha256',
                        'model_signature', 'tokenizer_hash', 'gpu_uuid', 'instance_id')}
        self.manifest = {'binding': self.binding, 'jobs': [], 'exact_capture_pairs': [dict(
            pair_id='T20_pair', reference_action_id='ref', candidate_action_id='candidate',
            relative_l2_limit=0., minimum_positions=3, require_predicted_token_ids_equal=True)]}
        for name, enabled in (('ref', False), ('candidate', True)):
            request = dict(request_id=name, token_ids=[11, 12, 13], max_new_tokens=3,
                           teacher_token_ids=[9, 8], capture_logits=True,
                           segments=[dict(segment_id='C', positions=[1, 2], token_ids=[12, 13])])
            self.manifest['jobs'].append(dict(action_id=name, operation='exact_capture_control',
                capture_enabled=enabled, target_ids=['C'], request=request, request_sha256=digest_json(request)))

    def audit(self, job, predictions=(1, 0, 1)):
        request = job['request']; positions = [0, 1, 2]
        recorder = DecodeInputRecorderV2(request, cached_prefix_tokens=0)
        for i, predicted in enumerate(predictions):
            recorder.append(predicted, fed_token=request['teacher_token_ids'][i-1] if i else None)
        trace = recorder.finish(logit_rows=3)
        recipe = dict(kind='exact_full_row_resumable_control',
            input_digest=request_input_digest(tuple(request['token_ids']), tuple(positions)),
            model_signature=self.binding['model_signature'], num_layers=3, prefix_tokens=0, source_imports=[],
            target_positions={'C': [1, 2]}, layers=[dict(layer=i, active_before=positions[:],
                active_after=positions[:], union_mask_digest=hashlib.sha256(
                    json.dumps(positions, separators=(',', ':')).encode('ascii')).hexdigest()) for i in range(1, 4)])
        audit = dict(kind='p0_exact_capture_control', status='COMPLETED', request_id=request['request_id'],
            capture_enabled=job['capture_enabled'],
            answer=dict(token_ids=list(predictions), generation_mode='teacher_forced_logit_diagnostic',
                        decode_input_trace_v2=trace),
            cleanup={'passed': True}, publication={'status': 'DIAGNOSTIC_NOT_PUBLISHED'}, extra_forward_count=0,
            execution=dict(recipe=recipe, recipe_sha256=digest_json(recipe)), capture=None)
        if job['capture_enabled']:
            audit['capture'] = dict(kind='actual_layer_target_capture_v2', layers_recorded=3, layers_submitted=3,
                target_reservations=['C'], rejected_targets={}, prefix_shadow_created=False,
                publication_performed=False, extra_forward_count=0,
                events=[dict(event='target_capture_after_layer', layer=i, projected_rows=3,
                             effective_current_rows=3, captured_rows=2) for i in range(1, 4)])
            proof = dict(request_id=request['request_id'], input_digest=recipe['input_digest'],
                model_signature=self.binding['model_signature'], positions=[1, 2], origin='EXACT_CONTEXT',
                generation=0, expected_layers=3, ledger_digest='b' * 64)
            audit['captured_targets'] = {'C': dict(proof=proof, artifact_digest='c' * 64,
                storage_audit=dict(parent_owned_kv_bytes=0, prefix_shadow_bytes=0,
                                   target_kv_bytes=96, selection_state_bytes=16),
                publication_state='VALIDATED_CANDIDATE_NOT_PUBLISHED')}
        return audit

    def write(self, *, mutate=None, predictions=None, logits=None, origins=None):
        writer = P0EvidenceWriter(self.root, binding=self.binding, manifest=self.manifest)
        for job in self.manifest['jobs']:
            name = job['action_id']
            audit = self.audit(job, predictions=(predictions or {}).get(name, (1, 0, 1)))
            if mutate: mutate(name, audit)
            writer.write_action(name, audit=audit,
                logits=(logits or {}).get(name, [torch.ones(2) for _ in range(3)]),
                origin=(origins or {}).get(name, 'cpu_fixture'))

    def evaluate(self, completed=('ref', 'candidate')):
        return evaluate_exact_pairs(self.root, self.manifest, completed)[0]

    def test_valid_cpu_fixture_checks_never_qualify_native_or_P1(self):
        before = copy.deepcopy(self.manifest)
        validate_exact_pairs(self.manifest)
        self.write(); result = self.evaluate()
        self.assertEqual(result['status'], 'CPU_ONLY')
        self.assertTrue(result['scoped_checks_passed'])
        for key in ('native_runtime_qualified', 'P0_qualified', 'P1_execution_allowed', 'real_cuda_T20_passed'):
            self.assertFalse(result[key])
        self.assertEqual(self.manifest, before)

    def test_absent_pairs_return_empty_without_reading_evidence(self):
        self.assertEqual(validate_exact_pairs({'jobs': []}), [])
        self.assertEqual(evaluate_exact_pairs(self.root, {'jobs': []}, []), [])

    def test_unfinished_pair_is_pending_without_opening_any_files(self):
        result = self.evaluate(('ref',))
        self.assertEqual(result['status'], 'PENDING')
        self.assertEqual(result['pending_action_ids'], ['candidate'])
        self.assertFalse(self.root.exists())

    def test_strict_pair_contract_rejects_extra_missing_bad_or_duplicate_fields(self):
        pair = self.manifest['exact_capture_pairs'][0]
        for change in ({'pair_id': '../escape'}, {'relative_l2_limit': float('nan')},
                       {'relative_l2_limit': -1}, {'relative_l2_limit': True},
                       {'minimum_positions': 4}, {'minimum_positions': True},
                       {'require_predicted_token_ids_equal': 1}, {'new_policy': True},
                       {'candidate_action_id': 'ref'}):
            with self.subTest(change=change):
                other = copy.deepcopy(self.manifest); other['exact_capture_pairs'] = [{**pair, **change}]
                with self.assertRaises(ValueError): validate_exact_pairs(other)
        self.manifest['exact_capture_pairs'].append(copy.deepcopy(pair))
        with self.assertRaises(ValueError): validate_exact_pairs(self.manifest)

    def test_request_teacher_sampling_and_target_changes_are_not_capture_only(self):
        for key, value in (('teacher_token_ids', [9, 7]), ('max_new_tokens', 4), ('seed', 3)):
            with self.subTest(key=key):
                other = copy.deepcopy(self.manifest)
                job = other['jobs'][1]; job['request'][key] = value
                job['request_sha256'] = digest_json(job['request'])
                with self.assertRaises(ValueError): validate_exact_pairs(other)
        self.manifest['jobs'][1]['capture_enabled'] = False
        with self.assertRaises(ValueError): validate_exact_pairs(self.manifest)

    def test_recipe_digest_tampering_fails_with_preserved_reason(self):
        self.write(mutate=lambda name, audit: audit['execution'].update(recipe_sha256='d' * 64))
        result = self.evaluate()
        self.assertEqual(result['status'], 'FAILED')
        self.assertIn('recipe', result['failure']['detail'])

    def test_identical_hashed_fake_recipes_cannot_omit_prompt_rows(self):
        def mutation(name, audit):
            recipe = audit['execution']['recipe']
            recipe['layers'][0]['active_before'] = [1, 2]
            recipe['layers'][0]['active_after'] = [1, 2]
            audit['execution']['recipe_sha256'] = digest_json(recipe)
        self.write(mutate=mutation)
        self.assertEqual(self.evaluate()['status'], 'FAILED')

    def test_identical_hashed_fake_recipes_cannot_use_prefix_or_source_import(self):
        def mutation(name, audit):
            recipe = audit['execution']['recipe']; recipe['source_imports'] = ['hidden-source']
            audit['execution']['recipe_sha256'] = digest_json(recipe)
        self.write(mutate=mutation)
        self.assertEqual(self.evaluate()['status'], 'FAILED')

    def test_g0_target_must_match_request_positions_and_independent_storage(self):
        def mutation(name, audit):
            if name == 'candidate': audit['captured_targets']['C']['storage_audit']['parent_owned_kv_bytes'] = 96
        self.write(mutate=mutation)
        self.assertEqual(self.evaluate()['status'], 'FAILED')

    def test_extra_forward_or_publication_cannot_be_qualified(self):
        self.write(mutate=lambda name, audit: audit.update(extra_forward_count=1))
        self.assertEqual(self.evaluate()['status'], 'FAILED')

    def test_missing_or_nonconsecutive_capture_layer_fails(self):
        def mutation(name, audit):
            if name == 'candidate': audit['capture']['events'][1]['layer'] = 1
        self.write(mutate=mutation)
        self.assertEqual(self.evaluate()['status'], 'FAILED')

    def test_numeric_failure_survives_successful_recipe_validation(self):
        self.write(logits={'candidate': [torch.zeros(2) for _ in range(3)]})
        result = self.evaluate()
        self.assertEqual(result['status'], 'FAILED')
        self.assertTrue(result['recipe_alignment_verified'])
        self.assertFalse(result['numerical']['numeric_passed'])

    def test_predicted_ids_policy_is_explicit_not_hidden_in_teacher_comparison(self):
        self.write(predictions={'candidate': (0, 0, 0)})
        result = self.evaluate()
        self.assertEqual(result['status'], 'FAILED')
        self.assertTrue(result['numerical']['numeric_passed'])
        self.assertFalse(result['numerical']['predicted_token_ids_identical'])

    def test_record_hash_is_checked_against_event_chain(self):
        self.write()
        record = self.root / 'candidate' / 'record.json'
        record.write_bytes(record.read_bytes() + b' ')
        result = self.evaluate()
        self.assertEqual(result['status'], 'FAILED')
        self.assertIn('digest mismatch', result['failure']['detail'])

    def test_raw_logit_corruption_is_failed_not_pending_or_passed(self):
        self.write()
        path = self.root / 'candidate' / 'logits.npy'
        path.write_bytes(path.read_bytes() + b'corruption')
        result = self.evaluate()
        self.assertEqual(result['status'], 'FAILED')
        self.assertIn('logit digest', result['failure']['detail'])

    def test_actual_control_audits_feed_pair_checker_without_handbuilt_recipe(self):
        # Real CPU fake blocks execute through execute_exact_capture_control;
        # only decode/logits below are fixture stubs, explicitly CPU_ONLY.
        from tests import test_p0_exact_control_v2 as control_tests
        fixture = control_tests.P0ExactControlTests()
        self.addCleanup(fixture.doCleanups)
        contexts = []
        self.manifest['jobs'] = []
        self.binding['model_signature'] = 'model'
        for name, enabled in (('ref', False), ('candidate', True)):
            _, context, model, job = fixture.context(enabled)
            actual_block = model.probekv_advance_prefill
            # The pinned real patch returns dict. This CPU fixture originally
            # returned LayerAdvanceResult directly, bypassing the real adapter's
            # target-active-position mask digest. Exercise its normal dict path.
            model.probekv_advance_prefill = lambda original=actual_block, **kw: asdict(original(**kw))
            context.request['request_id'] = name
            context.request['token_ids'] = list(context.request['token_ids'])
            original_finish = context.finish_from_prefill_hidden
            def finish(hidden, callback, original=original_finish, request=context.request):
                answer = original(hidden, callback)
                recorder = DecodeInputRecorderV2(request, cached_prefix_tokens=0)
                predictions = [1, 0, 1]
                for i, predicted in enumerate(predictions):
                    recorder.append(predicted, fed_token=request['teacher_token_ids'][i-1] if i else None)
                answer.update(token_ids=predictions, generation_mode='teacher_forced_logit_diagnostic',
                              decode_input_trace_v2=recorder.finish(logit_rows=3))
                return answer
            context.finish_from_prefill_hidden = finish
            job.update(action_id=name, request_sha256=digest_json(context.request))
            contexts.append((context, model, job))
            self.manifest['jobs'].append(job)
        validate_exact_pairs(self.manifest)
        writer = P0EvidenceWriter(self.root, binding=self.binding, manifest=self.manifest)
        for context, model, job in contexts:
            audit = fixture.execute(context, job)
            self.assertEqual(model.real_block_calls, 3)
            writer.write_action(job['action_id'], audit=audit,
                                logits=[torch.ones(2) for _ in range(3)], origin='cpu_fixture')
        result = self.evaluate()
        self.assertEqual(result['status'], 'CPU_ONLY', result)
        self.assertFalse(result['real_cuda_T20_passed'])


if __name__ == '__main__':
    unittest.main()
