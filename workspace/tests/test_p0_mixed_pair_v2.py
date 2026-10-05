"""Raw CPU evidence fixtures only; no native/GPU numerical qualification."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from probekv.p0_decode_evidence_v2 import DecodeInputRecorderV2
from probekv.p0_diagnostic_context_v2 import P0DiagnosticContextV2
from probekv.p0_evidence_v2 import P0EvidenceWriter
from probekv.p0_mixed_pair_v2 import validate_mixed_pairs, evaluate_mixed_pairs
from probekv.source_manifest_v2 import request_input_digest
from probekv.v8_schema10_execution import digest_json


def target_digest(layers):
    digest = hashlib.sha256()
    for layer, pair in enumerate(layers, 1):
        digest.update(str((layer, tuple(pair[0].shape), 'BF16 pre-RoPE')).encode('ascii'))
        for tensor in pair:
            digest.update(memoryview(tensor.view(torch.uint8).numpy()).cast('B'))
    return digest.hexdigest()


class MixedPairTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / 'batch'
        self.binding = {k: 'cpu-fixture-only' for k in ('code_commit', 'runtime_digest', 'patch_sha256',
            'model_signature', 'tokenizer_hash', 'gpu_uuid', 'instance_id')}
        self.manifest = dict(binding=self.binding, jobs=[], mixed_reference_pairs=[dict(
            pair_id='T21_pair', reference_action_id='reference', candidate_action_id='candidate',
            relative_l2_limit=0., minimum_positions=3, require_predicted_token_ids_equal=True,
            target_relative_l2_limit=0., target_absolute_max_limit=0., target_read_bytes=4096)])
        for name, operation in (('reference', 'explicit_mixed_reference'), ('candidate', 'mixed_sparse_control')):
            request = dict(request_id=name, token_ids=[11, 12, 13, 14, 15, 16],
                max_new_tokens=3, teacher_token_ids=[9, 8], capture_logits=True,
                segments=[dict(segment_id='U', positions=[0, 1], token_ids=[11, 12]),
                          dict(segment_id='T', positions=[3, 4], token_ids=[14, 15])])
            self.manifest['jobs'].append(dict(action_id=name, operation=operation,
                request=request, request_sha256=digest_json(request),
                target_id='T', upstream_segment_id='U',
                source=dict(source_id='source1', artifact_digest='a'*64),
                first_reuse_layer=2, repair_positions_by_layer={'2':[1], '3':[1]}))
        self.layers = tuple((torch.ones(2, 2, 2, dtype=torch.bfloat16)*i,
                             torch.ones(2, 2, 2, dtype=torch.bfloat16)*(i+1)) for i in range(1,4))

    def audit(self, job, *, layers=None, predictions=(1, 0, 1)):
        layers = layers if layers is not None else self.layers
        reference = job['operation'] == 'explicit_mixed_reference'
        request = job['request']; rows = list(range(6))
        recorder = DecodeInputRecorderV2(request, cached_prefix_tokens=0)
        for i, predicted in enumerate(predictions):
            recorder.append(predicted, fed_token=request['teacher_token_ids'][i-1] if i else None)
        recipe = dict(kind='explicit_mixed_full_query_reference' if reference else 'explicit_mixed_sparse_control',
            request_id=request['request_id'], input_digest=request_input_digest(request['token_ids'], rows),
            model_signature=self.binding['model_signature'], target_id='T', target_positions=[3,4],
            upstream_segment_id='U', source_id='source1', artifact_digest='a'*64, source_generation=0,
            source_birth_positions=[8,9], source_current_positions=[0,1], first_reuse_layer=2,
            repair_positions_by_layer={'2':[1], '3':[1]}, num_layers=3, prefix_tokens=0,
            target_execution='FULL_ALL_LAYERS', production_publication_allowed=False)
        report = dict(kind='p0_explicit_mixed_reference' if reference else 'p0_mixed_sparse_control',
            status='COMPLETED', request_id=request['request_id'], recipe=recipe,
            recipe_sha256=digest_json(recipe), cleanup=dict(passed=True, failures=[]),
            publication=dict(status='DIAGNOSTIC_NOT_PUBLISHED'), extra_forward_count=1,
            production_reuse_commit_observed=False, native_runtime_qualified=False,
            P1_execution_allowed=False, paper_evidence=False,
            integrity={**{key:'a'*64 for key in ('source_before','destination','source_after','destination_after')},
                       'passed':True},
            answer=dict(token_ids=list(predictions), generation_mode='teacher_forced_logit_diagnostic',
                decode_input_trace_v2=recorder.finish(logit_rows=3),
                whole_request_origin='p0_explicit_mixed_reference' if reference else 'p0_mixed_sparse_control',
                p0_diagnostic_context_v2=P0DiagnosticContextV2(request['request_id'],
                    recipe['input_digest'],self.binding['model_signature'],digest_json(recipe),
                    'explicit_mixed_reference' if reference else 'mixed_sparse_control').audit()))
        execution = dict(completed=True, failure=None, cleanup_failure=None, device_references_released=True,
            executed_layers=[], target_logical_digest=target_digest(layers),
            target_owned_host_bytes=sum(t.numel()*t.element_size() for pair in layers for t in pair),
            parent_owned_kv_bytes=0, prefix_shadow_bytes=0, publication_allowed=False,
            historical_kv_rows_replaced=2)
        for layer in range(1,4):
            projected = rows if layer <= 2 else [1,2,3,4,5]
            queries = rows if layer < 2 else [1,2,3,4,5]
            event = dict(layer_1based=layer, target_positions=[3,4],
                historical_kv_positions=[] if layer < 2 else [0],
                repair_positions=[0,1] if layer < 2 else [1],
                target_full_projection=True, completed_block=True)
            if reference:
                event.update(projected_rows=6, attention_query_rows=6)
            else:
                event.update(projected_positions=projected, attention_query_positions=queries,
                    actual_qkv_invocations=1, actual_attention_output_rows=len(queries),
                    native_runtime_status=dict(status=0 if layer < 2 else (1 if layer == 2 else 2),
                                               dense_full_repair=False))
            execution['executed_layers'].append(event)
        if reference:
            hook_recipe = dict(kind='independent_explicit_mixed_full_query_reference', token_count=6,
                target_positions=[3,4], source_positions=[0,1], first_reuse_layer=2,
                repair_positions_by_layer={'2':[1], '3':[1]}, geometry=[[8,4,4,2,2]]*3)
            execution.update(kind='p0_mixed_reference_hooks', recipe=hook_recipe,
                recipe_sha256=digest_json(hook_recipe), projection_counts=[1,1,1],
                historical_kv_rows_replaced=2, transient_device_bytes_upper=1000,
                caller_source_lease_and_integrity_guard_required=True,
                reference_scope='fixed_upstream_recipe_only_not_dense_equivalence')
            report['reference']=execution
        else:
            execution.update(kind='p0_actual_cacheblend_sparse_execution', geometry=[[8,4,4,2,2]]*3,
                             projection_counts=[1,1,1], target_r1_endpoint_exercised=False)
            report['execution']=execution
        if job.get('target_execution') == 'R1_ALL_LAYERS':
            recipe.update(target_execution='R1_ALL_LAYERS', target_source=dict(
                **job['target_source'], source_generation=0, source_birth_positions=[10,11],
                source_current_positions=[3,4]))
            report['recipe_sha256']=digest_json(recipe)
            report['answer']['p0_diagnostic_context_v2']['recipe_sha256']=digest_json(recipe)
            report['target_source_integrity']={**{k:job['target_source']['artifact_digest'] for k in
                ('source_before','destination','source_after','destination_after')}, 'passed':True}
            if not reference:
                execution['target_r1_endpoint_exercised']=True
                for row in execution['executed_layers']:
                    active=row['layer_1based']>=2
                    row.update(target_r1_commit_active=active, target_r1_source_installed=active,
                        target_r1_current_kv_writeback_verified=active,
                        target_r1_repair_positions=[3,4] if active else [])
        return report

    def r1_fixture(self):
        for job in self.manifest['jobs']:
            job.update(target_execution='R1_ALL_LAYERS',
                       target_source=dict(source_id='target-source',artifact_digest='b'*64))

    def write(self, *, mutate=None, layers=None, logits=None, predictions=None, origins=None):
        writer = P0EvidenceWriter(self.root, binding=self.binding, manifest=self.manifest)
        for job in self.manifest['jobs']:
            name = job['action_id']; values = (layers or {}).get(name, self.layers)
            audit = self.audit(job, layers=values, predictions=(predictions or {}).get(name,(1,0,1)))
            if mutate: mutate(name,audit)
            writer.write_action(name, audit=audit,
                logits=(logits or {}).get(name,[torch.ones(2) for _ in range(3)]),
                origin=(origins or {}).get(name,'cpu_fixture'), target_layers=values)

    def evaluate(self, completed=('reference','candidate')):
        return evaluate_mixed_pairs(self.root,self.manifest,completed)[0]

    def test_valid_cpu_fixture_never_qualifies_native_P0_or_P1(self):
        before=copy.deepcopy(self.manifest)
        validate_mixed_pairs(self.manifest); self.write(); result=self.evaluate()
        self.assertEqual(result['status'],'CPU_ONLY',result)
        self.assertTrue(result['scoped_checks_passed'])
        for key in ('native_runtime_qualified','gpu_runtime_qualified','P0_qualified','P1_execution_allowed','real_cuda_T21_passed'):
            self.assertFalse(result[key])
        self.assertEqual(self.manifest,before)

    def test_absent_pairs_do_not_read_files(self):
        self.assertEqual(validate_mixed_pairs({'jobs':[]}),[])
        self.assertEqual(evaluate_mixed_pairs(self.root,{'jobs':[]},[]),[])

    def test_unfinished_pair_pending_without_any_files(self):
        result=self.evaluate(('reference',))
        self.assertEqual(result['status'],'PENDING')
        self.assertEqual(result['pending_action_ids'],['candidate'])
        self.assertFalse(self.root.exists())

    def test_no_default_numerical_tolerance_or_reader_budget(self):
        for key in ('relative_l2_limit','target_relative_l2_limit','target_absolute_max_limit','target_read_bytes'):
            with self.subTest(key=key):
                manifest=copy.deepcopy(self.manifest); del manifest['mixed_reference_pairs'][0][key]
                with self.assertRaises(ValueError): validate_mixed_pairs(manifest)

    def test_invalid_policy_fields_rejected(self):
        pair=self.manifest['mixed_reference_pairs'][0]
        for change in ({'pair_id':'../escape'}, {'relative_l2_limit':True},
                       {'target_relative_l2_limit':float('nan')}, {'target_absolute_max_limit':-1},
                       {'target_absolute_max_limit':True}, {'target_read_bytes':0}, {'target_read_bytes':True},
                       {'minimum_positions':4}, {'minimum_positions':True},
                       {'require_predicted_token_ids_equal':1}, {'candidate_action_id':'reference'},
                       {'unexpected_policy':True}):
            with self.subTest(change=change):
                other=copy.deepcopy(self.manifest); other['mixed_reference_pairs']=[{**pair,**change}]
                with self.assertRaises(ValueError): validate_mixed_pairs(other)

    def test_duplicate_or_unplanned_pair_ids_rejected(self):
        other=copy.deepcopy(self.manifest); other['mixed_reference_pairs']*=2
        with self.assertRaises(ValueError): validate_mixed_pairs(other)
        other=copy.deepcopy(self.manifest); other['mixed_reference_pairs'][0]['candidate_action_id']='absent'
        with self.assertRaises(ValueError): validate_mixed_pairs(other)

    def test_only_matching_explicit_reference_and_sparse_control_allowed(self):
        for operation in ('explicit_mixed_reference','source_request','exact_capture_control'):
            other=copy.deepcopy(self.manifest); other['jobs'][1]['operation']=operation
            with self.assertRaises(ValueError): validate_mixed_pairs(other)

    def test_teacher_request_source_and_mask_changes_rejected(self):
        mutations=[lambda j:j['request'].update(teacher_token_ids=[9,7]),
                   lambda j:j['request'].update(seed=7),
                   lambda j:j['source'].update(source_id='other'),
                   lambda j:j['source'].update(artifact_digest='b'*64),
                   lambda j:j.update(repair_positions_by_layer={'2':[0],'3':[0]}),
                   lambda j:j.update(first_reuse_layer=1,repair_positions_by_layer={'1':[1],'2':[1],'3':[1]})]
        for mutate in mutations:
            other=copy.deepcopy(self.manifest); job=other['jobs'][1]; mutate(job)
            job['request_sha256']=digest_json(job['request'])
            with self.assertRaises(ValueError): validate_mixed_pairs(other)

    def test_raw_target_corruption_fails_with_evidence_preserved(self):
        self.write(); path=self.root/'candidate'/'target-kv'/'layer-0002-K.npy'
        path.write_bytes(path.read_bytes()+b'corruption')
        result=self.evaluate(); self.assertEqual(result['status'],'FAILED',result)
        self.assertTrue(path.exists())

    def test_missing_target_layer_fails_not_pending(self):
        self.write(); path=self.root/'candidate'/'target-kv'/'layer-0003-V.npy'; path.unlink()
        self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_raw_target_shape_must_match_recipe(self):
        wrong=tuple((torch.ones(1,2,2,dtype=torch.bfloat16),torch.ones(1,2,2,dtype=torch.bfloat16)) for _ in range(3))
        self.write(layers={'candidate':wrong})
        self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_reader_budget_is_enforced_not_ignored(self):
        self.manifest['mixed_reference_pairs'][0]['target_read_bytes']=1
        self.write(); self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_target_numerical_failure_even_when_logits_match(self):
        changed=tuple((k+1,v) for k,v in self.layers)
        self.write(layers={'candidate':changed})
        result=self.evaluate(); self.assertEqual(result['status'],'FAILED',result)

    def test_zero_reference_norm_has_no_invented_epsilon(self):
        zero=tuple((torch.zeros_like(k),torch.zeros_like(v)) for k,v in self.layers)
        changed=tuple((torch.ones_like(k)*.001,torch.zeros_like(v)) for k,v in self.layers)
        self.manifest['mixed_reference_pairs'][0].update(target_relative_l2_limit=1e10,target_absolute_max_limit=1)
        self.write(layers={'reference':zero,'candidate':changed})
        self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_explicit_absolute_tolerance_not_replaced_by_relative_only(self):
        self.manifest['mixed_reference_pairs'][0].update(target_relative_l2_limit=1.,target_absolute_max_limit=.1)
        self.write(layers={'candidate':tuple((k+.5,v) for k,v in self.layers)})
        self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_logit_numeric_failure_even_when_raw_targets_match(self):
        self.write(logits={'candidate':[torch.zeros(2) for _ in range(3)]})
        self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_token_id_equality_is_a_separate_explicit_condition(self):
        self.write(predictions={'candidate':(0,0,0)})
        self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_execution_must_not_omit_or_duplicate_layers(self):
        def mutate(name,audit):
            if name=='candidate': audit['execution']['executed_layers'][1]['layer_1based']=1
        self.write(mutate=mutate); self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_sparse_control_must_really_omit_historical_queries(self):
        def mutate(name,audit):
            if name=='candidate': audit['execution']['executed_layers'][1]['attention_query_positions']=list(range(6))
        self.write(mutate=mutate); self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_target_must_be_full_at_every_layer(self):
        def mutate(name,audit):
            if name=='candidate': audit['execution']['executed_layers'][1]['projected_positions']=[1,2,4,5]
        self.write(mutate=mutate); self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_historical_source_injection_must_match_frozen_mask(self):
        def mutate(name,audit):
            if name=='reference': audit['reference']['executed_layers'][1]['historical_kv_positions']=[1]
        self.write(mutate=mutate); self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_source_before_destination_and_after_are_all_checked(self):
        def mutate(name,audit):
            if name=='candidate': audit['integrity']['destination_after']='b'*64
        self.write(mutate=mutate); self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_ref_recipe_sha_and_target_digest_cannot_be_only_asserted(self):
        def mutate(name,audit):
            if name=='reference': audit['reference']['target_logical_digest']='f'*64
        self.write(mutate=mutate); self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_declared_target_digest_is_recomputed_from_raw_bits(self):
        def mutate(name,audit):
            if name=='candidate': audit['execution']['target_logical_digest']='f'*64
        self.write(mutate=mutate); self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_recipe_hash_tampering_fails(self):
        self.write(mutate=lambda name,audit:audit.update(recipe_sha256='b'*64))
        self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_publication_or_prefix_shadow_cannot_pass(self):
        def mutate(name,audit):
            if name=='candidate': audit['execution']['prefix_shadow_bytes']=32
        self.write(mutate=mutate); self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_cleanup_failure_fails_even_with_identical_logits_and_targets(self):
        self.write(mutate=lambda name,audit:audit['cleanup'].update(passed=False))
        self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_torn_event_chain_fails(self):
        self.write(); path=self.root/'actions.jsonl'; path.write_bytes(path.read_bytes().rstrip(b'\n'))
        self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_record_digest_tampering_fails(self):
        self.write(); path=self.root/'candidate'/'record.json'; path.write_bytes(path.read_bytes()+b' ')
        self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_reference_hook_geometry_must_match_raw_target_heads_and_dimensions(self):
        def mutate(name,audit):
            if name=='reference':
                hook=audit['reference']; hook['recipe']['geometry'][1]=[8,4,4,1,4]
                hook['recipe_sha256']=digest_json(hook['recipe'])
        self.write(mutate=mutate); self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_reference_hook_recipe_digest_and_masks_are_validated(self):
        def mutate(name,audit):
            if name=='reference':
                hook=audit['reference']; hook['recipe']['repair_positions_by_layer']['2']=[0]
                hook['recipe_sha256']=digest_json(hook['recipe'])
        self.write(mutate=mutate); self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_one_actual_projection_per_reference_layer_required(self):
        def mutate(name,audit):
            if name=='reference': audit['reference']['projection_counts']=[1,2,1]
        self.write(mutate=mutate); self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_actual_target_owned_bytes_must_match_raw_arrays(self):
        def mutate(name,audit):
            if name=='candidate': audit['execution']['target_owned_host_bytes']=0
        self.write(mutate=mutate); self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_execution_positions_must_be_integer_not_boolean_aliases(self):
        def mutate(name,audit):
            if name=='candidate': audit['execution']['executed_layers'][1]['historical_kv_positions']=[False]
        self.write(mutate=mutate); self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_r1_cpu_pair_remains_scoped_not_endpoint_qualification(self):
        self.r1_fixture(); self.write(); result=self.evaluate()
        self.assertEqual(result['status'],'CPU_ONLY',result)
        self.assertEqual(result['test_id'],'T31')
        self.assertFalse(result['real_cuda_target_r1_passed'])
        self.assertFalse(result['r1_endpoint_qualified'])

    def test_r1_pair_must_bind_same_target_source_and_execution_mode(self):
        self.r1_fixture(); self.manifest['jobs'][1]['target_source']['artifact_digest']='c'*64
        with self.assertRaises(ValueError): validate_mixed_pairs(self.manifest)

    def test_r1_cannot_relabel_full_target_without_actual_endpoint(self):
        self.r1_fixture()
        def mutate(name,audit):
            if name=='candidate': audit['execution']['target_r1_endpoint_exercised']=False
        self.write(mutate=mutate); self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_r1_actual_current_kv_writeback_required(self):
        self.r1_fixture()
        def mutate(name,audit):
            if name=='candidate':
                audit['execution']['executed_layers'][1]['target_r1_current_kv_writeback_verified']=False
        self.write(mutate=mutate); self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_r1_actual_all_target_repair_rows_required(self):
        self.r1_fixture()
        def mutate(name,audit):
            if name=='candidate': audit['execution']['executed_layers'][2]['target_r1_repair_positions']=[3]
        self.write(mutate=mutate); self.assertEqual(self.evaluate()['status'],'FAILED')

    def test_r1_target_source_integrity_cannot_be_omitted(self):
        self.r1_fixture()
        def mutate(name,audit):
            if name=='candidate': audit['target_source_integrity']['destination_after']='c'*64
        self.write(mutate=mutate); self.assertEqual(self.evaluate()['status'],'FAILED')


if __name__=='__main__': unittest.main()
