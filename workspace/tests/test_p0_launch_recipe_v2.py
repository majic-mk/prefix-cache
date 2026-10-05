"""CPU-only recipe/file tests; fake tiny model assets never authorize a GPU."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import torch

from probekv.p0_launch_recipe_v2 import build_controlled_p0_manifest
from probekv.p0_batch_v2 import validate_p0_batch
from probekv.segment_capture_v2 import SegmentCapture
from probekv.source_comparison_v2 import ComparisonProfileBindingV2, runtime_binding_digest, scorer_digest
from probekv.source_manifest_v2 import RequestManifestRegistry, TargetOccurrence, request_input_digest
from probekv.source_provenance_v2 import RequestExecutionLedger, PublicationScope
from probekv.source_store_v2 import TargetSourceStoreV2
from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema10_storage import file_digest


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return file_digest(path)


class LaunchRecipeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.registry = RequestManifestRegistry(max_bytes=2_000_000, max_manifest_bytes=200_000,
            persistent_root=self.root/'registry')
        self.store = TargetSourceStoreV2(self.root/'store', registry=self.registry,
            model_signature='cpu-model', tokenizer_hash='cpu-tokenizer', authorization_domain='cpu-domain',
            policy='ALLOW_MIXED_G1', max_bytes=2_000_000, staging_bytes=1_000_000)
        self.addCleanup(self.store.close)
        self.tokens = [10,11,12,13,14,15]
        self.request = dict(request_id='controlled-input', token_ids=self.tokens,
            max_new_tokens=3, teacher_token_ids=[9,8], capture_logits=True,
            segments=[dict(segment_id='U', positions=[0,1], token_ids=[10,11]),
                      dict(segment_id='T', positions=[3,4], token_ids=[13,14])])
        self.model = self.root/'model'; self.model.mkdir()
        config_sha = write_json(self.model/'config.json', dict(num_hidden_layers=32, vocab_size=100,
            max_position_embeddings=128, num_attention_heads=4, num_key_value_heads=2, hidden_size=8))
        model_key = 'mistralai/Mistral-7B-Instruct-v0.3'
        audit_path = self.root/'model-audit.json'
        audit_sha = write_json(audit_path, dict(model_id=model_key, revision='CPU-TEST-NOT-REAL-REVISION',
            tokenizer_assets_sha256='cpu-tokenizer', files={'config.json':config_sha}))
        source = dict(model_id=model_key, model_revision='CPU-TEST-NOT-REAL-REVISION',
            tokenizer_hash='cpu-tokenizer', model_signature='cpu-model', code_commit='c'*40)
        native = dict(binding=dict(code_commit='c'*40,patch_sha256='a'*64,
                                  model_signature='cpu-model',config_sha256='b'*64),
            native_runtime=dict(model_path=str(self.model),model_key=model_key,
                model_audit_path=str(audit_path),model_audit_sha256=audit_sha,
                cost_provenance=dict(model='cpu-model',code='c'*40,patch='a'*64,config='b'*64,gpu='CPU-NO-GPU'),
                source_provenance=source,allocator_capacity_bytes=1000000,
                prefix_shadow_capacity_bytes=0,max_model_len=128,gpu_memory_utilization=.8,
                storage_root=str(self.root/'unused'),cpu_backing_bytes=1000000,
                selector_parameters={},sentinel_evidence_paths={},installed_runtime_source_files_sha256={}))
        native['manifest_sha256'] = digest_json(native)
        native_path = self.root/'native.json'
        native_sha = write_json(native_path,native)
        self.args = dict(native_manifest_path=native_path,native_manifest_sha256=native_sha,
            instance_id='CPU-TEST-NO-SERVER',store=self.store,now_unix=1000.,
            authority=dict(approval_reference='CPU FIXTURE; NOT AUTHORIZATION',instance_id='CPU-TEST-NO-SERVER',
                gpu_uuid='CPU-NO-GPU',starts_at_unix=900.,expires_at_unix=10000.,
                unit_price_per_hour=1.,maximum_cost=1.,maximum_gpu_hours=1.),
            limits=dict(initialization_upper_seconds=10.,cleanup_seconds=10.,maximum_actions=2,
                host_capture_bytes=1000000,host_comparison_bytes=1000000,cuda_comparison_bytes=1000000,
                mixed_reference_host_bytes=1000000,mixed_reference_cuda_bytes=1000000),
            registry_budget=dict(max_bytes=2000000,max_manifest_bytes=200000),
            numerical_policy=dict(policy_id='cpu-bitexact-fixture',rationale='CPU test only, no BF16 GPU policy',
                exact=dict(relative_l2_limit=0.,minimum_positions=3,require_predicted_token_ids_equal=True),
                mixed=None),
            exact_controls=[dict(control_id='T20',request=deepcopy(self.request),target_ids=['T'],
                                 upper_seconds_by_arm=dict(off=20.,on=20.))],mixed_controls=[])

    def source(self, target, birth):
        positions = tuple(range(6))
        ledger = RequestExecutionLedger(request_id=birth,
            input_digest=request_input_digest(self.tokens,positions), model_signature='cpu-model',
            token_count=6,num_layers=32,exact_input_proof='CPU-FIXTURE-NOT-GPU')
        capture = SegmentCapture(ledger,target,byte_budget=100000,selection_depths=(1,2))
        for layer in range(1,33):
            ledger.record_layer(layer_1based=layer,qkv_rows=positions,attention_rows=positions,
                output_mlp_rows=positions,effective_current_kv_rows=positions,imported_kv=(),
                completion_reference='cpu-'+str(layer))
            k = (torch.arange(24).reshape(6,2,2)+layer).to(torch.bfloat16)
            capture.record_layer(layer,k,k+1,row_positions=positions)
        occurrence = TargetOccurrence('target',target[0],target[-1]+1)
        manifest = self.registry.register_actual_execution(ledger=ledger,token_ids=self.tokens,
            absolute_positions=positions,authorization_domain='cpu-domain',occurrences=(occurrence,),
            parent_sources=(),request_completed=True,input_reference='cpu:'+birth)
        ref = self.registry.capture_reference(manifest,'target',model_signature='cpu-model',
            authorization_domain='cpu-domain',target_token_ids=tuple(self.tokens[p] for p in target),
            target_positions=target)
        candidate = capture.finalize(ref,request_completed=True)
        snapshot = self.store.begin_request(birth)
        tokens = tuple(self.tokens[p] for p in target)
        plan = self.store.plan_publication(snapshot,candidate,token_ids=tokens,
            scope=PublicationScope('content_miss',0,0,0,0))
        result = self.store.commit_publication(snapshot,plan,candidate,token_ids=tokens)
        self.store.end_request(snapshot)
        row = self.store._catalog['rows'][result['source_id']]
        return dict(source_id=row['source_id'],artifact_digest=row['artifact_digest'])

    def mixed(self, *, r1=False):
        source = self.source((0,1),'earlier-upstream')
        control = dict(control_id='T21',request=deepcopy(self.request),upstream_segment_id='U',target_id='T',
            source=source,first_reuse_layer=3,repair_positions_by_layer={str(i):[1] for i in range(3,33)},
            upper_seconds_by_arm=dict(reference=20.,sparse=20.))
        if r1:
            control['target_r1_source'] = self.source((3,4),'earlier-target')
            control['upper_seconds_by_arm'].update(r1_reference=20.,r1_sparse=20.)
        self.args['exact_controls'] = []
        self.args['mixed_controls'] = [control]
        self.args['numerical_policy']['exact'] = None
        self.args['numerical_policy']['mixed'] = dict(relative_l2_limit=0.,minimum_positions=3,
            require_predicted_token_ids_equal=True,target_relative_l2_limit=0.,target_absolute_max_limit=0.,
            target_read_bytes=1000000)
        self.args['limits']['maximum_actions'] = 4 if r1 else 2
        return control

    def build(self):
        return build_controlled_p0_manifest(**self.args)

    def test_r0_endpoint_is_explicit_frozen_and_not_runtime_policy(self):
        control=self.mixed()
        control['upstream_repair_endpoint']='R0_DIAGNOSTIC'
        control['repair_positions_by_layer']={str(i):[] for i in range(3,33)}
        result=self.build()
        for job in result['manifest']['jobs']:
            self.assertEqual(job['upstream_repair_endpoint'],'R0_DIAGNOSTIC')
            self.assertTrue(all(not x for x in job['repair_positions_by_layer'].values()))
        self.assertFalse(result['preflight']['P0_qualified'])
        del control['upstream_repair_endpoint']
        with self.assertRaises(ValueError): self.build()

    def test_T20_pair_is_signed_no_mutation_and_not_gpu_permission(self):
        before = deepcopy(self.args['exact_controls'])
        hashes = (file_digest(self.store.root/'catalog.json'),file_digest(self.registry._root/'registry.json'))
        result = self.build(); m = result['manifest']
        self.assertEqual([j['capture_enabled'] for j in m['jobs']],[False,True])
        self.assertEqual(m['limits']['maximum_actions'],2)
        self.assertEqual(m['binding']['initial_pool_sha256'],hashes[0])
        self.assertEqual(m['binding']['initial_registry_sha256'],hashes[1])
        self.assertEqual(m['manifest_sha256'],digest_json({k:v for k,v in m.items() if k!='manifest_sha256'}))
        self.assertEqual(self.args['exact_controls'],before)
        for field in ('gpu_execution_allowed','P0_qualified','P1_execution_allowed','paper_evidence'):
            self.assertIs(result['preflight'][field],False)
        self.assertEqual(validate_p0_batch(m,actual_binding=m['binding'],now_unix=1000.)['status'],'INPUTS_VALIDATED')
        self.assertEqual(hashes,(file_digest(self.store.root/'catalog.json'),file_digest(self.registry._root/'registry.json')))

    def test_T21_binds_actual_published_source_and_all_model_layers(self):
        control = self.mixed()
        result = self.build(); jobs = result['manifest']['jobs']
        self.assertEqual([j['operation'] for j in jobs],['explicit_mixed_reference','mixed_sparse_control'])
        self.assertEqual(jobs[0]['source'],control['source'])
        self.assertEqual(len(result['preflight']['source_files_verified']),1)
        self.assertEqual(self.store._snapshots,{})
        self.assertEqual(self.store._lease_counts,{})

    def test_generated_requests_have_native_occurrence_keys_without_changing_inputs(self):
        from probekv.v8_schema10_canonical import request_occurrences
        original=deepcopy(self.args['exact_controls'][0]['request'])
        result=self.build();manifest=result['manifest']
        self.assertEqual(self.args['exact_controls'][0]['request'],original)
        self.assertEqual(manifest['controlled_recipe']['original_inputs'][0]['input_request_sha256'],digest_json(original))
        for job in manifest['jobs']:
            request=job['request']
            occurrences,targets,_=request_occurrences(request)
            self.assertEqual(set(targets),{'U','T'})
            self.assertEqual(request['token_ids'],original['token_ids'])
            self.assertEqual(request['teacher_token_ids'],original['teacher_token_ids'])
            for segment in request['segments']:
                self.assertEqual(segment['content_key'],self.store.content_key(segment['token_ids']))

    def test_wrong_existing_content_key_is_not_silently_replaced(self):
        self.args['exact_controls'][0]['request']['segments'][0]['content_key']='legacy-label'
        with self.assertRaisesRegex(ValueError,'actual v2 namespace'):self.build()

    def test_preexisting_matching_content_key_is_accepted(self):
        request=self.args['exact_controls'][0]['request']
        for segment in request['segments']:
            segment['content_key']=self.store.content_key(segment['token_ids'])
        self.assertEqual(self.build()['manifest']['jobs'][0]['request']['segments'],request['segments'])

    def test_r1_is_an_additional_pair_with_explicit_target_source_not_relabelled(self):
        control = self.mixed(r1=True)
        result = self.build(); jobs = result['manifest']['jobs']
        self.assertEqual(len(jobs),4)
        self.assertEqual([j['target_execution'] for j in jobs],['FULL_ALL_LAYERS']*2+['R1_ALL_LAYERS']*2)
        self.assertNotIn('target_source',jobs[0])
        self.assertEqual(jobs[2]['target_source'],control['target_r1_source'])
        self.assertEqual(len(result['manifest']['mixed_reference_pairs']),2)

    def test_missing_current_authority_never_inherits_historical_budget(self):
        for field,value in (('approval_reference',None),('maximum_cost',0),('unit_price_per_hour',0),
                            ('gpu_uuid','other'),('expires_at_unix',999)):
            with self.subTest(field=field):
                old = self.args['authority'][field]; self.args['authority'][field] = value
                with self.assertRaises(ValueError): self.build()
                self.args['authority'][field] = old

    def test_resource_and_action_caps_cannot_expand_implicitly(self):
        for field,value in (('maximum_actions',3),('host_capture_bytes',None),('cleanup_seconds',0)):
            with self.subTest(field=field):
                old = self.args['limits'][field]; self.args['limits'][field] = value
                with self.assertRaises(ValueError): self.build()
                self.args['limits'][field] = old
        self.args['authority']['maximum_gpu_hours'] = .001
        with self.assertRaises(ValueError): self.build()

    def test_policy_missing_nan_or_unknown_extra_fields_fail(self):
        original = deepcopy(self.args['numerical_policy'])
        for mutate in (lambda p:p.pop('rationale'),lambda p:p['exact'].pop('relative_l2_limit'),
                       lambda p:p['exact'].update(relative_l2_limit=float('nan')),
                       lambda p:p['exact'].update(passed=True)):
            self.args['numerical_policy'] = deepcopy(original); mutate(self.args['numerical_policy'])
            with self.assertRaises((ValueError,TypeError)): self.build()

    def test_native_file_changed_rejected(self):
        self.args['native_manifest_path'].write_text('{}',encoding='utf-8')
        with self.assertRaises(ValueError): self.build()

    def test_model_asset_changed_rejected(self):
        (self.model/'config.json').write_text('{}',encoding='utf-8')
        with self.assertRaises(ValueError): self.build()

    def test_registry_limits_must_match_actual_opened_registry(self):
        self.args['registry_budget']['max_bytes'] += 1
        with self.assertRaises(ValueError): self.build()

    def test_segment_token_mismatch_overlap_and_boolean_positions_rejected(self):
        original = deepcopy(self.args['exact_controls'])
        for mutate in (lambda q:q['segments'][0]['token_ids'].__setitem__(0,20),
                       lambda q:q['segments'][0]['positions'].__setitem__(0,False),
                       lambda q:q['segments'].append(deepcopy(q['segments'][0]))):
            self.args['exact_controls'] = deepcopy(original)
            mutate(self.args['exact_controls'][0]['request'])
            with self.assertRaises(ValueError): self.build()

    def test_request_geometry_identity_and_decode_policy_rejected(self):
        original = deepcopy(self.args['exact_controls'])
        for mutate in (lambda q:q['token_ids'].__setitem__(0,100),
                       lambda q:q.update(teacher_token_ids=[1]),
                       lambda q:q.update(publish_exact_prefix_shadow=True),
                       lambda q:q.update(native_dense_continuation=True)):
            self.args['exact_controls'] = deepcopy(original); mutate(self.args['exact_controls'][0]['request'])
            with self.assertRaises(ValueError): self.build()

    def test_duplicate_or_unbounded_control_identity_rejected(self):
        self.args['exact_controls'] *= 2
        with self.assertRaises(ValueError): self.build()

    def test_mixed_incomplete_checkpoint_end_fails_actual_model_geometry(self):
        control = self.mixed(); control['repair_positions_by_layer'].pop('32')
        with self.assertRaises(ValueError): self.build()

    def test_mixed_unknown_or_wrong_literal_source_rejected(self):
        control = self.mixed()
        original = deepcopy(control['source'])
        for ref in (dict(source_id='not-published',artifact_digest='a'*64),
                    dict(source_id=original['source_id'],artifact_digest='a'*64),
                    dict(birth_action_id='future',segment_id='U')):
            control['source'] = ref
            with self.assertRaises(ValueError): self.build()

    def test_full_source_backing_corruption_rejected(self):
        control = self.mixed()
        row = self.store._catalog['rows'][control['source']['source_id']]
        self.store._path(row['kv_file']).write_bytes(b'bad')
        with self.assertRaises(ValueError): self.build()

    def test_nonquiescent_snapshot_rejected_without_releasing_user_snapshot(self):
        snapshot = self.store.begin_request('active')
        try:
            with self.assertRaises(ValueError): self.build()
            self.assertIn(snapshot.snapshot_id,self.store._snapshots)
        finally: self.store.end_request(snapshot)

    def test_r1_missing_second_pair_upper_bound_fails(self):
        control = self.mixed(r1=True); control['upper_seconds_by_arm'].pop('r1_sparse')
        with self.assertRaises(ValueError): self.build()

    def birth(self):
        request = deepcopy(self.request); request.pop('teacher_token_ids'); request.pop('capture_logits')
        profile = ComparisonProfileBindingV2('cpu-model','ALLOW_MIXED_G1',2,.15,1.,1.,'d'*64,
                                            runtime_binding_digest(),scorer_digest())
        self.args['birth_controls'] = [dict(control_id='birth',request=request,target_ids=['U','T'],
                                          upper_seconds=20.,comparison_profile=asdict(profile))]
        self.args['limits']['maximum_actions'] = 3

    def test_optional_birth_is_normal_action_not_exact_diagnostic_export(self):
        self.birth(); result = self.build(); jobs = result['manifest']['jobs']
        self.assertEqual([j['operation'] for j in jobs],['exact_capture_control']*2+['source_request'])
        self.assertEqual(jobs[-1]['sources_by_segment'],{'U':[],'T':[]})
        self.assertNotIn('teacher_token_ids',jobs[-1]['request'])
        self.assertEqual(self.store._catalog['rows'],{})

    def test_birth_must_not_use_teacher_diagnostic_or_existing_pool(self):
        self.birth(); birth = self.args['birth_controls'][0]
        birth['request']['capture_logits'] = True
        with self.assertRaises(ValueError): self.build()
        birth['request'].pop('capture_logits'); self.source((0,1),'earlier')
        with self.assertRaises(ValueError): self.build()

    def test_mixed_cannot_reference_same_batch_birth(self):
        self.birth(); self.mixed()
        with self.assertRaises(ValueError): self.build()


if __name__ == '__main__':
    unittest.main()
