"""CPU evidence orchestration tests; mocks never count as GPU qualification."""
from copy import deepcopy
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from probekv.p1_qualification_v2 import assess_p1_preparation,verify_birth_witness,IDENTITY
from probekv.p0_acceptance_v2 import file_sha
from probekv.v8_schema10_execution import digest_json


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.identity={k:'a'*64 for k in IDENTITY}
        self.contract=dict(kind='P1_preparation_evidence_v2',runtime_identity=self.identity,
            prefix_mode='off',locked_test_accessed=False,
            p0_batches=[dict(root='cpu-fixture',binding=self.identity,manifest_sha256='b'*64)],
            frozen_inputs={},expected_input_graph_sha256='graph',p0_source_witnesses=[{'n':0},{'n':1}])
        self.graph=dict(model_signature='a'*64,runtime_tokenizer_identity='a'*64,graph_sha256='graph',
            partition_digest='partition',file_index_sha256='index',
            source_builds=[dict(build_id='pending-source')],consumer_actions=[{}],
            maximum_build_seconds=180,maximum_consumer_seconds=180)
        def mocked(name,**kwargs):
            p=patch('probekv.p1_qualification_v2.'+name,**kwargs);obj=p.start();self.addCleanup(p.stop);return obj
        self.raw=mocked('verify_batch',return_value=dict(manifest={},audits={},manifest_file_sha256='b'*64,events_sha256='event'))
        self.numeric=mocked('assess_numerical_correctness',return_value=dict(status='PASS',blockers=[],counts={'exact_pairs':1}))
        self.lineage=mocked('verify_lineage_scope',return_value=[dict(expected_generation={'a':1,'b':2})])
        self.input=mocked('load_frozen_input_graph',return_value=self.graph)
        self.store=mocked('verify_birth_witness',side_effect=lambda w,*args:dict(source_id=str(w['n']),generation=w['n']))

    def run_check(self):return assess_p1_preparation(self.contract,observed_runtime_digest='a'*64)

    def test_prepared_is_not_permission_and_unbuilt_sources_stay_pending(self):
        before=deepcopy(self.contract);r=self.run_check()
        self.assertTrue(r['build_plan_preparation_ready']);self.assertEqual(before,self.contract)
        for key in ('P1_execution_allowed','GPU_execution_allowed','P1_QA_execution_allowed','source_artifacts_for_P1_verified','full_P0_complete'):
            self.assertFalse(r[key])
        self.assertEqual(r['details']['FROZEN_INPUT_GRAPH']['source_builds_pending'],['pending-source'])
        self.assertEqual(r['report_sha256'],digest_json({k:v for k,v in r.items() if k!='report_sha256'}))

    def test_wrong_current_runtime_blocks_without_reading_sources(self):
        self.contract['runtime_identity']=dict(self.identity,runtime_digest='b'*64)
        self.assertFalse(self.run_check()['build_plan_preparation_ready']);self.store.assert_not_called()

    def test_old_gpu_runtime_not_silently_relabelled(self):
        self.contract['p0_batches'][0]['binding']=dict(self.identity,runtime_digest='old')
        r=self.run_check();self.assertFalse(r['build_plan_preparation_ready']);self.raw.assert_not_called()

    def test_duplicate_batches_not_extra_evidence(self):
        self.contract['p0_batches']*=2
        self.assertIn('duplicate P0',str(self.run_check()['blockers']))

    def test_bad_raw_digest_blocks(self):
        self.raw.side_effect=ValueError('raw digest corrupted')
        self.assertFalse(self.run_check()['build_plan_preparation_ready']);self.store.assert_not_called()

    def test_missing_strict_numerics_not_passed_summary(self):
        self.numeric.return_value=dict(status='BLOCKED',blockers=['missing T31'])
        r=self.run_check();self.assertIn('missing T31',str(r['blockers']));self.store.assert_not_called()

    def test_missing_g2_rejection_blocks(self):
        self.lineage.return_value=[dict(expected_generation={'a':1})]
        self.assertIn('G2 rejection',str(self.run_check()['blockers']))

    def test_missing_mixed_storage_not_exact_inheritance(self):
        self.store.side_effect=lambda w,*args:dict(source_id=str(w['n']),generation=0)
        self.assertIn('both G0 and G1',str(self.run_check()['blockers']))

    def test_duplicate_source_witness_blocks(self):
        self.store.side_effect=lambda w,*args:dict(source_id='same',generation=w['n'])
        self.assertIn('duplicate birth',str(self.run_check()['blockers']))

    def test_changed_graph_and_wrong_tokenizer_block(self):
        self.graph['graph_sha256']='changed'
        self.assertIn('frozen input graph',str(self.run_check()['blockers']))
        self.graph['graph_sha256']='graph';self.graph['runtime_tokenizer_identity']='other'
        self.assertIn('data/model identity',str(self.run_check()['blockers']))

    def test_future_data_or_bad_partition_propagates_block(self):
        self.input.side_effect=ValueError('future Source visibility')
        self.assertIn('future Source',str(self.run_check()['blockers']))

    def test_prefix_on_not_inherited(self):
        self.contract['prefix_mode']='on'
        self.assertIn('Prefix-on',str(self.run_check()['blockers']))

    def test_locked_test_refused(self):
        self.contract['locked_test_accessed']=True
        self.assertFalse(self.run_check()['build_plan_preparation_ready'])


class DiskWitnessTests(unittest.TestCase):
    def setUp(self):
        import torch
        from probekv.source_manifest_v2 import RequestManifestRegistry,TargetOccurrence,request_input_digest
        from probekv.source_provenance_v2 import RequestExecutionLedger,PublicationScope
        from probekv.source_store_v2 import TargetSourceStoreV2
        from probekv.segment_capture_v2 import SegmentCapture
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name)
        budget=dict(max_bytes=8*1024**2,max_manifest_bytes=4*1024**2)
        registry=RequestManifestRegistry(persistent_root=self.root/'registry',**budget)
        store=TargetSourceStoreV2(self.root/'store',registry=registry,model_signature='cpu-model',
            tokenizer_hash='cpu-tokenizer',authorization_domain='cpu',policy='ALLOW_MIXED_G1',
            max_bytes=32*1024**2,staging_bytes=4*1024**2)
        tokens=list(range(514));positions=tuple(range(514));target=tuple(range(1,513));birth='cpu-birth'
        ledger=RequestExecutionLedger(request_id=birth,input_digest=request_input_digest(tokens,positions),
            model_signature='cpu-model',token_count=514,num_layers=32,exact_input_proof='CPU-FIXTURE-NOT-GPU')
        capture=SegmentCapture(ledger,target,byte_budget=4*1024**2,selection_depths=(1,2))
        for layer in range(1,33):
            ledger.record_layer(layer_1based=layer,qkv_rows=positions,attention_rows=positions,
                output_mlp_rows=positions,effective_current_kv_rows=positions,imported_kv=(),completion_reference='cpu')
            k=torch.ones((514,1,2),dtype=torch.bfloat16)*layer
            capture.record_layer(layer,k,k+1,row_positions=positions)
        manifest=registry.register_actual_execution(ledger=ledger,token_ids=tokens,absolute_positions=positions,
            authorization_domain='cpu',occurrences=(TargetOccurrence('C',1,513),),parent_sources=(),
            request_completed=True,input_reference='cpu-fixture')
        ref=registry.capture_reference(manifest,'C',model_signature='cpu-model',authorization_domain='cpu',
            target_token_ids=tuple(tokens[1:513]),target_positions=target)
        candidate=capture.finalize(ref,request_completed=True);snapshot=store.begin_request(birth)
        plan=store.plan_publication(snapshot,candidate,token_ids=tokens[1:513],scope=PublicationScope('content_miss',0,0,0,0))
        event=store.commit_publication(snapshot,plan,candidate,token_ids=tokens[1:513]);store.end_request(snapshot)
        row=deepcopy(store._catalog['rows'][event['source_id']]);store.close()
        q=dict(request_id=birth,token_ids=tokens,max_new_tokens=1,segments=[dict(segment_id='C',token_ids=tokens[1:513],positions=list(target))])
        self.job=dict(action_id='birth',operation='source_request',request=q,request_sha256=digest_json(q))
        self.batch=dict(manifest=dict(jobs=[self.job]),audits={'birth':dict(publication=dict(
            extra_forward_count=0,prefix_shadow_created=False,targets={'C':dict(publication_performed=True,
                source_id=row['source_id'],generation=0,origin=row['origin'],publication_epoch=row['publication_epoch'])}))})
        spec=self.root/'pool_spec.json';spec.write_text(json.dumps(dict(registry_budget=budget)))
        def reference(p):return dict(path=str(p),sha256=file_sha(p))
        self.w=dict(batch_root=str(self.root/'batch'),action_id='birth',target_id='C',source_id=row['source_id'],
            catalog=reference(self.root/'store/catalog.json'),registry=reference(self.root/'registry/registry.json'),pool_spec=reference(spec))
        self.identity=dict(model_signature='cpu-model',tokenizer_hash='cpu-tokenizer')

    def verify(self):return verify_birth_witness(self.w,{str((self.root/'batch').resolve()):self.batch},self.identity)

    def test_real_cpu_backing_and_context_verified_without_mutation(self):
        before={str(p):file_sha(p) for p in self.root.rglob('*') if p.is_file()}
        r=self.verify();self.assertEqual(r['generation'],0);self.assertEqual(r['target_tokens'],512)
        self.assertEqual(before,{str(p):file_sha(p) for p in self.root.rglob('*') if p.is_file()})
        self.assertEqual(r['evidence_scope'],'P0_representative_birth_not_a_P1_build_receipt')

    def test_same_target_different_prefix_rejected(self):
        self.job['request']['token_ids'][0]=999
        self.assertRaisesRegex(ValueError,'historical context',self.verify)

    def test_teacher_cannot_be_birth(self):
        self.job['request']['teacher_token_ids']=[]
        self.assertRaisesRegex(ValueError,'teacher/reference',self.verify)

    def test_bad_catalog_sha_refused(self):
        self.w['catalog']['sha256']='f'*64
        self.assertRaisesRegex(ValueError,'digest mismatch',self.verify)

    def test_wrong_publication_source_refused(self):
        self.batch['audits']['birth']['publication']['targets']['C']['source_id']='other'
        self.assertRaisesRegex(ValueError,'not published',self.verify)

    def test_missing_registry_is_not_initialized(self):
        missing=self.root/'absent/registry.json';self.w['registry']['path']=str(missing)
        self.assertRaises(ValueError,self.verify);self.assertFalse(missing.parent.exists())


if __name__=='__main__':unittest.main()
