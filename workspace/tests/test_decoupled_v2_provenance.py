import gc
import unittest
import weakref
from dataclasses import replace
import torch
from probekv.source_provenance_v2 import (
    SourceOrigin,KVImport,RequestExecutionLedger,PublicationScope,publication_rejection,visible)
from probekv.segment_capture_v2 import SegmentCapture,ManifestReference


def ledger(n=10, layers=3, exact=True, exact_ref='verified-input'):
    return RequestExecutionLedger(request_id='birth',input_digest='input',model_signature='model',
        token_count=n,num_layers=layers,exact_input_proof=exact_ref if exact else None)


def step(x, *, imported=(), active=None):
    if active is None:
        active=tuple(i for i in range(x.token_count) if i not in {v.position for v in imported})
    return x.record_layer(layer_1based=len(x.layers)+1,qkv_rows=active,attention_rows=active,
        output_mlp_rows=active,imported_kv=imported,completion_reference='cpu-executed')


def accept(proof, policy='ALLOW_MIXED_G1',scope=None,**kwargs):
    flags=dict(permission_verified=True,request_completed=True,artifact_complete=True,
               budget_ok=True,capacity_ok=True,snapshot_current=True)
    flags.update(kwargs)
    return publication_rejection(proof,scope or PublicationScope('content_miss',0,0,0,0),policy=policy,**flags)


class ProvenanceV2Tests(unittest.TestCase):
    def test_legacy_materialization_cannot_accept_nonfinite_values(self):
        from probekv.v8_schema10_materialization import VariantMaterializationRequestV10
        from probekv.v8_schema10_contracts import DenseKVProvenance, VariantMaterializationReasonV10
        request = dict(reason=VariantMaterializationReasonV10.CONTENT_MISS,
            correctness_eligible_k=0, compared_k=0, best_residual=None, absolute_threshold=None,
            dense_kv_provenance=DenseKVProvenance.DENSE_EXACT, existing_variant_count=0,
            dense_reference_total_ms=100.0, estimated_materialization_ms=1.0)
        VariantMaterializationRequestV10(**request)
        for field in ('best_residual', 'absolute_threshold', 'dense_reference_total_ms',
                      'estimated_materialization_ms', 'estimated_replacement_ms'):
            for value in (float('nan'), float('inf'), -1.0, True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    VariantMaterializationRequestV10(**dict(request, **{field:value}))

    def test_T01_exact_and_T09_full_r1_by_ledger(self):
        x=ledger()
        for _ in range(3):step(x)
        p=x.target_proof((8,9))
        self.assertEqual((p.origin,p.generation),(SourceOrigin.EXACT,0))
        self.assertIsNone(accept(p))

    def test_T02_nine_approx_then_full_and_T03_same_request_no_increment(self):
        x=ledger(12)
        for _ in range(3):step(x,imported=tuple(KVImport(i,'s'+str(i),0) for i in range(9)))
        for target in ((9,),(10,11)):
            p=x.target_proof(target)
            self.assertEqual((p.origin,p.generation),(SourceOrigin.MIXED,1))
            self.assertIsNone(accept(p))
            self.assertEqual(accept(p,'EXACT_ONLY'),'generation_limit')

    def test_T04_T05_indirect_G2_not_washed(self):
        x=ledger(4)
        for _ in range(3):step(x,imported=(KVImport(0,'M1',1),))
        self.assertEqual(x.target_proof((1,)).generation,2)
        self.assertEqual(x.target_proof((2,3)).generation,2)
        self.assertEqual(accept(x.target_proof((2,3))),'generation_limit')

    def test_T06_unknown_and_missing_dependency(self):
        for exact in (True,False):
            x=ledger(4,exact=exact)
            for _ in range(3):step(x,active=(1,2,3))
            p=x.target_proof((2,3))
            self.assertEqual(p.origin,SourceOrigin.UNKNOWN)
            self.assertEqual(accept(p),'incomplete_or_unknown_provenance')

    def test_T07_suffix_cannot_taint_prefix(self):
        x=ledger(4)
        for _ in range(3):step(x,imported=(KVImport(3,'M1',1),))
        self.assertEqual(x.target_proof((0,1)).generation,0)

    def test_T08_partial_then_full_does_not_complete_history(self):
        x=ledger(4)
        step(x,imported=(KVImport(2,'old',0),))
        step(x);step(x)
        self.assertEqual(x.target_proof((2,3)).origin,SourceOrigin.PARTIAL)

    def test_transition_qkv_projected_but_old_rows_effective(self):
        x=ledger(3,1)
        x.record_layer(layer_1based=1,qkv_rows=(0,1,2),attention_rows=(1,2),output_mlp_rows=(1,2),
            effective_current_kv_rows=(1,2),imported_kv=(KVImport(0,'G0',0),),completion_reference='layer')
        self.assertEqual(x.target_proof((1,2)).generation,1)
        self.assertEqual(x.target_proof((0,)).origin,SourceOrigin.PARTIAL)

    def test_T10_exact_prefix_rejects_mixed(self):
        with self.assertRaises(ValueError):KVImport(0,'M1',1,exact_prefix_proof='p')
        x=ledger(4)
        for _ in range(3):step(x,imported=(KVImport(0,'E',0,exact_prefix_proof='verified-prefix'),))
        self.assertEqual(x.target_proof((2,3)).generation,0)

    def test_T12_nonfinite_generation_and_scope_rejected(self):
        x=ledger(2,1);step(x);p=x.target_proof((1,))
        for v in (float('nan'),float('inf'),.5,True,-1):
            with self.assertRaises(ValueError):replace(p,generation=v,origin=SourceOrigin.MIXED)
        for v in (float('nan'),float('inf'),-1):
            self.assertEqual(PublicationScope('complete_scope_mismatch',4,4,4,4,v,.1).rejection(),'nonfinite_or_negative_score')

    def test_T13_T14_T17_scope_and_triggers(self):
        for scope in (PublicationScope('content_miss',1,0,0,0),
                      PublicationScope('complete_scope_mismatch',4,2,2,2,.3,.1),
                      PublicationScope('complete_scope_mismatch',4,4,4,2,.3,.1),
                      PublicationScope('economic_rejection',0,0,0,0)):
            self.assertIsNotNone(scope.rejection())
        self.assertIsNone(PublicationScope('complete_scope_mismatch',4,4,4,4,.3,.1).rejection())

    def test_T15_T16_flags_and_duplicate(self):
        x=ledger(2,1);step(x);p=x.target_proof((1,))
        for flag in ('permission_verified','request_completed','artifact_complete','budget_ok','capacity_ok','snapshot_current'):
            self.assertEqual(accept(p,**{flag:False}),flag)
        self.assertEqual(accept(p,duplicate=True),'duplicate')

    def test_T18_T19_visibility(self):
        self.assertFalse(visible(birth_request_id='a',publication_epoch=2,reader_request_id='a',snapshot_epoch=3))
        self.assertFalse(visible(birth_request_id='a',publication_epoch=2,reader_request_id='b',snapshot_epoch=1))
        self.assertTrue(visible(birth_request_id='a',publication_epoch=2,reader_request_id='b',snapshot_epoch=2))

    def test_proof_binds_input_verifier_and_no_duplicate_layers(self):
        x=ledger(2,1);step(x)
        with self.assertRaises(ValueError):step(x)
        y=ledger(2,1,exact_ref='different-proof');step(y)
        self.assertNotEqual(x.target_proof((1,)).ledger_digest,y.target_proof((1,)).ledger_digest)

    def test_ledger_cannot_relabel_completed_execution(self):
        for field in ('request_id','input_digest','model_signature','exact_input_proof'):
            x=ledger(2,1);step(x)
            setattr(x,field,'changed')
            with self.assertRaisesRegex(ValueError,'identity changed'):
                x.target_proof((1,))


class SegmentCaptureV2Tests(unittest.TestCase):
    def test_T23_T26_T40_independent_mixed_target_and_manifest_reference(self):
        x=ledger(4,3); capture=SegmentCapture(x,(2,3),byte_budget=128,selection_depths=(1,2))
        parent=torch.arange(4,dtype=torch.bfloat16).reshape(4,1,1)
        ref=weakref.ref(parent)
        for i in range(1,4):
            step(x,imported=(KVImport(0,'non-owning-parent-id',0),))
            capture.record_layer(i,parent[1:],parent[1:]+1,row_positions=(1,2,3))
        manifest=ManifestReference('shared-request','hash','C',2,'domain')
        result=capture.finalize(manifest,request_completed=True)
        del parent;gc.collect()
        self.assertIsNone(ref())
        self.assertEqual(result['generation'],1)
        self.assertEqual(result['storage_audit'],dict(target_kv_bytes=24,selection_state_bytes=8,parent_owned_kv_bytes=0,prefix_shadow_bytes=0))
        self.assertFalse(result['native_runtime_qualified'])
        self.assertEqual(result['layers'][0][0].flatten().tolist(),[2.,3.])
        self.assertIs(result['manifest_reference'],manifest)

    def test_T15_missing_early_capture_and_memory_fail_closed(self):
        x=ledger(4,3); c=SegmentCapture(x,(2,3),byte_budget=1,selection_depths=(1,))
        step(x);t=torch.zeros((4,1,1),dtype=torch.bfloat16)
        with self.assertRaises(MemoryError):c.record_layer(1,t,t,row_positions=range(4))
        with self.assertRaises(ValueError):c.finalize(ManifestReference('m','d','C',2,'a'),request_completed=True)
        c.close()

    def test_T22_same_prompt_execution_manifest_changes_identity(self):
        def capture(mid):
            x=ledger(2,2);c=SegmentCapture(x,(1,),byte_budget=64,selection_depths=(1,))
            t=torch.ones((2,1,1),dtype=torch.bfloat16)
            for i in (1,2):step(x);c.record_layer(i,t,t,row_positions=range(2))
            return c.finalize(ManifestReference(mid,'d','C',1,'a'),request_completed=True)
        a,b=capture('recipe-E'),capture('recipe-M')
        self.assertNotEqual(a['source_artifact_id'],b['source_artifact_id'])
        self.assertEqual(a['artifact_digest'],b['artifact_digest'])

    def test_nonfinite_capture_rejected(self):
        x=ledger(2,1);c=SegmentCapture(x,(1,),byte_budget=16,selection_depths=());step(x)
        t=torch.full((2,1,1),float('nan'),dtype=torch.bfloat16)
        with self.assertRaises(ValueError):c.record_layer(1,t,t,row_positions=(0,1))

    def test_composite_rows_cannot_be_mislabeled_as_current_projection(self):
        x=ledger(4,1);c=SegmentCapture(x,(2,3),byte_budget=64,selection_depths=())
        step(x,imported=(KVImport(0,'old',0),))
        t=torch.ones((4,1,1),dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError,'projection layout'):
            c.record_layer(1,t,t,row_positions=(0,1,2,3))
