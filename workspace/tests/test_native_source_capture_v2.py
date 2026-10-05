"""CPU fake-block integration: actual torch hooks, not real LLM/GPU evidence."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
from threading import RLock
from dataclasses import asdict
import torch

from probekv.model_adapters import PinnedCacheBlendResumableAdapter
from probekv.resumable_prefill import ProbeKVResumablePrefillSession, LayerAdvanceResult
from probekv.native_source_capture_v2 import NativeTargetCaptureV2
from probekv.source_manifest_v2 import RequestManifestRegistry, ParentSourceMetadata
from probekv.source_provenance_v2 import SourceOrigin
from probekv.v8_schema10_native_adapter import NativeRequestContext
from probekv.v8_schema6_hbm import UnifiedHBMReservationManager


class Projection(torch.nn.Module):
    def __init__(self):
        super().__init__();self.calls=0

    def forward(self, h):
        self.calls+=1
        return torch.cat((h,h[:,:2]+10,h[:,:2]+20),dim=1),None


class CPUFakePinnedModel:
    def __init__(self,nlayers=3):
        self.layers=[NS(self_attn=NS(qkv_proj=Projection(),q_size=4,kv_size=2,
            num_heads=2,num_kv_heads=1,head_dim=2)) for _ in range(nlayers)]
        self.cache_fuse_metadata={};self.real_block_calls=0;self.fail_at=None;self.bad_rows=False

    def probekv_begin_prefill(self,tokens,positions,attention,working,signature):
        return torch.tensor(tokens,dtype=torch.bfloat16).reshape(-1,1).repeat(1,4),None

    def probekv_advance_prefill(self,**kw):
        self.real_block_calls+=1
        block=self.layers[kw['layer']-1]
        qkv,_=block.self_attn.qkv_proj(kw['hidden_states'])
        # Emulate a rotary implementation mutating the projection after hook.
        qkv[:,4:6].add_(100)
        if kw['layer']==self.fail_at:
            raise RuntimeError('real block failed after projection')
        rows=[kw['active_positions'].index(p) for p in kw['target_active_positions']]
        h=kw['hidden_states'][rows]+1
        if self.bad_rows:h=h[:-1]
        return LayerAdvanceResult(h,None,kw['working_kv'])

    def probekv_finish_prefill(self,**kw):
        return kw['hidden_states']


def fixture(*,capture=True,host_budget=10000, targets=None):
    model=CPUFakePinnedModel()
    adapter=PinnedCacheBlendResumableAdapter(model,NS(adapter_name='cpu_fake',num_layers=3))
    session=ProbeKVResumablePrefillSession(adapter,'model',tuple(range(6)),None,[])
    session.begin_prefill()
    registry=RequestManifestRegistry(max_bytes=1000000,max_manifest_bytes=100000)
    tracker=None
    if capture:
        tracker=NativeTargetCaptureV2(request_id='birth',token_ids=tuple(range(6)),
            model_signature='model',authorization_domain='domain',num_layers=3,
            targets=targets or {'C':(2,3,4)},selection_depths=(1,2),host_budget_bytes=host_budget,
            kv_heads=1,head_dim=2,registry=registry)
        session.provenance_capture=tracker
    return model,session,tracker,registry


def commit(session,tracker,generation=0,positions=(0,1),repair=(0,),bound=True):
    if tracker and bound:
        tracker.bind_parent(ParentSourceMetadata('source','a'*64,'model','domain',
            SourceOrigin.EXACT if generation==0 else SourceOrigin.MIXED,generation))
    session.register_source_handle('A','source',object())
    session.commit_segment_reuse(segment_id='A',source_id='source',boundary=session.current_layer+1,
                                 segment_positions=positions,repair_positions=repair)


class NativeSourceCaptureTests(unittest.TestCase):
    def finish(self,session,tracker):
        hidden=session.finish_prefill()
        return hidden,tracker.finalize(request_completed=True,actual_token_ids=tuple(range(6)))

    def test_same_layer_capture_no_extra_forward_and_pre_rope_slice(self):
        m,s,t,r=fixture(); hidden,out=self.finish(s,t)
        self.assertEqual(m.real_block_calls,3)
        self.assertEqual([b.self_attn.qkv_proj.calls for b in m.layers],[1,1,1])
        self.assertEqual(out['C']['origin'],SourceOrigin.EXACT.value)
        self.assertEqual(out['C']['layers'][0][0][:,0,0].tolist(),[12.,13.,14.])
        self.assertEqual(out['C']['selection_states'][1][:,0,0].tolist(),[13.,14.,15.])
        self.assertEqual(r.storage_audit()['manifest_count'],1)
        self.assertEqual(out['C']['storage_audit']['parent_owned_kv_bytes'],0)
        self.assertFalse(out['C']['native_runtime_qualified'])
        self.assertTrue(all(not b.self_attn.qkv_proj._forward_hooks for b in m.layers))

    def test_mixed_full_target_survives_upstream_selective_commit(self):
        m,s,t,r=fixture();s.advance_to_layer(1);commit(s,t)
        _,out=self.finish(s,t)
        self.assertEqual(out['C']['generation'],1)
        self.assertEqual(out['C']['origin'],SourceOrigin.MIXED.value)
        layer=t.ledger.layers[1]
        self.assertEqual(layer.qkv_rows,(0,1,2,3,4,5))
        self.assertEqual(layer.current_kv_rows,(0,2,3,4,5))
        self.assertEqual([(x.position,x.generation) for x in layer.imported_kv],[(1,0)])

    def test_capture_on_off_does_not_change_cpu_block_output(self):
        _,s,t,_=fixture();s.advance_to_layer(1);commit(s,t)
        actual,_=self.finish(s,t)
        _,plain,_,_=fixture(capture=False);plain.advance_to_layer(1);commit(plain,None)
        self.assertTrue(torch.equal(actual,plain.finish_prefill()))

    def test_r1_uses_effective_rows_not_commit_function_name(self):
        _,s,t,_=fixture();commit(s,t,positions=(0,1),repair=(0,1))
        _,out=self.finish(s,t)
        self.assertEqual(out['C']['generation'],0)

    def test_G2_and_unknown_parent_do_not_publish_candidate(self):
        for generation,bound in ((1,True),(0,False)):
            _,s,t,_=fixture();commit(s,t,generation=generation,bound=bound)
            _,out=self.finish(s,t)
            self.assertFalse(out)
            self.assertIn('C',t.audit()['rejected_targets'])

    def test_late_parent_binding_cannot_relabel_already_unknown_execution(self):
        _,s,t,_=fixture();commit(s,t,bound=False)
        s.advance_to_layer(1)
        t.bind_parent(ParentSourceMetadata('source','a'*64,'model','domain',SourceOrigin.EXACT,0))
        _,out=self.finish(s,t)
        self.assertFalse(out)
        self.assertIsNone(t.ledger.layers[0].imported_kv[0].generation)

    def test_removed_target_row_cannot_reenter_or_get_captured(self):
        m,s,t,_=fixture();commit(s,t,positions=(2,3,4),repair=(2,))
        _,out=self.finish(s,t)
        self.assertFalse(out)
        self.assertEqual(m.real_block_calls,3)
        self.assertEqual(t.rejections['C'],'target_not_full_this_layer')

    def test_cancelled_unknown_suffix_target_does_not_taint_earlier_exact_target(self):
        _,s,t,registry=fixture(targets={'early':(0,1),'late':(3,4)})
        commit(s,t,positions=(3,4),repair=(3,),bound=False)
        _,out=self.finish(s,t)
        self.assertEqual(set(out),{'early'})
        self.assertEqual(out['early']['generation'],0)
        self.assertEqual(t.rejections['late'],'target_not_full_this_layer')
        self.assertEqual(registry.storage_audit()['manifest_count'],1)

    def test_budget_failure_keeps_answer_without_second_forward(self):
        m,s,t,_=fixture(host_budget=1);_,out=self.finish(s,t)
        self.assertFalse(out);self.assertEqual(m.real_block_calls,3)
        self.assertEqual(t.rejections['C'],'host_capture_budget_exceeded')

    def test_observation_projection_is_not_executed_layer(self):
        m,s,t,_=fixture()
        m.layers[0].self_attn.qkv_proj(s.hidden_states)
        self.assertEqual(len(t.ledger.layers),0)
        _,out=self.finish(s,t)
        self.assertEqual(m.layers[0].self_attn.qkv_proj.calls,2)
        self.assertEqual(out['C']['generation'],0)

    def test_block_failure_removes_hook_and_invalidates_capture(self):
        m,s,t,_=fixture();m.fail_at=2
        with self.assertRaisesRegex(RuntimeError,'real block failed'):
            s.advance_to_layer(3)
        self.assertEqual(len(t.ledger.layers),1)
        self.assertFalse(t.captures)
        self.assertTrue(all(not b.self_attn.qkv_proj._forward_hooks for b in m.layers))

    def test_incomplete_birth_and_changed_input_are_rejected(self):
        for complete,tokens in ((False,tuple(range(6))),(True,(6,5,4,3,2,1))):
            _,s,t,_=fixture();s.finish_prefill()
            self.assertFalse(t.finalize(request_completed=complete,actual_token_ids=tokens))

    def test_legacy_builder_cannot_run_as_v2_capture_fallback(self):
        _,s,t,_=fixture()
        ctx=object.__new__(NativeRequestContext)
        ctx.source_capture_v2=t;ctx.committed={};ctx.cached_prefix_tokens=0
        self.assertEqual(ctx.deferred_canonical_builders(),{})
        self.assertEqual(ctx.materialization_metadata(),{})
        self.assertEqual(ctx.export_exact_dense(),{})
        ctx.finished=False;ctx.target_candidates_v2={}
        with self.assertRaises(RuntimeError):ctx.export_target_candidates_v2()

    def test_target_packet_rowcount_and_request_identity_fail_closed(self):
        m,s,t,_=fixture();m.bad_rows=True
        s.advance_to_layer(1)
        self.assertFalse(t.captures)

    def test_native_dense_continuation_records_remaining_layers_without_replay(self):
        m,s,t,_=fixture();s.working_kv=[object() for _ in range(3)]
        s.advance_to_layer(1)
        class DirectLayer:
            def __init__(self,attn):self.self_attn=attn;self.calls=0
            def __call__(self,positions,hidden,kv,attention,residual,status,metadata,old):
                self.calls+=1
                self.self_attn.qkv_proj(hidden)
                return hidden+1,residual
        m.layers=[DirectLayer(b.self_attn) for b in m.layers]
        a=NS(inner=m,spec=NS(num_layers=3),check_deadline=lambda:None,kv=s.working_kv,
             torch=NS(cuda=NS(Event=lambda **kw:Mock())))
        context=NS(adapter=a,engine=NS(session=s),cached_prefix_tokens=0,
            request={'token_ids':tuple(range(6))},committed={},_prepared_inputs=(None,torch.arange(6)),
            generation=1,attention=s.attention_metadata)
        NativeRequestContext._advance_native_dense_remaining(context)
        self.assertEqual([b.calls for b in m.layers],[0,1,1])
        self.assertEqual(len(t.ledger.layers),3)
        self.assertEqual(t.finalize(request_completed=True,actual_token_ids=tuple(range(6)))['C']['generation'],0)

    def native_context(self):
        m,s,_,registry=fixture(capture=False)
        ctx=object.__new__(NativeRequestContext)
        ctx.request={'request_id':'birth','token_ids':tuple(range(6))}
        ctx.segments={'C':{'positions':(2,3,4)}}
        ctx.finished=False;ctx.source_capture_v2=None;ctx.source_capture_workspace_v2=None
        ctx.cached_prefix_tokens=0;ctx.probe_fallback_reason=None
        ctx.engine=NS(session=s);ctx._begin=Mock()
        ctx.adapter=NS(inner=m,spec=NS(num_layers=3,checkpoints=(1,2)),
            provenance={'model_signature':'model'},
            hbm=UnifiedHBMReservationManager(allocator_capacity_bytes=4096,safety_bytes=0))
        return ctx,registry

    def test_native_context_reserves_only_one_target_layer_and_releases_on_close(self):
        ctx,registry=self.native_context()
        audit=ctx.configure_target_capture_v2(target_ids=('C',),registry=registry,
            authorization_domain='domain',host_budget_bytes=10000)
        self.assertIs(ctx.engine.session.provenance_capture,ctx.source_capture_v2)
        self.assertEqual(ctx.adapter.hbm.active_reserved_bytes,24)
        self.assertEqual(audit['peak_target_packet_budget_bytes'],24)
        self.assertFalse(audit['prefix_shadow_created'])
        ctx.closed=False;ctx.synchronize=Mock();ctx.hot_leases=NS(close=Mock())
        ctx.hot_replicas={};ctx.prepared={};ctx._observation={}
        ctx.workspace=ctx.capture_reservation=None
        ctx.adapter.store_provider=lambda:NS(pool=NS(mutation_lock=RLock()))
        ctx.close()
        self.assertEqual(ctx.adapter.hbm.active_reserved_bytes,0)
        self.assertFalse(ctx.source_capture_v2.captures)

    def test_native_context_rejects_prefix_or_late_capture_before_hook(self):
        for condition in ('prefix','late','legacy_capture'):
            ctx,registry=self.native_context()
            if condition=='prefix':ctx.cached_prefix_tokens=2
            elif condition=='late':ctx.engine.session.current_layer=1
            else:ctx.request['capture_original_full_prefill']=True
            with self.assertRaises(ValueError):
                ctx.configure_target_capture_v2(target_ids=('C',),registry=registry,
                    authorization_domain='domain',host_budget_bytes=10000)
            self.assertEqual(ctx.adapter.hbm.active_reserved_bytes,0)
            ctx._begin.assert_not_called()

    def test_native_context_budget_failure_skips_capture_no_extra_forward(self):
        ctx,registry=self.native_context()
        ctx.adapter.hbm=UnifiedHBMReservationManager(allocator_capacity_bytes=1,safety_bytes=0)
        audit=ctx.configure_target_capture_v2(target_ids=('C',),registry=registry,
            authorization_domain='domain',host_budget_bytes=10000)
        self.assertIn('C',audit['rejected_targets'])
        self.assertIsNone(ctx.engine.session.provenance_capture)
        ctx._begin.assert_not_called()
        self.assertEqual(ctx.adapter.hbm.active_reserved_bytes,0)
        m,s,t,_=fixture();s.token_ids=tuple(reversed(s.token_ids))
        s.advance_to_layer(1)
        self.assertFalse(t.captures)

    def parent_binding_context(self):
        _,session,tracker,registry=fixture()
        _,out=self.finish(session,tracker)
        candidate=out['C']
        ctx,_=self.native_context()
        ctx.configure_target_capture_v2(target_ids=('C',),registry=registry,
            authorization_domain='domain',host_budget_bytes=10000)
        ctx.segments['C']['content_key']='content'
        parent=ParentSourceMetadata('source',candidate['artifact_digest'],'model','domain',SourceOrigin.EXACT,0)
        ctx.frozen={'C':'source'}
        ctx.replica_reservations={'C':NS(released=False)}
        ticket=NS(source_id='source',expected_artifact_digest=parent.artifact_digest,
            segment_positions=(2,3,4),transfer_failed=False,preparation_cancelled=False)
        ctx.prepared={'C':ticket}
        row=NS(runtime_available=True,canonical_source_state_digest=parent.artifact_digest,
               healthy_backing_replicas=[NS(busy=True)])
        metadata={'token_ids':(2,3,4),'source_provenance_v2':{
            'manifest_reference':asdict(candidate['manifest_reference']),
            'birth_target_positions':(2,3,4),'proof_digest':candidate['proof'].proof_digest}}
        store=NS(objects={'source':NS(metadata=metadata)},pool=NS(
            mutation_lock=RLock(),_get=lambda *args:row,logical_lease_counts={'source':1}))
        ctx.adapter.store_provider=lambda:store
        return ctx,parent,store,row

    def test_native_parent_binding_uses_actual_ticket_and_manifest(self):
        ctx,parent,_,_=self.parent_binding_context()
        ctx.bind_capture_parent_v2(parent)
        self.assertEqual(ctx.source_capture_v2.parents,{'source':parent})

    def test_native_parent_binding_rejects_mismatched_or_unleased_artifact(self):
        for case in ('digest','frozen','logical','physical','reservation','cancelled','positions','missing_ticket'):
            with self.subTest(case=case):
                ctx,parent,store,row=self.parent_binding_context()
                if case=='digest':ctx.prepared['C'].expected_artifact_digest='b'*64
                elif case=='frozen':ctx.frozen['C']='other'
                elif case=='logical':store.pool.logical_lease_counts['source']=0
                elif case=='physical':row.healthy_backing_replicas[0].busy=False
                elif case=='reservation':ctx.replica_reservations['C'].released=True
                elif case=='cancelled':ctx.prepared['C'].preparation_cancelled=True
                elif case=='positions':ctx.prepared['C'].segment_positions=(1,2,3)
                else:ctx.prepared={}
                with self.assertRaises(ValueError):ctx.bind_capture_parent_v2(parent)
                self.assertFalse(ctx.source_capture_v2.parents)

    def test_native_parent_binding_does_not_infer_G0_from_legacy_tag(self):
        ctx,parent,store,_=self.parent_binding_context()
        store.objects['source'].metadata={'canonical_provenance':'dense_exact_full_prefill'}
        with self.assertRaisesRegex(ValueError,'verified v2'):
            ctx.bind_capture_parent_v2(parent)

    def test_native_parent_binding_rejects_tampered_generation_or_domain(self):
        for case in ('generation','domain','proof','tokens'):
            with self.subTest(case=case):
                ctx,parent,store,_=self.parent_binding_context()
                if case=='generation':parent=ParentSourceMetadata('source',parent.artifact_digest,'model','domain',SourceOrigin.MIXED,1)
                elif case=='domain':parent=ParentSourceMetadata('source',parent.artifact_digest,'model','other',SourceOrigin.EXACT,0)
                elif case=='proof':store.objects['source'].metadata['source_provenance_v2']['proof_digest']='b'*64
                else:store.objects['source'].metadata['token_ids']=(4,3,2)
                with self.assertRaises(ValueError):ctx.bind_capture_parent_v2(parent)
                self.assertFalse(ctx.source_capture_v2.parents)


if __name__=='__main__':
    unittest.main()
