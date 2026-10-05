"""CPU engine-protocol integration, not real CacheBlend kernels/GPU evidence."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from probekv.cacheblend_v6_online_engine import LayerwiseLoadTicket, CacheBlendV6OnlineEngine
from probekv.native_p0_request_v2 import P0RequestFailure
from probekv.p0_mixed_sparse_v2 import execute_mixed_sparse_control
from tests import test_p0_mixed_control_v2 as control_fixtures
from tests.test_p0_mixed_reference_v2 import ToyProjection, attention_math


class Projection(ToyProjection):
    def forward(self, hidden):
        value, aux, _ = super().forward(hidden)
        return value, aux


class Attention(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.q_size=8; self.kv_size=4; self.num_heads=4; self.num_kv_heads=2; self.head_dim=2
        self.qkv_proj=Projection(generator)
        self.hack_kv=[]
        self.attn=AttentionKernel()

    def forward(self, hidden, *, before, after, total, old, status):
        current, _ = self.qkv_proj(hidden)
        return self.attn(current[:,:8],current[:,8:12],current[:,12:],before=before,
                         after=after,total=total,old=old,status=status)


class AttentionKernel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.skip_target_writeback=False

    def forward(self,q,k,v,*,before,after,total,old,status):
        full = torch.zeros((total,16),dtype=q.dtype)
        full[list(before)] = torch.cat((q,k,v),dim=-1)
        if status:
            local = [before.index(p) for p in after]
            for dst, current in zip(old,(k,v)):
                if self.skip_target_writeback:
                    rows=[p for p in after if p not in (3,4)]
                    dst[rows]=current[[before.index(p) for p in rows]].reshape(len(rows),2,2)
                else:
                    dst[list(after)] = current[local].reshape(len(after),2,2)
            full[:,8:12] = old[0].reshape(total,4)
            full[:,12:16] = old[1].reshape(total,4)
        return attention_math(full,list(after),total)


class Block(torch.nn.Module):
    def __init__(self,generator):
        super().__init__()
        self.self_attn=Attention(generator)
        self.fail=False

    def forward(self, hidden, **kwargs):
        normalized=hidden.float()*torch.rsqrt(hidden.float().square().mean(-1,keepdim=True)+1e-5)
        out=self.self_attn(normalized.to(torch.bfloat16),**kwargs)
        if self.fail: raise RuntimeError('injected CPU model block failure')
        selected=[kwargs['before'].index(p) for p in kwargs['after']]
        h=hidden[selected].float()
        return (h+.15*out+.1*torch.tanh(h)).to(torch.bfloat16)


class PatchedToy(torch.nn.Module):
    """CPU math implements pinned method ABI; production engine is unmocked."""
    def __init__(self):
        super().__init__()
        generator=torch.Generator().manual_seed(531)
        self.layers=torch.nn.ModuleList([Block(generator) for _ in range(3)])
        self.cache_fuse_metadata={}; self.old_kvs=[[None,None] for _ in range(3)]
        self.calls=[]; self.bad_status=False; self.bad_rows=False

    def forward(self, hidden):
        for block in self.layers:
            rows=tuple(range(len(hidden)))
            hidden=block(hidden,before=rows,after=rows,total=len(hidden),old=None,status=0)
        return hidden

    def probekv_begin_prefill(self, token_ids, absolute_positions, attention_metadata, working_kv, model_signature):
        ids=torch.tensor(token_ids)
        hidden=torch.sin(ids.float()[:,None]*.1+torch.arange(8)[None,:]*.21).to(torch.bfloat16)
        self.cache_fuse_metadata.update(org_seq_len=len(ids),reuse_active=False,probekv_resumable=True)
        return hidden,None

    def probekv_advance_prefill(self, **kwargs):
        layer=kwargs['layer']; before=kwargs['active_positions']; after=kwargs['target_active_positions']
        status=1 if kwargs['reuse_commit'] else (2 if self.cache_fuse_metadata['reuse_active'] else 0)
        self.calls.append((layer,before,after,status))
        hidden=self.layers[layer-1](kwargs['hidden_states'],before=before,after=after,
            total=self.cache_fuse_metadata['org_seq_len'],old=self.old_kvs[layer-1],status=status)
        self.cache_fuse_metadata['reuse_active'] = status != 0
        if self.bad_rows: hidden=hidden[:-1]
        return dict(hidden_states=hidden,residual=None,working_kv=kwargs['working_kv'],gpu_ms=0,
            runtime_debug=dict(status=0 if self.bad_status else status,dense_full_repair=False))

    def probekv_finish_prefill(self, **kwargs):
        self.cache_fuse_metadata.update(probekv_resumable=False,reuse_active=False)
        return kwargs['hidden_states']


class _Event:
    def query(self): return True
    def synchronize(self): pass
    def elapsed_time(self, other): return 0.0  # Explicit CPU fake, never cost evidence.


class _CPUTestLoader:
    def __init__(self, pool=None, *, authorize=None, integrity_mode='online_immutable', device='cpu'):
        self.torch=torch; self.device='cpu'; self.capture_hardware_trace=False
        self.pool, self.authorize, self.integrity_mode = pool, authorize, integrity_mode

    def begin(self, *, segment_id, source_id, canonical_layers, segment_positions,
              expected_artifact_digest, request_id, replica_id, prefetch_window, resident_layers):
        self.authorize(source_id=source_id, segment_id=segment_id,
            bytes_required=sum(t.numel()*t.element_size() for pair in canonical_layers for t in pair))
        return LayerwiseLoadTicket(segment_id,source_id,0,0,resident_layers,_Event(),
            {l:_Event() for l in resident_layers},'', '',tuple(segment_positions),
            integrity_mode='online_immutable',expected_artifact_digest=expected_artifact_digest)

    def prefetch_pending(self, ticket, through_layer, **kwargs):
        if ticket.pending_layers: raise AssertionError('CPU test has no pending copy')


class MixedSparseControlTests(unittest.TestCase):
    def fixture(self):
        base=control_fixtures.P0MixedControlTests(); base.setUp(); self.addCleanup(base.doCleanups)
        c,store,snapshot,job,_,_=base.fixture()
        model=PatchedToy(); outer=control_fixtures._ToyOuter(model)
        c.adapter.inner=model; c.adapter.outer=outer
        c.adapter.spec=NS(num_layers=3,adapter_name='cpu_test_pinned_ABI')
        c.adapter.loader=_CPUTestLoader()
        job['operation']='mixed_sparse_control'
        self.addCleanup(patch.stopall)
        patch('probekv.p0_mixed_sparse_v2.PhysicalLayerwiseSourceLoader',_CPUTestLoader).start()
        patch.object(torch.cuda,'current_stream',return_value=NS(wait_event=Mock())).start()
        return c,store,snapshot,job,model,outer

    def r1_fixture(self):
        from tests import test_source_store_v2 as store_fixtures
        c,store,snapshot,job,model,outer=self.fixture()
        store.end_request(snapshot)
        f=store_fixtures.TargetSourceStoreTests(); f.setUp(); self.addCleanup(f.doCleanups)
        f.registry=store.registry
        f.tokens=(70,71,91,92); f.target_tokens=(91,92)
        target=f.publish(store,f.candidate('old-target-birth',base=90))
        snapshot=store.begin_request(c.request['request_id'])
        self.addCleanup(lambda:store.end_request(snapshot) if snapshot.snapshot_id in store._snapshots else None)
        row=store._visible_row(snapshot,target['source_id'])
        job.update(target_execution='R1_ALL_LAYERS',
            target_source=dict(source_id=row['source_id'],artifact_digest=row['artifact_digest']))
        return c,store,snapshot,job,model,outer

    def execute(self,c,store,snapshot,job,**overrides):
        options=dict(host_reference_bytes=4096,cuda_reference_bytes=4096)
        options.update(overrides)
        return execute_mixed_sparse_control(c,store,snapshot,job,**options)

    def test_actual_engine_sparse_boundary_target_capture_and_teacher_decode(self):
        c,store,snapshot,job,model,outer=self.fixture()
        before=deepcopy(store._catalog)
        with patch.object(c,'_begin',side_effect=AssertionError('no generic begin/production bypass')):
            result=self.execute(c,store,snapshot,job)
        self.assertEqual(result['status'],'COMPLETED')
        self.assertIsInstance(c.p0_diagnostic_resources_v2.hooks.engine,CacheBlendV6OnlineEngine)
        self.assertEqual(model.calls,[(1,(0,1,2,3,4,5),(0,1,2,3,4,5),0),
                                     (2,(0,1,2,3,4,5),(1,2,3,4,5),1),
                                     (3,(1,2,3,4,5),(1,2,3,4,5),2)])
        rows=result['execution']['executed_layers']
        self.assertEqual(rows[1]['projected_positions'],[0,1,2,3,4,5])
        self.assertEqual(rows[1]['attention_query_positions'],[1,2,3,4,5])
        self.assertEqual(rows[2]['projected_positions'],[1,2,3,4,5])
        self.assertEqual([r['historical_kv_positions'] for r in rows],[[],[0],[0]])
        self.assertTrue(all(r['target_full_projection'] and r['completed_block'] for r in rows))
        self.assertEqual(result['execution']['historical_kv_rows_replaced'],2)
        self.assertFalse(result['execution']['target_r1_endpoint_exercised'])
        self.assertTrue(result['execution']['device_references_released'])
        self.assertTrue(result['integrity']['passed'])
        self.assertEqual(result['publication']['status'],'DIAGNOSTIC_NOT_PUBLISHED')
        self.assertEqual(result['answer']['whole_request_origin'],'p0_mixed_sparse_control')
        self.assertEqual([x[0] for x in outer.calls],[[7],[8]])  # Only decode uses full outer.
        self.assertEqual(outer.decode_hook_counts,[[(0,0,0)]*3]*2)
        self.assertEqual(len(c.p0_reference_target_layers_v2),3)
        self.assertEqual(store._catalog,before)
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)
        self.assertEqual(sum(store._lease_counts.values()),0)
        self.assertTrue(c.closed)
        self.assertFalse(result['production_reuse_commit_observed'])

    def test_target_and_suffix_equal_independent_full_query_reference(self):
        from probekv.p0_mixed_control_v2 import execute_explicit_mixed_reference
        c,store,snapshot,job,model,outer=self.fixture()
        sparse=self.execute(c,store,snapshot,job)
        sparse_kv=c.p0_reference_target_layers_v2; sparse_logits=tuple(c.logit_trace)
        c2,store2,snapshot2,job2,model2,outer2=self.fixture()
        job2['operation']='explicit_mixed_reference'
        explicit=execute_explicit_mixed_reference(c2,store2,snapshot2,job2,
            host_reference_bytes=4096,cuda_reference_bytes=4096)
        self.assertEqual(sparse['answer']['token_ids'],explicit['answer']['token_ids'])
        for left,right in zip(sparse_logits,c2.logit_trace):
            self.assertTrue(torch.equal(left,right))
        for left,right in zip(sparse_kv,c2.p0_reference_target_layers_v2):
            for a,b in zip(left,right): self.assertTrue(torch.equal(a,b))

    def test_v2_diagnostic_loader_checks_live_ownership_without_legacy_pool(self):
        c,store,snapshot,job,model,outer=self.r1_fixture()
        original=c.adapter.loader
        original.authorize=Mock(side_effect=AssertionError('legacy pool must not be consulted'))
        checks=[]
        def factory(*args,**kwargs):
            authorize=kwargs['authorize']; owner=authorize.__self__
            for sid,row in owner.bindings.items():
                call=dict(source_id=row['source_id'],segment_id=sid,bytes_required=row['target_kv_bytes'])
                authorize(**call)
                for field,value in [('source_id','unknown'),('segment_id','unknown'),('bytes_required',1)]:
                    with self.assertRaises(ValueError): authorize(**{**call,field:value})
                r=owner.reservation
                r.released=True
                try:
                    with self.assertRaises(ValueError): authorize(**call)
                finally:r.released=False
                count=store._lease_counts[row['source_id']]
                store._lease_counts[row['source_id']]=0
                try:
                    with self.assertRaises(ValueError): authorize(**call)
                finally:store._lease_counts[row['source_id']]=count
                checks.append(sid)
            return _CPUTestLoader(*args,**kwargs)
        with patch('probekv.p0_mixed_sparse_v2.PhysicalLayerwiseSourceLoader',side_effect=factory):
            self.assertEqual(self.execute(c,store,snapshot,job)['status'],'COMPLETED')
        self.assertEqual(checks,['U','T'])
        self.assertIs(c.adapter.loader,original)
        original.authorize.assert_not_called()
        with self.assertRaises(ValueError):
            c.p0_diagnostic_resources_v2.authorize(source_id=job['source']['source_id'],
                segment_id='U',bytes_required=48)

    def test_diagnostic_loader_constructor_failure_cleans_lease_and_hbm(self):
        c,store,snapshot,job,model,outer=self.fixture()
        with patch('probekv.p0_mixed_sparse_v2.PhysicalLayerwiseSourceLoader',side_effect=RuntimeError('injected loader fault')):
            with self.assertRaises(P0RequestFailure):self.execute(c,store,snapshot,job)
        self.assertEqual(model.calls,[])
        self.assertEqual(sum(store._lease_counts.values()),0)
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)
        self.assertTrue(c.closed)

    def test_actual_drivers_raw_writer_and_T21_checker_end_to_end_CPU_only(self):
        self._run_evidence_pair(r1=False)

    def test_actual_target_r1_drivers_raw_writer_and_checker_CPU_only(self):
        self._run_evidence_pair(r1=True)

    def _run_evidence_pair(self, *, r1, r0=False):
        from probekv.p0_mixed_control_v2 import execute_explicit_mixed_reference
        from probekv.p0_evidence_v2 import P0EvidenceWriter
        from probekv.p0_mixed_pair_v2 import evaluate_mixed_pairs, validate_mixed_pairs
        from probekv.v8_schema10_execution import digest_json
        from tests.test_p0_mixed_pair_v2 import target_digest
        cases=[]
        for action,operation in (('reference','explicit_mixed_reference'),('candidate','mixed_sparse_control')):
            c,store,snapshot,job,model,outer=(self.r1_fixture() if r1 else self.fixture())
            store.end_request(snapshot)
            c.request['request_id']=action
            snapshot=store.begin_request(action)
            self.addCleanup(lambda s=store,p=snapshot:s.end_request(p) if p.snapshot_id in s._snapshots else None)
            job.update(action_id=action,operation=operation,request_sha256=digest_json(c.request))
            if r0:
                job['upstream_repair_endpoint']='R0_DIAGNOSTIC'
                job['repair_positions_by_layer']={k:[] for k in job['repair_positions_by_layer']}
            cases.append((c,store,snapshot,job))
        self.assertEqual(cases[0][3]['source'],cases[1][3]['source'])
        binding={k:'cpu-test-not-GPU' for k in ('code_commit','runtime_digest','patch_sha256','gpu_uuid','instance_id')}
        binding.update(model_signature='model',tokenizer_hash='tokenizer')
        manifest=dict(binding=binding,jobs=[case[3] for case in cases],mixed_reference_pairs=[dict(
            pair_id='T21_actual_CPU_pipeline',reference_action_id='reference',candidate_action_id='candidate',
            relative_l2_limit=0.,target_relative_l2_limit=0.,target_absolute_max_limit=0.,
            minimum_positions=3,require_predicted_token_ids_equal=True,target_read_bytes=4096)])
        validate_mixed_pairs(manifest)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'evidence'
            writer=P0EvidenceWriter(root,binding=binding,manifest=manifest)
            for c,store,snapshot,job in cases:
                run=(execute_explicit_mixed_reference if job['operation']=='explicit_mixed_reference'
                     else execute_mixed_sparse_control)
                audit=run(c,store,snapshot,job,host_reference_bytes=4096,cuda_reference_bytes=4096)
                detail=audit['reference' if job['operation']=='explicit_mixed_reference' else 'execution']
                self.assertEqual(detail['target_logical_digest'],target_digest(c.p0_reference_target_layers_v2))
                writer.write_action(job['action_id'],audit=audit,logits=c.logit_trace,
                    origin='cpu_fixture',target_layers=c.p0_reference_target_layers_v2)
            result=evaluate_mixed_pairs(root,manifest,['reference','candidate'])[0]
            self.assertEqual(result['status'],'CPU_ONLY',result)
            self.assertTrue(result['scoped_checks_passed'],result)
            self.assertFalse(result['real_cuda_T21_passed'])
            self.assertFalse(result['P0_qualified'])
            if r0:
                self.assertEqual(result['evidence_scope'],'upstream_r0_full_target_reference_only')
                self.assertFalse(result['real_cuda_upstream_r0_passed'])

    def test_upstream_r0_actual_empty_repair_reference_and_sparse_agree(self):
        self._run_evidence_pair(r1=False,r0=True)

    def test_empty_repair_requires_explicit_endpoint_and_cannot_combine_r1(self):
        from probekv.p0_mixed_control_v2 import validate_mixed_reference_job
        _,_,_,job,_,_=self.fixture()
        job['operation']='mixed_sparse_control'
        job['repair_positions_by_layer']={k:[] for k in job['repair_positions_by_layer']}
        with self.assertRaises(ValueError):
            validate_mixed_reference_job(job,_operation='mixed_sparse_control')
        job['upstream_repair_endpoint']='R0_DIAGNOSTIC'
        validate_mixed_reference_job(job,_operation='mixed_sparse_control')
        job['upstream_repair_endpoint']='unknown'
        with self.assertRaises(ValueError):
            validate_mixed_reference_job(job,_operation='mixed_sparse_control')
        _,_,_,r1job,_,_=self.r1_fixture()
        r1job['operation']='mixed_sparse_control'
        r1job['upstream_repair_endpoint']='R0_DIAGNOSTIC'
        with self.assertRaises(ValueError):
            validate_mixed_reference_job(r1job,_operation='mixed_sparse_control')

    def test_target_r1_actual_second_ticket_allrow_commit_with_upstream_still_sparse(self):
        c,store,snapshot,job,model,outer=self.r1_fixture()
        result=self.execute(c,store,snapshot,job)
        self.assertEqual(result['status'],'COMPLETED')
        self.assertTrue(result['execution']['target_r1_endpoint_exercised'])
        self.assertTrue(result['target_source_integrity']['passed'])
        self.assertEqual(result['recipe']['target_execution'],'R1_ALL_LAYERS')
        self.assertEqual(result['recipe']['target_source']['source_id'],job['target_source']['source_id'])
        trace=result['execution']['executed_layers']
        self.assertFalse(trace[0]['target_r1_commit_active'])
        for row in trace[1:]:
            self.assertEqual(row['target_r1_repair_positions'],[3,4])
            self.assertTrue(row['target_r1_source_installed'])
            self.assertTrue(row['target_r1_current_kv_writeback_verified'])
            self.assertNotIn(0,row['attention_query_positions'])
            self.assertFalse(row['native_runtime_status']['dense_full_repair'])
        self.assertFalse(c.native.finish_prefill.call_args.kwargs['exact_dense'])
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)
        self.assertEqual(sum(store._lease_counts.values()),0)
        self.assertEqual(result['answer']['whole_request_origin'],'p0_mixed_sparse_control')

    def test_target_r1_missing_current_writeback_is_not_full_recompute_proof(self):
        c,store,snapshot,job,model,outer=self.r1_fixture()
        model.layers[1].self_attn.attn.skip_target_writeback=True
        with self.assertRaises(P0RequestFailure) as failure: self.execute(c,store,snapshot,job)
        self.assertIn('failed actual current K/V overwrite',str(failure.exception))
        self.assertFalse(failure.exception.audit['execution']['target_r1_endpoint_exercised'])
        self.assertEqual(c.p0_reference_target_layers_v2,())
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)
        self.assertEqual(sum(store._lease_counts.values()),0)
        self.assertEqual(outer.calls,[])

    def test_target_r1_backend_hook_removal_failure_keeps_both_leases(self):
        c,store,snapshot,job,model,outer=self.r1_fixture()
        kernel=model.layers[1].self_attn.attn
        register=kernel.register_forward_hook; retained=[]
        def faulty(*args,**kwargs):
            handle=register(*args,**kwargs); retained.append(handle)
            return NS(remove=Mock(side_effect=RuntimeError('backend hook remove failed')))
        def recover():
            for handle in retained: handle.remove()
            owner=c.p0_diagnostic_resources_v2
            if owner and owner.hooks:
                owner.hooks.cleanup_failure=None
                owner.hooks._kernel_handle=None
            if owner: owner.quarantined=False
            c.close()
        self.addCleanup(recover)
        with patch.object(kernel,'register_forward_hook',side_effect=faulty):
            with self.assertRaises(P0RequestFailure) as failure:self.execute(c,store,snapshot,job)
        self.assertFalse(failure.exception.audit['cleanup']['passed'])
        self.assertFalse(failure.exception.audit['execution']['device_references_released'])
        self.assertEqual(sum(store._lease_counts.values()),2)
        self.assertGreater(c.adapter.hbm.active_reserved_bytes,0)
        self.assertEqual(outer.calls,[])

    def test_source_relocation_uses_current_absolute_rows_not_birth_offset(self):
        c,store,snapshot,job,model,outer=self.fixture()
        c.request['token_ids']=[70,12,13,90,91,92,93]
        c.request['segments'][0]['positions']=[1,2]
        c.request['segments'][1]['positions']=[4,5]
        c._prepared_inputs=(torch.tensor(c.request['token_ids']),torch.arange(7))
        job['repair_positions_by_layer']={'2':[2],'3':[2]}
        result=self.execute(c,store,snapshot,job)
        self.assertEqual(result['recipe']['source_current_positions'],[1,2])
        self.assertEqual(result['recipe']['source_birth_positions'],[2,3])
        for row in result['execution']['executed_layers'][1:]:
            self.assertEqual(row['historical_kv_positions'],[1])
            self.assertEqual(row['target_positions'],[4,5])
            self.assertIn(0,row['attention_query_positions'])
            self.assertNotIn(1,row['attention_query_positions'])

    def test_working_composite_budget_is_not_hidden(self):
        c,store,snapshot,job,model,outer=self.fixture()
        with self.assertRaises(P0RequestFailure) as failure:
            self.execute(c,store,snapshot,job,cuda_reference_bytes=220)
        self.assertIn('exceeds explicit bound',str(failure.exception))
        self.assertEqual(model.calls,[])
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)
        self.assertEqual(sum(store._lease_counts.values()),0)

    def test_bad_runtime_status_cannot_claim_sparse_execution(self):
        c,store,snapshot,job,model,outer=self.fixture(); model.bad_status=True
        with self.assertRaises(P0RequestFailure) as failure:
            self.execute(c,store,snapshot,job)
        self.assertEqual(failure.exception.audit['status'],'FAILED')
        self.assertEqual(outer.calls,[])
        self.assertEqual(c.p0_reference_target_layers_v2,())
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)

    def test_incomplete_block_and_wrong_row_count_fail_closed(self):
        for mode in ('block','rows'):
            with self.subTest(mode=mode):
                c,store,snapshot,job,model,outer=self.fixture()
                if mode=='block': model.layers[1].fail=True
                else: model.bad_rows=True
                with self.assertRaises(P0RequestFailure): self.execute(c,store,snapshot,job)
                self.assertEqual(outer.calls,[])
                self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)
                self.assertEqual(sum(store._lease_counts.values()),0)
                self.assertEqual(c.p0_reference_target_layers_v2,())

    def test_source_rows_must_actually_be_installed(self):
        c,store,snapshot,job,model,outer=self.fixture()
        with patch.object(CacheBlendV6OnlineEngine,'_install_ready_source_rows',return_value=None):
            with self.assertRaises(P0RequestFailure) as failure: self.execute(c,store,snapshot,job)
        self.assertIn('not installed',str(failure.exception))
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)

    def test_corrupted_installed_rows_fail_before_execution(self):
        original=CacheBlendV6OnlineEngine._install_ready_source_rows
        def corrupt(engine,layer):
            original(engine,layer)
            if layer==2: engine._composite_old_kvs[layer-1][1][0].add_(1)
        c,store,snapshot,job,model,outer=self.fixture()
        with patch.object(CacheBlendV6OnlineEngine,'_install_ready_source_rows',new=corrupt):
            with self.assertRaises(P0RequestFailure) as failure: self.execute(c,store,snapshot,job)
        self.assertIn('differs before RoPE',str(failure.exception))
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)

    def test_projection_hook_removal_failure_quarantines_not_cleanup_pass(self):
        c,store,snapshot,job,model,outer=self.fixture()
        projection=model.layers[0].self_attn.qkv_proj
        register=projection.register_forward_hook
        retained=[]
        def faulty_register(*args,**kwargs):
            real=register(*args,**kwargs); retained.append(real)
            return NS(remove=Mock(side_effect=RuntimeError('hook removal failed')))
        # Test recovery is explicit after retaining evidence; no runtime retry
        # declares a still-hooked model healthy.
        def recover():
            for handle in retained: handle.remove()
            owner=c.p0_diagnostic_resources_v2
            if owner and owner.hooks:
                owner.hooks.cleanup_failure=None
                owner.hooks._attention_handle=None
                owner.hooks._current_capture=None
            if owner: owner.quarantined=False
        self.addCleanup(recover)
        with patch.object(projection,'register_forward_hook',side_effect=faulty_register):
            with self.assertRaises(P0RequestFailure) as failure: self.execute(c,store,snapshot,job)
        audit=failure.exception.audit
        self.assertFalse(audit['cleanup']['passed'])
        self.assertIsNotNone(audit['execution']['cleanup_failure'])
        self.assertFalse(audit['execution']['device_references_released'])
        self.assertTrue(c.p0_diagnostic_resources_v2.quarantined)
        self.assertGreater(c.adapter.hbm.active_reserved_bytes,0)
        self.assertGreater(sum(store._lease_counts.values()),0)
        self.assertEqual(outer.calls,[])

    def test_failed_fence_retains_working_composite_and_lease(self):
        c,store,snapshot,job,model,outer=self.fixture()
        calls=[0]
        def failing_fence():
            calls[0]+=1
            if calls[0]>=2: raise RuntimeError('CPU simulated CUDA fence failure')
        c.synchronize.side_effect=failing_fence
        with self.assertRaises(P0RequestFailure) as failure: self.execute(c,store,snapshot,job)
        self.assertFalse(failure.exception.audit['cleanup']['passed'])
        self.assertTrue(c.engine._composite_old_kvs)
        self.assertGreater(c.adapter.hbm.active_reserved_bytes,0)
        self.assertGreater(sum(store._lease_counts.values()),0)
        self.assertEqual(outer.calls,[])

    def test_target_never_receives_own_old_source_or_repair_subset(self):
        c,store,snapshot,job,model,outer=self.fixture()
        result=self.execute(c,store,snapshot,job)
        for row in result['execution']['executed_layers']:
            self.assertEqual(row['target_positions'],[3,4])
            self.assertTrue({3,4} <= set(row['projected_positions']))
            self.assertTrue({3,4} <= set(row['attention_query_positions']))
            self.assertFalse({3,4} & set(row['historical_kv_positions']))
        self.assertFalse(c.native.finish_prefill.call_args.kwargs['exact_dense'])
        self.assertEqual(c.adapter.warm_history,[])
