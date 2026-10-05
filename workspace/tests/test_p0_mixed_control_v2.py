"""CPU driver integration with actual files/hooks; never GPU evidence."""
from contextlib import ExitStack
from copy import deepcopy
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from probekv.native_p0_request_v2 import P0RequestFailure
from probekv.p0_mixed_control_v2 import (
    execute_explicit_mixed_reference, validate_mixed_reference_job)
from probekv.v8_schema10_native_adapter import NativeRequestContext
from probekv.v8_schema6_hbm import UnifiedHBMReservationManager
from tests import test_p0_mixed_reference_v2 as model_fixtures
from tests import test_source_store_v2 as store_fixtures


class _ToyOuter:
    def __init__(self, model):
        self.model = model
        self.calls = []
        self.decode_hook_counts = []

    def __call__(self, *, input_ids, positions, kv_caches, attn_metadata):
        self.calls.append((input_ids.tolist(), positions.tolist()))
        if len(input_ids) == 1:
            counts = [(len(b._forward_pre_hooks), len(b._forward_hooks),
                       len(b.self_attn.qkv_proj._forward_hooks)) for b in self.model.layers]
            self.decode_hook_counts.append(counts)
            if any(any(row) for row in counts):
                raise AssertionError('reference hook survived into decode')
        hidden = torch.sin(input_ids.float()[:,None]*.1 + torch.arange(8)[None,:]*.21)
        return self.model(hidden.to(torch.bfloat16))

    def compute_logits(self, hidden, sampling):
        return hidden[-1:].float()


class P0MixedControlTests(unittest.TestCase):
    def fixture(self, r1=False, target_mixed=False):
        f = store_fixtures.TargetSourceStoreTests(); f.setUp()
        self.addCleanup(f.doCleanups)
        store = f.store()
        published = f.publish(store, f.candidate('historical-birth', base=3))
        target_published = None
        if r1:
            f.tokens = (10,11,91,92)
            target_published = f.publish(store,
                f.candidate('historical-target-birth',base=100,mixed=target_mixed),tokens=(91,92))
        snapshot = store.begin_request('mixed-reference')
        self.addCleanup(lambda:store.end_request(snapshot) if snapshot.snapshot_id in store._snapshots else None)
        row = store._visible_row(snapshot, published['source_id'])
        model = model_fixtures.ToyModel(); model.cache_fuse_metadata = {}
        model.old_kvs = [[None,None] for _ in range(3)]
        outer = _ToyOuter(model)
        request = dict(request_id='mixed-reference', token_ids=[12,13,90,91,92,93],
            segments=[dict(segment_id='U',positions=[0,1],token_ids=[12,13]),
                      dict(segment_id='T',positions=[3,4],token_ids=[91,92])],
            capture_logits=True, teacher_token_ids=[7,8], max_new_tokens=3)
        context = object.__new__(NativeRequestContext)
        context.request = request; context.arrival_ns = time.perf_counter_ns()
        context.finished = context.closed = False
        context.cached_prefix_tokens = 0; context.probe_fallback_reason = None
        context.prepared = {}; context.frozen = {}; context.committed = {}
        context.engine = None; context.repair_ratio = .15
        context.source_capture_v2 = context.source_consumption_v2 = None
        context.capture_reservation = context.capture_collector = None
        context.target_candidates_v2 = {}; context.canonical_exports = {}
        context.p0_diagnostic_v2 = context.p0_diagnostic_resources_v2 = None
        context._prepared_inputs = (torch.tensor(request['token_ids']),torch.arange(6))
        context.attention = NS(); context.sampling = NS(selected_token_indices=torch.tensor([5]))
        context.sampling_signature = dict(max_new_tokens=3)
        context.finish_timing_landmarks = {}; context.hot_leases = ExitStack()
        context.hot_replicas = {}; context._observation = {}; context.workspace = None
        context.synchronize = Mock()
        context.native = NS(finish_prefill=Mock(), finish_decode_step=Mock(),
            append_for_decode=Mock(side_effect=lambda token:NS(token=token)))
        decoded = []
        def prepare(metadata):
            decoded.append(metadata.token)
            return (torch.tensor([metadata.token]), torch.tensor([5+len(decoded)]),
                    NS(), NS(selected_token_indices=torch.tensor([0])))
        context.adapter = NS(torch=torch, spec=NS(num_layers=3), inner=model, outer=outer,
            kv=[], provenance=dict(model_signature='model',tokenizer_hash='tokenizer'),
            hbm=UnifiedHBMReservationManager(allocator_capacity_bytes=100000, safety_bytes=0),
            check_deadline=Mock(), prepare=prepare, warm_history=[],
            llm=NS(get_tokenizer=lambda:NS(eos_token_id=999)))
        job = dict(operation='explicit_mixed_reference', request=request, target_id='T',
            upstream_segment_id='U', source=dict(source_id=row['source_id'], artifact_digest=row['artifact_digest']),
            first_reuse_layer=2, repair_positions_by_layer={'2':[1],'3':[1]})
        if r1:
            target_row=store._visible_row(snapshot,target_published['source_id'])
            job.update(target_execution='R1_ALL_LAYERS',target_source=dict(
                source_id=target_row['source_id'],artifact_digest=target_row['artifact_digest']))
        def recover_fixture():
            # Test-only explicit recovery after the simulated fence is healthy.
            context.synchronize.side_effect = None
            owner = context.p0_diagnostic_resources_v2
            if owner is not None: owner.quarantined = False
            context.close()
        self.addCleanup(recover_fixture)
        return context,store,snapshot,job,model,outer

    def execute(self, context, store, snapshot, job, **overrides):
        options = dict(host_reference_bytes=4096,cuda_reference_bytes=4096)
        options.update(overrides)
        return execute_explicit_mixed_reference(context,store,snapshot,job,**options)

    def test_actual_full_prefill_teacher_endpoint_target_integrity_and_readonly_pool(self):
        c,store,snapshot,job,model,outer = self.fixture()
        catalog = deepcopy(store._catalog); catalog_bytes = (store.root/'catalog.json').read_bytes()
        audit_before = store.storage_audit(); events = deepcopy(store.events)
        snapshots = deepcopy(store._snapshots)
        with patch.object(store,'commit_publication',side_effect=AssertionError('reference cannot publish')):
            result = self.execute(c,store,snapshot,job)
        self.assertEqual(result['status'],'COMPLETED')
        self.assertEqual(result['publication']['status'],'DIAGNOSTIC_NOT_PUBLISHED')
        # The separate reference prefill is charged as diagnostic work, not
        # hidden as online reuse or Source materialization.
        self.assertEqual(result['extra_forward_count'],1)
        self.assertEqual(outer.calls[0],(job['request']['token_ids'],list(range(6))))
        self.assertEqual([row[0] for row in outer.calls[1:]],[[7],[8]])
        self.assertEqual(len(outer.calls),3)  # One prefill plus two teacher decode steps.
        self.assertEqual(outer.decode_hook_counts,[[(0,0,0)]*3]*2)
        self.assertEqual(result['answer']['whole_request_origin'],'p0_explicit_mixed_reference')
        self.assertFalse(c.native.finish_prefill.call_args.kwargs['exact_dense'])
        self.assertEqual(c.adapter.warm_history,[])
        self.assertEqual(result['answer']['decode_input_trace_v2']['fed_token_ids'],[7,8])
        self.assertEqual(len(c.logit_trace),3)
        self.assertEqual(len(c.p0_reference_target_layers_v2),3)
        for pair in c.p0_reference_target_layers_v2:
            for value in pair:
                self.assertEqual(tuple(value.shape),(2,2,2))
                self.assertEqual(value.dtype,torch.bfloat16)
                self.assertEqual(value.device.type,'cpu')
                self.assertIsNone(value._base)
        self.assertTrue(result['integrity']['passed'])
        self.assertEqual({result['integrity'][key] for key in
            ('source_before','destination','source_after','destination_after')},{job['source']['artifact_digest']})
        self.assertEqual(store._catalog,catalog)
        self.assertEqual((store.root/'catalog.json').read_bytes(),catalog_bytes)
        self.assertEqual(store.storage_audit(),audit_before)
        self.assertEqual(store.events,events); self.assertEqual(store._snapshots,snapshots)
        self.assertEqual(sum(store._lease_counts.values()),0)
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)
        self.assertTrue(c.closed); self.assertTrue(result['cleanup']['passed'])
        self.assertTrue(c.p0_diagnostic_resources_v2.closed)
        self.assertFalse(c.p0_diagnostic_resources_v2.source_layers)
        self.assertFalse(result['production_reuse_commit_observed'])
        self.assertFalse(result['native_runtime_qualified']); self.assertFalse(result['P1_execution_allowed'])

    def test_wrong_source_identity_or_geometry_rejected_before_forward(self):
        for mode in ('source_id','artifact_digest','geometry','model','tokenizer','positions'):
            with self.subTest(mode=mode):
                c,store,snapshot,job,model,outer = self.fixture()
                if mode=='source_id': job['source']['source_id']='absent'
                elif mode=='artifact_digest': job['source']['artifact_digest']='f'*64
                elif mode=='geometry': model.layers[0].self_attn.num_kv_heads=3
                elif mode=='model': c.adapter.provenance['model_signature']='different'
                elif mode=='tokenizer': c.adapter.provenance['tokenizer_hash']='different'
                else: c._prepared_inputs=(c._prepared_inputs[0],torch.arange(1,7))
                with self.assertRaises((ValueError,KeyError)):
                    self.execute(c,store,snapshot,job)
                self.assertEqual(outer.calls,[])
                self.assertEqual(sum(store._lease_counts.values()),0)
                self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)

    def test_explicit_resource_caps_reject_before_forward(self):
        for options in (dict(host_reference_bytes=1),dict(cuda_reference_bytes=1)):
            c,store,snapshot,job,model,outer = self.fixture()
            with self.assertRaises(MemoryError): self.execute(c,store,snapshot,job,**options)
            self.assertEqual(outer.calls,[])
            self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)

    def test_invalid_masks_target_overlap_and_flags_reject_before_forward(self):
        mutations = [lambda j:j.update(repair_positions_by_layer={'2':[1]}),
            lambda j:j.update(repair_positions_by_layer={'2':[1],'3':[0]}),
            lambda j:j.update(repair_positions_by_layer={'2':[0,1],'3':[0,1]}),
            lambda j:j['request']['segments'][1].update(positions=[1,2],token_ids=[13,90]),
            lambda j:j['request'].update(native_dense_continuation=True),
            lambda j:j['request'].update(publish_exact_prefix_shadow=True),
            lambda j:j['request'].update(correctness_repair_ratio=1.),
            lambda j:j.update(sources_by_segment={})]
        for mutate in mutations:
            c,store,snapshot,job,model,outer = self.fixture(); mutate(job)
            with self.assertRaises(ValueError): self.execute(c,store,snapshot,job)
            self.assertEqual(outer.calls,[])
            self.assertEqual(sum(store._lease_counts.values()),0)

    def test_actual_block_failure_detaches_hooks_and_releases_all_resources(self):
        c,store,snapshot,job,model,outer = self.fixture(); model.layers[1].fail=True
        with self.assertRaises(P0RequestFailure) as raised: self.execute(c,store,snapshot,job)
        audit=raised.exception.audit
        self.assertEqual(audit['status'],'FAILED'); self.assertIsNone(audit['answer'])
        self.assertTrue(audit['cleanup']['passed']); self.assertTrue(c.closed)
        self.assertEqual(c.p0_reference_target_layers_v2,())
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)
        self.assertEqual(sum(store._lease_counts.values()),0)
        for block in model.layers:
            self.assertFalse(block._forward_pre_hooks); self.assertFalse(block._forward_hooks)
            self.assertFalse(block.self_attn.qkv_proj._forward_hooks)

    def test_cleanup_fence_failure_preserves_quarantine_and_lease(self):
        c,store,snapshot,job,model,outer = self.fixture()
        count=[]
        def fence():
            count.append(1)
            if len(count)>=3: raise RuntimeError('simulated final fence failure')
        c.synchronize.side_effect=fence
        with self.assertRaises(P0RequestFailure) as raised: self.execute(c,store,snapshot,job)
        audit=raised.exception.audit
        self.assertEqual(audit['status'],'FAILED'); self.assertFalse(audit['cleanup']['passed'])
        self.assertIsNotNone(audit['answer'])
        self.assertTrue(c.p0_diagnostic_resources_v2.quarantined)
        self.assertFalse(c.closed); self.assertGreater(c.adapter.hbm.active_reserved_bytes,0)
        self.assertEqual(sum(store._lease_counts.values()),1)
        self.assertEqual(c.p0_reference_target_layers_v2,())

    def test_missing_first_token_callback_cannot_complete_reference(self):
        c,store,snapshot,job,model,outer = self.fixture()
        endpoint=c.finish_from_prefill_hidden
        c.finish_from_prefill_hidden=lambda hidden,callback:endpoint(hidden,lambda:None)
        with self.assertRaises(P0RequestFailure) as raised: self.execute(c,store,snapshot,job)
        self.assertEqual(raised.exception.audit['status'],'FAILED')
        self.assertIsNotNone(raised.exception.audit['answer'])
        self.assertTrue(raised.exception.audit['cleanup']['passed'])

    def test_hbm_reservation_failure_releases_source_lease_before_any_block(self):
        c,store,snapshot,job,model,outer = self.fixture()
        c.adapter.hbm=UnifiedHBMReservationManager(allocator_capacity_bytes=128,safety_bytes=0)
        with self.assertRaises(P0RequestFailure) as raised: self.execute(c,store,snapshot,job)
        self.assertIsInstance(raised.exception.cause,MemoryError)
        self.assertEqual(outer.calls,[])
        self.assertTrue(raised.exception.audit['cleanup']['passed'])
        self.assertEqual(sum(store._lease_counts.values()),0)
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)

    def test_wrong_source_file_digest_rejects_without_model_execution(self):
        c,store,snapshot,job,model,outer = self.fixture()
        with patch('probekv.p0_mixed_control_v2.file_digest',return_value='f'*64):
            with self.assertRaises(P0RequestFailure) as raised: self.execute(c,store,snapshot,job)
        self.assertIn('file digest',str(raised.exception.cause))
        self.assertEqual(outer.calls,[])
        self.assertTrue(raised.exception.audit['cleanup']['passed'])
        self.assertEqual(sum(store._lease_counts.values()),0)

    def test_deadline_checked_between_model_blocks(self):
        c,store,snapshot,job,model,outer = self.fixture()
        def deadline():
            if sum(b.self_attn.qkv_proj.calls for b in model.layers)>=1:
                raise TimeoutError('test deadline after first block')
        c.adapter.check_deadline.side_effect=deadline
        with self.assertRaises(P0RequestFailure) as raised: self.execute(c,store,snapshot,job)
        self.assertIsInstance(raised.exception.cause,TimeoutError)
        self.assertEqual([b.self_attn.qkv_proj.calls for b in model.layers],[1,0,0])
        self.assertTrue(raised.exception.audit['cleanup']['passed'])
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)
        self.assertEqual(sum(store._lease_counts.values()),0)

    def test_hook_removal_failure_quarantines_before_decode_and_is_not_cleanup_pass(self):
        c,store,snapshot,job,model,outer = self.fixture()
        original = torch.utils.hooks.RemovableHandle.remove
        removed = []
        def faulty_remove(handle):
            removed.append(handle.id)
            if len(removed)==1: raise RuntimeError('simulated hook removal failure')
            return original(handle)
        try:
            with patch.object(torch.utils.hooks.RemovableHandle,'remove',faulty_remove):
                with self.assertRaises(P0RequestFailure) as raised:
                    self.execute(c,store,snapshot,job)
            audit=raised.exception.audit
            self.assertEqual(audit['status'],'FAILED')
            self.assertFalse(audit['cleanup']['passed']); self.assertIsNone(audit['answer'])
            self.assertEqual(len(outer.calls),1); self.assertFalse(outer.decode_hook_counts)
            self.assertFalse(c.closed)
            self.assertTrue(c.p0_diagnostic_resources_v2.quarantined)
            self.assertGreater(c.adapter.hbm.active_reserved_bytes,0)
            self.assertEqual(sum(store._lease_counts.values()),1)
        finally:
            # Explicit CPU-test recovery only: production requires independent
            # recovery rather than silently forgetting failed hook removal.
            for block in model.layers:
                block._forward_pre_hooks.clear(); block._forward_hooks.clear()
                block.self_attn.qkv_proj._forward_hooks.clear()
            owner=c.p0_diagnostic_resources_v2
            if owner is not None and owner.hooks is not None:
                owner.hooks.cleanup_failure=None

    def test_r1_reference_leases_both_inputs_but_does_not_inject_historical_target(self):
        c,store,snapshot,job,model,outer=self.fixture(r1=True)
        observations=[]
        def during():
            owner=c.p0_diagnostic_resources_v2
            if owner is not None and owner.target_r1_layers:
                observations.append((sum(store._lease_counts.values()),len(owner.target_r1_layers)))
        c.adapter.check_deadline.side_effect=during
        audit=self.execute(c,store,snapshot,job)
        self.assertEqual(audit['recipe']['target_execution'],'R1_ALL_LAYERS')
        self.assertEqual(audit['recipe']['target_source']['source_id'],job['target_source']['source_id'])
        self.assertEqual(audit['recipe']['target_source']['source_generation'],0)
        self.assertEqual(audit['recipe']['target_source']['source_current_positions'],[3,4])
        self.assertEqual({audit['target_source_integrity'][k] for k in
            ('source_before','destination','source_after','destination_after')},{job['target_source']['artifact_digest']})
        self.assertTrue(audit['target_source_integrity']['passed'])
        self.assertTrue(observations); self.assertTrue(all(n==2 for n,_ in observations))
        for event in audit['reference']['executed_layers']:
            self.assertFalse(set(event['historical_kv_positions'])&{3,4})
            self.assertTrue(event['target_full_projection'])
        owner=c.p0_diagnostic_resources_v2
        self.assertFalse(owner.target_r1_layers); self.assertFalse(owner.target_r1_acquired)
        self.assertEqual(sum(store._lease_counts.values()),0)
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)
        self.assertFalse(audit['production_reuse_commit_observed'])
        self.assertEqual(audit['publication']['status'],'DIAGNOSTIC_NOT_PUBLISHED')

    def test_r1_reference_output_does_not_depend_on_unused_target_source_values(self):
        c,store,snapshot,job,model,outer=self.fixture(r1=True)
        audit=self.execute(c,store,snapshot,job)
        actual=tuple(tuple(t.clone() for t in pair) for pair in c.p0_reference_target_layers_v2)
        # A separate deterministic CPU toy request has identical upstream but
        # no target Source at all. Normal-full reference must be identical.
        d,pool,snap,plain,_,_=self.fixture()
        other=self.execute(d,pool,snap,plain)
        self.assertEqual(audit['answer']['token_ids'],other['answer']['token_ids'])
        for left,right in zip(actual,d.p0_reference_target_layers_v2):
            for a,b in zip(left,right): self.assertTrue(torch.equal(a,b))

    def test_r1_target_source_required_only_in_r1_contract(self):
        mutations=[lambda j:j.update(target_execution='R1_ALL_LAYERS'),
            lambda j:j.update(target_execution='invalid'),
            lambda j:j.update(target_source={'source_id':'other','artifact_digest':'b'*64}),
            lambda j:j.update(target_execution='R1_ALL_LAYERS',target_source={'source_id':'other'}),
            lambda j:j.update(target_execution='R1_ALL_LAYERS',target_source={'source_id':'other','artifact_digest':'bad'})]
        for change in mutations:
            c,store,snapshot,job,model,outer=self.fixture(); change(job)
            with self.assertRaises(ValueError): self.execute(c,store,snapshot,job)
            self.assertEqual(outer.calls,[])

    def test_r1_target_wrong_source_or_g1_rejected_before_forward(self):
        for mode in ('wrong_digest','wrong_content','mixed'):
            c,store,snapshot,job,model,outer=self.fixture(r1=True,target_mixed=mode=='mixed')
            if mode=='wrong_digest': job['target_source']['artifact_digest']='f'*64
            elif mode=='wrong_content': job['target_source']=dict(job['source'])
            with self.assertRaises(ValueError): self.execute(c,store,snapshot,job)
            self.assertEqual(outer.calls,[])
            self.assertEqual(sum(store._lease_counts.values()),0)

    def test_r1_memory_budgets_include_both_sources(self):
        for options in (dict(host_reference_bytes=250),dict(cuda_reference_bytes=192)):
            c,store,snapshot,job,model,outer=self.fixture(r1=True)
            with self.assertRaises(MemoryError): self.execute(c,store,snapshot,job,**options)
            self.assertEqual(outer.calls,[])
            self.assertEqual(sum(store._lease_counts.values()),0)

    def test_r1_partial_target_copy_failure_releases_both_after_fence(self):
        c,store,snapshot,job,model,outer=self.fixture(r1=True)
        original=torch.Tensor.to; copies=[]
        def copy(tensor,*args,**kwargs):
            if kwargs.get('copy') is True:
                copies.append(1)
                if len(copies)==8: raise MemoryError('simulated target second tensor copy failure')
            return original(tensor,*args,**kwargs)
        with patch.object(torch.Tensor,'to',copy):
            with self.assertRaises(P0RequestFailure) as raised: self.execute(c,store,snapshot,job)
        self.assertTrue(raised.exception.audit['cleanup']['passed'])
        self.assertEqual(outer.calls,[])
        self.assertEqual(sum(store._lease_counts.values()),0)
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)
        self.assertFalse(c.p0_diagnostic_resources_v2.target_r1_layers)

    def test_r1_failed_fence_preserves_both_leases_and_partial_allocations(self):
        c,store,snapshot,job,model,outer=self.fixture(r1=True)
        original=torch.Tensor.to; copies=[]
        def copy(tensor,*args,**kwargs):
            if kwargs.get('copy') is True:
                copies.append(1)
                if len(copies)==8: raise MemoryError('simulated target copy failure')
            return original(tensor,*args,**kwargs)
        c.synchronize.side_effect=RuntimeError('failed completion fence')
        with patch.object(torch.Tensor,'to',copy):
            with self.assertRaises(P0RequestFailure) as raised: self.execute(c,store,snapshot,job)
        owner=c.p0_diagnostic_resources_v2
        self.assertFalse(raised.exception.audit['cleanup']['passed'])
        self.assertTrue(owner.quarantined)
        self.assertEqual(sum(store._lease_counts.values()),2)
        self.assertGreater(c.adapter.hbm.active_reserved_bytes,0)
        self.assertTrue(owner.source_layers); self.assertTrue(owner.target_r1_layers)

    def test_r1_both_leases_end_before_hbm_reservation_release(self):
        c,store,snapshot,job,model,outer=self.fixture(r1=True)
        original=c.adapter.hbm.release; observed=[]
        def release(identifier):
            observed.append(sum(store._lease_counts.values()))
            return original(identifier)
        c.adapter.hbm.release=release
        self.execute(c,store,snapshot,job)
        self.assertEqual(observed,[0])


if __name__=='__main__': unittest.main()
