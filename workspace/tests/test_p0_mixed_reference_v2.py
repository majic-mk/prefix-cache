"""CPU mathematical hooks only: not real-model, CUDA, timing or P0 qualification."""
import unittest
from unittest.mock import patch

import torch

from probekv.p0_mixed_reference_v2 import ExplicitMixedReferenceHooks


class ToyProjection(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.register_buffer('weight', torch.randn((8,16), generator=generator)*.2)
        self.calls = 0
        self.last = None
        self.aux = object()

    def forward(self, hidden):
        self.calls += 1
        self.last = (hidden.float() @ self.weight).to(torch.bfloat16)
        return self.last, self.aux, 'aux-preserved'


def attention_math(qkv, rows, total_rows):
    """Post-RoPE causal GQA; rows indexes queries, all K/V rows are visible."""
    q = qkv[rows, :8].float().reshape(len(rows),4,2)
    k = qkv[:, 8:12].float().reshape(total_rows,2,2)
    v = qkv[:, 12:].float().reshape(total_rows,2,2)
    def rope(t, positions):
        angle = torch.tensor(positions,dtype=torch.float32)[:,None]*.13
        cosine, sine = torch.cos(angle), torch.sin(angle)
        return torch.stack((t[:,:,0]*cosine-t[:,:,1]*sine,
                            t[:,:,0]*sine+t[:,:,1]*cosine),dim=-1)
    q = rope(q,rows)
    k = rope(k,list(range(total_rows))).repeat_interleave(2,dim=1)
    v = v.repeat_interleave(2,dim=1)
    scores = torch.einsum('qhd,khd->hqk',q,k)/(2**.5)
    causal = torch.arange(total_rows)[None,:] > torch.tensor(rows)[:,None]
    scores.masked_fill_(causal[None,:,:],float('-inf'))
    return torch.einsum('hqk,khd->qhd',scores.softmax(dim=-1),v).reshape(len(rows),8)


class ToyAttention(torch.nn.Module):
    def __init__(self,generator):
        super().__init__()
        self.q_size=8; self.kv_size=4; self.num_heads=4; self.num_kv_heads=2; self.head_dim=2
        self.qkv_proj=ToyProjection(generator)
        self.returned=None

    def forward(self, hidden):
        qkv,aux,tag=self.qkv_proj(hidden)
        if aux is not self.qkv_proj.aux or tag!='aux-preserved':
            raise AssertionError('projection auxiliaries changed')
        self.returned=qkv
        return attention_math(qkv,list(range(len(hidden))),len(hidden))


class ToyBlock(torch.nn.Module):
    def __init__(self,generator):
        super().__init__()
        self.self_attn=ToyAttention(generator)
        self.fail=False

    def forward(self, hidden):
        normalized=hidden.float()*torch.rsqrt(hidden.float().square().mean(-1,keepdim=True)+1e-5)
        result=self.self_attn(normalized.to(torch.bfloat16))
        if self.fail:
            raise RuntimeError('intentional block failure')
        return (hidden.float()+.15*result+.1*torch.tanh(hidden.float())).to(torch.bfloat16)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        gen=torch.Generator().manual_seed(531)
        self.layers=torch.nn.ModuleList([ToyBlock(gen) for _ in range(3)])

    def forward(self, hidden):
        for block in self.layers:
            hidden=block(hidden)
        return hidden


def prescribed_reference(model, hidden, sources, *, reduced):
    """Independent direct math, no hook and no production execution ledger."""
    hidden=hidden.clone(); targets=[]
    for i,block in enumerate(model.layers,1):
        replacements=() if i==1 else ((0,) if i==2 else (0,1))
        active=[p for p in range(len(hidden)) if not reduced or p not in replacements]
        norm=hidden[active].float()*torch.rsqrt(hidden[active].float().square().mean(-1,keepdim=True)+1e-5)
        current=(norm.to(torch.bfloat16).float() @ block.self_attn.qkv_proj.weight).to(torch.bfloat16)
        # Full query reference computes the discarded rows; reduced reference
        # does not. Both construct exactly the same attended K/V tensor.
        qkv=torch.zeros((len(hidden),16),dtype=torch.bfloat16)
        qkv[active]=current
        for p in replacements:
            qkv[p,8:12]=sources[i-1][0][p].reshape(4)
            qkv[p,12:]=sources[i-1][1][p].reshape(4)
        targets.append((qkv[3:5,8:12].clone().reshape(2,2,2),
                        qkv[3:5,12:].clone().reshape(2,2,2)))
        out=attention_math(qkv,active,len(hidden))
        hidden[active]=(hidden[active].float()+.15*out+.1*torch.tanh(hidden[active].float())).to(torch.bfloat16)
    return hidden,targets


class MixedReferenceHooksTests(unittest.TestCase):
    def fixture(self):
        generator=torch.Generator().manual_seed(114)
        model=ToyModel()
        hidden=torch.randn((6,8),generator=generator).to(torch.bfloat16)
        sources=tuple(tuple(torch.randn((3,2,2),generator=generator).to(torch.bfloat16)
                            for _ in range(2)) for _ in range(3))
        return model,hidden,sources

    def scope(self,model,sources,**overrides):
        kwargs=dict(token_count=6,target_positions=(3,4),source_positions=(0,1,2),
            repair_positions_by_layer={2:(1,2),3:(2,)}, first_reuse_layer=2,
            source_layers=sources,max_target_host_bytes=96,max_transient_device_bytes=1024)
        kwargs.update(overrides)
        return ExplicitMixedReferenceHooks(model,**kwargs)

    def assert_no_hooks(self,model):
        for block in model.layers:
            self.assertFalse(block._forward_pre_hooks)
            self.assertFalse(block._forward_hooks)
            self.assertFalse(block.self_attn.qkv_proj._forward_hooks)

    def test_gqa_rope_matches_manual_full_and_reduced_causal_reference(self):
        model,hidden,sources=self.fixture()
        manual,captured=prescribed_reference(model,hidden,sources,reduced=False)
        reduced,reduced_captured=prescribed_reference(model,hidden,sources,reduced=True)
        originals=tuple(tuple(t.clone() for t in pair) for pair in sources)
        with patch.object(torch.cuda,'synchronize',side_effect=AssertionError('CPU must not call CUDA')):
            with self.scope(model,sources) as hooks:
                result=model(hidden)
                with self.assertRaisesRegex(RuntimeError,'successfully completed'):
                    _=hooks.target_layers
        self.assertTrue(torch.equal(result,manual))
        self.assertTrue(torch.equal(result[3:],reduced[3:]))
        for layer,(actual,expected,recaptured) in enumerate(zip(hooks.target_layers,captured,reduced_captured)):
            for a,e,r in zip(actual,expected,recaptured):
                self.assertTrue(torch.equal(a,e)); self.assertTrue(torch.equal(a,r))
                self.assertIsNone(a._base)
                self.assertEqual(a.untyped_storage().nbytes(),a.numel()*a.element_size())
            self.assertEqual(model.layers[layer].self_attn.qkv_proj.calls,1)
            for a,b in zip(sources[layer],originals[layer]):
                self.assertTrue(torch.equal(a,b))
        audit=hooks.audit()
        self.assertTrue(audit['completed']); self.assertEqual(audit['historical_kv_rows_replaced'],3)
        self.assertEqual(audit['target_owned_host_bytes'],96)
        self.assertEqual(audit['parent_owned_kv_bytes'],0); self.assertEqual(audit['prefix_shadow_bytes'],0)
        self.assertEqual(len(audit['target_logical_digest']),64)
        self.assertFalse(audit['publication_allowed']); self.assertFalse(audit['native_runtime_qualified'])
        self.assert_no_hooks(model)

    def test_original_projection_query_target_and_repaired_rows_not_modified(self):
        model,hidden,sources=self.fixture()
        with self.scope(model,sources):
            model(hidden)
        for layer,block in enumerate(model.layers,1):
            original=block.self_attn.qkv_proj.last; mixed=block.self_attn.returned
            self.assertNotEqual(original.data_ptr(),mixed.data_ptr())
            self.assertTrue(torch.equal(original[:,:8],mixed[:,:8]))
            unchanged=list(range(6)) if layer==1 else ([1,2,3,4,5] if layer==2 else [2,3,4,5])
            self.assertTrue(torch.equal(original[unchanged],mixed[unchanged]))
            if layer>1:
                self.assertTrue(torch.equal(mixed[0,8:12],sources[layer-1][0][0].reshape(4)))

    def test_mixed_target_differs_from_whole_request_dense(self):
        model,hidden,sources=self.fixture()
        dense=model(hidden)
        with self.scope(model,sources) as hooks:
            mixed=model(hidden)
        self.assertFalse(torch.equal(dense[3:],mixed[3:]))
        self.assertEqual(hooks.audit()['reference_scope'],'fixed_upstream_recipe_only_not_dense_equivalence')

    def test_invalid_causal_or_noncontiguous_ranges(self):
        for change in ({'source_positions':(2,3,4)}, {'target_positions':(3,5)},
                       {'source_positions':(0,2)}, {'token_count':True}):
            model,_,sources=self.fixture()
            with self.subTest(change=change), self.assertRaises(ValueError):
                with self.scope(model,sources,**change):
                    pass
            self.assert_no_hooks(model)

    def test_boundary_recipe_and_reentry_validation(self):
        for change in ({'first_reuse_layer':0},{'first_reuse_layer':4},
                {'repair_positions_by_layer':{2:(1,2)}},
                {'repair_positions_by_layer':{1:(),2:(),3:()}},
                {'repair_positions_by_layer':{2:(2,),3:(1,2)}},
                {'repair_positions_by_layer':{2:(1,1),3:()}},
                {'repair_positions_by_layer':{2:(0,1,2),3:(0,1,2)}}):
            model,_,sources=self.fixture()
            with self.subTest(change=change), self.assertRaises(ValueError):
                with self.scope(model,sources,**change):
                    pass
            self.assert_no_hooks(model)

    def test_budgets_fail_before_hook_install_or_source_tensor_operation(self):
        for budget in ({'max_target_host_bytes':95},{'max_transient_device_bytes':100}):
            model,_,sources=self.fixture()
            with patch.object(torch,'isfinite',side_effect=AssertionError('operation before budget')):
                with self.subTest(budget=budget), self.assertRaises(MemoryError):
                    with self.scope(model,sources,**budget):
                        pass
            self.assert_no_hooks(model)

    def test_source_geometry_dtype_finite_and_full_layer_count(self):
        for kind in ('shape','dtype','nan','missing','gqa'):
            model,_,sources=self.fixture(); sources=list(sources)
            if kind=='shape': sources[1]=(sources[1][0][:2],sources[1][1])
            elif kind=='dtype': sources[1]=(sources[1][0].float(),sources[1][1])
            elif kind=='nan': sources[1][0][0,0,0]=float('nan')
            elif kind=='missing': sources.pop()
            elif kind=='gqa': model.layers[1].self_attn.num_heads=3
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                with self.scope(model,sources): pass
            self.assert_no_hooks(model)

    def test_model_failure_discards_partial_targets_and_removes_hooks(self):
        model,hidden,sources=self.fixture(); model.layers[1].fail=True
        hooks=self.scope(model,sources)
        with self.assertRaisesRegex(RuntimeError,'intentional block failure'):
            with hooks: model(hidden)
        self.assertFalse(hooks.audit()['completed']); self.assertEqual(hooks._targets,[])
        self.assertIn('REFERENCE_EXECUTION_FAILED',hooks.failure)
        with self.assertRaises(RuntimeError): _=hooks.target_layers
        self.assert_no_hooks(model)

    def test_absent_or_incomplete_forward_cannot_succeed(self):
        for partial_forward in (False,True):
            model,hidden,sources=self.fixture(); hooks=self.scope(model,sources)
            with self.subTest(partial_forward=partial_forward), self.assertRaisesRegex(ValueError,'incomplete'):
                with hooks:
                    if partial_forward: model.layers[0](hidden)
            self.assertFalse(hooks.audit()['completed']); self.assert_no_hooks(model)

    def test_duplicate_layer_or_decode_inside_scope_rejected(self):
        model,hidden,sources=self.fixture(); hooks=self.scope(model,sources)
        with self.assertRaisesRegex(ValueError,'no decode'):
            with hooks:
                model(hidden)
                model(hidden[:1])
        self.assertFalse(hooks.audit()['completed']); self.assert_no_hooks(model)

    def test_projection_outside_block_and_wrong_prompt_layout_rejected(self):
        for direct in (True,False):
            model,hidden,sources=self.fixture(); hooks=self.scope(model,sources)
            with self.subTest(direct=direct), self.assertRaises(ValueError):
                with hooks:
                    if direct: model.layers[0].self_attn.qkv_proj(hidden)
                    else: model(hidden[:1])
            self.assert_no_hooks(model)

    def test_source_mutation_and_scope_reuse_rejected(self):
        model,hidden,sources=self.fixture(); hooks=self.scope(model,sources)
        with self.assertRaisesRegex(ValueError,'Source mutated'):
            with hooks:
                sources[1][0].zero_()
                model(hidden)
        self.assert_no_hooks(model)
        with self.assertRaisesRegex(RuntimeError,'cannot be reused'):
            with hooks: pass

    def test_finite_target_check_and_no_cuda_qualification(self):
        model,hidden,sources=self.fixture(); hidden[3,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite current target'):
            with self.scope(model,sources) as hooks: model(hidden)
        self.assertFalse(hooks.audit()['P1_execution_allowed']); self.assert_no_hooks(model)

    def test_inference_tensors_do_not_claim_version_integrity(self):
        model,hidden,sources=self.fixture()
        with torch.inference_mode():
            sources=tuple(tuple(t.clone() for t in pair) for pair in sources)
            with self.scope(model,sources) as hooks:
                model(hidden)
        self.assertTrue(hooks.audit()['completed'])
        self.assertFalse(hooks.audit()['source_version_counters_available'])
        self.assertTrue(hooks.audit()['caller_source_lease_and_integrity_guard_required'])
        self.assert_no_hooks(model)

    def test_target_copy_survives_inplace_rope_on_returned_qkv(self):
        model,hidden,sources=self.fixture()
        # Native rotary implementations may rotate returned K in place. This
        # synthetic mutation occurs after projection hooks, before block exit.
        original_forward=model.layers[0].self_attn.forward
        def inplace_forward(h):
            result=original_forward(h)
            model.layers[0].self_attn.returned[:,8:12].zero_()
            return result
        model.layers[0].self_attn.forward=inplace_forward
        with self.scope(model,sources) as hooks:
            model(hidden)
        expected=model.layers[0].self_attn.qkv_proj.last[3:5,8:12].reshape(2,2,2)
        self.assertTrue(torch.equal(hooks.target_layers[0][0],expected))
        self.assertGreater(torch.count_nonzero(hooks.target_layers[0][0]).item(),0)
        self.assert_no_hooks(model)

    def test_tensor_projection_output_supported_without_aux_tuple(self):
        model,hidden,sources=self.fixture()
        for block in model.layers:
            proj=block.self_attn.qkv_proj
            forward=proj.forward
            proj.forward=lambda h,forward=forward:forward(h)[0]
            block.self_attn.forward=lambda h,proj=proj:attention_math(proj(h),list(range(len(h))),len(h))
        with self.scope(model,sources) as hooks:
            model(hidden)
        self.assertTrue(hooks.audit()['completed']); self.assert_no_hooks(model)

    def test_explicit_fenced_release_preserves_completed_cpu_targets(self):
        model,hidden,sources=self.fixture(); hooks=self.scope(model,sources)
        with hooks:
            with self.assertRaisesRegex(RuntimeError,'completion fence'):
                hooks.release_device_references(caller_fenced=True)
            model(hidden)
        cpu_targets=hooks.target_layers
        with self.assertRaisesRegex(RuntimeError,'completion fence'):
            hooks.release_device_references(caller_fenced=False)
        self.assertIs(hooks._model,model)
        self.assertTrue(hooks._source_layers)
        hooks.release_device_references(caller_fenced=True)
        hooks.release_device_references(caller_fenced=True)
        self.assertIsNone(hooks._model); self.assertEqual(hooks._source_layers,())
        self.assertIsNone(hooks._pending)
        self.assertEqual(hooks.target_layers,cpu_targets)
        self.assertTrue(hooks.audit()['device_references_released'])

    def test_failed_block_retains_pending_gpu_ownership_until_fenced_release(self):
        model,hidden,sources=self.fixture(); model.layers[1].fail=True
        hooks=self.scope(model,sources)
        with self.assertRaisesRegex(RuntimeError,'intentional block failure'):
            with hooks: model(hidden)
        self.assertIsNotNone(hooks._pending)
        self.assertTrue(hooks._source_layers)
        self.assert_no_hooks(model)
        hooks.release_device_references(caller_fenced=True)
        self.assertIsNone(hooks._pending); self.assertEqual(hooks._source_layers,())
        self.assertFalse(hooks.audit()['completed']); self.assertEqual(hooks._targets,[])

    def test_deadline_checked_before_each_layer_and_failure_removes_hooks(self):
        model,hidden,sources=self.fixture(); calls=[]
        def deadline():
            calls.append(len(calls)+1)
            if len(calls)==2: raise TimeoutError('bounded reference deadline')
        hooks=self.scope(model,sources,check_deadline=deadline)
        with self.assertRaisesRegex(TimeoutError,'bounded reference deadline'):
            with hooks: model(hidden)
        self.assertEqual(calls,[1,2])
        self.assertEqual(hooks.audit()['projection_counts'],[1,0,0])
        self.assert_no_hooks(model)
        hooks.release_device_references(caller_fenced=True)
        self.assertIsNone(hooks._check_deadline)


if __name__=='__main__':
    unittest.main()
