"""CPU-local issued K comparison receipts; no GPU/quality qualification."""
from dataclasses import asdict, replace
import gc
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import Mock, patch

import torch

from tests import test_source_store_v2 as fixtures
from probekv import selection_comparison
from probekv.source_comparison_v2 import (ComparisonProfileBindingV2,
    ComparisonSessionV2, runtime_binding_digest, scorer_digest)
from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema10_storage import tensor_digest


class SourceComparisonReceiptTests(unittest.TestCase):
    setUp=fixtures.TargetSourceStoreTests.setUp
    close_stores=fixtures.TargetSourceStoreTests.close_stores
    store=fixtures.TargetSourceStoreTests.store
    candidate=fixtures.TargetSourceStoreTests.candidate
    plan=fixtures.TargetSourceStoreTests.plan
    publish=fixtures.TargetSourceStoreTests.publish
    read_rows=fixtures.TargetSourceStoreTests.read_rows

    def profile(self, **changes):
        values=dict(model_signature='model',provenance_policy='ALLOW_MIXED_G1',
            completed_depth=1,trim_ratio=0.15,tau_reuse=0.0,tau_add=0.0,
            repair_policy_digest='a'*64,runtime_digest=runtime_binding_digest(),
            scoring_function_digest=scorer_digest())
        values.update(changes)
        return ComparisonProfileBindingV2(**values)

    def context(self, request='consumer', current=None):
        if current is None: current=torch.ones((2,2,2),dtype=torch.bfloat16)
        return SimpleNamespace(request=dict(request_id=request,token_ids=list(self.tokens)),
            adapter=SimpleNamespace(provenance={'model_signature':'model'}),
            segments={'target':dict(positions=(2,3))},
            execution_inventory={'target':SimpleNamespace(comparison_eligible=True,remaining_positions=(2,3))},
            current_completed_depth=1,generation=7,finished=False,
            observe_current_k=Mock(return_value=current))

    def setup_session(self, *, count=2, profile=None, workspace_bytes=100000,
                      store=None, request='consumer'):
        store=self.store() if store is None else store
        for i in range(count): self.publish(store,self.candidate('birth-'+str(i),base=20*i))
        snapshot=store.begin_request(request)
        session=ComparisonSessionV2(store,snapshot,profile or self.profile(),workspace_bytes=workspace_bytes)
        self.addCleanup(session.close)
        return store,snapshot,session,self.context(request)

    def reidentify(self, candidate):
        candidate['artifact_digest']=tensor_digest(t for pair in candidate['layers'] for t in pair)
        candidate['source_artifact_id']=digest_json(dict(manifest=asdict(candidate['manifest_reference']),
            proof=asdict(candidate['proof']),artifact_digest=candidate['artifact_digest']))
        return candidate

    def test_real_selection_reads_match_legacy_scores_without_full_kv_or_lru(self):
        store,snapshot,session,context=self.setup_session()
        before=store.lookup(snapshot,self.target_tokens)
        raw=selection_comparison.prepare_current_k
        with patch.object(store,'read_selection',wraps=store.read_selection) as reads, \
             patch.object(selection_comparison,'prepare_current_k',wraps=raw) as prepared, \
             patch('probekv.source_store_v2.LayerFile',side_effect=AssertionError('full KV forbidden')):
            receipt=session.compare_native(context,'target')
        self.assertEqual(reads.call_count,2)
        self.assertEqual(prepared.call_count,1)
        context.observe_current_k.assert_called_once_with('target',1)
        self.assertEqual(receipt.counts,dict(stored=2,eligible=2,available=2,compared=2))
        self.assertEqual(receipt.observation_layer,2)
        self.assertGreater(receipt.host_ms,0)
        self.assertTrue(receipt.diagnostic_only)
        current=context.observe_current_k.return_value
        normalized=selection_comparison.prepare_current_k(current)
        for sid,actual in receipt.scores:
            source=store.read_selection(snapshot,sid,1)
            expected=float(selection_comparison.residual_scores(source.unsqueeze(0),*normalized,0.15)[0])
            self.assertEqual(actual,expected)
        self.assertEqual(before,store.lookup(snapshot,self.target_tokens))
        self.assertFalse(any(store._lease_counts.values()))

    def test_miss_receipt_requires_real_empty_pool_and_no_model_observation(self):
        _,_,session,context=self.setup_session(count=0)
        context.observe_current_k.side_effect=AssertionError('miss should not project')
        receipt=session.compare_native(context,'target')
        scope=session.publication_scope(receipt)
        self.assertEqual(scope.reason,'content_miss')
        self.assertEqual(receipt.counts,dict(stored=0,eligible=0,available=0,compared=0))
        self.assertIsNone(receipt.current_k_digest)
        self.assertIsNone(receipt.winner_source_id)

    def test_forged_scores_counts_identity_profile_and_unknown_receipt_rejected(self):
        _,_,session,context=self.setup_session()
        receipt=session.compare_native(context,'target')
        for changes in (dict(scores=((receipt.stored_ids[0],9999.0),)),
                        dict(stored_ids=()),dict(receipt_id='unissued'),dict(request_id='other'),
                        dict(profile_binding_digest='b'*64),dict(current_k_digest='c'*64),
                        dict(winner_source_id='fake'),dict(diagnostic_only=False)):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                session.verify(replace(receipt,**changes))
        with self.assertRaises(TypeError):
            session.compare_native(context,'target',scores={'fake':0})

    def test_receipt_cross_issuer_snapshot_and_store_rejected(self):
        store,snapshot,session,context=self.setup_session()
        receipt=session.compare_native(context,'target')
        other=ComparisonSessionV2(store,snapshot,self.profile(),workspace_bytes=100000)
        self.addCleanup(other.close)
        with self.assertRaises(ValueError): other.verify(receipt)
        store.end_request(snapshot)
        later=store.begin_request('later')
        another=ComparisonSessionV2(store,later,self.profile(),workspace_bytes=100000)
        self.addCleanup(another.close)
        with self.assertRaises(ValueError): another.verify(receipt)
        store2=self.store('independent')
        ss2=store2.begin_request('consumer')
        independent=ComparisonSessionV2(store2,ss2,self.profile(),workspace_bytes=100000)
        self.addCleanup(independent.close)
        with self.assertRaises(ValueError): independent.verify(receipt)

    def test_receipt_single_observation_and_single_growth_use(self):
        _,_,session,context=self.setup_session()
        receipt=session.compare_native(context,'target')
        self.assertIsNone(session.publication_scope(receipt).rejection())
        session.mark_publication_used(receipt)
        with self.assertRaises(ValueError): session.publication_scope(receipt)
        with self.assertRaises(RuntimeError): session.compare_native(context,'target')

    def test_partial_candidate_scope_preserves_stored_and_forbids_growth(self):
        store,snapshot,session,context=self.setup_session()
        selected=store.lookup(snapshot,self.target_tokens)[0]['source_id']
        receipt=session.compare_native(context,'target',candidate_ids=(selected,))
        self.assertEqual(receipt.counts,dict(stored=2,eligible=2,available=2,compared=1))
        with self.assertRaisesRegex(ValueError,'incomplete'):
            session.publication_scope(receipt)

    def test_provenance_filtered_row_stays_in_stored_scope_not_content_miss(self):
        store,snapshot,session,context=self.setup_session(count=1)
        rows=store.lookup(snapshot,self.target_tokens)
        filtered=tuple(dict(row,generation=2) for row in rows)
        with patch.object(store,'lookup',return_value=filtered):
            receipt=session.compare_native(context,'target')
            self.assertEqual(receipt.counts,dict(stored=1,eligible=0,available=0,compared=0))
            with self.assertRaises(ValueError): session.publication_scope(receipt)
        self.assertEqual(receipt.rejected_states[0][1],'provenance_filtered')

    def test_missing_depth_or_corrupt_state_preserves_stored_without_full_kv(self):
        for failure in ('depth','file'):
            with self.subTest(failure=failure):
                store=self.store(failure)
                self.publish(store,self.candidate(failure+'-birth'))
                profile=self.profile(completed_depth=3) if failure=='depth' else self.profile()
                _,snapshot,session,context=self.setup_session(count=0,store=store,profile=profile)
                context.current_completed_depth=profile.completed_depth
                if failure=='file':
                    row=store.lookup(snapshot,self.target_tokens)[0]
                    (store.root/row['selection_file']).write_bytes(b'broken state')
                with patch('probekv.source_store_v2.LayerFile',side_effect=AssertionError('full KV forbidden')):
                    receipt=session.compare_native(context,'target')
                    self.assertEqual(receipt.counts,dict(stored=1,eligible=1,available=0,compared=0))
                    with self.assertRaises(ValueError): session.publication_scope(receipt)
                self.assertTrue(receipt.rejected_states)

    def test_invalid_current_k_never_issues_growth_receipt(self):
        variants=[torch.ones((2,2),dtype=torch.bfloat16),torch.ones((2,2,2)),
                  torch.ones((1,2,2),dtype=torch.bfloat16)]
        for nonfinite in (float('nan'),float('inf'),-float('inf')):
            value=torch.ones((2,2,2),dtype=torch.bfloat16);value[0,0,0]=nonfinite;variants.append(value)
        _,_,session,context=self.setup_session()
        for current in variants:
            context.observe_current_k.return_value=current
            with self.subTest(shape=current.shape),self.assertRaises(ValueError):
                session.compare_native(context,'target')
        self.assertEqual(session._issued,{})

    def test_nonfinite_or_wrongshape_source_is_unavailable_not_new_content(self):
        store,_,session,context=self.setup_session(count=1)
        source=torch.ones((2,2,2),dtype=torch.bfloat16);source[0,0,0]=float('nan')
        with patch.object(store,'read_selection',return_value=source):
            receipt=session.compare_native(context,'target')
        self.assertEqual(receipt.counts,dict(stored=1,eligible=1,available=0,compared=0))
        with self.assertRaises(ValueError): session.publication_scope(receipt)

    def test_invalid_profiles_and_stale_runtime_rejected(self):
        for changes in (dict(trim_ratio=float('nan')),dict(tau_add=float('inf')),
                        dict(tau_reuse=-1),dict(trim_ratio=True),dict(completed_depth=0),
                        dict(completed_depth=True),dict(trim_ratio=1.0),dict(tau_add=0.1),
                        dict(diagnostic_only=False),dict(repair_policy_digest='not-a-sha')):
            with self.subTest(changes=changes),self.assertRaises(ValueError): self.profile(**changes)
        store=self.store()
        snapshot=store.begin_request('consumer')
        for changes in (dict(provenance_policy='EXACT_ONLY'),dict(model_signature='other'),
                        dict(scoring_function_digest='0'*64),dict(runtime_digest='0'*64)):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                ComparisonSessionV2(store,snapshot,self.profile(**changes),workspace_bytes=100000)

    def test_wrong_context_identity_depth_prefix_ownership_rejected(self):
        _,_,session,context=self.setup_session()
        mutations=(lambda c:c.request.update(request_id='other'),
                   lambda c:c.adapter.provenance.update(model_signature='wrong'),
                   lambda c:setattr(c,'finished',True),
                   lambda c:setattr(c,'current_completed_depth',2),
                   lambda c:setattr(c,'generation',True),
                   lambda c:setattr(c,'generation',-1),
                   lambda c:setattr(c.execution_inventory['target'],'comparison_eligible',False),
                   lambda c:setattr(c.execution_inventory['target'],'remaining_positions',(3,)),
                   lambda c:c.segments['target'].update(positions=(3,2)))
        for mutate in mutations:
            context=self.context();mutate(context)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                session.compare_native(context,'target')
            context.observe_current_k.assert_not_called()

    def test_generation_or_depth_changed_during_observation_rejected(self):
        _,_,session,context=self.setup_session()
        for field,value in (('generation',8),('current_completed_depth',2)):
            context=self.context()
            def observe(*args):
                setattr(context,field,value)
                return torch.ones((2,2,2),dtype=torch.bfloat16)
            context.observe_current_k.side_effect=observe
            with self.subTest(field=field),self.assertRaises(ValueError): session.compare_native(context,'target')
        self.assertEqual(session._issued,{})

    def test_completed_depth_cannot_observe_past_real_model_layer_count(self):
        _,_,session,context=self.setup_session(profile=self.profile(completed_depth=3))
        context.current_completed_depth=3
        context.adapter.spec=SimpleNamespace(num_layers=3)
        with self.assertRaisesRegex(ValueError,'observation layer'):
            session.compare_native(context,'target')
        context.observe_current_k.assert_not_called()

    def test_input_identity_changed_without_generation_is_not_signable(self):
        _,_,session,context=self.setup_session()
        def observe(*args):
            context.request['token_ids'][2]=999
            return torch.ones((2,2,2),dtype=torch.bfloat16)
        context.observe_current_k.side_effect=observe
        with self.assertRaises(ValueError): session.compare_native(context,'target')
        self.assertEqual(session._issued,{})

    def test_upstream_only_token_mutation_during_observation_is_not_signable(self):
        _,_,session,context=self.setup_session()
        target_before=tuple(context.request['token_ids'][2:])
        def observe(*args):
            context.request['token_ids'][0]=999
            return torch.ones((2,2,2),dtype=torch.bfloat16)
        context.observe_current_k.side_effect=observe
        with self.assertRaises(ValueError): session.compare_native(context,'target')
        self.assertEqual(tuple(context.request['token_ids'][2:]),target_before)
        self.assertEqual(session._issued,{})

    def test_upstream_only_mutation_before_winner_freeze_invalidates_receipt(self):
        _,_,session,context=self.setup_session()
        receipt=session.compare_native(context,'target')
        context.request['token_ids'][0]=999
        self.assertEqual(tuple(context.request['token_ids'][2:]),receipt.token_ids)
        with self.assertRaisesRegex(ValueError,'stale observation'):
            session.verify(receipt,context)

    def test_same_target_different_full_birth_input_cannot_use_comparison_to_publish(self):
        store,snapshot,session,context=self.setup_session(count=1)
        receipt=session.compare_native(context,'target')
        original=self.tokens
        self.tokens=(999,11,12,13)
        try:
            candidate=self.candidate('consumer',base=50)
        finally:
            self.tokens=original
        self.assertNotEqual(candidate['proof'].input_digest,receipt.request_input_digest)
        self.assertEqual(candidate['proof'].positions,receipt.absolute_positions)
        with self.assertRaisesRegex(ValueError,'birth input/positions'):
            store.plan_from_comparison(snapshot,candidate,token_ids=self.target_tokens,
                comparison=session,receipt=receipt)
        self.assertEqual(store.storage_audit()['source_count'],1)

    def test_exact_score_tie_prefers_g0_then_publication_order(self):
        store=self.store()
        mixed=self.candidate('mixed-first',mixed=True)
        exact=self.candidate('exact-later')
        exact['layers'][2][1].add_(1);self.reidentify(exact)
        first=self.publish(store,mixed)['source_id']
        second=self.publish(store,exact)['source_id']
        _,_,session,context=self.setup_session(count=0,store=store)
        receipt=session.compare_native(context,'target')
        self.assertEqual(dict(receipt.scores)[first],dict(receipt.scores)[second])
        self.assertEqual(receipt.winner_source_id,second)
        # A separate all-exact tie uses birth/publication order, not hash order.
        exact_store=self.store('exact-tie')
        a=self.candidate('exact-a');b=self.candidate('exact-b')
        b['layers'][2][1].add_(2);self.reidentify(b)
        earliest=self.publish(exact_store,a)['source_id'];self.publish(exact_store,b)
        _,_,session2,context2=self.setup_session(count=0,store=exact_store)
        self.assertEqual(session2.compare_native(context2,'target').winner_source_id,earliest)

    def test_near_but_unequal_scores_not_replaced_by_exact_origin_preference(self):
        store=self.store()
        exact=self.candidate('exact',base=20)
        mixed=self.candidate('mixed',mixed=True)
        self.publish(store,exact)
        best=self.publish(store,mixed)['source_id']
        _,_,session,context=self.setup_session(count=0,store=store)
        context.observe_current_k.return_value=mixed['selection_states'][1].clone()
        receipt=session.compare_native(context,'target')
        self.assertEqual(receipt.winner_source_id,best)

    def test_zero_norm_n2_ceil_trim_remains_finite_and_legacy_identical(self):
        store,snapshot,session,context=self.setup_session(count=1,profile=self.profile(trim_ratio=0.75))
        current=torch.zeros((2,2,2),dtype=torch.bfloat16)
        context.observe_current_k.return_value=current
        receipt=session.compare_native(context,'target')
        source=store.read_selection(snapshot,receipt.compared_ids[0],1)
        score=float(selection_comparison.residual_scores(source.unsqueeze(0),
            *selection_comparison.prepare_current_k(current),0.75)[0])
        self.assertEqual(receipt.scores[0][1],score)
        self.assertTrue(torch.isfinite(torch.tensor(score)))

    def test_compatible_score_does_not_authorize_mismatch_growth(self):
        _,_,session,context=self.setup_session(profile=self.profile(tau_reuse=1000,tau_add=1000))
        receipt=session.compare_native(context,'target')
        with self.assertRaisesRegex(ValueError,'compatible'):
            session.publication_scope(receipt)

    def test_ended_snapshot_revoked_permission_and_changed_state_invalidate_receipt(self):
        store,snapshot,session,context=self.setup_session(count=1)
        receipt=session.compare_native(context,'target')
        row=store.lookup(snapshot,self.target_tokens)[0]
        path=store.root/row['selection_file'];original=path.read_bytes();path.write_bytes(b'changed')
        with self.assertRaises(ValueError): session.verify(receipt)
        path.write_bytes(original)
        context.generation+=1
        with self.assertRaises(ValueError): session.verify(receipt,context)
        store.end_request(snapshot)
        with self.assertRaises(ValueError): session.verify(receipt)
        self.registry.revoke_authorization_domain('domain')
        with self.assertRaises(ValueError): session.verify(receipt)

    def test_no_budget_workspace_receipt_or_unknown_candidate(self):
        _,_,session,context=self.setup_session(workspace_bytes=32)
        with self.assertRaises(MemoryError): session.compare_native(context,'target')
        self.assertEqual(session._issued,{})
        with self.assertRaises(ValueError): session.compare_native(context,'target',candidate_ids=('unknown',))

    def test_all_checkpoint_file_payload_is_budgeted_before_selection_read(self):
        store,snapshot,session,context=self.setup_session(count=1)
        current=context.observe_current_k.return_value
        per_source=current.numel()*32+current.shape[0]*32
        one_layer_bytes=current.numel()*current.element_size()
        row=store.lookup(snapshot,self.target_tokens)[0]
        self.assertGreater(row['selection_state_bytes'],one_layer_bytes)
        # Enough for the old single-depth calculation, but not the actual
        # all-checkpoint file payload loaded by read_selection.
        session.workspace_bytes=2*per_source+one_layer_bytes
        with patch.object(store,'read_selection',wraps=store.read_selection) as read:
            with self.assertRaisesRegex(MemoryError,'file payload'):
                session.compare_native(context,'target')
        read.assert_not_called()
        self.assertEqual(session._issued,{})

    def test_receipt_bound_real_mismatch_publication_and_consumption(self):
        store,snapshot,session,context=self.setup_session(count=1)
        receipt=session.compare_native(context,'target')
        candidate=self.candidate('consumer',base=50)
        plan=store.plan_from_comparison(snapshot,candidate,token_ids=self.target_tokens,
            comparison=session,receipt=receipt)
        result=store.commit_publication(snapshot,plan,candidate,token_ids=self.target_tokens)
        self.assertEqual(result['status'],'PUBLISHED')
        self.assertEqual(store.events[-1]['comparison_receipt_id'],receipt.receipt_id)
        self.assertTrue(store.events[-1]['verified_comparison_binding'])
        self.assertIn(receipt.receipt_id,session._used_for_publication)
        self.assertEqual(len(store.lookup(snapshot,self.target_tokens)),1)
        with self.assertRaises(ValueError): session.publication_scope(receipt)

    def test_real_miss_publication_requires_same_issuer_and_target_tokens(self):
        store,snapshot,session,context=self.setup_session(count=0)
        receipt=session.compare_native(context,'target')
        candidate=self.candidate('consumer')
        wrong=ComparisonSessionV2(store,snapshot,self.profile(),workspace_bytes=100000)
        self.addCleanup(wrong.close)
        with self.assertRaises(ValueError):
            store.plan_from_comparison(snapshot,candidate,token_ids=self.target_tokens,
                comparison=wrong,receipt=receipt)
        with self.assertRaises(ValueError):
            store.plan_from_comparison(snapshot,candidate,token_ids=(99,100),
                comparison=session,receipt=receipt)
        plan=store.plan_from_comparison(snapshot,candidate,token_ids=self.target_tokens,
            comparison=session,receipt=receipt)
        result=store.commit_publication(snapshot,plan,candidate,token_ids=self.target_tokens)
        self.assertEqual(result['status'],'PUBLISHED')
        self.assertEqual(context.observe_current_k.call_count,0)

    def test_selection_corruption_after_plan_blocks_commit_and_preserves_old_pool(self):
        store,snapshot,session,context=self.setup_session(count=1)
        receipt=session.compare_native(context,'target')
        candidate=self.candidate('consumer',base=50)
        plan=store.plan_from_comparison(snapshot,candidate,token_ids=self.target_tokens,
            comparison=session,receipt=receipt)
        row=store.lookup(snapshot,self.target_tokens)[0]
        path=store.root/row['selection_file'];body=path.read_bytes()
        path.write_bytes(b'corrupt-after-plan')
        result=store.commit_publication(snapshot,plan,candidate,token_ids=self.target_tokens)
        self.assertEqual(result['status'],'REJECTED')
        self.assertEqual(store.storage_audit()['source_count'],1)
        self.assertIn(receipt.receipt_id,session._used_for_publication)
        path.write_bytes(body)

    def test_pool_publication_after_receipt_is_stale_even_if_old_snapshot_ids_still_visible(self):
        store,snapshot,session,context=self.setup_session(count=1)
        receipt=session.compare_native(context,'target')
        candidate=self.candidate('consumer',base=50)
        # Low-level CPU fixture publication is intentionally separate from the
        # receipt-authorized path; it advances actual pool membership here.
        plan=self.plan(store,snapshot,candidate)
        result=store.commit_publication(snapshot,plan,candidate,token_ids=self.target_tokens)
        self.assertEqual(result['status'],'PUBLISHED')
        self.assertEqual(tuple(r['source_id'] for r in store.lookup(snapshot,self.target_tokens)),receipt.stored_ids)
        with self.assertRaisesRegex(ValueError,'pool changed'):
            session.verify(receipt)

    def test_failed_bound_commit_consumes_receipt_without_model_retry(self):
        store,snapshot,session,context=self.setup_session(count=1)
        receipt=session.compare_native(context,'target')
        candidate=self.candidate('consumer',base=50)
        plan=store.plan_from_comparison(snapshot,candidate,token_ids=self.target_tokens,
            comparison=session,receipt=receipt)
        with patch.object(store,'_stage',side_effect=OSError('disk failed')):
            result=store.commit_publication(snapshot,plan,candidate,token_ids=self.target_tokens)
        self.assertEqual(result['status'],'REJECTED')
        self.assertEqual(store.storage_audit()['source_count'],1)
        with self.assertRaisesRegex(ValueError,'consumed'): session.publication_scope(receipt)
        context.observe_current_k.assert_called_once()
        self.assertEqual(store.events[-1]['extra_forward_count'],0)

    def test_closed_issuer_before_commit_is_optional_publication_rejection_not_uncaught_error(self):
        store,snapshot,session,context=self.setup_session(count=1)
        receipt=session.compare_native(context,'target')
        candidate=self.candidate('consumer',base=50)
        plan=store.plan_from_comparison(snapshot,candidate,token_ids=self.target_tokens,
            comparison=session,receipt=receipt)
        session.close()
        result=store.commit_publication(snapshot,plan,candidate,token_ids=self.target_tokens)
        self.assertEqual(result['status'],'REJECTED')
        self.assertEqual(store.storage_audit()['source_count'],1)
        self.assertEqual(store.events[-1]['publication_performed'],False)

    def test_profile_replaced_after_plan_rejects_commit(self):
        store,snapshot,session,context=self.setup_session(count=1)
        receipt=session.compare_native(context,'target')
        candidate=self.candidate('consumer',base=50)
        plan=store.plan_from_comparison(snapshot,candidate,token_ids=self.target_tokens,
            comparison=session,receipt=receipt)
        session.profile=replace(session.profile,repair_policy_digest='b'*64)
        result=store.commit_publication(snapshot,plan,candidate,token_ids=self.target_tokens)
        self.assertEqual(result['status'],'REJECTED')
        self.assertEqual(store.storage_audit()['source_count'],1)

    def test_receipts_retain_digests_not_current_or_historical_tensor_owners(self):
        _,_,session,context=self.setup_session(count=1)
        tensor=context.observe_current_k.return_value
        weak=weakref.ref(tensor)
        receipt=session.compare_native(context,'target')
        context.observe_current_k.reset_mock(return_value=True)
        del tensor,context
        gc.collect()
        self.assertIsNone(weak())
        self.assertTrue(all(not torch.is_tensor(value) for value in asdict(receipt).values()))


if __name__=='__main__': unittest.main()
