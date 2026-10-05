"""Issued comparison -> native capture -> durable publication, CPU fakes only.

The fake projection includes a prefix-dependent offset so historical contexts
produce genuine different captured tensors. This is not real-model evidence.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from probekv import selection_comparison
from probekv.native_publication_v2 import publish_compared_targets_v2
from probekv.native_source_capture_v2 import NativeTargetCaptureV2
from probekv.resumable_prefill import ProbeKVResumablePrefillSession
from probekv.source_comparison_v2 import (ComparisonProfileBindingV2,
    ComparisonSessionV2, runtime_binding_digest, scorer_digest)
from probekv.source_provenance_v2 import PublicationScope
from tests import test_native_publication_v2 as fixtures


class NativeComparisonPublicationTests(unittest.TestCase):
    def setUp(self):
        # Composition avoids inheriting/rerunning the fixture's test methods.
        self.fixture = fixtures.NativePublicationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def context(self, request_id='birth', prefix=0):
        ctx, model, old_session, _ = self.fixture.context()
        tokens = (prefix, 1, 2, 3, 4, 5)
        session = ProbeKVResumablePrefillSession(old_session.adapter, 'model', tokens, None, [])
        session.begin_prefill()
        capture = NativeTargetCaptureV2(request_id=request_id, token_ids=tokens,
            model_signature='model', authorization_domain='domain', num_layers=3,
            targets={'C': (2, 3, 4)}, selection_depths=(1, 2), host_budget_bytes=10000,
            kv_heads=1, head_dim=2, registry=self.fixture.registry)
        session.provenance_capture = capture
        session.adapter.spec.num_attention_heads = 2
        session.adapter.spec.num_kv_heads = 1
        for block in model.layers:
            block.input_layernorm = lambda hidden, residual: (hidden, residual)

            def causal_cpu_projection(module, args, result, prefix=prefix):
                result[0][:, 4:].add_(prefix)
                return result

            # Installed before the actual-capture hook, so the retained K/V
            # are precisely what this fake request's block generated.
            block.self_attn.qkv_proj.register_forward_hook(causal_cpu_projection)
        ctx.request.update(request_id=request_id, token_ids=tokens)
        ctx.source_capture_v2 = capture
        ctx.engine = NS(session=session, finish_prefill=session.finish_prefill)
        ctx.advance_to_depth = session.advance_to_layer
        ctx.execution_inventory = {'C': NS(comparison_eligible=True, remaining_positions=(2, 3, 4))}
        ctx.generation = 1
        ctx._observation = {}
        return ctx, model, session

    def profile(self):
        return ComparisonProfileBindingV2(model_signature='model',
            provenance_policy='ALLOW_MIXED_G1', completed_depth=1, trim_ratio=0.15,
            tau_reuse=0.0, tau_add=0.0, repair_policy_digest='a'*64,
            runtime_digest=runtime_binding_digest(), scoring_function_digest=scorer_digest())

    def seed(self, store, count):
        """Historical CPU fixtures, never claimed to be online decisions."""
        for index in range(count):
            ctx, _, _ = self.context('history-' + str(index), prefix=10 + 20 * index)
            snapshot = store.begin_request(ctx.request['request_id'])
            self.fixture.finish(ctx)
            rows = store.lookup(snapshot, (2, 3, 4))
            scope = (self.fixture.miss if not rows else
                PublicationScope('complete_scope_mismatch', len(rows), len(rows), len(rows), len(rows), 2.0, 1.0))
            candidate = ctx.target_candidates_v2['C']
            plan = store.plan_publication(snapshot, candidate, token_ids=(2, 3, 4), scope=scope)
            result = store.commit_publication(snapshot, plan, candidate, token_ids=(2, 3, 4))
            self.assertEqual(result['status'], 'PUBLISHED')
            store.end_request(snapshot)

    def prepared(self, count=0, name='pool'):
        store = self.fixture.store(name)
        self.seed(store, count)
        ctx, model, session = self.context()
        snapshot = store.begin_request('birth')
        comparison = ComparisonSessionV2(store, snapshot, self.profile(), workspace_bytes=100000)
        self.addCleanup(comparison.close)
        session.advance_to_layer(1)
        return ctx, model, store, snapshot, comparison

    def assert_answer_preserved(self, answer, expected, model):
        self.assertEqual(answer, expected)
        self.assertEqual(model.real_block_calls, 3)

    def test_real_miss_before_finish_publishes_after_finish_only_for_future_requests(self):
        ctx, model, store, snapshot, comparison = self.prepared()
        with patch.object(ctx, 'observe_current_k', side_effect=AssertionError('real miss needs no projection')):
            receipt = comparison.compare_native(ctx, 'C')
        self.assertEqual(receipt.counts, dict(stored=0, eligible=0, available=0, compared=0))
        self.assertEqual(model.real_block_calls, 1)
        with self.assertRaisesRegex(RuntimeError, 'completed'):
            publish_compared_targets_v2(ctx, comparison, {'C': receipt})
        self.assertEqual(store.storage_audit()['source_count'], 0)
        answer = self.fixture.finish(ctx)
        expected = deepcopy(answer)
        audit = publish_compared_targets_v2(ctx, comparison, {'C': receipt})
        result = audit['targets']['C']
        self.assertEqual(result['status'], 'PUBLISHED')
        self.assertEqual(audit['publication_evidence_mode'], 'ISSUED_K_COMPARISON')
        self.assertEqual(audit['extra_forward_count'], 0)
        self.assertGreater(audit['post_request_host_ms'], 0)
        self.assertFalse(audit['native_runtime_qualified'])
        self.assertTrue(store.events[-1]['verified_comparison_binding'])
        self.assertEqual(store.events[-1]['comparison_receipt_id'], receipt.receipt_id)
        self.assertIn(receipt.receipt_id, comparison._used_for_publication)
        self.assertEqual(store.lookup(snapshot, (2, 3, 4)), ())
        with self.assertRaises(ValueError):
            store.read_selection(snapshot, result['source_id'], 1)
        store.end_request(snapshot)
        reader = store.begin_request('future')
        self.assertEqual(len(store.lookup(reader, (2, 3, 4))), 1)
        with store.leased_target(reader, result['source_id']) as layers:
            self.assertEqual(tuple(layers[0][0].shape), (3, 1, 2))
        store.end_request(reader)
        self.assert_answer_preserved(answer, expected, model)
        self.assertEqual([b.self_attn.qkv_proj.calls for b in model.layers], [1, 1, 1])

    def test_full_mismatch_computes_real_K_scores_then_publishes(self):
        ctx, model, store, snapshot, comparison = self.prepared(count=2)
        score = selection_comparison.residual_scores
        with patch.object(selection_comparison, 'residual_scores', wraps=score) as scored, \
                patch.object(store, 'read_selection', wraps=store.read_selection) as reads, \
                patch('probekv.source_store_v2.LayerFile', side_effect=AssertionError('full KV forbidden in compare')):
            receipt = comparison.compare_native(ctx, 'C')
        self.assertEqual(scored.call_count, 2)
        self.assertEqual(reads.call_count, 2)
        self.assertEqual(receipt.counts, dict(stored=2, eligible=2, available=2, compared=2))
        current = ctx.observe_current_k('C', 1)
        normalized = selection_comparison.prepare_current_k(current)
        for source_id, actual in receipt.scores:
            source = store.read_selection(snapshot, source_id, 1)
            self.assertEqual(actual, float(score(source.unsqueeze(0), *normalized, 0.15)[0]))
            self.assertGreater(actual, comparison.profile.tau_add)
        self.assertEqual(model.real_block_calls, 1)
        answer = self.fixture.finish(ctx)
        expected = deepcopy(answer)
        audit = publish_compared_targets_v2(ctx, comparison, {'C': receipt})
        self.assertEqual(audit['targets']['C']['status'], 'PUBLISHED')
        self.assertEqual(store.storage_audit()['source_count'], 3)
        self.assertEqual(len(store.lookup(snapshot, (2, 3, 4))), 2)
        self.assert_answer_preserved(answer, expected, model)
        # Exactly one permitted read-only d1 observation, not another block.
        self.assertEqual([b.self_attn.qkv_proj.calls for b in model.layers], [1, 2, 1])
        store.end_request(snapshot)

    def test_bare_scope_dict_and_forged_receipts_reject_without_store_commit(self):
        for index, kind in enumerate(('scope', 'dict', 'unissued', 'altered_score')):
            with self.subTest(kind=kind):
                ctx, model, store, snapshot, comparison = self.prepared(name='invalid-' + str(index))
                receipt = comparison.compare_native(ctx, 'C')
                invalid = {'scope': self.fixture.miss, 'dict': asdict(receipt),
                    'unissued': replace(receipt, receipt_id='not-issued'),
                    'altered_score': replace(receipt, scores=(('fake-source', 99.0),))}[kind]
                answer = self.fixture.finish(ctx)
                expected = deepcopy(answer)
                with patch.object(store, 'commit_publication', side_effect=AssertionError('invalid evidence')) as commit:
                    audit = publish_compared_targets_v2(ctx, comparison, {'C': invalid})
                commit.assert_not_called()
                self.assertEqual(audit['targets']['C']['status'], 'REJECTED')
                self.assertFalse(audit['targets']['C']['publication_performed'])
                self.assertEqual(store.storage_audit()['source_count'], 0)
                self.assert_answer_preserved(answer, expected, model)
                store.end_request(snapshot)

    def test_missing_receipt_is_skip_not_inferred_content_miss(self):
        ctx, model, store, snapshot, comparison = self.prepared()
        answer = self.fixture.finish(ctx)
        expected = deepcopy(answer)
        with patch.object(store, 'plan_from_comparison', side_effect=AssertionError('no receipt')) as plan:
            audit = publish_compared_targets_v2(ctx, comparison, {})
        plan.assert_not_called()
        self.assertEqual(audit['targets']['C']['status'], 'SKIPPED')
        self.assertEqual(audit['targets']['C']['reason'], 'missing_publication_scope')
        self.assertEqual(store.storage_audit()['source_count'], 0)
        self.assert_answer_preserved(answer, expected, model)
        store.end_request(snapshot)

    def test_partial_comparison_cannot_grow(self):
        ctx, model, store, snapshot, comparison = self.prepared(count=2)
        selected = store.lookup(snapshot, (2, 3, 4))[0]['source_id']
        receipt = comparison.compare_native(ctx, 'C', candidate_ids=(selected,))
        self.assertEqual(receipt.counts, dict(stored=2, eligible=2, available=2, compared=1))
        answer = self.fixture.finish(ctx)
        expected = deepcopy(answer)
        audit = publish_compared_targets_v2(ctx, comparison, {'C': receipt})
        self.assertEqual(audit['targets']['C']['status'], 'REJECTED')
        self.assertIn('incomplete_stored_scope', audit['targets']['C']['reason'])
        self.assertEqual(store.storage_audit()['source_count'], 2)
        self.assert_answer_preserved(answer, expected, model)
        store.end_request(snapshot)

    def test_unavailable_selection_state_cannot_be_laundered_into_miss(self):
        ctx, model, store, snapshot, comparison = self.prepared(count=1)
        with patch.object(store, 'read_selection', side_effect=KeyError('missing shallow K')), \
                patch('probekv.source_store_v2.LayerFile', side_effect=AssertionError('no full KV fallback')):
            receipt = comparison.compare_native(ctx, 'C')
        self.assertEqual(receipt.counts, dict(stored=1, eligible=1, available=0, compared=0))
        self.assertTrue(receipt.rejected_states)
        answer = self.fixture.finish(ctx)
        expected = deepcopy(answer)
        audit = publish_compared_targets_v2(ctx, comparison, {'C': receipt})
        self.assertEqual(audit['targets']['C']['status'], 'REJECTED')
        self.assertEqual(store.storage_audit()['source_count'], 1)
        self.assert_answer_preserved(answer, expected, model)
        store.end_request(snapshot)

    def test_receipt_target_position_or_tokens_must_match_actual_capture(self):
        mutations = (dict(segment_id='other'), dict(absolute_positions=(1, 2, 3)),
                     dict(token_ids=(20, 30, 40)))
        for index, changes in enumerate(mutations):
            with self.subTest(changes=changes):
                ctx, model, store, snapshot, comparison = self.prepared(name='identity-' + str(index))
                receipt = comparison.compare_native(ctx, 'C')
                answer = self.fixture.finish(ctx)
                expected = deepcopy(answer)
                audit = publish_compared_targets_v2(ctx, comparison, {'C': replace(receipt, **changes)})
                self.assertEqual(audit['targets']['C']['status'], 'REJECTED')
                self.assertEqual(store.storage_audit()['source_count'], 0)
                self.assert_answer_preserved(answer, expected, model)
                store.end_request(snapshot)

    def test_different_issuer_cannot_reuse_an_issued_receipt(self):
        ctx, model, store, snapshot, comparison = self.prepared()
        receipt = comparison.compare_native(ctx, 'C')
        other = ComparisonSessionV2(store, snapshot, self.profile(), workspace_bytes=100000)
        self.addCleanup(other.close)
        answer = self.fixture.finish(ctx)
        expected = deepcopy(answer)
        audit = publish_compared_targets_v2(ctx, other, {'C': receipt})
        self.assertEqual(audit['targets']['C']['status'], 'REJECTED')
        self.assertIn('unissued', audit['targets']['C']['reason'])
        self.assertEqual(store.storage_audit()['source_count'], 0)
        self.assert_answer_preserved(answer, expected, model)
        store.end_request(snapshot)

    def test_closed_issuer_rejects_without_rerun_or_answer_loss(self):
        ctx, model, store, snapshot, comparison = self.prepared()
        receipt = comparison.compare_native(ctx, 'C')
        comparison.close()
        answer = self.fixture.finish(ctx)
        expected = deepcopy(answer)
        audit = publish_compared_targets_v2(ctx, comparison, {'C': receipt})
        self.assertEqual(audit['targets']['C']['status'], 'REJECTED')
        self.assertIn('closed', audit['targets']['C']['reason'])
        self.assertEqual(store.storage_audit()['source_count'], 0)
        self.assert_answer_preserved(answer, expected, model)
        store.end_request(snapshot)

    def test_ended_snapshot_rejects_before_attempt_without_answer_loss(self):
        ctx, model, store, snapshot, comparison = self.prepared()
        receipt = comparison.compare_native(ctx, 'C')
        answer = self.fixture.finish(ctx)
        expected = deepcopy(answer)
        store.end_request(snapshot)
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            publish_compared_targets_v2(ctx, comparison, {'C': receipt})
        self.assertEqual(store.storage_audit()['source_count'], 0)
        self.assert_answer_preserved(answer, expected, model)

    def test_pool_changed_after_receipt_rejects_without_overwriting_published_state(self):
        ctx, model, store, snapshot, comparison = self.prepared()
        receipt = comparison.compare_native(ctx, 'C')
        answer = self.fixture.finish(ctx)
        expected = deepcopy(answer)
        candidate = ctx.target_candidates_v2['C']
        # A separate rule-level diagnostic transaction deliberately makes the
        # issued observation stale; the strict bridge must not silently rebase.
        plan = store.plan_publication(snapshot, candidate, token_ids=(2, 3, 4), scope=self.fixture.miss)
        first = store.commit_publication(snapshot, plan, candidate, token_ids=(2, 3, 4))
        self.assertEqual(first['status'], 'PUBLISHED')
        before = {p.name: p.stat().st_size for p in store.root.iterdir()}
        audit = publish_compared_targets_v2(ctx, comparison, {'C': receipt})
        self.assertEqual(audit['targets']['C']['status'], 'REJECTED')
        self.assertIn('pool changed', audit['targets']['C']['reason'])
        self.assertEqual(store.storage_audit()['source_count'], 1)
        self.assertEqual(before, {p.name: p.stat().st_size for p in store.root.iterdir()})
        self.assert_answer_preserved(answer, expected, model)
        store.end_request(snapshot)


if __name__ == '__main__':
    unittest.main()
