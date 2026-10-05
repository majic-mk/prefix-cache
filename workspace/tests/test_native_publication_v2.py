"""CPU fake native blocks + durable manifests + real SSD-file publication.

These tests do not qualify GPU correctness, QA, native parent consumption, or
production dispatch. The completed answer endpoint is deliberately a CPU fake.
"""
from dataclasses import replace
import gc
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import weakref

import torch

from probekv.source_manifest_v2 import RequestManifestRegistry
from probekv.source_provenance_v2 import PublicationScope, SourceOrigin
from probekv.source_store_v2 import TargetSourceStoreV2
from probekv.v8_schema10_native_adapter import NativeRequestContext
from tests.test_native_source_capture_v2 import fixture, commit


class NativePublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = self.registry_at('manifests')
        self.stores = []
        self.addCleanup(lambda: [store.close() for store in reversed(self.stores)])
        self.miss = PublicationScope('content_miss', 0, 0, 0, 0)

    def registry_at(self, name):
        return RequestManifestRegistry(max_bytes=2000000, max_manifest_bytes=200000,
            persistent_root=self.root/name)

    def store(self, name='pool', **overrides):
        arguments = dict(registry=self.registry, model_signature='model',
            authorization_domain='domain', tokenizer_hash='tokenizer', policy='ALLOW_MIXED_G1',
            max_bytes=4000000, staging_bytes=1000000, max_variants=4, probation_opportunities=0)
        arguments.update(overrides)
        store = TargetSourceStoreV2(self.root/name, **arguments)
        self.stores.append(store)
        return store

    def context(self, *, targets=None, host_budget=10000):
        model, session, capture, _ = fixture(targets=targets, host_budget=host_budget)
        capture.registry = self.registry
        ctx = object.__new__(NativeRequestContext)
        ctx.request = dict(request_id='birth', token_ids=tuple(range(6)))
        ctx.segments = {sid: dict(positions=positions) for sid, positions in capture.targets.items()}
        ctx.adapter = NS(inner=model, spec=NS(num_layers=3),
                         provenance=dict(model_signature='model'), torch=torch)
        ctx.source_capture_v2 = capture
        ctx.target_candidates_v2 = {}
        ctx.finished = False
        ctx.engine = NS(session=session, finish_prefill=session.finish_prefill)
        ctx.advance_to_depth = session.advance_to_layer
        ctx.finish_timing_landmarks = {}
        ctx.capture_reservation = None
        ctx.committed = {}
        ctx.cached_prefix_tokens = 0
        ctx.synchronize = Mock()

        def completed_cpu_answer(hidden, on_first_token):
            on_first_token()
            ctx.finished = True
            return dict(answer=hidden.tolist(), token_ids=[int(hidden[-1, 0])])

        ctx.finish_from_prefill_hidden = completed_cpu_answer
        return ctx, model, session, capture

    def finish(self, ctx):
        callback = Mock()
        result = ctx.finish(callback)
        callback.assert_called_once()
        return result

    def publish(self, ctx, store, snapshot, scopes=None):
        return ctx.publish_target_candidates_v2(store, snapshot,
            {'C': self.miss} if scopes is None else scopes)

    def test_completed_native_capture_publishes_only_target_for_next_request(self):
        ctx, model, _, capture = self.context()
        store = self.store()
        snapshot = store.begin_request('birth')
        answer = self.finish(ctx)
        answer_before = dict(answer)
        candidate = ctx.target_candidates_v2['C']
        expected = candidate['layers'][0][0].clone()
        owner = weakref.ref(candidate['layers'][0][0])
        with patch.object(ctx.adapter, 'build_exact_dense_source', create=True,
                          side_effect=AssertionError('extra forward forbidden')), \
                patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('GPU forbidden')):
            audit = self.publish(ctx, store, snapshot)
        row = audit['targets']['C']
        self.assertEqual(row['status'], 'PUBLISHED')
        self.assertEqual(row['origin'], SourceOrigin.EXACT.value)
        self.assertEqual(model.real_block_calls, 3)
        self.assertEqual([b.self_attn.qkv_proj.calls for b in model.layers], [1, 1, 1])
        self.assertEqual(answer, answer_before)
        self.assertEqual(store.lookup(snapshot, (2, 3, 4)), ())
        with self.assertRaises(ValueError):
            store.read_selection(snapshot, row['source_id'], 1)
        with self.assertRaises(ValueError):
            with store.leased_target(snapshot, row['source_id']):
                pass
        store.end_request(snapshot)
        ctx.target_candidates_v2.clear()
        del candidate
        gc.collect()
        self.assertIsNone(owner())
        reader = store.begin_request('next-request')
        rows = store.lookup(reader, (2, 3, 4))
        self.assertEqual(len(rows), 1)
        with store.leased_target(reader, row['source_id']) as layers:
            self.assertEqual(len(layers), 3)
            self.assertTrue(torch.equal(layers[0][0], expected))
            self.assertEqual(tuple(layers[0][0].shape), (3, 1, 2))
        store.end_request(reader)
        storage = store.storage_audit()
        self.assertEqual(storage['target_kv_bytes'], 3 * 2 * 3 * 1 * 2 * 2)
        self.assertEqual(storage['selection_state_bytes'], 2 * 3 * 1 * 2 * 2)
        for key in ('parent_owned_kv_bytes', 'parent_lease_count', 'prefix_shadow_bytes'):
            self.assertEqual(storage[key], 0)
        self.assertEqual(storage['physical_backing'], 'SSD_ONLY_P0')
        self.assertEqual(audit['extra_forward_count'], 0)
        self.assertGreater(audit['post_request_host_ms'], 0)
        self.assertFalse(audit['native_runtime_qualified'])
        self.assertFalse(audit['production_dispatch_integrated'])
        self.assertFalse(audit['native_parent_consumption_qualified'])
        self.assertIs(capture.events[-1], audit)

    def test_mixed_parent_rank_preserved_and_G2_not_laundered(self):
        for generation in (0, 1):
            with self.subTest(parent_generation=generation):
                # Separate durable namespaces avoid conflicting birth identity.
                self.registry = self.registry_at('manifest-g' + str(generation))
                ctx, model, session, _ = self.context()
                store = self.store('pool-g' + str(generation))
                snapshot = store.begin_request('birth')
                session.advance_to_layer(1)
                commit(session, ctx.source_capture_v2, generation=generation)
                answer = self.finish(ctx)
                audit = self.publish(ctx, store, snapshot)
                self.assertEqual(model.real_block_calls, 3)
                self.assertTrue(answer['answer'])
                result = audit['targets']['C']
                if generation == 0:
                    self.assertEqual(result['status'], 'PUBLISHED')
                    self.assertEqual(result['origin'], SourceOrigin.MIXED.value)
                    self.assertEqual(result['generation'], 1)
                else:
                    self.assertEqual(result['status'], 'SKIPPED')
                    self.assertFalse(result['publication_performed'])
                    self.assertEqual(store.storage_audit()['source_count'], 0)
                store.end_request(snapshot)

    def test_early_unfinalized_or_unbound_publication_is_rejected(self):
        ctx, model, session, capture = self.context()
        store = self.store()
        snapshot = store.begin_request('birth')
        with self.assertRaisesRegex(RuntimeError, 'completed'):
            self.publish(ctx, store, snapshot)
        session.advance_to_layer(1)
        with self.assertRaisesRegex(RuntimeError, 'completed'):
            self.publish(ctx, store, snapshot)
        ctx.finished = True
        with self.assertRaisesRegex(ValueError, 'finalized actual native capture'):
            self.publish(ctx, store, snapshot)
        ctx.source_capture_v2 = None
        with self.assertRaisesRegex(ValueError, 'no legacy build fallback'):
            self.publish(ctx, store, snapshot)
        self.assertEqual(model.real_block_calls, 1)
        self.assertFalse(store.events)
        ctx.source_capture_v2 = capture
        store.end_request(snapshot)

    def test_missing_scope_skips_without_inferred_miss_or_repeat_attempt(self):
        ctx, model, _, _ = self.context()
        store = self.store()
        snapshot = store.begin_request('birth')
        self.finish(ctx)
        with patch.object(store, 'plan_publication', side_effect=AssertionError('no scope')):
            audit = self.publish(ctx, store, snapshot, scopes={})
        self.assertEqual(audit['targets']['C']['reason'], 'missing_publication_scope')
        with self.assertRaisesRegex(RuntimeError, 'not repeatable'):
            self.publish(ctx, store, snapshot)
        self.assertEqual(model.real_block_calls, 3)
        self.assertEqual(store.storage_audit()['source_count'], 0)
        store.end_request(snapshot)

    def test_mismatched_store_or_snapshot_never_reaches_plan(self):
        ctx, _, _, _ = self.context()
        self.finish(ctx)
        configs = (
            dict(model_signature='other-model'),
            dict(authorization_domain='other-domain'),
            dict(registry=self.registry_at('other-registry')),
        )
        for i, config in enumerate(configs):
            store = self.store('wrong-' + str(i), **config)
            snapshot = store.begin_request('birth')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                self.publish(ctx, store, snapshot)
            self.assertEqual(store.storage_audit()['source_count'], 0)
            store.end_request(snapshot)
        store = self.store()
        snapshot = store.begin_request('different-request')
        with self.assertRaisesRegex(ValueError, 'snapshot must belong'):
            self.publish(ctx, store, snapshot)
        store.end_request(snapshot)
        snapshot = store.begin_request('birth')
        with self.assertRaises(ValueError):
            self.publish(ctx, store, replace(snapshot, epoch=snapshot.epoch + 1))
        store.end_request(snapshot)
        with self.assertRaises(ValueError):
            self.publish(ctx, store, snapshot)

    def test_single_use_success_does_not_rewrite_or_recompute(self):
        ctx, model, _, _ = self.context()
        store = self.store()
        snapshot = store.begin_request('birth')
        self.finish(ctx)
        self.assertEqual(self.publish(ctx, store, snapshot)['targets']['C']['status'], 'PUBLISHED')
        file_sizes = {p.name: p.stat().st_size for p in store.root.iterdir()}
        with self.assertRaisesRegex(RuntimeError, 'not repeatable'):
            self.publish(ctx, store, snapshot)
        self.assertEqual(file_sizes, {p.name: p.stat().st_size for p in store.root.iterdir()})
        self.assertEqual(len(store.events), 1)
        self.assertEqual(model.real_block_calls, 3)
        store.end_request(snapshot)

    def test_failure_of_one_target_keeps_answer_and_other_publication(self):
        ctx, model, _, _ = self.context(targets={'bad': (1, 2), 'good': (3, 4)})
        store = self.store()
        snapshot = store.begin_request('birth')
        answer = self.finish(ctx)
        expected = dict(answer)
        ctx.target_candidates_v2['bad']['selection_states'] = {}
        audit = self.publish(ctx, store, snapshot, scopes={'bad': self.miss, 'good': self.miss})
        self.assertEqual(audit['targets']['bad']['status'], 'REJECTED')
        self.assertEqual(audit['targets']['good']['status'], 'PUBLISHED')
        self.assertEqual(answer, expected)
        self.assertEqual(model.real_block_calls, 3)
        self.assertEqual(store.storage_audit()['source_count'], 1)
        store.end_request(snapshot)

    def test_io_commit_failure_is_reported_without_answer_loss_or_retry(self):
        ctx, model, _, _ = self.context()
        store = self.store()
        snapshot = store.begin_request('birth')
        answer = self.finish(ctx)
        expected = dict(answer)
        with patch.object(store, '_stage', side_effect=OSError('disk full')):
            audit = self.publish(ctx, store, snapshot)
        self.assertEqual(audit['targets']['C']['status'], 'REJECTED')
        self.assertIn('disk full', audit['targets']['C']['reason'])
        self.assertFalse(audit['targets']['C']['publication_performed'])
        self.assertEqual(answer, expected)
        self.assertEqual(model.real_block_calls, 3)
        self.assertEqual(store.storage_audit()['source_count'], 0)
        store.end_request(snapshot)

    def test_unexpected_commit_exception_does_not_claim_definite_non_write(self):
        ctx, model, _, _ = self.context()
        store = self.store()
        snapshot = store.begin_request('birth')
        answer = self.finish(ctx)
        with patch.object(store, 'commit_publication', side_effect=RuntimeError('unknown outcome')):
            audit = self.publish(ctx, store, snapshot)
        self.assertEqual(audit['targets']['C']['status'], 'COMMIT_UNCERTAIN')
        self.assertIsNone(audit['targets']['C']['publication_performed'])
        self.assertTrue(audit['targets']['C']['recovery_required'])
        self.assertTrue(answer['answer'])
        self.assertEqual(model.real_block_calls, 3)
        store.end_request(snapshot)

    def test_post_commit_exception_blocks_remaining_targets_without_false_rollback(self):
        ctx, model, _, _ = self.context(targets={'first': (1, 2), 'second': (3, 4)})
        store = self.store()
        snapshot = store.begin_request('birth')
        answer = self.finish(ctx)
        actual_commit = store.commit_publication

        def committed_then_failed(*args, **kwargs):
            result = actual_commit(*args, **kwargs)
            self.assertEqual(result['status'], 'PUBLISHED')
            raise OSError('response failed after durable catalog commit')

        with patch.object(store, 'commit_publication', side_effect=committed_then_failed) as wrapped:
            audit = self.publish(ctx, store, snapshot,
                scopes={'first': self.miss, 'second': self.miss})
        wrapped.assert_called_once()
        self.assertEqual(audit['targets']['first']['status'], 'COMMIT_UNCERTAIN')
        self.assertIsNone(audit['targets']['first']['publication_performed'])
        self.assertEqual(audit['targets']['second']['reason'], 'prior_commit_outcome_uncertain')
        self.assertTrue(audit['recovery_required'])
        self.assertEqual(store.storage_audit()['source_count'], 1)
        self.assertTrue(answer['answer'])
        self.assertEqual(model.real_block_calls, 3)
        store.end_request(snapshot)

    def test_capture_budget_skip_never_calls_store_or_legacy_builder(self):
        ctx, model, _, _ = self.context(host_budget=1)
        store = self.store()
        snapshot = store.begin_request('birth')
        answer = self.finish(ctx)
        with patch.object(store, 'plan_publication', side_effect=AssertionError('no candidate')):
            audit = self.publish(ctx, store, snapshot)
        self.assertEqual(audit['targets']['C']['status'], 'SKIPPED')
        self.assertEqual(audit['targets']['C']['reason'], 'host_capture_budget_exceeded')
        self.assertEqual(ctx.deferred_canonical_builders(), {})
        self.assertEqual(ctx.export_exact_dense(), {})
        self.assertEqual(model.real_block_calls, 3)
        self.assertTrue(answer['answer'])
        store.end_request(snapshot)

    def test_partial_scope_or_tampered_occurrence_never_grows_pool(self):
        for index, kind in enumerate(('scope', 'occurrence', 'proof')):
            self.registry = self.registry_at('tamper-manifests-' + str(index))
            ctx, model, _, _ = self.context()
            store = self.store('tamper-pool-' + str(index))
            snapshot = store.begin_request('birth')
            self.finish(ctx)
            scope = self.miss
            candidate = ctx.target_candidates_v2['C']
            if kind == 'scope':
                scope = PublicationScope('temporary_unavailable', 0, 0, 0, 0)
            elif kind == 'occurrence':
                candidate['manifest_reference'] = replace(candidate['manifest_reference'], target_occurrence='other')
            else:
                candidate['proof'] = replace(candidate['proof'], request_id='other')
            audit = self.publish(ctx, store, snapshot, scopes={'C': scope})
            self.assertEqual(audit['targets']['C']['status'], 'REJECTED')
            self.assertEqual(store.storage_audit()['source_count'], 0)
            self.assertEqual(model.real_block_calls, 3)
            store.end_request(snapshot)


if __name__ == '__main__':
    unittest.main()
