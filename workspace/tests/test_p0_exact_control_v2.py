"""T20 CPU control integration; actual fake blocks/hooks, never GPU evidence."""
from copy import deepcopy
import time
import unittest
from unittest.mock import patch

from probekv.p0_exact_control_v2 import execute_exact_capture_control, validate_exact_control_job
from probekv.native_p0_request_v2 import P0RequestFailure
from probekv.v8_schema10_native_adapter import NativeRequestContext
from tests import test_native_p0_request_v2 as request_fixtures


class P0ExactControlTests(unittest.TestCase):
    def context(self, capture_enabled=True):
        fixture = request_fixtures.NativeP0RequestTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        store, context, snapshot, _, _, model = fixture.captured_request()
        self.addCleanup(lambda: store.end_request(snapshot) if snapshot.snapshot_id in store._snapshots else None)
        # The source fixture starts an actual CPU resumable session, but the
        # operation under test must reserve its own fresh PRIVATE capture.
        previous = context.engine.session.provenance_capture
        previous.close()
        context.engine.session.provenance_capture = None
        context.configure_target_capture_v2 = NativeRequestContext.configure_target_capture_v2.__get__(context)
        context.request.update(segments=[dict(segment_id='C', positions=[2, 3, 4], token_ids=[2, 3, 4])],
                               capture_logits=True, teacher_token_ids=[7, 8], max_new_tokens=3,
                               correctness_repair_ratio=.15)
        context.arrival_ns = time.perf_counter_ns()
        job = dict(operation='exact_capture_control', capture_enabled=capture_enabled,
                   target_ids=['C'], request=context.request)
        return store, context, model, job

    def execute(self, context, job, **overrides):
        args = dict(authorization_domain='domain', host_capture_bytes=10000,
                    registry_budget=dict(max_bytes=1000000, max_manifest_bytes=100000))
        args.update(overrides)
        return execute_exact_capture_control(context, job, **args)

    def test_capture_on_off_same_actual_full_row_recipe_and_no_publication(self):
        results = []
        for enabled in (False, True):
            store, context, model, job = self.context(enabled)
            original_registry = store.registry
            before = deepcopy(store.storage_audit())
            with patch.object(store, 'commit_publication', side_effect=AssertionError('diagnostic cannot publish')):
                result = self.execute(context, job)
            results.append(result)
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(result['publication']['status'], 'DIAGNOSTIC_NOT_PUBLISHED')
            self.assertEqual(model.real_block_calls, 3)
            self.assertEqual([layer.self_attn.qkv_proj.calls for layer in model.layers], [1, 1, 1])
            self.assertEqual([row['layer'] for row in result['execution']['recipe']['layers']], [1, 2, 3])
            self.assertEqual(store.storage_audit(), before)
            self.assertEqual(original_registry.storage_audit()['manifest_count'], 0)
            self.assertTrue(context.closed)
            self.assertIsNone(context.engine)
            self.assertFalse(context.target_candidates_v2)
            self.assertEqual(context.adapter.hbm.active_reserved_bytes, 0)
            self.assertTrue(result['cleanup']['passed'])
            self.assertFalse(result['native_runtime_qualified'])
            self.assertFalse(result['P1_execution_allowed'])
            self.assertEqual(result['extra_forward_count'], 0)
            # One persistent fixture projection hook remains; no capture hook leaks.
            self.assertEqual([len(layer.self_attn.qkv_proj._forward_hooks) for layer in model.layers], [1, 1, 1])
            if enabled:
                capture = context.source_capture_v2
                self.assertTrue(capture._finalized)
                self.assertEqual(capture.audit()['layers_recorded'], 3)
                self.assertIsNot(capture.registry, original_registry)
                self.assertFalse(capture.registry.durable)
                self.assertEqual(capture.registry.storage_audit()['manifest_count'], 1)
                target = result['captured_targets']['C']
                self.assertEqual(target['proof']['generation'], 0)
                self.assertEqual(target['storage_audit']['parent_owned_kv_bytes'], 0)
                self.assertEqual(target['storage_audit']['prefix_shadow_bytes'], 0)
            else:
                self.assertIsNone(result['capture'])
                self.assertFalse(context.target_candidates_v2)
        self.assertEqual(results[0]['execution'], results[1]['execution'])
        self.assertEqual(results[0]['answer']['token_ids'], results[1]['answer']['token_ids'])

    def test_capture_capacity_failure_keeps_answer_but_cannot_complete_control(self):
        store, context, model, job = self.context()
        with self.assertRaises(P0RequestFailure) as caught:
            self.execute(context, job, host_capture_bytes=1)
        audit = caught.exception.audit
        self.assertEqual(audit['status'], 'FAILED')
        self.assertIsNotNone(audit['answer'])
        self.assertEqual(model.real_block_calls, 3)
        self.assertTrue(audit['cleanup']['passed'])
        self.assertEqual(store.storage_audit()['source_count'], 0)
        self.assertFalse(audit['native_runtime_qualified'])

    def test_actual_block_failure_preserves_failure_and_cleans_capture_hook(self):
        store, context, model, job = self.context()
        model.fail_at = 2
        with self.assertRaises(P0RequestFailure) as caught:
            self.execute(context, job)
        audit = caught.exception.audit
        self.assertEqual(model.real_block_calls, 2)
        self.assertEqual([layer.self_attn.qkv_proj.calls for layer in model.layers], [1, 1, 0])
        self.assertIsNone(audit['answer'])
        self.assertTrue(audit['cleanup']['passed'])
        self.assertEqual(context.adapter.hbm.active_reserved_bytes, 0)
        self.assertEqual([len(layer.self_attn.qkv_proj._forward_hooks) for layer in model.layers], [1, 1, 1])
        self.assertEqual(store.storage_audit()['source_count'], 0)

    def test_source_decision_fields_rejected_before_any_block(self):
        for field, value in (('sources_by_segment', {'C': []}), ('comparison_profile', {})):
            with self.subTest(field=field):
                _, context, model, job = self.context()
                job[field] = value
                with self.assertRaises(ValueError): self.execute(context, job)
                self.assertEqual(model.real_block_calls, 0)
                self.assertFalse(context.closed)

    def test_prefix_and_existing_source_context_rejected_before_execution(self):
        for field, value in (('cached_prefix_tokens', 1), ('probe_fallback_reason', 'missing shadow'),
                             ('frozen', {'C': 'unvalidated'}), ('prepared', {'C': object()}),
                             ('committed', {'C': 2})):
            with self.subTest(field=field):
                _, context, model, job = self.context()
                setattr(context, field, value)
                with self.assertRaises(ValueError): self.execute(context, job)
                self.assertEqual(model.real_block_calls, 0)

    def test_teacher_and_executor_flags_rejected_before_execution(self):
        variants = (('capture_logits', False), ('teacher_token_ids', []), ('teacher_token_ids', [True, 8]),
                    ('teacher_token_ids', [7, -1]), ('native_dense_continuation', True),
                    ('publish_exact_prefix_shadow', True), ('capture_original_full_prefill', True),
                    ('use_gpu_hot_cache', True), ('retain_gpu_hot_cache', True),
                    ('prefetch_window', 1), ('correctness_repair_ratio', 1.))
        for field, value in variants:
            with self.subTest(field=field):
                _, context, model, job = self.context()
                context.request[field] = value
                with self.assertRaises(ValueError): self.execute(context, job)
                self.assertEqual(model.real_block_calls, 0)

    def test_invalid_targets_and_toggle_rejected(self):
        for field, value in (('target_ids', []), ('target_ids', ['C', 'C']), ('target_ids', ['UNKNOWN']),
                             ('capture_enabled', 1), ('operation', 'mixed_reference')):
            with self.subTest(field=field):
                _, context, model, job = self.context()
                job[field] = value
                with self.assertRaises(ValueError): validate_exact_control_job(job)
                self.assertEqual(model.real_block_calls, 0)

    def test_actual_arrival_and_repair_ratio_rejected_before_blocks(self):
        for field, value in (('arrival_ns', 0), ('repair_ratio', 1.)):
            with self.subTest(field=field):
                _, context, model, job = self.context()
                setattr(context, field, value)
                with self.assertRaises(ValueError): self.execute(context, job)
                self.assertEqual(model.real_block_calls, 0)

    def test_private_registry_cannot_be_redirected_to_persistent_root(self):
        store, context, model, job = self.context()
        with self.assertRaises(ValueError):
            self.execute(context, job, registry_budget=dict(max_bytes=1000000,
                max_manifest_bytes=100000, persistent_root=store.root/'forbidden-diagnostic-registry'))
        self.assertEqual(model.real_block_calls, 0)

    def test_missing_first_token_cannot_be_completed_control(self):
        _, context, model, job = self.context(False)
        endpoint = context.finish_from_prefill_hidden
        context.finish_from_prefill_hidden = lambda hidden, callback: endpoint(hidden, lambda: None)
        with self.assertRaises(P0RequestFailure) as caught:
            self.execute(context, job)
        self.assertIsNotNone(caught.exception.audit['answer'])
        self.assertEqual(model.real_block_calls, 3)
        self.assertTrue(caught.exception.audit['cleanup']['passed'])

    def test_cleanup_failure_not_relabelled_as_success(self):
        _, context, model, job = self.context(False)
        # A failed device fence is retained for explicit recovery, not hidden.
        context.synchronize.side_effect = RuntimeError('fixture device fence failed')
        try:
            with self.assertRaises(P0RequestFailure) as caught:
                self.execute(context, job)
            audit = caught.exception.audit
            self.assertEqual(model.real_block_calls, 3)
            self.assertEqual(audit['status'], 'FAILED')
            self.assertFalse(audit['cleanup']['passed'])
            self.assertFalse(context.closed)
        finally:
            context.synchronize.side_effect = None
            context.close()


if __name__ == '__main__':
    unittest.main()
