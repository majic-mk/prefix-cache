"""CPU provenance restrictions, not actual-model mixed-reference evidence."""
from contextlib import ExitStack
from dataclasses import FrozenInstanceError
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import tests.test_native_v2_prefix_publication as prefix_helpers
from probekv.p0_diagnostic_context_v2 import (
    P0DiagnosticContextV2, bind_p0_diagnostic_context, diagnostic_context_v2)


class P0DiagnosticContextTests(unittest.TestCase):
    def fresh(self):
        context, _ = prefix_helpers.NativeV2PrefixPublicationTests().fixture(v2=False)
        context.engine = None
        context.closed = False
        context.prepared = {}; context.frozen = {}
        context.target_candidates_v2 = {}; context.canonical_exports = {}
        context.p0_diagnostic_v2 = None
        return context

    def bind(self, context):
        return bind_p0_diagnostic_context(context, recipe_sha256='a'*64)

    def test_marker_is_immutable_and_grants_no_authority(self):
        context = self.fresh(); marker = self.bind(context)
        self.assertIs(diagnostic_context_v2(context), marker)
        self.assertIs(type(marker), P0DiagnosticContextV2)
        for field in ('grants_execution_authority', 'exact_prefix_publication_allowed',
                      'source_publication_allowed', 'native_gpu_qualified'):
            self.assertIs(marker.audit()[field], False)
        with self.assertRaises(FrozenInstanceError):
            marker.kind = 'exact'
        with self.assertRaises(ValueError):
            self.bind(context)

    def test_binding_rejects_nonfresh_or_existing_capture_contexts(self):
        updates = [dict(finished=True), dict(closed=True), dict(cached_prefix_tokens=1),
                   dict(engine=NS(session=NS(current_layer=0))),
                   dict(source_capture_v2=object()), dict(source_consumption_v2=object()),
                   dict(prepared={'A':object()}), dict(frozen={'A':'source'}),
                   dict(committed={'A':2}), dict(target_candidates_v2={'A':object()}),
                   dict(canonical_exports={'A':object()}), dict(capture_reservation=object()),
                   dict(capture_collector=object()), dict(probe_fallback_reason='missing'),
                   dict(p0_diagnostic_resources_v2=object())]
        for values in updates:
            with self.subTest(values=tuple(values)):
                context = self.fresh(); context.__dict__.update(values)
                with self.assertRaises(ValueError): self.bind(context)
        for field in ('capture_original_full_prefill', 'publish_exact_prefix_shadow'):
            context = self.fresh(); context.request[field] = True
            with self.assertRaises(ValueError): self.bind(context)

    def test_invalid_recipe_or_kind_and_untyped_context_rejected(self):
        for recipe in ('', 'G'*64, True, 'a'*63):
            with self.assertRaises(ValueError):
                bind_p0_diagnostic_context(self.fresh(), recipe_sha256=recipe)
        with self.assertRaises(ValueError):
            bind_p0_diagnostic_context(self.fresh(), recipe_sha256='a'*64, kind='exact')
        with self.assertRaises(ValueError):
            bind_p0_diagnostic_context(NS(), recipe_sha256='a'*64)

    def test_stale_or_untyped_marker_fails_before_native_sampling(self):
        for mode in ('tokens', 'request_id', 'model', 'prefix', 'untyped', 'bad_recipe'):
            with self.subTest(mode=mode):
                context = self.fresh(); self.bind(context)
                if mode == 'tokens': context.request['token_ids'] = (11, 12, 99)
                elif mode == 'request_id': context.request['request_id'] = 'different'
                elif mode == 'model': context.adapter.provenance['model_signature'] = 'other'
                elif mode == 'prefix': context.cached_prefix_tokens = 1
                elif mode == 'untyped': context.p0_diagnostic_v2 = {'kind':'explicit_mixed_reference'}
                else: object.__setattr__(context.p0_diagnostic_v2, 'recipe_sha256', 'bad')
                with self.assertRaises(ValueError):
                    context.finish_from_prefill_hidden(None, Mock())
                context.native.finish_prefill.assert_not_called()
                for method in (context.export_exact_dense, context.deferred_canonical_builders,
                               context.materialization_metadata):
                    with self.assertRaises(ValueError): method()

    def test_complete_capture_cannot_override_restrictive_marker(self):
        helpers = prefix_helpers.NativeV2PrefixPublicationTests()
        context, _ = helpers.fixture()
        from probekv.source_manifest_v2 import request_input_digest
        context.p0_diagnostic_v2 = P0DiagnosticContextV2('request',
            request_input_digest((11,12,13), (0,1,2)), 'model', 'a'*64)
        self.assertFalse(helpers.decision(context))
        self.assertEqual(context.v2_prefix_publication_audit['reason'],
                         'p0_explicit_mixed_reference_not_exact')

    def test_full_endpoint_mixed_label_never_warms_exact_history(self):
        helpers = prefix_helpers.NativeV2PrefixPublicationTests()
        for capture in (False, True):
            for teacher in (False, True):
                context, _ = helpers.fixture(v2=capture)
                from probekv.source_manifest_v2 import request_input_digest
                context.p0_diagnostic_v2 = P0DiagnosticContextV2('request',
                    request_input_digest((11,12,13), (0,1,2)), 'model', 'a'*64)
                if teacher: context.request['teacher_token_ids'] = [1]
                result = helpers.full_endpoint(context)
                self.assertEqual(result['whole_request_origin'], 'p0_explicit_mixed_reference')
                self.assertEqual(context.adapter.warm_history, [])
                self.assertFalse(context.native.finish_prefill.call_args.kwargs['exact_dense'])
                self.assertEqual(result['p0_diagnostic_context_v2']['recipe_sha256'], 'a'*64)
                self.assertFalse(result['p0_diagnostic_context_v2']['native_gpu_qualified'])

    def test_publication_capture_and_ordinary_finish_are_restricted(self):
        context = self.fresh(); self.bind(context)
        context.canonical_exports = {'injected':object()}
        context.target_candidates_v2 = {'injected':object()}
        for method in (context.export_exact_dense, context.deferred_canonical_builders,
                       context.materialization_metadata):
            self.assertEqual(method(), {})
        calls = [lambda:context.configure_target_capture_v2(target_ids=[], registry=None,
                      authorization_domain='none', host_budget_bytes=1),
                 context._enable_original_capture, context.export_target_candidates_v2,
                 lambda:context.bind_capture_parent_v2(None),
                 lambda:context.configure_source_consumption_v2(None),
                 lambda:context.publish_target_candidates_v2(None,None,{}),
                 lambda:context.publish_compared_targets_v2(None,{}),
                 lambda:context.finish(Mock())]
        for call in calls:
            with self.assertRaises(ValueError): call()

    def close_fixture(self):
        context = self.fresh()
        context.synchronize = Mock()
        context.hot_leases = ExitStack(); context.hot_replicas = {}
        context._observation = {}; context.workspace = None
        context.capture_reservation = None
        context.adapter.inner.layers = []
        context.p0_diagnostic_resources_v2 = NS(quarantined=False, close_after_fence=Mock())
        return context

    def test_resources_release_only_after_fence_and_only_once(self):
        context = self.close_fixture(); calls = []
        context.synchronize.side_effect = lambda:calls.append('fence')
        context.p0_diagnostic_resources_v2.close_after_fence.side_effect = lambda:calls.append('release')
        context.close(); context.close()
        self.assertEqual(calls, ['fence','release'])
        self.assertTrue(context.closed)

    def test_failed_fence_quarantines_and_never_releases(self):
        context = self.close_fixture()
        context.synchronize.side_effect = RuntimeError('fence failed')
        with self.assertRaisesRegex(RuntimeError, 'fence failed'): context.close()
        self.assertTrue(context.p0_diagnostic_resources_v2.quarantined)
        context.p0_diagnostic_resources_v2.close_after_fence.assert_not_called()
        self.assertFalse(context.closed)

    def test_resource_cleanup_error_does_not_mark_context_closed(self):
        context = self.close_fixture()
        context.p0_diagnostic_resources_v2.close_after_fence.side_effect = RuntimeError('cleanup failed')
        with self.assertRaisesRegex(RuntimeError, 'cleanup failed'): context.close()
        self.assertFalse(context.closed)


if __name__ == '__main__':
    unittest.main()
