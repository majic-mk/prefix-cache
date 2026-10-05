"""CPU decision tests only; native block completion is deliberately mocked."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from probekv.source_manifest_v2 import request_input_digest
from probekv.source_provenance_v2 import KVImport, RequestExecutionLedger
from probekv.v8_schema10_native_adapter import NativeRequestContext


class _AfterPrefixDecision(Exception):
    """Stop before sampling: this suite only tests native publication input."""


class NativeV2PrefixPublicationTests(unittest.TestCase):
    def fixture(self, *, v2=True, complete=True, mixed=False, unknown=False):
        tokens = (11, 12, 13)
        positions = tuple(range(len(tokens)))
        ledger = RequestExecutionLedger(request_id='request',
            input_digest=request_input_digest(tokens, positions),
            model_signature='model', token_count=3, num_layers=2,
            exact_input_proof='cpu-input-proof' if not unknown else None)
        if complete:
            for layer in (1, 2):
                rows = (1, 2) if mixed else positions
                ledger.record_layer(layer_1based=layer, qkv_rows=rows,
                    attention_rows=rows, output_mlp_rows=rows,
                    imported_kv=(KVImport(0, 'source', 0),) if mixed else (),
                    completion_reference='cpu-fake-block')
        session = NS(commits={}, exact_prefix_tokens=0, token_ids=tokens,
                     absolute_positions=positions, model_signature='model')
        context = object.__new__(NativeRequestContext)
        context.finished = False
        context.committed = {}
        context.cached_prefix_tokens = 0
        context.request = dict(request_id='request', token_ids=tokens)
        context.adapter = NS(torch=None, inner=NS(cache_fuse_metadata={}),
                             provenance=dict(model_signature='model'), spec=NS(num_layers=2))
        context.engine = NS(session=session)
        context.source_capture_v2 = NS(ledger=ledger, _failed=False) if v2 else None
        context.native = NS(finish_prefill=Mock(side_effect=_AfterPrefixDecision))
        return context, ledger

    def decision(self, context):
        with self.assertRaises(_AfterPrefixDecision):
            context.finish_from_prefill_hidden(None, Mock())
        return context.native.finish_prefill.call_args.kwargs['exact_dense']

    def test_complete_unchanged_whole_request_G0_allows_exact_publication(self):
        context, _ = self.fixture()
        self.assertTrue(self.decision(context))
        self.assertEqual(context.v2_prefix_publication_audit['reason'], 'complete_known_G0_request')

    def test_session_only_commit_does_not_poison_native_exact_prefix(self):
        context, _ = self.fixture()
        context.engine.session.commits['A'] = object()
        self.assertFalse(self.decision(context))
        self.assertEqual(context.v2_prefix_publication_audit['reason'], 'context_or_session_reuse_commit')

    def test_context_only_or_matching_commits_also_decline_publication(self):
        for session_commit in (False, True):
            context, _ = self.fixture()
            context.committed['A'] = 2
            if session_commit:
                context.engine.session.commits['A'] = object()
            self.assertFalse(self.decision(context))

    def test_incomplete_unknown_or_mixed_ledger_is_not_exact(self):
        for settings in (dict(complete=False), dict(unknown=True), dict(mixed=True)):
            with self.subTest(settings=settings):
                context, _ = self.fixture(**settings)
                self.assertFalse(self.decision(context))

    def test_changed_request_model_or_bound_ledger_identity_declines(self):
        for field in ('tokens', 'request_id', 'model', 'ledger_identity', 'session_positions'):
            with self.subTest(field=field):
                context, ledger = self.fixture()
                if field == 'tokens':
                    context.request['token_ids'] = (11, 12, 99)
                elif field == 'request_id':
                    context.request['request_id'] = 'other'
                elif field == 'model':
                    context.adapter.provenance['model_signature'] = 'other-model'
                elif field == 'ledger_identity':
                    ledger.exact_input_proof = 'mutated-proof'
                else:
                    context.engine.session.absolute_positions = (1, 2, 3)
                self.assertFalse(self.decision(context))

    def test_failed_capture_prefix_hit_or_missing_ledger_declines(self):
        for field in ('failed', 'prefix', 'ledger'):
            with self.subTest(field=field):
                context, _ = self.fixture()
                if field == 'failed':
                    context.source_capture_v2._failed = True
                elif field == 'prefix':
                    context.cached_prefix_tokens = 1
                else:
                    context.source_capture_v2.ledger = None
                self.assertFalse(self.decision(context))

    def test_historical_default_is_unchanged_without_v2(self):
        context, _ = self.fixture(v2=False)
        context.engine.session.commits['diagnostic'] = object()
        self.assertTrue(self.decision(context))
        self.assertFalse(hasattr(context, 'v2_prefix_publication_audit'))
        context, _ = self.fixture(v2=False)
        context.committed['A'] = 2
        self.assertFalse(self.decision(context))

    def full_endpoint(self, context):
        context.native.finish_prefill = Mock()
        context.finish_timing_landmarks = {}
        context.sampling = NS(selected_token_indices=torch.tensor([2]))
        context.sampling_signature = {'max_new_tokens': 1}
        context.adapter.torch = torch
        context.adapter.outer = NS(compute_logits=lambda *args:torch.tensor([[0.,1.]]))
        context.adapter.llm = NS(get_tokenizer=lambda:NS(eos_token_id=99))
        context.adapter.warm_history = []
        context.engine.session.resolve_completed_layer_timings = Mock()
        context.engine.session.layer_audit = []
        context.engine.overlap_trace = lambda:[]
        with patch('probekv.v8_schema10_qa.answer_evidence',return_value={'token_ids':[1]}):
            return context.finish_from_prefill_hidden(torch.zeros(3,4),Mock())

    def test_full_endpoint_never_warms_or_labels_unverified_execution_exact(self):
        for case in ('session_commit','context_commit','failed_capture','mixed','incomplete'):
            with self.subTest(case=case):
                context,_=self.fixture(mixed=case=='mixed',complete=case!='incomplete')
                if case=='session_commit':context.engine.session.commits['A']=object()
                elif case=='context_commit':context.committed['A']=1
                elif case=='failed_capture':context.source_capture_v2._failed=True
                result=self.full_endpoint(context)
                self.assertNotEqual(result['whole_request_origin'],'exact_dense_full_prefill')
                self.assertEqual(context.adapter.warm_history,[])
                self.assertFalse(result['v2_prefix_publication_audit']['exact_prefix_publication_allowed'])

    def test_full_endpoint_verified_G0_warms_exact_but_teacher_does_not(self):
        for teacher in (False,True):
            context,_=self.fixture()
            if teacher:context.request['teacher_token_ids']=[1]
            result=self.full_endpoint(context)
            self.assertEqual(result['whole_request_origin'],'exact_dense_full_prefill')
            self.assertEqual(len(context.adapter.warm_history),0 if teacher else 1)
            self.assertTrue(result['v2_prefix_publication_audit']['exact_prefix_publication_allowed'])

    def test_consumption_without_ledger_never_publishes_exact_prefix(self):
        for session_commit in (False,True):
            context,_=self.fixture(v2=False)
            context.source_consumption_v2=object()
            if session_commit:context.engine.session.commits['A']=object()
            self.assertFalse(self.decision(context))
            self.assertEqual(context.v2_prefix_publication_audit['reason'],
                             'v2_consumption_without_whole_request_proof')

    def test_consumption_without_ledger_never_warms_or_labels_exact(self):
        for session_commit in (False,True):
            context,_=self.fixture(v2=False)
            context.source_consumption_v2=object()
            if session_commit:context.engine.session.commits['A']=object()
            result=self.full_endpoint(context)
            self.assertFalse(context.native.finish_prefill.call_args.kwargs['exact_dense'])
            self.assertEqual(context.adapter.warm_history,[])
            self.assertNotEqual(result['whole_request_origin'],'exact_dense_full_prefill')


if __name__ == '__main__':
    unittest.main()
