"""Real local store/SelectionState integration, not GPU qualification."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from tests import test_source_comparison_v2 as comparison_fixtures
from probekv.source_comparison_v2 import ComparisonSessionV2
from probekv.source_metadata_selection import (MetadataSelectionPolicy,
    build_prefix_manifest, MetadataError)


class MetadataReceiptIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=comparison_fixtures.SourceComparisonReceiptTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    @staticmethod
    def prefix(tokens):
        return build_prefix_manifest(tokens,target_start=2,role_spans=((0,2,'document'),),
                                     tokenizer_signature='tokenizer')

    def setup_selection(self):
        f=self.fixture
        store=f.store()
        f.tokens=(20,21,12,13)
        first=f.publish(store,f.candidate('old-unrelated'))['source_id']
        old_first=self.prefix(f.tokens)
        f.tokens=(10,11,12,13)
        second=f.publish(store,f.candidate('old-related',base=20))['source_id']
        current=self.prefix(f.tokens)
        snapshot=store.begin_request('consumer')
        policy=MetadataSelectionPolicy(enabled=True,tau=100.,delta=100.)
        session=ComparisonSessionV2(store,snapshot,f.profile(tau_reuse=100.,tau_add=100.),
                                    workspace_bytes=100000,metadata_policy=policy)
        f.addCleanup(session.close)
        return store,snapshot,session,f.context(),first,second,current,{first:old_first,second:current}

    def test_real_comparison_reranks_without_full_kv_leases_lru_or_extra_projection(self):
        store,snapshot,session,context,a,b,current,old=self.setup_selection()
        before=store.lookup(snapshot,self.fixture.target_tokens)
        with patch('probekv.source_store_v2.LayerFile',side_effect=AssertionError('full KV forbidden')):
            receipt=session.compare_native(context,'target',current_prefix=current,historical_prefixes=old)
        self.assertEqual(receipt.metadata_selection.original_source_id,a)
        self.assertEqual(receipt.winner_source_id,b)
        self.assertLess(dict(receipt.scores)[a],dict(receipt.scores)[b])
        self.assertLessEqual(dict(receipt.scores)[b],session.profile.tau_reuse)
        self.assertEqual(receipt.counts,dict(stored=2,eligible=2,available=2,compared=2))
        self.assertEqual(store.lookup(snapshot,self.fixture.target_tokens),before)
        self.assertFalse(any(store._lease_counts.values()))
        context.observe_current_k.assert_called_once()
        session.verify(receipt,context)

    def test_missing_metadata_preserves_issued_state_winner(self):
        _,_,session,context,a,_,current,_=self.setup_selection()
        receipt=session.compare_native(context,'target',current_prefix=current)
        self.assertEqual(receipt.winner_source_id,a)
        self.assertEqual(receipt.metadata_selection.reason,'BAND_METADATA_UNAVAILABLE')

    def test_historical_identity_mismatch_does_not_issue_receipt(self):
        _,_,session,context,a,_,current,old=self.setup_selection()
        old[a]=current
        with self.assertRaisesRegex(MetadataError,'actual input'):
            session.compare_native(context,'target',current_prefix=current,historical_prefixes=old)
        self.assertFalse(session._issued)

    def test_policy_is_bound_and_tau_cannot_bypass_original_admission(self):
        _,_,session,context,_,_,current,old=self.setup_selection()
        receipt=session.compare_native(context,'target',current_prefix=current,historical_prefixes=old)
        session.metadata_policy=replace(session.metadata_policy,delta=99.)
        with self.assertRaisesRegex(ValueError,'policy changed'): session.verify(receipt,context)
        session.metadata_policy=replace(session.metadata_policy,tau=101.)
        with self.assertRaisesRegex(ValueError,'threshold'): session.verify(receipt,context)

    def test_current_target_alignment_is_verified(self):
        _,_,session,context,_,_,current,old=self.setup_selection()
        with self.assertRaisesRegex(MetadataError,'target/tokenizer'):
            session.compare_native(context,'target',current_prefix=replace(current,target_start=3),
                                   historical_prefixes=old)


if __name__=='__main__': unittest.main()
