"""CPU arithmetic/contracts only; not evidence of GPU speed or QA quality."""
from dataclasses import asdict, replace
import random
import unittest

from probekv.source_metadata_selection import (MetadataError, PrefixOccurrence,
    PrefixManifest, MetadataSelectionPolicy, SharedPrefixManifests, StateCandidate,
    build_prefix_manifest, score_prefixes, select_source, validate_prefix_tokens)


def prefix(keys, lengths=None, roles=None):
    lengths = lengths or [1]*len(keys)
    roles = roles or ['document']*len(keys)
    rows, start = [], 0
    for i, (key, n, role) in enumerate(zip(keys, lengths, roles)):
        rows.append(PrefixOccurrence(key, role, n, i, start, start+n))
        start += n
    return PrefixManifest('tok', start, tuple(rows))


class PrefixScoreTests(unittest.TestCase):
    def test_user_example(self):
        old = prefix('ABCDEFGH', [10]*8)
        new = prefix('BACDXY', [10]*6)
        result = score_prefixes(old, new)
        self.assertTrue(result.valid)
        self.assertEqual((result.L_old,result.L_new,result.W_match,result.matched_count), (80,60,40,4))
        self.assertEqual(result.overlap, .4)
        self.assertAlmostEqual(result.order_penalty, 1/6)
        self.assertAlmostEqual(result.metadata_score, 1/3)

    def test_duplicates_pair_in_occurrence_order_not_sets(self):
        result = score_prefixes(prefix('AABA'), prefix('ABA'))
        self.assertEqual(result.matched_count, 3)
        self.assertEqual(result.W_match, 3)
        self.assertEqual(result.overlap, .75)
        self.assertAlmostEqual(result.order_penalty, 1/3)
        self.assertAlmostEqual(result.metadata_score, .5)

    def test_length_weighted_inversion_not_unweighted(self):
        result = score_prefixes(prefix('ABC', [10,1,1]), prefix('BAC',[1,10,1]))
        self.assertAlmostEqual(result.order_penalty, 10/21)
        self.assertAlmostEqual(result.metadata_score, 11/21)

    def test_weighted_fenwick_matches_pairwise_reference(self):
        rng = random.Random(271)
        for _ in range(100):
            count = rng.randrange(2, 25)
            keys = list(map(str, range(count)))
            lengths = {key:rng.randrange(1, 1000) for key in keys}
            shuffled = keys[:]; rng.shuffle(shuffled)
            result = score_prefixes(prefix(keys,[lengths[k] for k in keys]),
                                    prefix(shuffled,[lengths[k] for k in shuffled]))
            ranks = {k:i for i,k in enumerate(shuffled)}
            denom = sum(lengths[a]*lengths[b] for i,a in enumerate(keys) for b in keys[i+1:])
            inv = sum(lengths[a]*lengths[b] for i,a in enumerate(keys) for b in keys[i+1:]
                      if ranks[a]>ranks[b])
            self.assertEqual(result.order_penalty, inv/denom)

    def test_empty_missing_incomplete_are_distinct(self):
        empty = prefix('')
        result = score_prefixes(empty, empty)
        self.assertEqual(result.metadata_score, 1)
        self.assertTrue(result.both_prefixes_empty)
        for left,right in ((empty,prefix('A')), (prefix('A'),empty), (prefix('A'),prefix('B'))):
            result = score_prefixes(left,right)
            self.assertTrue(result.valid)
            self.assertEqual(result.metadata_score, 0)
            self.assertEqual(result.order_penalty, 0)
        for left in (None, replace(empty,complete=False), PrefixManifest('tok',2,())):
            result = score_prefixes(left,empty)
            self.assertFalse(result.valid)
            self.assertIsNone(result.metadata_score)
            self.assertIsNone(result.overlap)
        self.assertEqual(score_prefixes(prefix('A'),prefix('AB')).order_penalty,0)

    def test_type_compatibility_explicit_and_one_to_one(self):
        old = prefix('AA', roles=['instruction','document'])
        new = prefix('AA', roles=['system','document'])
        self.assertEqual(score_prefixes(old,new).matched_count,1)
        result = score_prefixes(old,new,compatible_role_groups=(('instruction','system'),))
        self.assertEqual(result.matched_count,2)
        self.assertEqual(result.metadata_score,1)
        with self.assertRaises(MetadataError):
            score_prefixes(old,new,compatible_role_groups=(('a','b'),('b','c')))

    def test_identity_length_and_tokenizer_contradictions_raise(self):
        with self.assertRaises(MetadataError): score_prefixes(prefix('A',[2]),prefix('A',[3]))
        with self.assertRaises(MetadataError): prefix('AA',[2,3])
        with self.assertRaises(MetadataError):
            score_prefixes(prefix('A'),replace(prefix('A'),tokenizer_signature='other'))

    def test_overlap_wrong_order_target_padding_and_nonfinite_raise(self):
        base = PrefixOccurrence('a','document',2,0,0,2)
        for changes in (dict(token_count=3),dict(token_count=float('nan')),
                        dict(order_index=float('inf')),dict(role='padding'),dict(token_start=-1)):
            with self.subTest(changes=changes),self.assertRaises(MetadataError):
                replace(base,**changes)
        for second in (PrefixOccurrence('b','document',2,1,1,3),
                       PrefixOccurrence('b','document',2,0,2,4)):
            with self.assertRaises(MetadataError): PrefixManifest('tok',4,(base,second))
        with self.assertRaises(MetadataError): PrefixManifest('tok',1,(base,))

    def test_builder_covers_all_roles_and_retains_no_tokens(self):
        tokens = [1,2,3,4,5,6]
        spans = [(0,1,'system'),(1,2,'instruction'),(2,3,'document'),(3,4,'separator')]
        manifest = build_prefix_manifest(tokens,target_start=4,role_spans=spans,tokenizer_signature='tok')
        self.assertTrue(manifest.fully_covered)
        self.assertNotIn('token_ids',asdict(manifest))
        validate_prefix_tokens(manifest,tokens,target_start=4,tokenizer_signature='tok')
        with self.assertRaises(MetadataError):
            validate_prefix_tokens(manifest,[9]+tokens[1:],target_start=4,tokenizer_signature='tok')
        with self.assertRaises(MetadataError):
            validate_prefix_tokens(manifest,tokens,target_start=5,tokenizer_signature='tok')
        hole = build_prefix_manifest(tokens,target_start=4,role_spans=spans[1:],tokenizer_signature='tok')
        self.assertFalse(score_prefixes(hole,manifest).valid)

    def test_shared_manifest_roundtrip_dedup_budget_and_corruption(self):
        store = SharedPrefixManifests(max_bytes=2000)
        manifest = prefix('AB',[2,3])
        ref = store.register(manifest); used = store.bytes_used
        self.assertEqual(store.register(manifest),ref)
        self.assertEqual(store.bytes_used,used)
        restored = SharedPrefixManifests(max_bytes=2000)
        self.assertEqual(restored.import_payload(store.to_payload(ref)),ref)
        self.assertEqual(restored.resolve(ref),manifest)
        self.assertIsNone(restored.resolve('missing'))
        body = store.to_payload(ref); body['payload']['target_start']=99
        with self.assertRaises(MetadataError): restored.import_payload(body)
        with self.assertRaises(MemoryError): SharedPrefixManifests(max_bytes=1).register(manifest)


class MetadataSelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows=(StateCandidate('A',.1),StateCandidate('B',.12),StateCandidate('C',.3))
        self.current=prefix('AB')
        self.old={'A':prefix('BA'),'B':prefix('AB'),'C':prefix('AB')}
        self.policy=MetadataSelectionPolicy(enabled=True,tau=.2,delta=.03)

    def select(self, **changes):
        kwargs=dict(policy=self.policy,current_prefix=self.current,historical_prefixes=self.old)
        kwargs.update(changes)
        return select_source(self.rows,**kwargs)

    def test_only_near_compatible_candidates_reranked(self):
        result=self.select()
        self.assertEqual(result.original_source_id,'A')
        self.assertEqual(result.selected_source_id,'B')
        self.assertEqual(result.band_source_ids,('A','B'))
        self.assertTrue(result.requires_selected_source_admission)

    def test_disabled_does_not_access_metadata_or_change_state_winner(self):
        class Forbidden(dict):
            def get(self,*args): raise AssertionError('disabled must not read metadata')
        result=self.select(policy=MetadataSelectionPolicy(),historical_prefixes=Forbidden(x=1))
        self.assertEqual(result.selected_source_id,'A')
        self.assertEqual(result.scores,())

    def test_single_band_does_not_need_metadata(self):
        result=self.select(policy=replace(self.policy,delta=0),current_prefix=None,historical_prefixes={})
        self.assertEqual(result.selected_source_id,'A')
        self.assertEqual(result.reason,'SINGLE_NEAR_CANDIDATE')

    def test_missing_one_entire_band_falls_back_and_outside_band_ignored(self):
        for mapping in ({'B':self.old['B']},{'A':self.old['A']},{}):
            result=self.select(historical_prefixes=mapping)
            self.assertEqual(result.reason,'BAND_METADATA_UNAVAILABLE')
            self.assertEqual(result.selected_source_id,'A')
        self.assertEqual(self.select(historical_prefixes={k:v for k,v in self.old.items() if k!='C'}).selected_source_id,'B')

    def test_incomplete_and_missing_current_fall_back(self):
        for current in (None,replace(self.current,complete=False)):
            self.assertEqual(self.select(current_prefix=current).selected_source_id,'A')

    def test_metadata_ties_preserve_original_deterministic_state_order(self):
        old={k:self.current for k in self.old}
        self.assertEqual(self.select(historical_prefixes=old).selected_source_id,'A')
        result=select_source((StateCandidate('Z',.1),StateCandidate('A',.1)),
            policy=self.policy,current_prefix=self.current,historical_prefixes={'Z':self.current,'A':self.current})
        self.assertEqual(result.selected_source_id,'Z')

    def test_no_compatible_or_admissible_candidate(self):
        self.assertIsNone(self.select(policy=replace(self.policy,tau=.01)).selected_source_id)
        result=select_source(tuple(replace(r,admission_eligible=False) for r in self.rows),policy=self.policy)
        self.assertIsNone(result.selected_source_id)
        result=select_source((self.rows[0],replace(self.rows[1],admission_eligible=False)),
            policy=self.policy,current_prefix=self.current,historical_prefixes=self.old)
        self.assertEqual(result.selected_source_id,'A')

    def test_configuration_and_scores_fail_closed(self):
        for kwargs in (dict(enabled=True),dict(enabled=True,tau=.1,delta=None),
                       dict(enabled=True,tau=float('inf'),delta=0),dict(delta=-1),dict(enabled=1)):
            with self.subTest(kwargs=kwargs),self.assertRaises(MetadataError): MetadataSelectionPolicy(**kwargs)
        for value in (float('nan'),float('inf'),-1,True):
            with self.assertRaises(MetadataError): StateCandidate('A',value)
        with self.assertRaises(MetadataError): select_source(self.rows[::-1],policy=self.policy)
        with self.assertRaises(MetadataError): select_source((self.rows[0],self.rows[0]),policy=self.policy)


if __name__=='__main__': unittest.main()
