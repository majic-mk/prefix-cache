"""Abstract policy tests, not ProbeKV engine integration tests."""
import unittest
from dataclasses import replace
from provenance_reference import (
    Origin, DependencySpan, LayerLedger, Admission, imported_rank,
    inherited_rank, prefix_rank, full_target, classify_target, decide, visible,
)


def ledger():
    return tuple(LayerLedger(i, 8, 8, 8, 8) for i in range(1, 4))


def candidate(**kwargs):
    a = Admission('content_miss', 0, 0, 0, None, None, Origin.EXACT, 0)
    return replace(a, **kwargs)


class PlanRulesTest(unittest.TestCase):
    def test_exact_target(self):
        self.assertEqual(classify_target(ledger(), 3, [0, 0]), (Origin.EXACT, 0))

    def test_mixed_target_allowed(self):
        g = imported_rank(0)
        origin, generation = classify_target(ledger(), 3, [0, g])
        self.assertEqual((origin, generation), (Origin.MIXED, 1))
        self.assertTrue(decide(candidate(provenance=origin, generation=generation))[0])

    def test_nine_approximate_predecessors(self):
        deps = [DependencySpan(i*8, (i+1)*8, imported_rank(0)) for i in range(9)]
        self.assertEqual(prefix_rank(72, deps), 1)

    def test_same_request_full_chain_does_not_increment(self):
        b = inherited_rank([imported_rank(0)])
        c = inherited_rank([b, 0])
        self.assertEqual((b, c), (1, 1))

    def test_cross_request_second_generation_not_published(self):
        g = inherited_rank([imported_rank(1), 0])
        self.assertEqual(g, 2)
        self.assertEqual(decide(candidate(provenance=Origin.MIXED, generation=g))[1], 'generation_limit')

    def test_indirect_full_segment_cannot_wash_rank(self):
        middle_full = inherited_rank([imported_rank(1)])
        target_full = inherited_rank([middle_full])
        self.assertEqual(target_full, 2)

    def test_unknown_remains_unknown(self):
        self.assertEqual(classify_target(ledger(), 3, [None, 0]), (Origin.UNKNOWN, None))
        self.assertFalse(decide(candidate(provenance=Origin.UNKNOWN, generation=None))[0])

    def test_suffix_does_not_taint(self):
        self.assertEqual(prefix_rank(8, [DependencySpan(0, 8, 0), DependencySpan(16, 24, 2)]), 0)

    def test_overlap_requires_split(self):
        with self.assertRaises(ValueError):
            prefix_rank(8, [DependencySpan(0, 10, 0)])

    def test_partial_then_r1_not_full(self):
        rows = list(ledger())
        rows[0] = LayerLedger(1, 8, 8, 4, 4, 4)
        self.assertFalse(full_target(rows, 3))
        self.assertEqual(classify_target(rows, 3, [0])[0], Origin.PARTIAL)

    def test_missing_or_duplicate_layer_not_full(self):
        self.assertFalse(full_target(ledger()[1:], 3))
        self.assertFalse(full_target((ledger()[0], ledger()[0], ledger()[2]), 3))

    def test_all_rows_full_r1_can_be_eligible(self):
        self.assertTrue(full_target(ledger(), 3))

    def test_exact_prefix_proof(self):
        self.assertEqual(imported_rank(0, exact_prefix_proof=True), 0)
        with self.assertRaises(ValueError):
            imported_rank(1, exact_prefix_proof=True)

    def test_exact_only_rejects_mixed(self):
        self.assertFalse(decide(candidate(provenance=Origin.MIXED, generation=1, max_generation=0))[0])

    def test_corrupt_origin_rejected(self):
        self.assertEqual(decide(candidate(generation=1))[1], 'inconsistent_origin_and_generation')

    def test_nan_inf_rejected(self):
        for v in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(v=v):
                self.assertFalse(decide(candidate(best_score=v))[0])
                self.assertFalse(decide(candidate(add_threshold=v))[0])

    def test_pool_miss_cannot_hide_existing_object(self):
        self.assertFalse(decide(candidate(stored_count=1))[0])

    def test_complete_mismatch(self):
        a = candidate(reason='complete_scope_mismatch', stored_count=4,
                      eligible_count=4, compared_count=4, best_score=0.3, add_threshold=0.2)
        self.assertTrue(decide(a)[0])
        self.assertFalse(decide(replace(a, compared_count=2))[0])
        self.assertFalse(decide(replace(a, best_score=0.1))[0])
        self.assertFalse(decide(replace(a, eligible_count=2, compared_count=2))[0])

    def test_incomplete_capture_or_budget_rejected(self):
        for k in ('budget_ok','capacity_ok','artifact_complete','early_layers_captured','permission_ok','snapshot_current'):
            with self.subTest(k=k):
                self.assertFalse(decide(candidate(**{k:False}))[0])

    def test_duplicate_rejected(self):
        self.assertFalse(decide(candidate(duplicate=True))[0])

    def test_failure_not_novel_context(self):
        for reason in ('runtime_failure','economic_rejection','margin_failure','question_changed'):
            self.assertFalse(decide(candidate(reason=reason))[0])

    def test_birth_request_cannot_read_own_new_source(self):
        self.assertFalse(visible(birth_request_id='r1', publication_epoch=2,
                                 reading_request_id='r1', reader_snapshot_epoch=3, completed_publication=True))

    def test_future_and_partial_publication_invisible(self):
        self.assertFalse(visible(birth_request_id='r1', publication_epoch=2,
                                 reading_request_id='r2', reader_snapshot_epoch=1, completed_publication=True))
        self.assertFalse(visible(birth_request_id='r1', publication_epoch=2,
                                 reading_request_id='r2', reader_snapshot_epoch=3, completed_publication=False))
        self.assertTrue(visible(birth_request_id='r1', publication_epoch=2,
                                reading_request_id='r2', reader_snapshot_epoch=3, completed_publication=True))

    def test_invalid_counts_and_geometry_fail_closed(self):
        with self.assertRaises(ValueError):
            imported_rank(-1)
        with self.assertRaises(ValueError):
            LayerLedger(1, 8, 9, 8, 8)
        self.assertFalse(decide(candidate(stored_count=1, eligible_count=1, compared_count=2))[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
