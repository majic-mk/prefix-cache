import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from probekv.p0_partition_overlap_v2 import audit_partition_overlap, audit_partition_overlap_files


class P0PartitionOverlapTests(unittest.TestCase):
    def setUp(self):
        self.partition = 'a' * 64
        self.old = {'case_id': 'birth:old-c000', 'group_id': 'HotPotQA:old',
                    'source_dataset': 'hotpotqa', 'source_split': 'calibration',
                    'partition_role': 'development_profile_freeze', 'locked_test_accessed': False}
        self.snapshot = {'kind': 'readonly_server_metadata_audit', 'locked_test_accessed': False,
                         'raw_dataset_rows_read': False, 'mixed_split_cases_read': False,
                         'old_development_partition': {'sha256': self.partition,
                                                       'path': '/metadata/development.jsonl', 'rows': [self.old]}}
        self.row = {'case_id': 'new:case', 'group_id': 'HotPotQA:new', 'dataset': 'HotPotQA',
                    'source_origin_ids': ['birth'], 'source_ids_are_origin_ids_not_artifact_ids': True,
                    'targets': [{'origin_example_id': 'future', 'support_stratum': 'full_support_sentence'}],
                    'known_previously_observed': False, 'input_sha256': {'partition': self.partition}}

    def audit(self, rows=None):
        return audit_partition_overlap(self.snapshot, [self.row] if rows is None else rows,
                                       metadata_snapshot_sha256='b' * 64, census_sha256='c' * 64)

    def test_origin_overlap_is_found_even_with_different_content_group(self):
        result = self.audit()
        self.assertEqual(result['direct_group_overlap_count'], 0)
        self.assertEqual(result['direct_origin_overlap_count'], 1)
        self.assertEqual(result['annotations'][0]['direct_old_partition_origin_ids'], ['birth'])
        self.assertFalse(result['P1_execution_allowed'])
        self.assertEqual(result['qualified_target_count'], 0)

    def test_dependency_bridge_propagates_exposure_without_claiming_complete_lineage(self):
        second = copy.deepcopy(self.row)
        second.update(case_id='second:case', group_id='HotPotQA:second', source_origin_ids=['other'],
                      known_dependency_component='HotPotQA:new')
        result = self.audit([self.row, second])
        self.assertEqual(result['known_exposure_census_rows'], 2)
        self.assertEqual(result['direct_origin_overlap_count'], 1)
        self.assertFalse(result['annotations'][1]['full_lineage_verified'])

    def test_explicit_ancestry_origin_and_group_edges_are_respected(self):
        self.row['source_origin_ids'] = ['clean']
        self.row['lineage_origin_ids'] = ['birth']
        self.assertTrue(self.audit()['annotations'][0]['known_exposure_detected'])
        self.row['lineage_origin_ids'] = []
        self.row['lineage_group_ids'] = ['HotPotQA:old']
        self.assertTrue(self.audit()['annotations'][0]['known_exposure_detected'])

    def test_no_known_edge_is_not_freshness_and_prior_observed_census_stays_exposed(self):
        self.row['source_origin_ids'] = ['other']
        result = self.audit()
        self.assertEqual(result['annotations'][0]['status'], 'UNKNOWN_NOT_PROVEN_FRESH')
        self.assertFalse(result['annotations'][0]['freshness_established'])
        self.assertEqual(result['support_pairs_without_known_exposure'], 1)
        self.row['known_previously_observed'] = True
        self.assertEqual(self.audit()['support_pairs_without_known_exposure'], 0)

    def test_duplicate_partition_group_not_counted_as_independent_and_inputs_unchanged(self):
        other = dict(self.old, case_id='birth:old-c001')
        self.snapshot['old_development_partition']['rows'].append(other)
        before = copy.deepcopy((self.snapshot, self.row))
        result = self.audit()
        self.assertEqual(result['partition_rows'], 2)
        self.assertEqual(result['partition_unique_groups'], 1)
        self.assertEqual(result['partition_unique_case_origins'], 1)
        self.assertEqual(before, (self.snapshot, self.row))
        self.assertFalse(result['old_partitions_modified'])

    def test_different_dataset_origin_does_not_create_false_link(self):
        self.row.update(dataset='2WikiMultiHopQA', group_id='2WikiMultiHopQA:new')
        self.assertFalse(self.audit()['annotations'][0]['known_exposure_detected'])

    def test_rejects_partition_mismatch_locked_rows_or_untyped_origin_ids(self):
        for mutation in ('hash', 'locked', 'artifact', 'namespace'):
            with self.subTest(mutation=mutation):
                self.setUp()
                if mutation == 'hash': self.row['input_sha256']['partition'] = 'd' * 64
                if mutation == 'locked': self.old['source_split'] = 'test'
                if mutation == 'artifact': self.row['source_ids_are_origin_ids_not_artifact_ids'] = False
                if mutation == 'namespace': self.row['group_id'] = 'MuSiQue:new'
                with self.assertRaises(ValueError): self.audit()

    def test_file_reader_binds_digest_and_refuses_modified_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            meta = json.dumps(self.snapshot).encode()
            census = (json.dumps(self.row) + '\n').encode()
            (root / 'meta.json').write_bytes(meta)
            (root / 'census.jsonl').write_bytes(census)
            args = dict(expected_metadata_snapshot_sha256=hashlib.sha256(meta).hexdigest(),
                        expected_census_sha256=hashlib.sha256(census).hexdigest())
            result = audit_partition_overlap_files(root / 'meta.json', root / 'census.jsonl', **args)
            self.assertEqual(result['binding']['declared_original_partition_sha256'], self.partition)
            (root / 'meta.json').write_bytes(meta + b' ')
            with self.assertRaises(ValueError):
                audit_partition_overlap_files(root / 'meta.json', root / 'census.jsonl', **args)


if __name__ == '__main__':
    unittest.main()
