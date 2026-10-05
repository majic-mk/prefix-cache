"""Fault-injected CPU/disk publication checks, not runtime/GPU evidence."""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from tests import test_source_store_v2 as fixtures


class TargetSourceStoreFaultTests(unittest.TestCase):
    # Reuse only fixture helpers, not the base TestCase's test methods.
    setUp = fixtures.TargetSourceStoreTests.setUp
    close_stores = fixtures.TargetSourceStoreTests.close_stores
    store = fixtures.TargetSourceStoreTests.store
    candidate = fixtures.TargetSourceStoreTests.candidate
    plan = fixtures.TargetSourceStoreTests.plan
    publish = fixtures.TargetSourceStoreTests.publish
    read_rows = fixtures.TargetSourceStoreTests.read_rows

    def test_replace_completed_then_error_preserves_new_backing_for_recovery(self):
        store = self.store()
        candidate = self.candidate('uncertain-birth')
        expected = candidate['layers'][0][0].clone()
        snapshot = store.begin_request('uncertain-birth')
        plan = self.plan(store, snapshot, candidate)
        real_replace = os.replace

        def completed_replace_then_error(source, destination):
            real_replace(source, destination)
            if Path(destination) == store.root / 'catalog.json':
                raise OSError('simulated reported error after successful replace')

        with patch('probekv.source_store_v2.os.replace', side_effect=completed_replace_then_error):
            result = store.commit_publication(snapshot, plan, candidate, token_ids=self.target_tokens)
        self.assertEqual(result['status'], 'COMMIT_UNCERTAIN')
        self.assertIsNone(result['publication_performed'])
        self.assertTrue(result['recovery_required'])
        self.assertIsNone(store.events[-1]['publication_performed'])
        with self.assertRaises(RuntimeError):
            store.lookup(snapshot, self.target_tokens)
        payload = json.loads((store.root / 'catalog.json').read_text())['payload']
        row = payload['rows'][candidate['source_artifact_id']]
        self.assertTrue((store.root / row['kv_file']).is_file())
        self.assertTrue((store.root / row['selection_file']).is_file())
        store.close()
        recovered = self.store()
        reader = recovered.begin_request('after-recovery')
        self.assertEqual(len(recovered.lookup(reader, self.target_tokens)), 1)
        with recovered.leased_target(reader, row['source_id']) as layers:
            self.assertTrue(torch.equal(layers[0][0], expected))
        recovered.end_request(reader)
        self.assertFalse(recovered.storage_audit()['recovery_required'])

    def test_replace_failed_before_commit_preserves_old_catalog_and_removes_stage(self):
        store = self.store()
        candidate = self.candidate('failed-birth')
        snapshot = store.begin_request('failed-birth')
        plan = self.plan(store, snapshot, candidate)
        before = (store.root / 'catalog.json').read_bytes()
        with patch('probekv.source_store_v2.os.replace', side_effect=OSError('before replace')):
            result = store.commit_publication(snapshot, plan, candidate, token_ids=self.target_tokens)
        self.assertEqual(result['status'], 'REJECTED')
        self.assertFalse(result['publication_performed'])
        self.assertEqual((store.root / 'catalog.json').read_bytes(), before)
        self.assertEqual(store.storage_audit()['source_count'], 0)
        self.assertEqual(store.storage_audit()['quarantine_bytes'], 0)
        self.assertFalse(store.storage_audit()['recovery_required'])
        store.end_request(snapshot)

    def test_selection_state_stage_failure_removes_already_written_kv(self):
        store = self.store()
        candidate = self.candidate('second-stage-failure')
        snapshot = store.begin_request('second-stage-failure')
        plan = self.plan(store, snapshot, candidate)
        before = (store.root / 'catalog.json').read_bytes()
        real_stage = store._stage
        calls = []

        def fail_second_stage(suffix, writer, created):
            calls.append(suffix)
            if suffix == '.states':
                self.assertEqual(len(tuple(store.root.glob('*.kv'))), 1)
                raise OSError('SelectionState write failed after target KV staging')
            return real_stage(suffix, writer, created)

        with patch.object(store, '_stage', side_effect=fail_second_stage):
            result = store.commit_publication(snapshot, plan, candidate, token_ids=self.target_tokens)
        self.assertEqual(calls, ['.kv', '.states'])
        self.assertEqual(result['status'], 'REJECTED')
        self.assertFalse(result['publication_performed'])
        self.assertEqual((store.root / 'catalog.json').read_bytes(), before)
        self.assertEqual(store.storage_audit()['source_count'], 0)
        self.assertEqual(store.storage_audit()['quarantine_bytes'], 0)
        self.assertFalse(tuple(store.root.glob('*.kv')))
        self.assertFalse(tuple(store.root.glob('*.states')))
        store.end_request(snapshot)

    def test_victim_unlink_failure_is_explicit_and_quarantine_bytes_are_counted(self):
        store = self.store(max_variants=1)
        first = self.publish(store, self.candidate('first'))
        old = self.read_rows(store)[0]
        old_paths = {store.root / old[key] for key in ('kv_file', 'selection_file')}
        old_bytes = sum(path.stat().st_size for path in old_paths)
        candidate = self.candidate('replacement', base=50)
        snapshot = store.begin_request('replacement')
        plan = self.plan(store, snapshot, candidate)
        real_unlink = Path.unlink

        def blocked_old_unlink(path, *args, **kwargs):
            if path in old_paths:
                raise PermissionError('simulated delayed victim cleanup')
            return real_unlink(path, *args, **kwargs)

        with patch.object(Path, 'unlink', blocked_old_unlink):
            result = store.commit_publication(snapshot, plan, candidate, token_ids=self.target_tokens)
        self.assertEqual(result['status'], 'PUBLISHED_CLEANUP_PENDING')
        self.assertTrue(result['publication_performed'])
        self.assertEqual(result['evicted_source_id'], first['source_id'])
        self.assertEqual(set(result['cleanup_pending']), {path.name for path in old_paths})
        audit = store.storage_audit()
        self.assertEqual(audit['source_count'], 1)
        self.assertEqual(audit['quarantine_bytes'], old_bytes)
        self.assertTrue(all(path.is_file() for path in old_paths))
        store.end_request(snapshot)
        store.close()
        recovered = self.store(max_variants=1)
        self.assertEqual(recovered.storage_audit()['quarantine_bytes'], old_bytes)
        self.assertEqual(self.read_rows(recovered)[0]['source_id'], result['source_id'])

    def test_birth_accounting_does_not_consume_subsequent_probation_grace(self):
        store = self.store(probation_opportunities=2)
        candidate = self.candidate('birth')
        snapshot = store.begin_request('birth')
        plan = self.plan(store, snapshot, candidate)
        result = store.commit_publication(snapshot, plan, candidate, token_ids=self.target_tokens)
        sid = result['source_id']
        store.record_lookup_opportunity(snapshot, self.target_tokens)
        row = store._catalog['rows'][sid]
        key = row['content_key']
        self.assertEqual(row['grace_until'] - store._catalog['content_opportunities'][key], 2)
        store.record_lookup_opportunity(snapshot, self.target_tokens)
        self.assertEqual(row['grace_until'] - store._catalog['content_opportunities'][key], 2)
        store.end_request(snapshot)
        self.assertFalse(store._lookup_markers)
        for request, expected_protection in (('future-one', True), ('future-two', False)):
            reader = store.begin_request(request)
            store.record_lookup_opportunity(reader, self.target_tokens)
            self.assertEqual(store._protected(store._catalog['rows'][sid]), expected_protection)
            store.end_request(reader)
            self.assertFalse(store._lookup_markers)

    def test_writer_lock_rejects_second_store_without_changing_first(self):
        store = self.store()
        before = (store.root / 'catalog.json').read_bytes()
        with self.assertRaises(OSError):
            self.store()
        self.assertEqual((store.root / 'catalog.json').read_bytes(), before)
        snapshot = store.begin_request('still-usable')
        store.end_request(snapshot)
        store.close()
        reopened = self.store()
        self.assertEqual(reopened.storage_audit()['source_count'], 0)

    def test_external_catalog_change_is_fail_closed(self):
        store = self.store()
        snapshot = store.begin_request('reader')
        path = store.root / 'catalog.json'
        before = path.read_bytes()
        path.write_bytes(before + b'\n')
        with self.assertRaisesRegex(ValueError, 'catalog changed'):
            store.lookup(snapshot, self.target_tokens)
        self.assertEqual(path.read_bytes(), before + b'\n')

    def test_initial_persistent_budget_includes_empty_catalog_and_registry(self):
        with self.assertRaises(MemoryError):
            self.store(max_bytes=1)
        self.assertFalse((self.root / 'pool' / 'catalog.json').exists())
        self.assertFalse(tuple((self.root / 'pool').glob('*.tmp')))

    def test_lookup_catalog_growth_checks_budget_and_rolls_back(self):
        store = self.store(max_bytes=1800)
        snapshot = store.begin_request('many-content-lookups')
        failed = False
        for i in range(100):
            before = (store.root / 'catalog.json').read_bytes()
            old_opportunities = dict(store._catalog['content_opportunities'])
            try:
                store.record_lookup_opportunity(snapshot, (1000 + i,))
            except MemoryError:
                failed = True
                self.assertEqual((store.root / 'catalog.json').read_bytes(), before)
                self.assertEqual(store._catalog['content_opportunities'], old_opportunities)
                self.assertTrue(store.storage_audit()['persistent_budget_satisfied'])
                break
        self.assertTrue(failed, 'bounded catalog must stop growth before exhausting memory')
        store.end_request(snapshot)

    def test_bad_layer_header_does_not_acquire_a_leaked_lease(self):
        store = self.store()
        result = self.publish(store, self.candidate('birth'))
        row = self.read_rows(store)[0]
        (store.root / row['kv_file']).write_bytes(b'bad-layer-header')
        snapshot = store.begin_request('reader')
        with self.assertRaises(ValueError):
            with store.leased_target(snapshot, result['source_id']):
                self.fail('damaged layer file cannot be yielded')
        self.assertFalse(any(store._lease_counts.values()))
        store.end_request(snapshot)

    def test_recovery_rejects_over_budget_quarantine_before_loading_tensors(self):
        store = self.store()
        self.publish(store, self.candidate('birth'))
        root = store.root
        limit = store.config['max_bytes']
        store.close()
        orphan = root / 'unpublished-quarantine.kv'
        orphan.write_bytes(b'X' * limit)
        with patch('probekv.source_store_v2.LayerFile',
                   side_effect=AssertionError('over-budget restore must not load target tensors')) as layer_file:
            with self.assertRaises(MemoryError):
                self.store()
        layer_file.assert_not_called()
        self.assertTrue(orphan.is_file())
        self.assertEqual(orphan.stat().st_size, limit)


if __name__ == '__main__':
    unittest.main()
