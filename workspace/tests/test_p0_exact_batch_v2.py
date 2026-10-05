"""Bounded exact-control batch routing; real CPU fake blocks, never CUDA evidence."""
from contextlib import contextmanager
from copy import deepcopy
import json
import time
import unittest
from unittest.mock import Mock, patch

from probekv.p0_batch_v2 import run_p0_native_batch, validate_p0_batch
from probekv.p0_evidence_v2 import read_p0_events
from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema10_storage import file_digest
from tests import test_p0_batch_v2 as batch_fixtures
from tests import test_p0_exact_control_v2 as control_fixtures


class P0ExactBatchTests(unittest.TestCase):
    def setUp(self):
        base = batch_fixtures.P0BatchTests()
        base.setUp()
        self.addCleanup(base.doCleanups)
        control = control_fixtures.P0ExactControlTests()
        self.addCleanup(control.doCleanups)
        self.store, self.c, self.model, job = control.context()
        # Exact diagnostic actions must not hold a natural-pool request
        # snapshot; the fixture starts one for its ordinary publication tests.
        for snapshot in tuple(self.store._snapshots.values()):
            self.store.end_request(snapshot)
        self.source_job = deepcopy(base.manifest['jobs'][0])
        self.manifest = deepcopy(base.manifest)
        job.update(action_id='exact-on', upper_seconds=30.,
                   input_origin='controlled_provenance_diagnostic')
        self.manifest['jobs'] = [job]
        self.manifest['exact_capture_pairs'] = []
        self.output = base.output
        self.manifest['binding'].update(initial_pool_sha256=file_digest(self.store.root/'catalog.json'),
            initial_registry_sha256=file_digest(self.store.registry._root/'registry.json'))
        self.sign()
        adapter = self.c.adapter
        adapter.active = None
        adapter.deadline = float('inf')
        adapter.reset = Mock()

        @contextmanager
        def open_request(request, *, arrival_ns):
            self.assertEqual(request, self.c.request)
            self.c.arrival_ns = arrival_ns
            try:
                yield self.c
            finally:
                self.c.close()
        adapter.open_request = open_request

    def sign(self):
        for job in self.manifest['jobs']:
            job['request_sha256'] = digest_json(job['request'])
        self.manifest['limits']['maximum_actions'] = len(self.manifest['jobs'])
        self.manifest['binding']['input_manifest_sha256'] = digest_json([
            dict(action_id=j['action_id'], request_sha256=j['request_sha256'], input_origin=j['input_origin'])
            for j in self.manifest['jobs']])
        batch_fixtures.sign(self.manifest)
        self.binding = deepcopy(self.manifest['binding'])

    def validate(self):
        return validate_p0_batch(self.manifest, actual_binding=self.binding, now_unix=time.time())

    def run_batch(self, session_started_ns=None):
        with patch('probekv.p0_batch_v2._native_origin', return_value='cpu_fixture'):
            return run_p0_native_batch(self.c.adapter, self.store, self.manifest,
                actual_binding=self.binding, output=self.output,
                session_started_ns=session_started_ns or time.perf_counter_ns())

    def test_control_validates_without_comparison_profile_or_sources(self):
        self.assertNotIn('comparison_profile', self.manifest['jobs'][0])
        self.assertNotIn('sources_by_segment', self.manifest['jobs'][0])
        report = self.validate()
        self.assertEqual(report['status'], 'INPUTS_VALIDATED')
        self.assertFalse(report['P1_execution_allowed'])
        self.assertEqual(self.model.real_block_calls, 0)

    def test_private_capture_cannot_supply_later_source_birth_reference(self):
        later = deepcopy(self.source_job)
        later['action_id'] = 'later'
        later['request']['request_id'] = 'later-request'
        later['sources_by_segment'] = {'C': [dict(birth_action_id='exact-on', segment_id='C')]}
        self.manifest['jobs'].append(later)
        self.sign()
        with self.assertRaisesRegex(ValueError, 'publication reference'):
            self.validate()
        self.assertEqual(self.model.real_block_calls, 0)

    def test_unknown_operation_is_not_defaulted_to_source_request(self):
        self.manifest['jobs'][0]['operation'] = 'unknown_control'
        self.sign()
        with self.assertRaisesRegex(ValueError, 'unsupported P0 operation'):
            self.validate()
        self.c.adapter.reset.assert_not_called()

    def test_registry_persistent_destination_is_rejected_at_preflight(self):
        self.manifest['registry_budget']['persistent_root'] = 'not-a-permitted-registry'
        self.sign()
        with self.assertRaisesRegex(ValueError, 'bounded shared manifest registry'):
            self.validate()
        self.c.adapter.reset.assert_not_called()

    def test_actual_control_batch_never_opens_pool_snapshot_or_publishes(self):
        before = self.store.storage_audit()
        with patch.object(self.store, 'begin_request', side_effect=AssertionError('no pool snapshot for diagnostic')), \
                patch.object(self.store, 'commit_publication', side_effect=AssertionError('diagnostic cannot publish')), \
                patch('probekv.p0_batch_v2.execute_p0_request', side_effect=AssertionError('not a Source request')):
            result = self.run_batch()
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertEqual(result['completed_action_ids'], ['exact-on'])
        self.assertEqual(result['exact_capture_pairs'], [])
        self.assertEqual(result['evidence_origin'], 'cpu_fixture')
        self.assertFalse(result['gpu_runtime_qualified'])
        self.assertFalse(result['P1_execution_allowed'])
        raw = json.loads((self.output/'exact-on/request.json').read_text())
        record = json.loads((self.output/'exact-on/record.json').read_text())
        self.assertEqual(raw['kind'], 'p0_exact_capture_control')
        self.assertEqual(raw['capture']['layers_recorded'], 3)
        self.assertEqual(raw['publication']['status'], 'DIAGNOSTIC_NOT_PUBLISHED')
        self.assertFalse(raw['native_runtime_qualified'])
        self.assertEqual(record['numerical_verdict'], 'NOT_EVALUATED')
        self.assertEqual(self.model.real_block_calls, 3)
        self.assertEqual([b.self_attn.qkv_proj.calls for b in self.model.layers], [1, 1, 1])
        self.assertEqual(self.store.storage_audit(), before)
        self.assertEqual(self.store.registry.storage_audit()['manifest_count'], 0)
        self.assertFalse(self.c.target_candidates_v2)
        self.assertEqual(self.c.adapter.hbm.active_reserved_bytes, 0)

    def test_control_budget_stop_happens_before_reset_or_model_work(self):
        with patch('probekv.p0_batch_v2.time.perf_counter_ns', return_value=4000_000_000_000):
            result = self.run_batch(1)
        self.assertEqual(result['status'], 'BUDGET_STOP')
        self.assertEqual(result['pending_action_ids'], ['exact-on'])
        self.assertEqual(result['exact_capture_pairs'], [])
        self.c.adapter.reset.assert_not_called()
        self.assertEqual(self.model.real_block_calls, 0)
        self.assertFalse(self.c.closed)

    def test_pair_failure_is_preserved_separately_from_completed_action(self):
        # This mock exercises aggregation only; no numerical result is claimed.
        with patch('probekv.p0_batch_v2.evaluate_exact_pairs',
                   return_value=[dict(pair_id='cpu-aggregation-fixture', status='FAILED')]):
            result = self.run_batch()
        self.assertEqual(result['status'], 'NUMERICAL_PAIR_FAILED')
        self.assertEqual(result['completed_action_ids'], ['exact-on'])
        self.assertEqual(result['failed_action_ids'], [])
        self.assertFalse(result['P1_execution_allowed'])
        events = read_p0_events(self.output/'actions.jsonl', binding=self.binding)
        self.assertEqual(next(e['payload']['status'] for e in events
                              if e['kind'] == 'exact_capture_pair_evaluated'), 'FAILED')

    def test_capture_failure_retains_failed_action_without_publication(self):
        self.manifest['limits']['host_capture_bytes'] = 1
        self.sign()
        result = self.run_batch()
        self.assertEqual(result['status'], 'FAILED')
        self.assertEqual(result['failed_action_ids'], ['exact-on'])
        self.assertEqual(result['completed_action_ids'], [])
        events = read_p0_events(self.output/'actions.jsonl', binding=self.binding)
        partial = next(e['payload']['partial_request_audit'] for e in events if e['kind'] == 'action_failed')
        self.assertIsNotNone(partial['answer'])
        self.assertTrue(partial['cleanup']['passed'])
        self.assertEqual(self.model.real_block_calls, 3)
        self.assertEqual(self.store.storage_audit()['source_count'], 0)


if __name__ == '__main__':
    unittest.main()
