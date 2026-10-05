"""Tamper/rejection tests with CPU-generated files; never GPU evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from probekv.p0_acceptance_v2 import verify_batch, assess_numerical_correctness, file_sha, verify_lineage_scope
from probekv.p0_evidence_v2 import P0EvidenceWriter
from probekv.v8_schema10_execution import digest_json


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / 'batch'
        self.binding = dict(model_signature='fixture-model', tokenizer_hash='fixture-tok', patch_sha256='fixture-patch')

    def fixture(self, *, origin='cpu_fixture', cleanup=True, recorded=True, failed=False):
        manifest = dict(binding=self.binding, jobs=[dict(action_id='a')])
        manifest['manifest_sha256'] = digest_json(manifest)
        writer = P0EvidenceWriter(self.root, binding=self.binding, manifest=manifest)
        if recorded:
            writer.write_action('a', audit=dict(status='COMPLETED', cleanup=dict(passed=cleanup)),
                                logits=[], origin=origin)
        writer.finalize(dict(status='FAILED' if failed else 'COMPLETED',
            completed_action_ids=['a'] if not failed else [],
            failed_action_ids=['a'] if failed else [], pending_action_ids=[]))
        return dict(root=str(self.root), manifest_sha256=file_sha(self.root/'manifest.json'), binding=self.binding)

    def verify(self, spec):
        return verify_batch(spec['root'], manifest_sha256=spec['manifest_sha256'], expected_binding=spec['binding'])

    def test_cpu_fixture_cannot_qualify_even_with_completed_result(self):
        with self.assertRaisesRegex(ValueError, 'CPU/simulated'):
            self.verify(self.fixture())

    def test_record_chain_validation_does_not_itself_supply_numerical_pairs(self):
        spec=self.fixture(origin='real_cuda_execution')  # simulated metadata for parser test only
        before={p.name:file_sha(p) for p in self.root.iterdir() if p.is_file()}
        batch=self.verify(spec)
        self.assertEqual(batch['numerical_pairs'], [])
        self.assertEqual(before,{p.name:file_sha(p) for p in self.root.iterdir() if p.is_file()})
        result=assess_numerical_correctness([spec])
        self.assertEqual(result['status'],'BLOCKED')
        self.assertFalse(result['full_P0_complete'])
        self.assertFalse(result['P1_execution_allowed'])

    def test_manifest_byte_tamper_rejected(self):
        spec=self.fixture(origin='real_cuda_execution')
        with (self.root/'manifest.json').open('a') as f:f.write(' ')
        with self.assertRaisesRegex(ValueError,'manifest bytes'):self.verify(spec)

    def test_wrong_expected_binding_rejected(self):
        spec=self.fixture(origin='real_cuda_execution');spec['binding']={'model_signature':'other'}
        with self.assertRaisesRegex(ValueError,'binding'):self.verify(spec)

    def test_request_tamper_rejected(self):
        spec=self.fixture(origin='real_cuda_execution')
        (self.root/'a/request.json').write_text('{"passed":true}')
        with self.assertRaisesRegex(ValueError,'digest'):self.verify(spec)

    def test_missing_action_record_is_not_completion(self):
        spec=self.fixture(origin='real_cuda_execution',recorded=False)
        with self.assertRaisesRegex(ValueError,'completeness'):self.verify(spec)

    def test_cleanup_failure_rejected(self):
        with self.assertRaisesRegex(ValueError,'cleanup'):
            self.verify(self.fixture(origin='real_cuda_execution',cleanup=False))

    def test_failed_batch_rejected(self):
        with self.assertRaisesRegex(ValueError,'incomplete'):
            self.verify(self.fixture(origin='real_cuda_execution',failed=True))

    def test_result_cannot_hide_event_corruption(self):
        spec=self.fixture(origin='real_cuda_execution')
        with (self.root/'actions.jsonl').open('a') as f:f.write('{}\n')
        with self.assertRaisesRegex(ValueError,'event file'):self.verify(spec)

    def test_weak_policy_cannot_qualify_even_with_pair_passed(self):
        spec=self.fixture(origin='real_cuda_execution')
        pair=dict(status='PASSED',test_id='T20',minimum_positions=31,relative_l2_limit=1e-4,
                  require_predicted_token_ids_equal=True)
        with patch('probekv.p0_acceptance_v2.evaluate_exact_pairs',return_value=[pair]):
            result=assess_numerical_correctness([spec])
        self.assertFalse(result['scoped_numerical_correctness_verified'])
        self.assertIn('weakens',str(result['blockers']))

    def test_numerical_pass_never_grants_p1_or_full_p0(self):
        spec=self.fixture(origin='real_cuda_execution')
        pairs=[dict(status='PASSED',test_id=tid,minimum_positions=32,relative_l2_limit=1e-4,
                    require_predicted_token_ids_equal=True,target_relative_l2_limit=1e-4,
                    target_absolute_max_limit=1e-3,real_cuda_target_r1_passed=tid=='T31',
                    real_cuda_T21_passed=tid=='T21') for tid in ('T20','T21','T31')]
        with patch('probekv.p0_acceptance_v2.evaluate_exact_pairs',return_value=pairs[:1]), \
             patch('probekv.p0_acceptance_v2.evaluate_mixed_pairs',return_value=pairs[1:]):
            result=assess_numerical_correctness([spec])
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['counts'],dict(exact_pairs=1,mixed_pairs=2,mixed_r1_pairs=1))
        self.assertFalse(result['full_P0_complete']); self.assertFalse(result['P1_execution_allowed'])
        self.assertFalse(result['GPU_execution_allowed'])
        pairs[-1]['real_cuda_target_r1_passed']=False
        pairs[-1]['real_cuda_upstream_r0_passed']=True
        with patch('probekv.p0_acceptance_v2.evaluate_exact_pairs',return_value=pairs[:1]), \
             patch('probekv.p0_acceptance_v2.evaluate_mixed_pairs',return_value=pairs[1:]):
            result=assess_numerical_correctness([spec])
        self.assertEqual(result['counts']['mixed_r1_pairs'],0)
        self.assertIn('MIXED_R1_RAW_NUMERICAL_PAIR_MISSING',str(result['blockers']))
        pairs[1]['real_cuda_T21_passed']=False
        pairs[-1]['real_cuda_target_r1_passed']=True
        with patch('probekv.p0_acceptance_v2.evaluate_exact_pairs',return_value=pairs[:1]), \
             patch('probekv.p0_acceptance_v2.evaluate_mixed_pairs',return_value=pairs[1:]):
            result=assess_numerical_correctness([spec])
        self.assertIn('MIXED_RAW_NUMERICAL_PAIR_MISSING',str(result['blockers']))

    def test_legacy_single_target_and_explicit_g2_rejection(self):
        from tests.test_p0_lineage_control_v2 import job
        j=job();j['action_id']='a'
        audit=dict(controlled_recipe_commit_observed=True,economic_admission_evaluated=False,
                   production_reuse_commit_observed=False,birth_snapshot_sees_child=False,
                   target_proof=dict(generation=1,origin='MIXED_CONTEXT_FULL_SEGMENT',positions=list(range(11,21))),
                   publication=dict(targets={'T':dict(publication_performed=True)}))
        batch=dict(manifest=dict(jobs=[j]),audits={'a':audit})
        self.assertEqual(verify_lineage_scope(batch)[0]['expected_generation'],{'T':1})
        del audit['birth_snapshot_sees_child']
        with self.assertRaises(KeyError):verify_lineage_scope(batch)
        j['expected_parent_generation']=1;j['expected_target_generation']=2
        audit['target_proof']['generation']=2
        audit['publication']['targets']['T']['publication_performed']=False
        audit['g2_rejection_reason']='generation exceeds main pool limit'
        self.assertEqual(verify_lineage_scope(batch)[0]['expected_generation'],{'T':2})
        del audit['g2_rejection_reason']
        with self.assertRaisesRegex(ValueError,'rejection'):verify_lineage_scope(batch)


if __name__=='__main__':unittest.main()
