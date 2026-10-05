"""Execute the CPU-only readiness CLI, without mocked authority or model data."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


class StageReadinessCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = Path(__file__).resolve().parents[1]
        self.entry = self.workspace / 'scripts/check_decoupled_v2_stage_readiness.py'
        self.evidence = self.root / 'evidence.json'; self.evidence.write_text('{}', encoding='utf-8')
        self.output = self.root / 'result'

    def run_cli(self, *, stage='CONTROLLED_P0_E', extra=()):
        environment = dict(os.environ)
        environment['PYTHONPATH'] = os.pathsep.join((str(self.workspace / 'src'), str(self.workspace)))
        environment['CUDA_VISIBLE_DEVICES'] = ''
        environment['PYTHONIOENCODING'] = 'utf-8'
        return subprocess.run([sys.executable, str(self.entry), '--stage', stage,
            '--workspace', str(self.workspace), '--evidence', str(self.evidence),
            '--output', str(self.output), *extra], cwd=str(self.workspace), env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', timeout=30)

    def test_empty_refs_write_specific_BLOCKED_report_with_real_clock(self):
        before = self.evidence.read_bytes(); started = time.time()
        process = self.run_cli()
        self.assertEqual(process.returncode, 2, process.stderr)
        report = json.loads((self.output / 'stage_readiness.json').read_text(encoding='utf-8'))
        self.assertEqual(report['status'], 'BLOCKED')
        self.assertFalse(report['requires_natural_p1_cohort'])
        self.assertFalse(report['gpu_execution_allowed'])
        self.assertFalse(report['automatic_rental_allowed'])
        self.assertTrue(report['blockers'])
        self.assertLessEqual(started, report['inspection']['started_at_unix'])
        self.assertLessEqual(report['inspection']['completed_at_unix'], time.time())
        self.assertEqual(report['inspection']['evidence_sha256'], hashlib.sha256(before).hexdigest())
        self.assertFalse(report['inspection']['model_loaded'])
        self.assertFalse(report['inspection']['network_access_performed'])
        self.assertEqual(self.evidence.read_bytes(), before)

    def test_unknown_stage_rejects_before_output_creation(self):
        process = self.run_cli(stage='P1')
        self.assertEqual(process.returncode, 2)
        self.assertIn('invalid choice', process.stderr)
        self.assertFalse(self.output.exists())

    def test_second_execution_never_overwrites_existing_result(self):
        self.assertEqual(self.run_cli().returncode, 2)
        path = self.output / 'stage_readiness.json'; original = path.read_bytes()
        self.evidence.write_text('{"new": "value"}', encoding='utf-8')
        process = self.run_cli()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('fresh output directory', process.stderr)
        self.assertEqual(path.read_bytes(), original)

    def test_existing_empty_output_directory_also_rejected(self):
        self.output.mkdir()
        process = self.run_cli()
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_no_clock_override_or_execution_flag(self):
        for extra in (('--now-unix', '1'), ('--execute',), ('--rent',)):
            process = self.run_cli(extra=extra)
            self.assertEqual(process.returncode, 2)
            self.assertFalse(self.output.exists())

    def test_missing_or_malformed_input_is_not_replaced_with_template(self):
        self.evidence.unlink()
        process = self.run_cli()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(self.output.exists())
        self.evidence.write_text('[true]', encoding='utf-8')
        process = self.run_cli()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_p1_requires_own_data_without_promoting_controlled_diagnostics(self):
        process = self.run_cli(stage='P1_M')
        self.assertEqual(process.returncode, 2, process.stderr)
        report = json.loads((self.output / 'stage_readiness.json').read_text(encoding='utf-8'))
        self.assertTrue(report['requires_natural_p1_cohort'])
        self.assertFalse(report['P1_M_execution_allowed'])
        self.assertIn('PAIRED_M1_BIRTH_RECIPE', [b['code'] for b in report['blockers']])


if __name__ == '__main__':
    unittest.main()
