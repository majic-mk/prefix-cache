"""Report semantics only; these tests do not qualify the named runtime bridges."""
import ast
from pathlib import Path
import unittest

from scripts.validate_decoupled_v2_p0 import CPU_BRIDGE_TEST_IDS, scoped_cpu_bridge_evidence


class ScopedCPUReportTests(unittest.TestCase):
    def records(self):
        return [dict(test_id=tid, observed='PASS')
                for ids in CPU_BRIDGE_TEST_IDS.values() for tid in ids]

    def test_explicit_selectors_resolve_to_real_test_methods(self):
        root = Path(__file__).resolve().parent
        for ids in CPU_BRIDGE_TEST_IDS.values():
            for tid in ids:
                module, cls, method = tid.split('.')
                tree = ast.parse((root / (module + '.py')).read_text(encoding='utf-8'))
                classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls]
                self.assertEqual(len(classes), 1, tid)
                self.assertIn(method, [node.name for node in classes[0].body if isinstance(node, ast.FunctionDef)], tid)

    def test_local_pass_never_upgrades_native_or_gpu_gates(self):
        result = scoped_cpu_bridge_evidence(self.records())
        self.assertTrue(all(row['local_code_implemented'] and row['cpu_evidence_passed']
                            for row in result['bridges'].values()))
        self.assertTrue(all(row['hardware_qualification'] == 'NOT_RUN'
                            for row in result['bridges'].values()))
        for name in ('native_runtime_qualified', 'gpu_execution_allowed',
                     'P1_E_execution_allowed', 'P1_M_execution_allowed'):
            self.assertIs(result[name], False)
        self.assertIn('production_v2_pool_dispatch_integrated', result['legacy_aggregate_flag_semantics'])

    def test_missing_result_is_not_passed(self):
        records = self.records()
        missing = records.pop(0)['test_id']
        result = scoped_cpu_bridge_evidence(records)['bridges']['target_capture_and_provenance_bridge']
        self.assertFalse(result['cpu_evidence_passed'])
        self.assertEqual(result['missing_or_nonpassing_test_results'][missing], [])

    def test_skip_fail_unknown_and_duplicate_are_not_passed(self):
        for state in ('SKIP', 'FAIL', None, 'PASSED'):
            records = self.records()
            records[0]['observed'] = state
            result = scoped_cpu_bridge_evidence(records)['bridges']['target_capture_and_provenance_bridge']
            self.assertFalse(result['cpu_evidence_passed'])
        records = self.records()
        records.append(dict(records[0]))
        result = scoped_cpu_bridge_evidence(records)['bridges']['target_capture_and_provenance_bridge']
        self.assertFalse(result['cpu_evidence_passed'])

    def test_empty_run_cannot_report_bridge_success(self):
        result = scoped_cpu_bridge_evidence([])
        self.assertTrue(all(not row['cpu_evidence_passed'] for row in result['bridges'].values()))
        self.assertEqual(result['test_record_file'], 'test_results.json')


if __name__ == '__main__':
    unittest.main()
