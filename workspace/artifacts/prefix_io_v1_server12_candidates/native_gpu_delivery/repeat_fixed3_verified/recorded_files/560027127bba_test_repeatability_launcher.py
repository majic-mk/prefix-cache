"""CPU-only rejection and pure disposition checks, never native qualification.

Protocol bytes are real preregistration. Synthetic dictionaries used for pure
decision tests do not issue a receipt, create authority, or execute a launch.
"""
import ast
from copy import deepcopy
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ATTEMPTS = []
FORBIDDEN = ('torch', 'vllm', 'py_kvcache')


class NoBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in FORBIDDEN):
            ATTEMPTS.append(fullname)
            raise RuntimeError('CPU launcher tests cannot import a model/native backend')
        return None


def load(filename, name):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hook = NoBackend()
sys.meta_path.insert(0, hook)
P = load('prepare_repeatability_cpu.py', '_repeatability_prepare_CPU_test')
R = load('run_fixed_repeatability.py', '_repeatability_supervisor_CPU_test')
PROTOCOL = json.loads((HERE / 'PROTOCOL.json').read_bytes())


def fixture_record(index=0, duration=16238752):
    """Pure disposition fixture; never given to a public receipt or guard API."""
    covered = duration <= 16238752
    return dict(fixture_origin='synthetic_pure_disposition_only', diagnostic_index=index,
        native_execution_verified=True, diagnostic_evidence_valid=True, lifecycle_valid=True,
        full_output_tokens=128, full_capture_frames=128, selected_gpu_elapsed_ns=duration,
        frozen_cost_upper_ns=16238752, frozen_a_only_budget_ns=13171328, covered_original_upper=covered,
        status='VALID_RECORD_COST_COVERED' if covered else 'VALID_RECORD_COST_EXCEEDED',
        normal_qualification_passed=False, runtime_condition_qualified=False, qualification_passed=False,
        permits_next_mode=None, P4_strategy_effect_verified=False, performance_benefit_proved=False,
        calibration_refit=False, thresholds_changed=False)


class ProtocolTests(unittest.TestCase):
    def test_actual_preregistered_bytes_and_slots(self):
        self.assertEqual(hashlib.sha256((HERE / 'PROTOCOL.json').read_bytes()).hexdigest(),
            'ddfcacd283cc9f5fc6d2ec5bd75333c34c8dd956fe02487092b5702c7d420dcd')
        self.assertEqual(len(R.validate_protocol(PROTOCOL)), 3)

    def test_no_dynamic_repeat_count(self):
        for value in (True, 1, 2, 4, '3'):
            with self.subTest(value=value):
                protocol = deepcopy(PROTOCOL); protocol['repetitions'] = value
                with self.assertRaises(ValueError): R.validate_protocol(protocol)

    def test_upper_or_budget_widening_rejected(self):
        for key in ('cost_upper_ns', 'step_budget_ns'):
            protocol = deepcopy(PROTOCOL); protocol[key] += 1
            with self.assertRaises(ValueError): R.validate_protocol(protocol)

    def test_workload_or_offset_change_rejected(self):
        for key in ('seed', 'prompt_first_token', 'measured_offset', 'ssd_read_physical_bytes'):
            protocol = deepcopy(PROTOCOL); protocol[key] += 1
            with self.assertRaises(ValueError): R.validate_protocol(protocol)

    def test_dynamic_label_or_duplicate_slot_rejected(self):
        protocol = deepcopy(PROTOCOL); protocol['jobs'][1]['label'] = protocol['jobs'][0]['label']
        with self.assertRaises(ValueError): R.validate_protocol(protocol)

    def test_true_integer_slot_refused(self):
        protocol = deepcopy(PROTOCOL); protocol['jobs'][0]['diagnostic_index'] = False
        with self.assertRaises(ValueError): R.validate_protocol(protocol)

    def test_cost_or_pass_selection_stops_refused(self):
        for key in ('cost_exceedance_alone_stops_planned_repetitions', 'pass_alone_stops_planned_repetitions'):
            protocol = deepcopy(PROTOCOL); protocol['decision_rule'][key] = True
            with self.assertRaises(ValueError): R.validate_protocol(protocol)

    def test_retries_or_budget_increase_rejected(self):
        for key in ('replacement_or_retry_attempts', 'planned_max_reserved_seconds', 'original_cumulative_seconds'):
            protocol = deepcopy(PROTOCOL); protocol['budget'][key] += 1
            with self.assertRaises(ValueError): R.validate_protocol(protocol)


class DispositionTests(unittest.TestCase):
    def test_pure_cost_miss_continues_and_issues_no_GPU_authority(self):
        result = R.decision(fixture_record(duration=16893473), 0, PROTOCOL)
        self.assertEqual(result['action'], 'CONTINUE_FIXED_PROTOCOL')
        self.assertEqual(result['next_diagnostic_index'], 1)
        self.assertFalse(result['observed_coverage'])
        self.assertIs(result['GPU_authority_issued'], False)
        self.assertIsNone(result['permits_next_mode'])

    def test_pure_cost_covered_does_not_stop_early(self):
        result = R.decision(fixture_record(index=1, duration=15000000), 1, PROTOCOL)
        self.assertEqual(result['next_diagnostic_index'], 2)
        self.assertIs(result['normal_qualification_passed'], False)

    def test_exactly_third_slot_completes(self):
        result = R.decision(fixture_record(index=2), 2, PROTOCOL)
        self.assertEqual(result['action'], 'FIXED_PROTOCOL_COMPLETE')
        self.assertIsNone(result['next_diagnostic_index'])

    def test_index_mismatch_rejected(self):
        with self.assertRaises(ValueError): R.decision(fixture_record(), 1, PROTOCOL)

    def test_lifecycle_or_output_failure_cannot_continue(self):
        for key, value in (('lifecycle_valid', False), ('diagnostic_evidence_valid', False),
            ('native_execution_verified', False), ('full_output_tokens', 127), ('full_capture_frames', 127)):
            record = fixture_record(); record[key] = value
            with self.assertRaises(ValueError): R.decision(record, 0, PROTOCOL)

    def test_missing_or_boolean_duration_rejected(self):
        for value in (None, True, 0, -1, float('nan'), 16238752.0):
            record = fixture_record(); record['selected_gpu_elapsed_ns'] = value
            with self.assertRaises(ValueError): R.decision(record, 0, PROTOCOL)

    def test_forged_coverage_or_cost_refit_rejected(self):
        for key, value in (('covered_original_upper', 1), ('covered_original_upper', False),
            ('calibration_refit', True), ('thresholds_changed', True), ('frozen_cost_upper_ns', 17000000)):
            record = fixture_record(); record[key] = value
            with self.assertRaises(ValueError): R.decision(record, 0, PROTOCOL)

    def test_diagnostic_P4_or_normal_promotion_rejected(self):
        for key, value in (('normal_qualification_passed', True), ('permits_next_mode', 'shadow'),
            ('P4_strategy_effect_verified', True), ('performance_benefit_proved', True)):
            record = fixture_record(); record[key] = value
            with self.assertRaises(ValueError): R.decision(record, 0, PROTOCOL)


class SourceBoundaryTests(unittest.TestCase):
    def test_explicit_source_inventory_excludes_canceled_drafts(self):
        used = set(P.REQUIRED_SOURCES + P.OPTIONAL_SOURCES)
        self.assertFalse(used.intersection({'historical_calibration_binding.py', 'native_conditional_cost.py',
            'p4_single_file_receipt.py', 'prepare_and_verify_native_cost.py', 'SERVER12_LEDGER_SNAPSHOT.json'}))
        self.assertIn('PREREGISTRATION_LEDGER_SNAPSHOT.json', used)

    def test_negative_or_boolean_file_size_refused_and_empty_source_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True); (root / 'empty.py').write_bytes(b'')
            actual = P.ref(root, 'empty.py')
            self.assertEqual(actual['bytes'], 0)
            P.checked(root, actual)
            for size in (-1, True, 0.0, '0'):
                row = dict(actual, bytes=size)
                with self.assertRaises(ValueError): P.shape(row, root)

    def test_file_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True); path = root / 'source.py'; path.write_bytes(b'first')
            actual = P.ref(root, 'source.py'); path.write_bytes(b'second')
            with self.assertRaises(ValueError): P.checked(root, actual)

    def test_traversal_external_or_backslash_path_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for path in ('../secret', '/secret', 'a//b', 'C:/secret', 'a\\b', './a'):
                with self.assertRaises(ValueError): P.safe(root, path)
                with self.assertRaises(ValueError): R.safe(root, path)

    def test_active_or_insufficient_original_ledger_refused(self):
        for active, seconds in (({'id': 'fixture-only'}, 100), (None, 28800-959), (None, True), (None, float('inf'))):
            with self.assertRaises(ValueError): P.require_idle_ledger(dict(active_reservation=active,
                gpu_wall_seconds=seconds, events=[]))

    def test_helper_never_imports_backend_or_launches_external_process(self):
        for name in ('prepare_repeatability_cpu.py', 'run_fixed_repeatability.py'):
            tree = ast.parse((HERE / name).read_bytes())
            imports = [a.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                for a in (node.names if isinstance(node, ast.Import) else [])]
            self.assertFalse(any(value.split('.')[0] in FORBIDDEN for value in imports))
            self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr in
                ('Popen', 'system', 'execv', 'kill', 'killpg', 'extractall') for node in ast.walk(tree)))

    def test_duplicate_or_nonfinite_JSON_refused(self):
        for module in (P, R):
            for raw in (b'{"a":1,"a":2}', b'{"a":NaN}'):
                with self.assertRaises(ValueError): module.parse(raw)


def main():
    result = unittest.main(exit=False)
    imports = sorted(name for name in sys.modules if any(name == p or name.startswith(p + '.') for p in FORBIDDEN))
    print(json.dumps(dict(status='CPU_REPEATABILITY_LAUNCHER_TESTS_ONLY', tests=result.result.testsRun,
        passed=result.result.testsRun-len(result.result.failures)-len(result.result.errors)-len(result.result.skipped),
        failures=len(result.result.failures), errors=len(result.result.errors), skipped=len(result.result.skipped),
        forbidden_import_attempts=list(ATTEMPTS), actual_backend_imports=imports,
        actual_GPU_runs=0, native_qualification_issued=False)))
    return 0 if result.result.wasSuccessful() and not result.result.skipped and not imports and not ATTEMPTS else 1


if __name__ == '__main__':
    raise SystemExit(main())
