"""Symbolic CPU counterexamples, never native timing or receipt fixtures."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_cost_protocol_under_test", HERE / "check_cost_protocol.py")
C = importlib.util.module_from_spec(spec); sys.modules[spec.name] = C; spec.loader.exec_module(C)
PROTOCOL = json.loads((HERE / "PROTOCOL.json").read_text(encoding="utf-8"))


class CostProtocolTests(unittest.TestCase):
    def changed(self, part, key, value):
        result = deepcopy(PROTOCOL); result[part][key] = value; return result

    def test_finite_protocol_valid_but_cannot_authorize(self):
        result = C.check_protocol(PROTOCOL)
        self.assertTrue(result["protocol_valid"])
        self.assertFalse(result["gpu_launch_allowed"])
        self.assertFalse(result["cost_values_available"])
        self.assertFalse(result["receipt_issued"])

    def test_common_source_closure_cannot_claim_on_cost_measured(self):
        for key, value in (("bridge", "active"), ("notification_installed", True),
                           ("queue_observation_installed", True),
                           ("measured_paths", PROTOCOL["common"]["measured_paths"] + ["armed_queue_get"])):
            with self.subTest(key=key), self.assertRaises(ValueError): C.check_protocol(self.changed("common", key, value))

    def test_cost_values_cannot_be_filled_from_cpu_or_old_cell(self):
        for key in ("cost_upper_ns", "step_budget_ns"):
            with self.assertRaises(ValueError): C.check_protocol(self.changed("common", key, 1000))

    def test_original_math_and_holdout_are_not_refit(self):
        for key, value in (("formula", "new_formula"), ("holdout_refit", True),
                           ("budget_uses", ["calibration_A_only", "action_B"]), ("calibration_pairs", 3)):
            with self.subTest(key=key), self.assertRaises(ValueError): C.check_protocol(self.changed("common", key, value))

    def test_all_six_fixed_windows_and_split_required(self):
        for value in (C.WINDOWS[:-1], list(reversed(C.WINDOWS)), C.WINDOWS + [C.WINDOWS[0]]):
            with self.assertRaises(ValueError): C.check_protocol(self.changed("common", "windows", value))

    def test_fixed_load_threshold_and_integer_types(self):
        for key, value in (("output_tokens", 127), ("max_wait_ns", 200000000),
                           ("max_accepted_parents", 64), ("operations", True), ("physical_bytes", 2 * 917504)):
            with self.subTest(key=key), self.assertRaises(ValueError): C.check_protocol(self.changed("condition", key, value))

    def test_off_shadow_never_install_notification(self):
        value = deepcopy(PROTOCOL["runtime"]["arms"]); value[1]["notification"] = True
        with self.assertRaises(ValueError): C.check_protocol(self.changed("runtime", "arms", value))

    def test_common_and_runtime_namespaces_cannot_be_conflated(self):
        for part, value in (("common", "owner_LABEL"), ("runtime", "frontend_RID")):
            with self.assertRaises(ValueError): C.check_protocol(self.changed(part, "capture_run_namespace", value))

    def test_on_semantics_not_common_cost_coverage_or_improvement(self):
        for key, value in (("on_migration_gate", "covered"), ("instrumentation_cost_qualified", True),
                           ("performance_claim", True), ("upper_refit_after_runtime", True),
                           ("add_cpu_cost_to_upper", True), ("change_budget_to_force_defer", True)):
            with self.subTest(key=key), self.assertRaises(ValueError): C.check_protocol(self.changed("runtime", key, value))

    def test_budget_reference_never_authorization_or_remaining_guess(self):
        for key, value in (("new_authorization", "old-v6-job"), ("remaining_seconds", 5000),
                           ("new_allocation", 1220), ("cumulative_cap_seconds", 30000)):
            with self.subTest(key=key), self.assertRaises(ValueError): C.check_protocol(self.changed("budget_reference", key, value))

    def test_cpu_checker_cannot_create_jobs_or_native_receipts(self):
        for key in ("gpu_launch_allowed", "native_execution_verified", "create_jobs", "create_scope", "receipt_issued"):
            with self.subTest(key=key), self.assertRaises(ValueError): C.check_protocol(self.changed("execution", key, True))

    def test_receipt_depends_on_common_not_on_its_own_runtime(self):
        value = deepcopy(PROTOCOL); value["dependencies"]["strict_common_cost_receipt"].append("on_notification_native")
        with self.assertRaisesRegex(ValueError, "cycle"): C.check_protocol(value)

    def test_cannot_skip_common_receipt_before_shadow_or_on(self):
        for stage in ("strict_common_cost_receipt", "shadow_native", "on_notification_native"):
            with self.assertRaisesRegex(ValueError, "prior evidence"):
                C.simulated_next_obligation(dict(origin="synthetic_protocol_test", completed=[stage],
                    upper_relation="unavailable", notification_exercised=False))

    def test_symbolic_issue_relation_stops_without_forcing_defer(self):
        stages = C.topological_order(C.DEPENDENCIES)
        completed = stages[:stages.index("strict_common_cost_receipt") + 1]
        result = C.simulated_next_obligation(dict(origin="synthetic_protocol_test", completed=completed,
            upper_relation="le", notification_exercised=False))
        self.assertEqual(result["symbolic_status"], "NOT_EXERCISED_NO_FORCED_DEFER")
        self.assertFalse(result["gpu_launch_allowed"])

    def test_symbolic_completed_on_without_real_wait_not_demonstrated(self):
        result = C.simulated_next_obligation(dict(origin="synthetic_protocol_test",
            completed=C.topological_order(C.DEPENDENCIES), upper_relation="gt", notification_exercised=False))
        self.assertEqual(result["symbolic_status"], "ON_NOTIFICATION_NOT_DEMONSTRATED")
        self.assertFalse(result["native_execution_verified"])

    def test_even_all_symbolic_stages_cannot_issue_receipt_or_gpu_permission(self):
        result = C.simulated_next_obligation(dict(origin="synthetic_protocol_test",
            completed=C.topological_order(C.DEPENDENCIES), upper_relation="gt", notification_exercised=True))
        self.assertFalse(result["receipt_issued"]); self.assertFalse(result["gpu_launch_allowed"])
        self.assertFalse(result["performance_claim"])

    def test_no_unknown_field_or_real_evidence_impersonation(self):
        value = deepcopy(PROTOCOL); value["cost_receipt"] = "synthetic"
        with self.assertRaises(ValueError): C.check_protocol(value)
        with self.assertRaises(ValueError): C.simulated_next_obligation(dict(origin="native_gpu_recording",
            completed=[], upper_relation="unavailable", notification_exercised=False))

    def test_outer_lock_checks_all_sources_and_rejects_drift_missing_and_native_claim(self):
        spec = importlib.util.spec_from_file_location("_cpu_protocol_runner_test", HERE / "run_cpu_protocol.py")
        R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); source = root / "source.py"; lock = root / "lock.json"
            source.write_bytes(b"# synthetic protocol source\n")
            raw = source.read_bytes()
            value = dict(schema="c5_cost_binding_cpu_source_lock_v1", gpu_launch_allowed=False,
                native_execution_verified=False, files=[dict(path="source.py", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())])
            lock.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(R.outer_lock(root, lock, [source])["files"], 1)
            with self.assertRaises(ValueError): R.outer_lock(root, lock, [root / "missing.py"])
            source.write_bytes(raw + b"# drift\n")
            with self.assertRaises(ValueError): R.outer_lock(root, lock, [source])
            source.write_bytes(raw); value["gpu_launch_allowed"] = True
            lock.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError): R.outer_lock(root, lock, [source])


if __name__ == "__main__":
    unittest.main(verbosity=2)
