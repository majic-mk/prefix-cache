"""CPU rejection/causality tests; no fixture proves native execution."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / "native_cost_v5"
if not NATIVE.is_dir():
    NATIVE = HERE.parent / "server11-native-cost-v5-20261003"
V = load("runtime_verifier_cpu_test", HERE / "verify_p4_single_file.py")
F = load("native_fixture_for_runtime_cpu_test", NATIVE / "test_native_conditional_cost.py")


def policy_snapshot(shadow, *, previews=0, blocks=0, decisions=None):
    return dict(valid=True, fault=None, mode="interference", new_work_queues=0,
        held_job_or_resource_owners=False, interference_production_qualified=False,
        conditional_single_file=dict(enabled=True, shadow=shadow, generic_production_qualified=False,
            evidence_binding_sha256="a"*64, previews=previews, proposed_deferrals=previews,
            actual_blocked_attempts=blocks, actual_decisions=[] if decisions is None else decisions))


class RuntimeVerifierTests(unittest.TestCase):
    def setUp(self):
        self.selected = dict(native_step_ordinal=16, host_start_ns=50, host_end_ns=400,
            witness=dict(start_record_before_ns=10, start_completed_query_ns=100,
                         end_record_before_ns=410, end_completed_query_ns=500))
        self.event = dict(at_ns=600)
        self.trigger = dict(trigger_before_ns=120)
        self.decision = dict(key=[16, "preload:actual-native-key"], native_step_ordinal=16,
            start_record_before_ns=10, start_completed_query_ns=100,
            first_defer_ns=130, last_defer_ns=350, count=2,
            resource_release_credit=False, production_qualified=False)

    def check(self, mode, before, after, tail):
        return V.validate_deferral(mode, before, after, tail, selected=self.selected,
            accepted=self.event, trigger=self.trigger, binding_sha="a"*64)

    def on(self):
        before = policy_snapshot(True)  # Capture attaches only after this initial drain.
        after = policy_snapshot(False, previews=2, blocks=2, decisions=[self.decision])
        return before, after, deepcopy(after)

    def test_off_constructs_no_bridge(self):
        self.assertTrue(self.check("off", None, None, None)["off_bridge_is_none"])
        with self.assertRaises(ValueError):
            self.check("off", None, policy_snapshot(True), None)

    def test_shadow_proposes_without_blocking(self):
        after = policy_snapshot(True, previews=1)
        result = self.check("shadow", policy_snapshot(True), after, deepcopy(after))
        self.assertEqual(result["actual_blocked_attempts"], 0)
        after["conditional_single_file"]["actual_blocked_attempts"] = 1
        with self.assertRaises(ValueError):
            self.check("shadow", policy_snapshot(True), after, deepcopy(after))

    def test_shadow_without_qualified_proposal_is_not_validation(self):
        with self.assertRaisesRegex(ValueError, "no real qualified"):
            self.check("shadow", policy_snapshot(True), policy_snapshot(True), policy_snapshot(True))

    def test_on_can_issue_after_selected_step(self):
        result = self.check("on", *self.on())
        self.assertEqual(result["actual_blocked_attempts"], 2)

    def test_on_can_legally_resume_inside_same_step(self):
        self.event["at_ns"] = 380  # Real bounded defer; no claim that CUDA is finished.
        self.assertEqual(self.check("on", *self.on())["actual_blocked_attempts"], 2)

    def test_on_cannot_issue_before_last_defer(self):
        self.event["at_ns"] = 350
        with self.assertRaisesRegex(ValueError, "after native acceptance"):
            self.check("on", *self.on())

    def test_on_cannot_defer_after_end_record(self):
        self.decision["last_defer_ns"] = 410
        with self.assertRaises(ValueError):
            self.check("on", *self.on())

    def test_on_cannot_defer_before_actual_start_query(self):
        self.decision["first_defer_ns"] = 100
        with self.assertRaises(ValueError):
            self.check("on", *self.on())

    def test_on_missing_or_wrong_native_step_is_rejected(self):
        self.decision["native_step_ordinal"] = 17
        with self.assertRaisesRegex(ValueError, "current original step"):
            self.check("on", *self.on())

    def test_on_counters_and_shutdown_tail_must_match(self):
        before, after, tail = self.on()
        tail["conditional_single_file"]["actual_blocked_attempts"] += 1
        with self.assertRaisesRegex(ValueError, "changed after"):
            self.check("on", before, after, tail)

    def test_no_early_release_credit(self):
        self.decision["resource_release_credit"] = True
        with self.assertRaisesRegex(ValueError, "release claims"):
            self.check("on", *self.on())

    def test_frozen_migration_gate_never_widens(self):
        for mode in ("off", "shadow"):
            self.assertEqual(V.migration_gate(mode, V.LIMIT_NS), (True, True))
            self.assertEqual(V.migration_gate(mode, V.LIMIT_NS + 1), (False, False))
        self.assertEqual(V.migration_gate("on", V.LIMIT_NS + 1), (False, True))
        with self.assertRaises(ValueError):
            V.migration_gate("off", True)

    def test_original_accounting_prefix_remains_exact(self):
        plan, windows, journal, drain = F.fixture()
        checker = V.accounting_validator(F.M, NATIVE / "native_conditional_cost.py")
        args = dict(capture=windows[1]["capture"], frames=[], run_id=plan["journal_run_id"],
            native_source_sha256=plan["native_source_sha256"], arm="action", measured_offset=16,
            physical_bytes=8, operations=1, independent_payload=windows[1]["independent_payload"])
        accepted, completed, owners = checker(journal, drain, **args)
        self.assertEqual(len(accepted), 3)
        self.assertEqual(set(accepted), set(completed))
        self.assertTrue(owners)
        bad = deepcopy(journal)
        bad["events"][1]["result"] = 7
        with self.assertRaisesRegex(ValueError, "CQE"):
            checker(bad, drain, **args)

    def test_original_accounting_rejects_false_owner_counts(self):
        plan, windows, journal, drain = F.fixture()
        checker = V.accounting_validator(F.M, NATIVE / "native_conditional_cost.py")
        journal["frames"][1]["accepted"][0]["ops"] += 1
        with self.assertRaisesRegex(ValueError, "frame counters"):
            checker(journal, drain, capture={}, frames=[], run_id=plan["journal_run_id"],
                native_source_sha256=plan["native_source_sha256"], arm="action", measured_offset=16,
                physical_bytes=8, operations=1, independent_payload={})


if __name__ == "__main__":
    unittest.main()
