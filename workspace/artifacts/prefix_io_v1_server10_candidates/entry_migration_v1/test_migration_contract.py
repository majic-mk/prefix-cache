"""CPU-only negative contracts; these fixtures do not establish GPU provenance."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_migration_contract_under_test", HERE / "migration_contract.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)
SOURCE_ROOT = None


def ref(name, prefix=None):
    return dict(path=(C.DELIVERY if prefix is None else prefix) + "/" + name, bytes=1, sha256="a" * 64)


def source_rows():
    return [copy.deepcopy(C.PERMISSIONS_REF), ref("migration_contract.py"),
        ref("CUDA13_SOURCE_INVENTORY.json", C.SDK_CPU_DELIVERY),
        ref("CPU_COMPILE_LINK_RESULT.json", C.SDK_CPU_DELIVERY),
        ref("sdk_rebind.py", C.SDK_DELIVERY)]


def scope_fixture():
    scope = C.scope_template(ref("gpu-source-lock.json"), ref("PLAN.json"), C.INPUT_REF, ref("ANALYZER.json"))
    scope.update(status="USER_AUTHORIZED_G3_REFERENCE_DIAGNOSTIC",
        allow_gpu_initialization=True, allow_gpu_runs=True, human_authorization_record=ref("HUMAN.json"))
    return scope


def human_fixture(scope):
    human = copy.deepcopy(scope)
    human.update(authorization_origin="direct_human_reply", question="CPU fixture only, not human authorization",
        verbatim_user_answer="CPU fixture only", reply_observed_utc="2026-10-03 00:00:00 UTC",
        question_item_id="not_applicable:direct_typed_user_message", reply_medium="direct_typed_user_message",
        form_response_id_fabricated=False)
    return human


def scope_refs(scope):
    rows = [C.PERMISSIONS_REF, C.MIGRATION_ANCESTOR_REF]
    rows += [scope[k] for k in ("cpu_plan_ref", "input_manifest_ref", "analyzer_plan_ref")]
    return {r["path"]: copy.deepcopy(r) for r in rows}


def success_receipt(mode):
    return dict(mode=mode, label=C.JOB_NAMES[mode], guard_exit=0, child_exit=0, timed_out=False,
        error=None, session_drained=True, session_members_before_cleanup=[], session_members_after_cleanup=[],
        original_exit_code=0, original_acquisition_status="PASSED_NATIVE_" + mode.upper() + "_ACQUISITION",
        original_shutdown_completed=True)


def guard_fixture(mode="cold"):
    return dict(gpu_wall_seconds=17809.140367632266, events=[], active_reservation=dict(
        id="b" * 32, label=C.JOB_NAMES[mode], gpu_uuid=C.GPU_UUID, seconds_limit=300, reserved_seconds=320,
        session_id=123, process_group=123, permissions=copy.deepcopy(C.PERMISSIONS_REF), command=["python", "bounded.py"]))


class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ancestor_raw = Path(C.ORIGINAL.__file__).with_name("gpu-source-lock-reference-v1.json").read_bytes()

    def test_only_machine_identity_and_exclusive_cold_storage_change(self):
        old = C.ORIGINAL.fixed_context()
        old_snapshot = copy.deepcopy(old)
        expected = copy.deepcopy(old)
        expected["gpu_uuid"] = C.GPU_UUID
        expected["storage_by_mode"]["cold"] = C.STORAGE_RELATIVE
        C.same(C.fixed_context(), expected, "migration definition")
        C.same(C.ORIGINAL.fixed_context(), old_snapshot, "immutable parent globals")
        self.assertEqual(C.ORIGINAL.P.GPU_UUID, C.SOURCE_GPU_UUID)
        self.assertEqual(C.fixed_context()["sampling"]["temperature"], 0.0)
        self.assertIs(type(C.fixed_context()["sampling"]["temperature"]), float)

    def test_permission_bytes_match_actual_fixed_server_pin(self):
        path = (Path(SOURCE_ROOT) / C.PERMISSIONS) if SOURCE_ROOT else HERE / "permissions.server10.reference.yaml"
        actual = path.read_bytes()
        self.assertEqual(actual, C.PERMISSIONS_BYTES)
        self.assertEqual(len(actual), 685)
        self.assertEqual(hashlib.sha256(actual).hexdigest(), C.PERMISSIONS_REF["sha256"])
        self.assertIn(C.GPU_UUID.encode(), actual)
        self.assertNotIn(C.SOURCE_GPU_UUID.encode(), actual)
        self.assertIn(b"allow_driver_or_system_changes: false", actual)
        self.assertIn(b"max_gpu_hours: 8", actual)

    def test_ancestor_preserved_byte_refs_and_count(self):
        manifest = C.assemble_source_lock(self.ancestor_raw, source_rows())
        original = json.loads(self.ancestor_raw)
        C.same(manifest["files"][:4088], original["files"], "all immutable source rows")
        self.assertEqual(manifest["files"][4088], C.MIGRATION_ANCESTOR_REF)
        checked = C.validate_source_manifest(manifest, self.ancestor_raw)
        self.assertEqual(checked["files"], 4089 + len(source_rows()))
        self.assertFalse(checked["authorizes_gpu"])
        self.assertFalse(checked["full_source_bytes_verified"])

    def test_replaced_reordered_or_truncated_ancestor_is_rejected(self):
        base = C.assemble_source_lock(self.ancestor_raw, source_rows())
        for variant in ("hash", "order", "remove", "type"):
            changed = copy.deepcopy(base)
            if variant == "hash":
                changed["files"][0]["sha256"] = "0" * 64
            elif variant == "order":
                changed["files"][0], changed["files"][1] = changed["files"][1], changed["files"][0]
            elif variant == "remove":
                changed["files"].pop(0)
            else:
                changed["files"][0]["bytes"] = float(changed["files"][0]["bytes"])
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                C.validate_source_manifest(changed, self.ancestor_raw)

    def test_changed_ancestor_raw_and_ancestor_replacement_are_rejected(self):
        with self.assertRaises(ValueError):
            C.assemble_source_lock(self.ancestor_raw + b" ", source_rows())
        with self.assertRaises(ValueError):
            C.assemble_source_lock(self.ancestor_raw, source_rows() + [copy.deepcopy(C.GUARD_REF)])

    def test_append_bound_paths_and_no_authority_cycles(self):
        bad_rows = [ref("HUMAN_AUTHORIZATION.json"), ref("scope.json"), ref("gpu-source-lock.json"),
            ref("helper.py", "artifacts/prefix_io_v1/other"), ref("../escape.py"),
            ref("helper.py", C.SDK_CPU_DELIVERY + "-other")]
        for bad in bad_rows:
            with self.subTest(path=bad["path"]), self.assertRaises(ValueError):
                C.assemble_source_lock(self.ancestor_raw, source_rows() + [bad])
        too_many = [C.PERMISSIONS_REF] + [ref("file%02d.py" % i) for i in range(32)]
        with self.assertRaises(ValueError):
            C.assemble_source_lock(self.ancestor_raw, too_many)
        with self.assertRaises(ValueError):
            C.assemble_source_lock(self.ancestor_raw, source_rows() + [source_rows()[1]])

    def test_new_permission_must_be_present_and_exact(self):
        with self.assertRaises(ValueError):
            C.assemble_source_lock(self.ancestor_raw, source_rows()[1:])
        rows = source_rows()
        rows[0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            C.assemble_source_lock(self.ancestor_raw, rows)

    def test_plan_reuses_historical_cache_and_original_acquisition(self):
        plan = C.build_reference_plan(source_rows(), C.INPUT_REF, analyzer_plan_ref=ref("ANALYZER.json"))
        self.assertEqual(plan["baseline_source_lock"], C.ORIGINAL.BASELINE_REF)
        self.assertEqual(plan["migration_ancestor_source_lock"], C.MIGRATION_ANCESTOR_REF)
        self.assertEqual(plan["input_manifest_ref"], C.INPUT_REF)
        self.assertEqual([r["label"] for r in plan["jobs"]], list(C.JOB_NAMES.values()))
        self.assertEqual([r["requests"] for r in plan["jobs"]], [6, 6])
        self.assertTrue(plan["jobs"][1]["requires_prior_success_original_shutdown_and_OS_drain"])
        self.assertIn(C.PROJECT_ROOT + "/" + C.PAIRED_STORAGE_RELATIVE, plan["jobs"][1]["original_argv"])
        self.assertFalse(plan["authorizes_gpu"])
        with self.assertRaises(ValueError):
            C.build_reference_plan(source_rows(), ref("relabeled_old_input.json"))

    def test_template_and_records_remain_cpu_consistency_only(self):
        self.assertFalse(C.scope_template()["allow_gpu_runs"])
        scope = scope_fixture()
        result = C.validate_scope_records(scope, human_fixture(scope), scope_refs(scope))
        self.assertTrue(result["human_record_matches"])
        self.assertFalse(result["authorizes_gpu"])
        self.assertFalse(result["independent_human_provenance_verified"])

    def test_old_gpu_old_permission_and_old_output_identity_are_rejected(self):
        for key in ("gpu_uuid", "permission", "label", "historical"):
            scope = scope_fixture()
            if key == "gpu_uuid":
                scope["gpu_uuid"] = C.SOURCE_GPU_UUID
            elif key == "permission":
                scope["base_permissions"] = copy.deepcopy(C.ORIGINAL.PERMISSIONS_REF)
            elif key == "label":
                scope["permitted_run_names"]["cold"] = C.ORIGINAL.JOB_NAMES["cold"]
            else:
                scope["machine_migration"]["historical_evidence_modified"] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.validate_scope_records(scope, human_fixture(scope), scope_refs(scope))

    def test_strong_types_and_original_experiment_definition_cannot_drift(self):
        cases = [(("sampling", "temperature"), 0), (("sampling", "temperature"), -0.0),
            (("sampling", "logprobs"), 6), (("connector", "sync_on_store"), True),
            (("engine_overrides", "kv_cache_memory_bytes"), 2 * 1024 ** 3)]
        for keys, value in cases:
            scope = scope_fixture()
            scope["qualification_context"][keys[0]][keys[1]] = value
            with self.subTest(keys=keys, value=value), self.assertRaises(ValueError):
                C.validate_scope_records(scope, human_fixture(scope), scope_refs(scope))
        scope = scope_fixture()
        scope["maximum_jobs"] = 2.0
        with self.assertRaises(ValueError):
            C.validate_scope_records(scope, human_fixture(scope), scope_refs(scope))

    def test_human_must_bind_migration_context_and_new_source_refs(self):
        scope = scope_fixture()
        human = human_fixture(scope)
        human["migration_ancestor_source_lock"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            C.validate_scope_records(scope, human, scope_refs(scope))
        human = human_fixture(scope)
        human["allow_gpu_runs"] = 1
        with self.assertRaises(ValueError):
            C.validate_scope_records(scope, human, scope_refs(scope))
        refs = scope_refs(scope)
        refs.pop(C.MIGRATION_ANCESTOR_REF["path"])
        with self.assertRaises(ValueError):
            C.validate_scope_records(scope, human_fixture(scope), refs)

    def test_exact_guard_accepts_new_identity_and_preserves_reservation(self):
        witness = C.validate_execution_guard(scope_fixture(), guard_fixture(), "cold",
            actual_session_id=123, expected_command=["python", "bounded.py"], scope_sha256="c" * 64)
        self.assertEqual(witness["gpu_uuid"], C.GPU_UUID)
        self.assertEqual(witness["permissions_ref"], C.PERMISSIONS_REF)
        self.assertEqual(witness["reserved_seconds"], 320)

    def test_guard_rejects_wrong_gpu_permission_command_or_session(self):
        cases = [("gpu_uuid", C.SOURCE_GPU_UUID), ("permissions", C.ORIGINAL.PERMISSIONS_REF),
            ("command", ["python", "other.py"]), ("session_id", 124), ("process_group", 124),
            ("seconds_limit", 301), ("reserved_seconds", 320.0)]
        for key, value in cases:
            ledger = guard_fixture()
            ledger["active_reservation"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.validate_execution_guard(scope_fixture(), ledger, "cold", actual_session_id=123,
                    expected_command=["python", "bounded.py"], scope_sha256="c" * 64)

    def test_guard_rejects_reused_label_and_exhausted_budget(self):
        for kind in ("retry", "budget"):
            ledger = guard_fixture()
            if kind == "retry":
                ledger["events"] = [dict(label=C.JOB_NAMES["cold"])]
            else:
                ledger["gpu_wall_seconds"] = 28481
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                C.validate_execution_guard(scope_fixture(), ledger, "cold", actual_session_id=123,
                    expected_command=["python", "bounded.py"], scope_sha256="c" * 64)

    def test_order_requires_original_success_shutdown_and_drain(self):
        self.assertEqual(C.next_job([]), "cold")
        self.assertEqual(C.next_job([success_receipt("cold")]), "paired")
        self.assertIsNone(C.next_job([success_receipt("cold"), success_receipt("paired")]))
        for key, value in (("guard_exit", 1), ("child_exit", False), ("timed_out", True),
            ("error", "failure"), ("session_drained", False), ("original_shutdown_completed", False),
            ("session_members_before_cleanup", [123])):
            receipt = success_receipt("cold")
            receipt[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.next_job([receipt])
        with self.assertRaises(ValueError):
            C.next_job([success_receipt("paired")])
        with self.assertRaises(ValueError):
            C.next_job([success_receipt("cold")] * 3)

    def test_budget_storage_reuses_original_cumulative_contract(self):
        ledger = dict(gpu_wall_seconds=17809.140367632266, active_reservation=None, events=[])
        snapshot = dict(origin="actual_statvfs", captured_unix=1000.0,
            primary_root=C.PROJECT_ROOT, free_bytes=10 * 1024 ** 3)
        checked = C.validate_budget_and_storage(ledger, snapshot, now_unix=1001.0)
        self.assertAlmostEqual(checked["remaining_gpu_seconds"], 10990.859632367734)
        self.assertEqual(checked["reserved_gpu_seconds"], 640)
        self.assertFalse(checked["authorizes_gpu"])
        with self.assertRaises(ValueError):
            C.validate_budget_and_storage(ledger, snapshot, now_unix=1001.0, remaining_jobs=3)
        with self.assertRaises(ValueError):
            C.validate_budget_and_storage(ledger, dict(snapshot, free_bytes=10 * 1024 ** 3 - 1), now_unix=1001.0)
        with self.assertRaises(ValueError):
            C.validate_budget_and_storage(ledger, snapshot, now_unix=1121.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-root")
    args, remainder = parser.parse_known_args()
    SOURCE_ROOT = args.source_root
    unittest.main(argv=[sys.argv[0], *remainder])
