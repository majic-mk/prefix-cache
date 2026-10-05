"""CPU tests of two-job routing and unchanged original runtime/guard logic."""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--original-dir", type=Path)
parser.add_argument("--common-file", type=Path)
parser.add_argument("--failed-fixture-dir", type=Path)
ARGS, REST = parser.parse_known_args()
OLD = ARGS.original_dir or HERE.parents[1] / "prefix_io_v1_server09_candidates/g3_calibration_launcher_cpu_v1"
COMMON = ARGS.common_file or HERE.parents[1] / "prefix_io_v1_server09_candidates/g2_normal_worker_site_cache_v4_final/run_g2_normal_model_lifecycle.py"
FAILED = ARGS.failed_fixture_dir or HERE.parent / "kv_failure_01"
spec = importlib.util.spec_from_file_location("_kv_diagnostic_cpu_entry", HERE / "run_server10_kv_diagnostic_v2.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def failed_case():
    capture_path = FAILED / "native-kv-byte-receipt.json"
    report_path = FAILED / "acquisition-result.json"
    if not capture_path.exists():
        capture_path = FAILED / "kv-capture/native-kv-byte-receipt.json"
    if not report_path.exists():
        report_path = FAILED / "acquisition/result.json"
    return json.loads(capture_path.read_bytes()), json.loads(report_path.read_bytes())


@contextlib.contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        refs = {}
        def write(relative, raw):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            row = dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            refs[relative] = row
            return row
        for key, row in E.PINNED.items():
            write(row["path"], (OLD / Path(row["path"]).name).read_bytes())
        write("artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/run_g2_normal_model_lifecycle.py", COMMON.read_bytes())
        permission = write(E.PERMISSIONS, b"cpu-permission\n")
        plan = E.load(root, E.PINNED["plan"], "test_plan")
        baseline = {"files": [plan.GUARD_REF, plan.ACQUIRE_REF, permission] +
            [dict(path="cpu/source-%d" % i, bytes=1, sha256="a" * 64) for i in range(2049)]}
        ancestor = write("artifacts/prefix_io_v1/server10-reference-migration-v1-20261003/cpu-ancestor.json", json.dumps(baseline).encode())
        config = dict(ancestor_ref=ancestor, permissions_ref=permission,
            sdk_inventory_ref=write("artifacts/prefix_io_v1/server10-cuda13-cpu-20261003/inventory.json", b"{}\n"),
            sdk_proof_ref=write("artifacts/prefix_io_v1/server10-cuda13-cpu-20261003/proof.json", b"{}\n"))
        E.configure_plan(root, plan, config)
        try:
            yield root, refs, config, plan, write
        finally:
            sys.modules.pop(plan.__name__, None)


def receipt(mode):
    return dict(mode=mode, label=E.JOBS[mode], guard_exit=0, child_exit=0, timed_out=False,
        error=None, session_drained=True, session_members_before_cleanup=[], session_members_after_cleanup=[],
        original_exit_code=0, original_acquisition_status="PASSED_NATIVE_" + mode.upper() + "_ACQUISITION",
        original_shutdown_completed=True)


class DiagnosticContracts(unittest.TestCase):
    def test_full_real_request_prompt_cohort_binding_and_missing_capture_rejection(self):
        for mode in E.MODES:
            kinds = ("store", "g_mem") if mode == "populate" else ("g_ssd", "g_mem")
            rows, records, jobs = [], [], {}
            for ordinal in range(6):
                kind = kinds[ordinal % 2]
                prompt = [1000 + ordinal // 2] * 129
                rid = "fixture-" + str(ordinal)
                rows.append(dict(kind=kind, request_id=rid, prompt_token_ids=prompt, warmup=ordinal < 2))
                internal_id = rid + "-1a2b3c4d"
                records.append(dict(stage="producer_model_forward" if kind == "store" else "before_model_forward",
                    request_id=internal_id, prompt_sha256=hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest(),
                    prompt_tokens=129))
                route = "store" if kind == "store" else "cache" if kind == "g_mem" else "file"
                jobs[str(ordinal)] = dict(is_store=kind == "store", request_id=internal_id,
                    files={str(i): dict(source_kind=route) for i in range(8)})
            report, capture = dict(mode=mode, rows=rows), dict(records=records, native_jobs=jobs)
            self.assertEqual(E.validate_capture_cohort(capture, report, mode)["status"], "PASS_ACTUAL_ORIGINAL_REQUEST_CAPTURE_COHORT")
            with self.assertRaises(ValueError):
                E.validate_capture_cohort(dict(capture, records=records[1:]), report, mode)
            wrong = copy.deepcopy(capture)
            wrong["records"][0]["request_id"] = "wrong-but-full-count"
            with self.assertRaises(ValueError):
                E.validate_capture_cohort(wrong, report, mode)
            wrong = copy.deepcopy(capture)
            wrong["records"][0]["prompt_sha256"] = "a" * 64
            with self.assertRaises(ValueError):
                E.validate_capture_cohort(wrong, report, mode)
            if mode == "paired":
                wrong = copy.deepcopy(capture)
                # Whole internal-ID exchange preserves same prompt and both load roles.
                for collection in (wrong["records"], list(wrong["native_jobs"].values())):
                    collection[0]["request_id"], collection[1]["request_id"] = collection[1]["request_id"], collection[0]["request_id"]
                with self.assertRaisesRegex(ValueError, "native route"):
                    E.validate_capture_cohort(wrong, report, mode)

    def test_real_failed_run_old_mapping_fails_and_source_bound_v2_mapping_passes(self):
        capture, report = failed_case()
        self.assertEqual(capture["status"], "PASS_NATIVE_KV_BYTE_DIAGNOSTIC")
        self.assertEqual(report["status"], "PASSED_NATIVE_POPULATE_ACQUISITION")
        before = json.dumps([capture, report], sort_keys=True)
        # Exact v1 method kept frozen; it must reproduce the real bookkeeping failure.
        old_path = HERE.parent / "kv_diagnostic_v1/run_server10_kv_diagnostic.py"
        if not old_path.exists():
            old_path = OLD.parents[0] / "server10-kv-diagnostic-v1-20261003/run_server10_kv_diagnostic.py"
        old_spec = importlib.util.spec_from_file_location("_frozen_v1_cohort_cpu_replay", old_path)
        old = importlib.util.module_from_spec(old_spec)
        old_spec.loader.exec_module(old)
        with self.assertRaisesRegex(ValueError, "same actual original request IDs"):
            old.validate_capture_cohort(capture, report, "populate")
        result = E.validate_capture_cohort(capture, report, "populate")
        self.assertEqual(len(result["request_mappings"]), 6)
        self.assertEqual(len({row["external_request_id"] for row in result["request_mappings"]}), 6)
        self.assertEqual(len({row["internal_request_id"] for row in result["request_mappings"]}), 6)
        self.assertEqual(json.dumps([capture, report], sort_keys=True), before)

    def test_real_input_rejects_false_suffix_same_prefix_foreign_and_duplicate_mapping(self):
        capture, report = failed_case()
        first = next(row for row in capture["records"] if row["stage"] == "before_model_forward")
        source_id = first["request_id"]
        external = source_id[:-9]
        bad_ids = [source_id + "x", external + "-ABCDEF01", external + "-1234567",
                   external + "-123456789", external, external + "0-12345678", "foreign-12345678"]
        for wrong_id in bad_ids:
            altered = copy.deepcopy(capture)
            for row in altered["records"]:
                if row["stage"] == "before_model_forward" and row["request_id"] == source_id:
                    row["request_id"] = wrong_id
            for job in altered["native_jobs"].values():
                if job["request_id"] == source_id:
                    job["request_id"] = wrong_id
            with self.assertRaises(ValueError):
                E.validate_capture_cohort(altered, report, "populate")
        altered = copy.deepcopy(capture)
        consumers = [row for row in altered["records"] if row["stage"] == "before_model_forward"]
        consumers[1]["request_id"] = consumers[0]["request_id"]
        with self.assertRaises(ValueError):
            E.validate_capture_cohort(altered, report, "populate")
        altered_report = copy.deepcopy(report)
        altered_report["rows"][2]["request_id"] = altered_report["rows"][0]["request_id"]
        with self.assertRaises(ValueError):
            E.validate_capture_cohort(capture, altered_report, "populate")

    def test_actual_native_job_suffix_cannot_be_substituted_even_if_syntax_matches(self):
        capture, report = failed_case()
        altered = copy.deepcopy(capture)
        row = next(item for item in altered["records"] if item["stage"] == "before_model_forward")
        row["request_id"] = row["request_id"][:-8] + "deadbeef"
        with self.assertRaisesRegex(ValueError, "native jobs"):
            E.validate_capture_cohort(altered, report, "populate")

    def test_exact_two_job_order_and_no_third_job(self):
        with fixture() as (_, _, _, plan, _):
            self.assertEqual(plan.next_job([]), "populate")
            self.assertEqual(plan.next_job([receipt("populate")]), "paired")
            self.assertIsNone(plan.next_job([receipt("populate"), receipt("paired")]))
            for rows in ([receipt("paired")], [receipt("populate")] * 3,
                         [dict(receipt("populate"), session_drained=False)],
                         [dict(receipt("populate"), original_shutdown_completed=False)]):
                with self.assertRaises(ValueError):
                    plan.next_job(rows)

    def test_fixed_context_preserves_original_model_io_and_excludes_latency(self):
        with fixture() as (_, _, _, plan, _):
            context = plan.fixed_context()
            self.assertEqual((context["sizes"], context["acquisition_domain"], context["reps"]), ([128], 1024, 3))
            self.assertEqual(context["requests_by_mode"], {"populate": 9, "paired": 6})
            self.assertEqual(context["total_requests"], 15)
            self.assertEqual(context["connector"]["load_planner"], "off")
            self.assertTrue(context["asynchronous_author_io_preserved"])
            self.assertFalse(context["latency_fit_allowed"])
            self.assertFalse(context["production_qualified"])
            self.assertEqual(context["shared_storage_relative"], E.STORAGE)

    def test_two_job_scope_finite_budget_and_plan_predecessor(self):
        with fixture() as (_, _, _, plan, _):
            row = dict(path=E.DELIVERY + "/test.py", bytes=1, sha256="a" * 64)
            result = plan.build_calibration_plan(new_source_refs=[row])
            self.assertFalse(result["jobs"][0]["requires_prior_success_original_shutdown_and_OS_drain"])
            self.assertTrue(result["jobs"][1]["requires_prior_success_original_shutdown_and_OS_drain"])
            scope = plan.scope_template()
            self.assertEqual(scope["maximum_jobs"], 2)
            self.assertEqual(scope["maximum_total_planned_reserve_seconds"], 640)
            self.assertEqual(scope["allowed_modes"], ["populate", "paired"])
            self.assertFalse(scope["allow_gpu_runs"])

    def test_storage_reserve_reduced_with_unchanged_eight_GiB_floor(self):
        with fixture() as (_, _, _, plan, _):
            ledger = dict(active_reservation=None, gpu_wall_seconds=100, events=[])
            snapshot = dict(origin="actual_statvfs", captured_unix=time.time(), primary_root=plan.PROJECT_ROOT,
                free_bytes=9*1024**3)
            result = plan.validate_budget_and_storage(ledger, snapshot, remaining_jobs=2)
            self.assertEqual(result["remaining_storage_reserve_bytes"], 1024**3)
            self.assertEqual(result["storage_floor_bytes"], 8*1024**3)
            snapshot["free_bytes"] -= 1
            with self.assertRaises(ValueError):
                plan.validate_budget_and_storage(ledger, snapshot, remaining_jobs=2)

    def test_ancestor_preserved_and_authority_cycles_rejected(self):
        with fixture() as (root, _, config, plan, _):
            raw = (root / config["ancestor_ref"]["path"]).read_bytes()
            row = dict(path=E.DELIVERY + "/test.py", bytes=1, sha256="b" * 64)
            manifest = plan.assemble_source_lock(raw, [row])
            plan.validate_source_manifest(manifest, raw)
            self.assertEqual(manifest["files"][:-2], json.loads(raw)["files"])
            manifest["files"][0]["sha256"] = "f" * 64
            with self.assertRaises(ValueError):
                plan.validate_source_manifest(manifest, raw)
            with self.assertRaises(ValueError):
                plan.assemble_source_lock(raw, [dict(row, path=E.DELIVERY + "/HUMAN_AUTHORIZATION.json")])

    def test_actual_runtime_globals_and_original_fixed_argv_preserved(self):
        with fixture() as (root, refs, config, plan, _):
            runtime = E.DiagnosticRuntime(root, refs, config)
            with runtime.session() as original:
                common = original.load_source(root, refs, original.COMMON)
                self.assertIs(original.execute_original_acquisition.__globals__, original.__dict__)
                self.assertEqual(original.execute_original_acquisition.__globals__["GPU_UUID"], E.GPU_UUID)
                self.assertEqual(common.SDK_INVENTORY, config["sdk_inventory_ref"]["path"])
                argv = original.fixed_argv("populate", "out", "storage", "model", "plan")
                self.assertEqual(argv, ["acquire_native_aio_costs.py", "--output-dir", "out", "--storage", "storage",
                    "--model-dir", "model", "--model-plan", "plan", "--mode", "populate", "--domain", "1024",
                    "--sizes", "128", "--reps", "3", "--max-num-seqs", "1", "--iodepth", "4"])
            self.assertEqual(runtime.last_state["status"], "RESTORED")
            self.assertEqual(original.GPU_UUID, "GPU-54ace1eb-dc9d-3fee-d720-2051f5d337a2")
            self.assertNotIn(original.__name__, sys.modules)
            self.assertNotIn(common.__name__, sys.modules)

    def test_original_guard_new_identity_and_old_identity_rejection(self):
        with fixture() as (root, refs, config, plan, _):
            runtime = E.DiagnosticRuntime(root, refs, config)
            with runtime.session() as original:
                guard_globals = original.validate_guard.__globals__
                fake_os = types.SimpleNamespace(name="posix", getsid=lambda _: 88, environ={"CUDA_VISIBLE_DEVICES": E.GPU_UUID})
                old_os = guard_globals["os"]
                guard_globals["os"] = fake_os
                try:
                    witness = dict(schema_version=1, purpose=E.PURPOSE, gpu_uuid=E.GPU_UUID,
                        label=E.JOBS["populate"], source_lock_sha256="a"*64, scope_sha256="b"*64,
                        permissions_ref=config["permissions_ref"], session_id=88, guard_command=["python", "--execute"],
                        seconds_limit=300, reserved_seconds=320, authorized_scope_verified=True, active_reservation_verified=True)
                    original.validate_guard(witness, "populate")
                    with self.assertRaises(ValueError):
                        original.validate_guard(dict(witness, gpu_uuid="GPU-old"), "populate")
                finally:
                    guard_globals["os"] = old_os


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *REST])
