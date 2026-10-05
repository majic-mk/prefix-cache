"""Finite CPU-only contracts for metadata-first reference diagnosis.

All altered constants below are explicit temporary CPU fixtures. Production
source pins, driver, guard, old artifacts, scopes and budget are never changed.
The simulated original entry verifies fixture driver bytes before CPU readiness.
No framework, GPU, compiler, remote call or subprocess is used by these tests.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tested_fast_reference_CPU_only", HERE / "g3_fast_reference_preflight.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


class FastReferenceCPU(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="explicit-fast-reference-CPU-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.driver = self.root / "CPU-fixture-driver.bin"
        self.good_bytes = b"CPU_FIXTURE_NOT_A_REAL_DRIVER"
        self.driver.write_bytes(self.good_bytes)
        self.driver_ref = dict(path=str(self.driver), bytes=len(self.good_bytes),
            sha256=hashlib.sha256(self.good_bytes).hexdigest())
        self.pins = {}
        self.entry_data = dict(status="CPU_READY_NO_NEW_REFERENCE_HUMAN_SCOPE", source_count=4088,
            actual_gpu_runs=0, GPU_initialized=False, framework_imported=False,
            production_qualified=False, performance_claim=False)
        self.original_failure = dict(status="BLOCKED_REFERENCE_CPU_OR_AUTHORITY_GATE",
            reason="explicit CPU fixture same-size driver SHA drift", actual_gpu_runs=0,
            GPU_initialized=False, production_qualified=False, performance_claim=False)
        self.write_entry("checks_driver")
        self.write_pin("SDK_REF", "CPU-fixture-sdk.py", b'"""Unexecuted CPU-only fixture SDK bytes."""\n')
        inventory = dict(status="CPU_ONLY_EXISTING_CUDA13_SOURCE_INVENTORY", driver=self.driver_ref,
            fixture_origin="CPU_FIXTURE_NOT_GPU")
        proof = dict(status="PASS_CPU_CUDA13_SM120F_COMPILE_AND_HOST_LINK_ONLY", driver_ref=self.driver_ref,
            fixture_origin="CPU_FIXTURE_NOT_GPU")
        self.write_pin("INVENTORY_REF", "CPU-fixture-inventory.json", self.json_bytes(inventory))
        self.write_pin("PROOF_REF", "CPU-fixture-proof.json", self.json_bytes(proof))
        self.manifest = dict(allow_gpu_runs=False, allow_gpu_initialization=False,
            fixture_origin="CPU_FIXTURE_NOT_GPU", files=[self.pins[k] for k in
                ("ENTRY_REF", "INVENTORY_REF", "PROOF_REF", "SDK_REF")])
        # Declaration-only fixture leaf rows deliberately have no real files:
        # the early gate must not touch model trees. The original CPU entry
        # simulated below is the only positive completion authority in tests.
        self.manifest["files"] += [dict(path="unread-model/fixture-%04d.bin" % i,
            bytes=1024**3, sha256="0" * 64) for i in range(4084)]
        self.write_pin("LOCK_REF", "CPU-fixture-lock.json", self.json_bytes(self.manifest))
        self.entry_data["source_lock"] = dict(self.pins["LOCK_REF"])
        self.stack = contextlib.ExitStack(); self.addCleanup(self.stack.close)
        for key, value in self.pins.items():
            self.stack.enter_context(mock.patch.object(F, key, value))
        self.stack.enter_context(mock.patch.object(F, "DRIVER_REF", self.driver_ref))
        self.stack.enter_context(mock.patch.object(F, "resource_snapshot", return_value={
            "origin": "explicit_CPU_fixture", "cpu_quota_cores": 0.5, "memory_limit_bytes": 2 * 1024**3}))
        self.stack.enter_context(mock.patch.object(F, "CPU_FIXTURE_ORIGINAL_DATA", self.entry_data, create=True))
        self.stack.enter_context(mock.patch.object(F, "CPU_FIXTURE_ORIGINAL_FAILURE", self.original_failure, create=True))
        self.stack.enter_context(mock.patch.object(F, "CPU_FIXTURE_DRIVER", self.driver_ref, create=True))
        name = "_fast_reference_original_cpu_entry"
        self.assertNotIn(name, sys.modules)
        self.addCleanup(sys.modules.pop, name, None)

    @staticmethod
    def json_bytes(value):
        return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")

    def write_pin(self, name, relative, raw):
        path = self.root / relative
        path.write_bytes(raw)
        self.pins[name] = dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    def write_entry(self, behavior):
        # A source-bound fixture of the original entry, without backend imports.
        # Reading the test helper namespace supplies only explicit CPU data.
        header = "import hashlib,json,sys\nfrom pathlib import Path\n"
        header += "F = sys.modules[" + repr(spec.name) + "]\n"
        body = "def main(argv):\n    if '--preflight' not in argv:\n        raise AssertionError('fixture only permits original CPU preflight')\n"
        if behavior == "checks_driver":
            body += "    pin=F.CPU_FIXTURE_DRIVER\n    actual=hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()\n"
            body += "    code=0 if actual == pin['sha256'] else 78\n    data=F.CPU_FIXTURE_ORIGINAL_DATA if code == 0 else F.CPU_FIXTURE_ORIGINAL_FAILURE\n"
        elif behavior == "failure":
            body += "    code=78\n    data=F.CPU_FIXTURE_ORIGINAL_FAILURE\n"
        elif behavior == "bad_code_type":
            body += "    code=False\n    data=F.CPU_FIXTURE_ORIGINAL_DATA\n"
        else:
            body += "    code=0\n    data=F.CPU_FIXTURE_ORIGINAL_DATA\n"
        body += "    print(json.dumps(data))\n    return code\n"
        self.write_pin("ENTRY_REF", "CPU-fixture-original-entry.py", (header + body).encode("utf-8"))

    def entry_behavior(self, behavior):
        old = dict(F.ENTRY_REF)
        self.write_entry(behavior)
        F.ENTRY_REF = self.pins["ENTRY_REF"]
        self.manifest["files"] = [self.pins["ENTRY_REF"] if row == old else row for row in self.manifest["files"]]
        self.write_pin("LOCK_REF", "CPU-fixture-lock.json", self.json_bytes(self.manifest))
        F.LOCK_REF = self.pins["LOCK_REF"]
        self.entry_data["source_lock"] = dict(F.LOCK_REF)

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file() and not p.is_symlink()}

    def assert_no_gpu_qualification(self, result):
        for key in ("GPU_initialized", "framework_imported", "numerical_reference_verified",
                    "production_qualified", "performance_claim", "new_GPU_authorization_created"):
            self.assertIs(result[key], False)
        self.assertEqual(result["actual_gpu_runs"], 0)
        self.assertEqual(result["GPU_metadata_queries"], 0)

    def assert_blocked_no_original(self):
        before = self.snapshot()
        with mock.patch.object(F, "original_full_preflight") as original:
            code, result = F.diagnose(self.root)
            self.assertEqual(code, 78)
            self.assertEqual(result["status"], "BLOCKED_DRIVER_ASSET_GATE")
            self.assertIs(result["full_source_bytes_verified"], False)
            self.assertIs(result["runtime_assets_fully_verified"], False)
            self.assertIs(result["driver_SHA_verified"], False)
            self.assertEqual(result["profile"]["model_bytes_read_by_early_gate"], 0)
            self.assertEqual(result["profile"]["full_source_checks_called"], 0)
            self.assertGreaterEqual(result["early_elapsed_seconds"], 0)
            self.assert_no_gpu_qualification(result)
            original.assert_not_called()
        self.assertEqual(before, self.snapshot())
        return result

    def test_zero_byte_driver_blocks_before_original_and_reports_expected_actual(self):
        self.driver.write_bytes(b"")
        result = self.assert_blocked_no_original()
        self.assertEqual(result["driver_expected"], self.driver_ref)
        self.assertEqual(result["driver_observed"]["bytes"], 0)

    def test_missing_driver_blocks_before_original(self):
        self.driver.unlink()
        result = self.assert_blocked_no_original()
        self.assertFalse(result["driver_observed"]["exists"])
        self.assertEqual(result["driver_observed"]["error"], "FileNotFoundError")

    def test_symlink_driver_blocks_before_original_even_if_target_size_matches(self):
        target = self.root / "CPU-fixture-driver-target.bin"
        target.write_bytes(self.good_bytes)
        self.driver.unlink()
        try:
            self.driver.symlink_to(target)
        except OSError as error:
            self.skipTest("native test symlink unavailable: " + type(error).__name__)
        result = self.assert_blocked_no_original()
        self.assertIs(result["driver_observed"]["symlink"], True)

    def test_directory_driver_blocks_before_original(self):
        self.driver.unlink(); self.driver.mkdir()
        result = self.assert_blocked_no_original()
        self.assertIs(result["driver_observed"]["regular"], False)

    def test_metadata_match_is_only_continue_never_ready_or_byte_verified(self):
        before = self.snapshot()
        with mock.patch.object(F, "original_full_preflight") as original:
            code, result = F.diagnose(self.root, quick_only=True)
            self.assertEqual(code, 0)
            self.assertEqual(result["status"], "CONTINUE_ORIGINAL_FULL_PREFLIGHT_REQUIRED")
            for key in ("driver_SHA_verified", "full_source_bytes_verified", "runtime_assets_fully_verified"):
                self.assertIs(result[key], False)
            self.assertEqual(result["profile"]["model_bytes_read_by_early_gate"], 0)
            self.assertEqual(result["profile"]["full_source_checks_called"], 0)
            self.assert_no_gpu_qualification(result)
            original.assert_not_called()
        self.assertEqual(before, self.snapshot())

    def test_same_size_bad_driver_reaches_original_byte_check_once_and_is_rejected(self):
        self.driver.write_bytes(b"X" * len(self.good_bytes))
        before = self.snapshot()
        with mock.patch.object(F, "original_full_preflight", wraps=F.original_full_preflight) as original:
            code, result = F.diagnose(self.root)
            self.assertEqual(code, 78)
            self.assertEqual(result["original_full_preflight"], self.original_failure)
            self.assertEqual(result["status"], self.original_failure["status"])
            self.assertIs(result["full_source_bytes_verified"], False)
            self.assertIs(result["runtime_assets_fully_verified"], False)
            self.assertEqual(result["profile"]["full_source_checks_called"], 1)
            self.assert_no_gpu_qualification(result)
            original.assert_called_once()
        self.assertEqual(before, self.snapshot())

    def test_valid_match_runs_original_exactly_once_and_only_marks_complete_cpu_checks(self):
        before = self.snapshot()
        with mock.patch.object(F, "original_full_preflight", wraps=F.original_full_preflight) as original:
            code, result = F.diagnose(self.root)
            self.assertEqual(code, 0)
            self.assertEqual(result["status"], "CPU_READY_NO_NEW_REFERENCE_HUMAN_SCOPE")
            self.assertEqual(result["original_full_preflight"], self.entry_data)
            self.assertIs(result["full_source_bytes_verified"], True)
            self.assertIs(result["runtime_assets_fully_verified"], True)
            self.assertIs(result["driver_SHA_verified"], True)
            self.assertEqual(result["profile"]["full_source_checks_called"], 1)
            self.assert_no_gpu_qualification(result)
            original.assert_called_once()
        self.assertEqual(before, self.snapshot())

    def test_original_source_failure_reason_code_and_unknown_source_flags_are_preserved(self):
        self.entry_behavior("failure")
        self.original_failure["reason"] = "explicit CPU original full model/source drift"
        with mock.patch.object(F, "original_full_preflight", wraps=F.original_full_preflight) as original:
            code, result = F.diagnose(self.root)
            self.assertEqual(code, 78)
            self.assertEqual(result["original_full_preflight"], self.original_failure)
            self.assertIs(result["full_source_bytes_verified"], False)
            self.assert_no_gpu_qualification(result)
            original.assert_called_once()

    def test_frozen_lock_sha_drift_rejected_before_driver_metadata_or_original(self):
        path = self.root / F.LOCK_REF["path"]
        raw = path.read_bytes(); path.write_bytes(b" " + raw[1:])
        with mock.patch.object(F, "driver_metadata") as metadata, mock.patch.object(F, "original_full_preflight") as original:
            with self.assertRaisesRegex(ValueError, "CPU pin SHA drift"):
                F.diagnose(self.root)
            metadata.assert_not_called(); original.assert_not_called()

    def test_sdk_byte_drift_rejected_before_driver_metadata_or_original(self):
        path = self.root / F.SDK_REF["path"]
        path.write_bytes(b"X" * path.stat().st_size)
        with mock.patch.object(F, "driver_metadata") as metadata, mock.patch.object(F, "original_full_preflight") as original:
            with self.assertRaisesRegex(ValueError, "CPU pin SHA drift"):
                F.diagnose(self.root)
            metadata.assert_not_called(); original.assert_not_called()

    def test_inventory_and_proof_sha_drift_each_rejected_before_driver_metadata(self):
        for key in ("INVENTORY_REF", "PROOF_REF"):
            path = self.root / getattr(F, key)["path"]
            raw = path.read_bytes()
            try:
                path.write_bytes(b"X" * len(raw))
                with self.subTest(pin=key), mock.patch.object(F, "driver_metadata") as metadata, \
                     mock.patch.object(F, "original_full_preflight") as original:
                    with self.assertRaisesRegex(ValueError, "CPU pin SHA drift"):
                        F.diagnose(self.root)
                    metadata.assert_not_called(); original.assert_not_called()
            finally:
                path.write_bytes(raw)

    def test_declared_entry_pin_change_rejected_before_any_driver_or_full_check(self):
        self.manifest["files"][0] = dict(F.ENTRY_REF, sha256="9" * 64)
        self.write_pin("LOCK_REF", "CPU-fixture-lock.json", self.json_bytes(self.manifest))
        with mock.patch.object(F, "LOCK_REF", self.pins["LOCK_REF"]), mock.patch.object(F, "driver_metadata") as metadata, \
             mock.patch.object(F, "original_full_preflight") as original:
            with self.assertRaisesRegex(ValueError, "exact original entry/SDK pins"):
                F.diagnose(self.root)
            metadata.assert_not_called(); original.assert_not_called()

    def test_same_size_entry_source_drift_rejected_before_any_entry_code_executes(self):
        path = self.root / F.ENTRY_REF["path"]
        path.write_bytes(b"X" * path.stat().st_size)
        profile = dict(small_pin_bytes_read=0, full_source_checks_called=0)
        with self.assertRaisesRegex(ValueError, "CPU pin SHA drift"):
            F.original_full_preflight(self.root, profile)
        self.assertEqual(profile["full_source_checks_called"], 0)
        self.assertNotIn("_fast_reference_original_cpu_entry", sys.modules)

    def test_fake_original_ready_requires_exact_lock_count_init_status_and_no_GPU_runs(self):
        self.entry_behavior("always_ready")
        for key, bad in (("source_lock", dict(F.LOCK_REF, sha256="8" * 64)), ("source_count", 4087),
                         ("GPU_initialized", True), ("status", "PASSED_GPU_RUN"), ("actual_gpu_runs", 1)):
            saved = self.entry_data[key]
            try:
                self.entry_data[key] = bad
                with self.subTest(field=key), self.assertRaisesRegex(ValueError, "only original complete preflight"):
                    F.diagnose(self.root)
                self.assertNotIn("_fast_reference_original_cpu_entry", sys.modules)
            finally:
                self.entry_data[key] = saved
        self.entry_behavior("bad_code_type")
        with self.assertRaisesRegex(ValueError, "original CPU preflight result"):
            F.diagnose(self.root)

    def test_CLI_has_no_launch_or_execute_or_scope_option_and_calls_no_checks(self):
        for unsupported in ("--launch", "--execute", "--scope-record"):
            with self.subTest(option=unsupported), mock.patch.object(F, "diagnose") as diagnose, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                F.main(["--project", str(self.root), unsupported])
            self.assertEqual(error.exception.code, 2)
            diagnose.assert_not_called()

    def test_CLI_pin_failure_has_false_qualification_and_no_environment_module_or_file_mutation(self):
        import os
        before = self.snapshot(); env = dict(os.environ); modules = set(sys.modules)
        path = self.root / F.SDK_REF["path"]
        path.write_bytes(b"X" * path.stat().st_size)
        expected = self.snapshot()
        with mock.patch.object(F, "original_full_preflight") as original, contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(F.main(["--project", str(self.root)]), 78)
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["status"], "BLOCKED_FAST_CPU_PIN_OR_PATH_GATE")
            self.assertIn("CPU pin SHA drift", result["reason"])
            self.assertIs(result["full_source_bytes_verified"], False)
            self.assert_no_gpu_qualification(result)
            original.assert_not_called()
        self.assertEqual(expected, self.snapshot())
        self.assertNotEqual(before, expected)  # Only explicit fixture mutation.
        self.assertEqual(env, dict(os.environ))
        self.assertEqual(modules, set(sys.modules))


if __name__ == "__main__":
    # The only helper exposed to the source-bound fake original entry is this
    # CPU fixture module; the production entry never imports it.
    sys.modules[spec.name] = F
    unittest.main(verbosity=2)
