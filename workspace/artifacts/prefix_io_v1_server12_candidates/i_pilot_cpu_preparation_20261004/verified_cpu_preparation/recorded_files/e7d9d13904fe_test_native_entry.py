"""CPU rejection and source preservation checks; no native positive attestation."""
import ast
import importlib.util
import json
from hashlib import sha256
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SHARED_COMMON = Path(os.environ.get('C5_NATIVE_SHARED_COMMON',
    str(HERE.parents[2] / 'artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/common_candidate'))).resolve()
FORBIDDEN_IMPORTS = []


class NoGpuImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in ("torch", "vllm", "py_kvcache", "prefix_io_control"):
            FORBIDDEN_IMPORTS.append(fullname)
            raise ImportError("CPU boundary tests prohibit GPU/model/author imports: " + fullname)
        return None


sys.meta_path.insert(0, NoGpuImports())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


N = module("cpu_native_entry_contract", HERE / "native_conditional_cost.py")
S = module("cpu_native_entry_serializer", HERE / "prepare_and_verify_native_cost.py")
R = module("cpu_native_entry_runtime", HERE / "run_native_cost_experiment.py")
Q = module("cpu_native_entry_receipt", HERE / "p4_single_file_receipt.py")


def write(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return dict(path=relative, bytes=len(raw), sha256=sha256(raw).hexdigest())


class NativeEntryTests(unittest.TestCase):
    def test_preregistered_synthetic_origin_is_rejected_before_site_access(self):
        with patch.object(N, "binding_api", side_effect=AssertionError("site touched")):
            for value in ({}, {"evidence_origin": "synthetic_cpu_contract"},
                          {"evidence_origin": "native_runtime_preregistered", "cpu_preparation_only": True},
                          {"evidence_origin": "native_gpu_recording", "job_id": N.ENTRY_LABEL}):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "synthetic"):
                    N.validate_native_plan(HERE, value)

    def test_old_v6_and_preparation_jobs_cannot_be_promoted(self):
        for job in ("server11-native-cost-six-window-06", "server11-c5-native-common-cost-preparation01"):
            with self.subTest(job=job), self.assertRaisesRegex(ValueError, "old native plans"):
                N.validate_native_plan(HERE, dict(evidence_origin="native_runtime_preregistered",
                    cpu_preparation_only=False, job_id=job))

    def test_native_plan_requires_real_site_configuration(self):
        plan = dict(evidence_origin="native_runtime_preregistered", cpu_preparation_only=False,
                    job_id=N.ENTRY_LABEL, project_root=HERE.as_posix())
        with patch.object(N, "binding_api") as api:
            api.return_value.verify_configuration.side_effect = ValueError("unresolved actual GPU site")
            with self.assertRaisesRegex(ValueError, "unresolved actual GPU"):
                N.validate_native_plan(HERE, plan)
            self.assertEqual(api.return_value.verify_configuration.call_count, 1)

    def test_runtime_load_rejects_before_source_module_import(self):
        with patch.object(R, "binding_api") as api, patch.object(R, "load_source") as loader:
            api.return_value.verify_configuration.side_effect = ValueError("no fresh source-bound grant")
            with self.assertRaisesRegex(ValueError, "fresh source-bound"):
                R.load_configuration(HERE / "missing.json")
            loader.assert_not_called()

    def test_direct_parent_execution_obeys_site_gate(self):
        with patch.object(R, "ROOT", HERE), patch.object(R, "binding_api") as api, \
                patch.object(R.subprocess, "run") as process:
            api.return_value.verify_configuration.side_effect = ValueError("site configuration denied")
            with self.assertRaisesRegex(ValueError, "site configuration denied"):
                R.execute_parent(HERE, {}, {}, {})
            process.assert_not_called()

    def test_direct_window_execution_obeys_site_gate_before_storage(self):
        with patch.object(R, "ROOT", HERE), patch.object(R, "binding_api") as api, \
                patch.object(R, "prepare_window_storage") as storage:
            api.return_value.verify_configuration.side_effect = ValueError("unknown UUID")
            with self.assertRaisesRegex(ValueError, "unknown UUID"):
                R.execute_window(HERE, {}, {}, {}, 0)
            storage.assert_not_called()

    def test_guard_only_delegates_to_real_reservation_validator(self):
        config = dict(gpu_entry_binding_ref=dict(path="binding.json", bytes=1, sha256="0" * 64))
        with patch.object(R, "binding_api") as api:
            api.return_value.verify_active_guard.side_effect = ValueError("no active original reservation")
            with self.assertRaisesRegex(ValueError, "no active original"):
                R.verify_guard(config)

    def test_prepare_does_not_write_without_real_site_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(S.C, "binding_api") as api:
                api.return_value.verify_configuration.side_effect = ValueError("grant absent")
                with self.assertRaisesRegex(ValueError, "grant absent"):
                    S.create_plan(root, source_lock_relative="lock.json", config_relative="config.json",
                                  output_relative="plan.json")
            self.assertFalse((root / "plan.json").exists())

    def test_serializer_refuses_preparation_plan_before_runtime_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            ref = write(root, "plan.json", dict(evidence_origin="synthetic_cpu_contract"))
            with self.assertRaisesRegex(ValueError, "synthetic"):
                S.serialize_runtime_record(root, runtime_relative="missing-runtime.json", plan_ref=ref,
                                           source_verification_refs={})

    def test_raw_window_refuses_preparation_plan_before_runtime_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            ref = write(root, "plan.json", dict(cpu_preparation_only=True))
            with self.assertRaisesRegex(ValueError, "synthetic"):
                S.verify_raw_window(root, plan_ref=ref, child_relative="missing-child.json")

    def test_final_verifier_refuses_preparation_plan_before_guard_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            ref = write(root, "plan.json", dict(evidence_origin="synthetic_cpu_contract"))
            missing = dict(path="missing.json", bytes=1, sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "synthetic"):
                N.verify_native_cell(root, plan_ref=ref, measurements_ref=missing, guard_ref=missing,
                    expected_plan_ref=ref, expected_guard_ref=missing, original_source_path="missing.py")

    def test_serializer_has_no_synthetic_cpu_switch(self):
        with self.assertRaises(TypeError):
            S.serialize_runtime_record(HERE, runtime_relative="a", plan_ref={}, source_verification_refs={},
                                       synthetic_cpu=True)

    def test_final_verifier_rejects_plan_not_pinned_by_launch_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan = write(root, "plan.json", {"job_id": N.ENTRY_LABEL})
            missing = dict(path="missing.json", bytes=1, sha256="0" * 64)
            with patch.object(N, "validate_native_plan", return_value=({}, {}, {})), \
                    patch.object(N, "binding_api") as api:
                api.return_value.read.return_value = dict(plan_ref=missing)
                with self.assertRaisesRegex(ValueError, "independent prelaunch intent"):
                    N.verify_native_cell(root, plan_ref=plan, measurements_ref=missing, guard_ref=missing,
                        expected_plan_ref=plan, expected_guard_ref=missing, original_source_path="missing.py")
                api.return_value.verify_completed_guard.assert_not_called()

    def test_serializer_rejects_plan_not_pinned_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan = write(root, "plan.json", {"job_id": N.ENTRY_LABEL})
            with patch.object(S.C, "validate_native_plan", return_value=({}, {}, {})), \
                    patch.object(S.C, "binding_api") as api:
                api.return_value.read.return_value = dict(plan_ref={"old": True})
                with self.assertRaisesRegex(ValueError, "independent prelaunch intent"):
                    S.serialize_runtime_record(root, runtime_relative="missing-runtime.json", plan_ref=plan,
                                               source_verification_refs={})

    def test_raw_window_rejects_postlaunch_self_referencing_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan = write(root, "plan.json", {"job_id": N.ENTRY_LABEL})
            with patch.object(S.C, "validate_native_plan", return_value=({}, {}, {})), \
                    patch.object(S.C, "binding_api") as api:
                api.return_value.read.return_value = dict(plan_ref={"old": True})
                with self.assertRaisesRegex(ValueError, "independent prelaunch intent"):
                    S.verify_raw_window(root, plan_ref=plan, child_relative="missing-child.json")

    def test_exact_receipt_private_constructor_refuses_pass(self):
        for value in (None, {}, {"PASS": True}, {"native_execution_verified": True}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Q.ExactSingleFileReceipt(value)

    def test_canonical_issuer_rejects_synthetic_plan_before_serializer_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan = write(root, "plan.json", dict(evidence_origin="synthetic_cpu_contract",
                cpu_preparation_only=True, job_id=N.ENTRY_LABEL, units=1, operations=1,
                transfer_quantum_bytes=917504))
            data = write(root, "data.json", {"origin": "native_gpu_recording"})
            write(root, "binding.json", dict(schema_version=1, scope="p4_single_file_native_binding_v1",
                calibration=dict(plan_ref=plan, measurements_ref=data, guard_ref=data),
                runtime_common_refs=[], runtime_overlay_refs=[], kernel_mode=Q._KERNEL_MODE))
            with patch.object(Q, "_load_frozen_serializer", side_effect=AssertionError("serializer touched")):
                with self.assertRaisesRegex(ValueError, "synthetic"):
                    Q.load_verified_single_file(root, "binding.json")

    def test_canonical_verifier_and_serializer_pins_match_actual_sources(self):
        for row, name in ((Q._VERIFIER, "native_conditional_cost.py"),
                          (Q._SERIALIZER, "prepare_and_verify_native_cost.py")):
            raw = (HERE / name).read_bytes()
            self.assertEqual((row.bytes, row.sha256), (len(raw), sha256(raw).hexdigest()))
            self.assertEqual(row.path, N.ENTRY + "/" + name)

    def test_fixed_geometry_and_order_unchanged(self):
        self.assertEqual((R.ORDER, R.PROMPT_TOKENS, R.OPERATION_COUNT, R.FILE_BYTES,
                          R.MAX_ACCEPTED_PARENTS),
                         ((("A", "B"), ("B", "A"), ("A", "B")), 129, 1, 917504, 8))
        self.assertEqual(R.PROMPT_FIRST, (40100, 41100, 42100))
        self.assertEqual(R.SEEDS, (4029, 4030, 4031))
        self.assertNotIn("GPU_UUID", vars(R))

    def test_actual_identity_helper_copied_byte_exact(self):
        raw = (SHARED_COMMON / "single_file_runtime_binding.py").read_bytes()
        self.assertEqual(sha256(raw).hexdigest(),
                         "8b66b3aba1c936b85f4d82fe73c568650b712acf250aae454823cd35da060d60")

    def test_numerical_and_model_helpers_preserve_parent_asts(self):
        inheritance = json.loads((HERE / "NATIVE_SOURCE_INHERITANCE.json").read_bytes())
        for row in inheritance["unchanged_function_asts"]:
            tree = ast.parse((HERE / row["file"]).read_bytes())
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == row["function"])
            actual = sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            self.assertEqual(actual, row["sha256"], row["file"] + "::" + row["function"])

    def test_reactor_collector_policy_bridge_unchanged(self):
        inheritance = json.loads((HERE / "NATIVE_SOURCE_INHERITANCE.json").read_bytes())
        for row in inheritance["unchanged_common_files"]:
            raw = (SHARED_COMMON / row["path"]).read_bytes()
            self.assertEqual((len(raw), sha256(raw).hexdigest()), (row["bytes"], row["sha256"]), row["path"])

    def test_import_does_not_load_gpu_or_author_packages(self):
        names = [name for name in sys.modules if name == "torch" or name.startswith("torch.") or
                 name == "vllm" or name.startswith("vllm.") or name == "py_kvcache"]
        self.assertEqual(names, [])
        self.assertEqual(FORBIDDEN_IMPORTS, [])


if __name__ == "__main__":
    program = unittest.main(exit=False)
    loaded = sorted(name for name in sys.modules if name.split(".", 1)[0] in
                    ("torch", "vllm", "py_kvcache", "prefix_io_control"))
    attempts = sorted(set(FORBIDDEN_IMPORTS))
    success = program.result.wasSuccessful() and not program.result.skipped and not attempts and not loaded
    print(json.dumps(dict(scope="CPU_NATIVE_ENTRY_IMPORT_AUDIT", status="PASS" if success else "FAILED",
        forbidden_import_attempts=attempts, loaded_forbidden_modules=loaded,
        tests=program.result.testsRun, failures=len(program.result.failures),
        errors=len(program.result.errors), skipped=len(program.result.skipped)), sort_keys=True))
    raise SystemExit(0 if success else 1)
