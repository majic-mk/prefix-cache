"""Independent CPU refusal tests of the finite issuer; never mint GPU proof."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ACTIVATION = HERE.parent / "activation"
sys.path.insert(0, str(ACTIVATION / "source"))
from prefix_io_control import gpu_cell_issuer as I
from prefix_io_control import p4_cost_table as C
from prefix_io_control.dispatch_budget import ZERO

spec = importlib.util.spec_from_file_location("_independent_native_math_review", ACTIVATION / "native_conditional_cost.py")
V = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = V
spec.loader.exec_module(V)


def byte_ref(root, path):
    raw = path.read_bytes()
    return dict(path=path.relative_to(root).as_posix(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


class FiniteCapabilityRefusal(unittest.TestCase):
    def setUp(self):
        self.cell = C.CostCell(("CPU-fixture-only", 1), ZERO, "ssd_read", 917504,
                               "existing_io_plus_delta", 100, 20, 5)
        self.assertEqual(len(I._ISSUED), 0)

    def tearDown(self):
        self.assertEqual(len(I._ISSUED), 0, "a CPU refusal/maths test must never mint actual GPU evidence")

    def test_public_scope_cannot_claim_gpu(self):
        with self.assertRaises(ValueError):
            C.CostTable((self.cell,), scope="gpu_verified_exact_cells", source_sha256="0" * 64)

    def test_mock_lookup_keeps_original_closed_production(self):
        table = C.CostTable((self.cell,), scope="mock_only", source_sha256="0" * 64)
        self.assertFalse(table.production_qualified)
        self.assertIsNone(table.lookup(self.cell.load_signature, ZERO, "ssd_read", 917504))
        value = table.lookup(self.cell.load_signature, ZERO, "ssd_read", 917504, execution="cpu_mock")
        self.assertEqual(value.total_ns, 125)
        self.assertTrue(value.mock_only)
        self.assertFalse(value.production_qualified)

    def test_conditional_lookup_keeps_original_closed_production(self):
        table = C.CostTable((self.cell,), scope="conditional", source_sha256="0" * 64)
        self.assertFalse(table.production_qualified)
        self.assertIsNone(table.lookup(self.cell.load_signature, ZERO, "ssd_read", 917504))
        self.assertIsNone(table.lookup(self.cell.load_signature, ZERO, "ssd_read", 917504, execution="cpu_mock"))

    def test_report_pass_and_gpu_boolean_do_not_qualify_old_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(dict(status="PASS", origin="native_gpu_recording", production_qualified=True)), encoding="utf-8")
            table = C.load_conditional_table(path)
            self.assertFalse(table.production_qualified)
            self.assertIsNone(I.qualified_identity(table))

    def test_unissued_object_cannot_construct_real_table(self):
        with self.assertRaises(ValueError):
            C._from_verified_gpu_capability(object())

    def test_forged_scope_without_private_proof_cannot_open_lookup(self):
        table = C.CostTable((self.cell,), scope="conditional", source_sha256="0" * 64)
        object.__setattr__(table, "scope", "gpu_verified_exact_cells")
        self.assertFalse(table.production_qualified)
        self.assertIsNone(table.lookup(self.cell.load_signature, ZERO, "ssd_read", 917504))

    def test_origin_text_cannot_replace_complete_actual_raw_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "plan.json"
            path.write_text(json.dumps(dict(schema="strong_gpu_exact_cell_prelaunch_plan_v1",
                evidence_origin="native_runtime_preregistered", cpu_preparation_only=False, synthetic_fixture=False,
                project_root=root.as_posix(), gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc",
                source_lock_ref=dict(path="absent-lock.json", bytes=10, sha256="0" * 64),
                common_runtime_domain_sha256="1" * 64)), encoding="utf-8")
            plan_ref = byte_ref(root, path)
            absent = dict(path="absent.json", bytes=10, sha256="0" * 64)
            with self.assertRaises(ValueError):
                I.issue_verified_gpu_table(root, plan_ref=plan_ref, measurements_ref=absent,
                    guard_ref=absent, intent_ref=absent, expected_plan_ref=plan_ref,
                    expected_guard_ref=absent, expected_intent_ref=absent,
                    expected_runtime_domain_sha256="1" * 64,
                    expected_gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc",
                    expected_source_lock_sha256="0" * 64)

    def test_duplicate_raw_keys_and_byte_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "raw.json"
            path.write_bytes(b'{"origin":"native_gpu_recording","origin":"cpu_fixture"}')
            ref = byte_ref(root, path)
            with self.assertRaises(ValueError):
                I.json_ref(root, ref)
            path.write_bytes(b'{"origin":"native_gpu_recording","origin":"gpu_fixture"}')
            with self.assertRaises(ValueError):
                I.read_ref(root, ref)

    def test_heldout_never_changes_frozen_calibration_upper(self):
        source = ACTIVATION / "source" / "prefix_io_control" / "p4_paired_measurement_verifier.py"
        reference = dict(path="collector.py", bytes=1, sha256="0" * 64)
        common = dict(calibration_A_ns=[100, 110], calibration_B_ns=[120, 130], reference_source_ref=reference)
        low = I.replay_math_audit(V, source, heldout_B_ns=1, **common)
        high = I.replay_math_audit(V, source, heldout_B_ns=10000, **common)
        self.assertEqual(low["calibration"], high["calibration"])
        self.assertEqual(low["upper_ns"], high["upper_ns"])
        self.assertTrue(low["heldout_covered"])
        self.assertFalse(high["heldout_covered"])
        self.assertFalse(high["holdout_used_to_fit"])
        self.assertFalse(low["production_qualified"])
        self.assertFalse(high["actual_gpu_execution_proved"])


def module_refs():
    return [dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in (Path(I.__file__), Path(C.__file__), Path(V.__file__),
                         ACTIVATION / "source" / "prefix_io_control" / "p4_paired_measurement_verifier.py")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = module_refs()
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FiniteCapabilityRefusal))
    after = module_refs()
    forbidden = [name for name in sys.modules if name == "torch" or name == "vllm" or name == "py_kvcache" or name.startswith(("torch.", "vllm.", "py_kvcache."))]
    passed = result.wasSuccessful() and before == after and not forbidden and not I._ISSUED
    document = dict(schema="independent_finite_capability_CPU_refusal_review_v1",
        status="PASS_CPU_REFUSALS_AND_HELDOUT_MATH_ONLY" if passed else "FAIL",
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        source_refs_before=before, source_refs_after=after, source_bytes_unchanged=before == after,
        actual_GPU_runs=0, actual_RPC_calls=0, actual_issued_GPU_capabilities=len(I._ISSUED),
        actual_qualified_cells=0, actual_native_running_identity_verified=False,
        forbidden_runtime_imports=forbidden, test_log=log.getvalue())
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: document[key] for key in ("status", "tests_run", "failures", "errors", "skipped", "actual_GPU_runs", "actual_issued_GPU_capabilities")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
