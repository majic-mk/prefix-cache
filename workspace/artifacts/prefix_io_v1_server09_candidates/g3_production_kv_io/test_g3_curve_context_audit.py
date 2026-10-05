"""Synthetic negatives for context audit; no GPU curves are fabricated for use."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("g3_curve_context_audit", Path(__file__).with_name("g3_curve_context_audit.py"))
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


class ContextAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="g3_explicit_cpu_curve_fixture_")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "CPU_ONLY_FIXTURE.json"
        self.context = dict(gpu_uuid="GPU-CPU-FIXTURE", model_name="CPU_FIXTURE_PATH", kv_dtype="auto",
                            kv_bytes_per_token=57344, source_lock_sha256="0" * 64)
        self.data = dict(schema_version=2, model_name=self.context["model_name"], kv_dtype="auto",
                         kv_bytes_per_token=57344, break_even_ssd_tokens=64, break_even_mem_tokens=16,
                         provenance=dict(gpu_uuid=self.context["gpu_uuid"], source_lock_sha256="0" * 64,
                                         candidate_only=False, independent_content_validation=True),
                         curves={name: dict(knots={"16": 0.001, "128": 0.002}) for name in ("f", "g_ssd", "g_mem")})

    def freeze(self, raw=None):
        raw = raw if raw is not None else json.dumps(self.data).encode()
        self.path.write_bytes(raw)
        return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    def audit(self):
        return A.inspect_curve_context(self.path, self.freeze(), self.context)

    def test_matching_caller_metadata_never_grants_gpu_or_production(self):
        r = self.audit()
        self.assertEqual(r["status"], "CPU_METADATA_MATCH_ONLY_UNQUALIFIED")
        self.assertFalse(r["gpu_authorized"] or r["production_qualified"] or r["SSD_restore_verified"])
        self.assertEqual(r["independent_runtime_evidence_review"], "NOT_PERFORMED")

    def test_old_gpu_is_blocked(self):
        self.data["provenance"]["gpu_uuid"] = "GPU-OLD-FIXTURE"
        self.assertIn("gpu_uuid_mismatch_or_missing", self.audit()["rejection_reasons"])

    def test_model_alias_is_not_substituted(self):
        self.data["model_name"] = "Qwen/Qwen2.5-7B-Instruct"
        self.assertIn("model_name_mismatch", self.audit()["rejection_reasons"])

    def test_missing_provenance_is_unknown(self):
        del self.data["provenance"]
        reasons = self.audit()["rejection_reasons"]
        self.assertIn("candidate_only_or_qualification_unknown", reasons)
        self.assertIn("independent_content_validation_missing_or_false", reasons)

    def test_restricted_candidate_is_blocked(self):
        self.data["provenance"].update(candidate_only=True, independent_content_validation=False,
                                       allowed_consumer="same-budget calibration integration diagnostic only")
        self.assertEqual(len(self.audit()["rejection_reasons"]), 3)

    def test_threshold_above_domain_is_reported_without_extrapolation(self):
        self.data["break_even_ssd_tokens"] = 144
        self.assertIn("g_ssd_threshold_above_measured_support", self.audit()["rejection_reasons"])

    def test_boolean_density_is_not_integer(self):
        self.data["kv_bytes_per_token"] = True
        self.assertIn("kv_bytes_per_token_mismatch", self.audit()["rejection_reasons"])

    def test_source_lock_drift_is_blocked(self):
        self.data["provenance"]["source_lock_sha256"] = "1" * 64
        self.assertIn("current_source_lock_binding_missing_or_mismatch", self.audit()["rejection_reasons"])

    def test_curve_byte_change_is_rejected(self):
        ref = self.freeze()
        self.path.write_bytes(self.path.read_bytes().replace(b"0.001", b"0.003"))
        with self.assertRaisesRegex(ValueError, "SHA"):
            A.inspect_curve_context(self.path, ref, self.context)

    def test_duplicate_key_is_rejected(self):
        ref = self.freeze(b'{"schema_version":2,"schema_version":2}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            A.inspect_curve_context(self.path, ref, self.context)

    def test_nonfinite_cost_is_rejected(self):
        self.data["curves"]["f"]["knots"]["16"] = float("inf")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.audit()

    def test_noncanonical_or_out_of_band_knots_rejected(self):
        self.data["curves"]["f"]["knots"]["016"] = .01
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.audit()


if __name__ == "__main__":
    unittest.main(verbosity=2)
