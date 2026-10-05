"""Closed-gate CPU negatives only; no successful GPU-shaped fixture exists."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("_cpu_actual_pair_validator", HERE / "development_cost_pair.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class Driver:
    MAX_GPU_SECONDS = 28800

    @staticmethod
    def read(path):
        return json.loads(path.read_bytes())

    @staticmethod
    def check_ref(root, row):
        path = Path(root) / row["path"]
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("CPU_NEGATIVE_REFERENCE_BYTES_DRIFT")
        return path

    @staticmethod
    def load(*args):
        raise AssertionError("No negative fixture may reach native validator loading")


class ClosedGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="prefix_io_cpu_negative_")
        self.root = Path(self.tmp.name)
        self.refs = {}
        candidates = (PROJECT / M.VALIDATION_PATH,
            PROJECT / "artifacts/prefix_io_v1_server12_candidates/i_pilot_cpu_preparation_20261004/calibration_v2/native_conditional_cost.py")
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise FileNotFoundError("Pinned original validator missing from both server and local candidate paths")
        self.put_bytes(M.VALIDATION_PATH, source.read_bytes())
        self.material = {k: None for k in M.MATERIAL_FIELDS}
        for name in M.SOURCE_FIELDS:
            self.material[name] = self.put("cpu_only/" + name + ".json", {"fixture_origin": "CPU_NEGATIVE_ONLY"})
        self.material["gpu_uuid"] = "TEST_ONLY_CPU_NO_GPU"
        self.material["common_runtime_domain_sha256"] = "a" * 64
        self.document = dict(schema="bounded_I_actual_AB_cost_input_v1", material=self.material,
            evidence_refs=list(self.refs.values()))

    def tearDown(self):
        self.tmp.cleanup()

    def put_bytes(self, relative, raw):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        ref = dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        self.refs[relative] = ref
        return ref

    def put(self, relative, document):
        return self.put_bytes(relative, json.dumps(document, sort_keys=True).encode("utf-8"))

    def validate(self):
        row = self.put("cpu_only/binding.json", self.document)
        return M.validate_actual_development_pair(self.root, self.refs, row, driver=Driver,
            expected_gpu_uuid="TEST_ONLY_CPU_NO_GPU", expected_common_runtime_domain_sha256="a" * 64)

    def test_missing_real_AB_results_is_closed(self):
        with self.assertRaisesRegex(ValueError, "exact actual A and B results"):
            self.validate()

    def test_cpu_origin_cannot_open_gate(self):
        for mode in ("A", "B"):
            row = self.put("cpu_only/" + mode + ".json", dict(schema="bounded_original_model_result_v1",
                mode=mode, origin="CPU_NEGATIVE_ONLY"))
            self.document["evidence_refs"].append(row)
        with self.assertRaisesRegex(ValueError, "CPU or simulated origin cannot open gate"):
            self.validate()

    def test_binding_claimed_qualification_is_rejected(self):
        self.document["production_qualified"] = True
        with self.assertRaisesRegex(ValueError, "exact bounded A/B binding schema"):
            self.validate()

    def test_missing_material_source_is_closed(self):
        self.document["evidence_refs"].remove(self.material["collector_source_ref"])
        with self.assertRaisesRegex(ValueError, "material source absent from evidence"):
            self.validate()

    def test_duplicate_evidence_is_rejected(self):
        self.document["evidence_refs"].append(self.document["evidence_refs"][0])
        with self.assertRaisesRegex(ValueError, "unique actual evidence references"):
            self.validate()

    def test_source_byte_drift_is_rejected_before_origin(self):
        (self.root / M.VALIDATION_PATH).write_bytes(b"CPU_NEGATIVE_DRIFT")
        with self.assertRaisesRegex(ValueError, "CPU_NEGATIVE_REFERENCE_BYTES_DRIFT"):
            self.validate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
