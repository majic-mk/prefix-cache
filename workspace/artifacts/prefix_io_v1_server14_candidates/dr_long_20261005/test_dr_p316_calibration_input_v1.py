"""Actual frozen-input ancestry and exact original evaluator preservation."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


def load(filename, name):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


helper = load("dr_p316_calibration_input_v1.py", "_strict_actual_calibration")
contract = load("original_scripts/concurrent_pilot_contract_p316.py", "_actual_p316_contract")


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        original = self.root / helper.PARENT_PATH
        original.parent.mkdir(parents=True)
        original.write_bytes((HERE / "original_assets/all_hit-manifest.json").read_bytes())
        self.path = self.root / "calibration-single-family-manifest-v1.json"
        self.path.write_bytes((HERE / "calibration-single-family-manifest-v1.json").read_bytes())
        self.permit = json.loads((HERE / "original_assets/permit.json").read_text())
        self.curves = json.loads((HERE / "original_assets/curves-v2.json").read_text())
        self.manifest = json.loads(self.path.read_text())
        self.out = self.root / "experiments/prefix_io_v1/runs/server14-dr-cal05/details"

    def validate(self):
        return helper.validate_calibration_registration(self.root, self.permit,
            self.manifest, self.path, self.out)

    def test_actual_parent_contract_and_independent_child_ancestry(self):
        permit_before = deepcopy(self.permit)
        registered = self.validate()
        parent = registered["parent_manifest"]
        self.assertEqual(contract.validate(parent, self.curves, parent["gpu_uuid"]), parent["engine"])
        with self.assertRaisesRegex(ValueError, "frozen workload changed"):
            contract.validate(self.manifest, self.curves, self.manifest["gpu_uuid"])
        self.assertEqual(self.permit, permit_before)
        self.assertEqual(registered["record"]["differences"], ["scope", "requests[*].family_index=0"])
        self.assertEqual([r["request_id"] for r in self.manifest["requests"]], list(range(10)))
        self.assertEqual([r["scheduled_time"] for r in self.manifest["requests"]],
                         [r["scheduled_time"] for r in parent["requests"]])
        self.assertTrue(all(len(f["tokens"]) == 16257 for f in self.manifest["families"]))
        self.assertEqual(self.manifest["output_tokens"], 128)

    def test_engine_tokens_arrivals_and_output_changes_rejected(self):
        for field in ("engine", "tokens", "arrival", "output", "scope", "family"):
            changed = deepcopy(self.manifest)
            if field == "engine":
                changed["engine"]["max_num_seqs"] = 1
            elif field == "tokens":
                changed["families"][0]["tokens"].pop()
            elif field == "arrival":
                changed["requests"][1]["scheduled_time"] += .1
            elif field == "output":
                changed["output_tokens"] = 8
            elif field == "scope":
                changed["scope"] = "formal performance"
            else:
                changed["requests"][1]["family_index"] = 1
            self.path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ValueError):
                helper.validate_calibration_registration(self.root, self.permit,
                    changed, self.path, self.out)

    def test_original_parent_bytes_and_permit_cannot_be_changed(self):
        self.permit["manifests"]["all_hit"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.validate()
        self.permit = json.loads((HERE / "original_assets/permit.json").read_text())
        (self.root / helper.PARENT_PATH).write_bytes(b"changed parent")
        with self.assertRaises(ValueError):
            self.validate()

    def test_calibration_entry_rejects_effect_arms_before_model(self):
        helper.validate_calibration_adapter_arm(dict(current_device_adapter=dict(arm="CAL")))
        for arm in ("U", "F8+D_R", "I", "J"):
            with self.assertRaises(ValueError):
                helper.validate_calibration_adapter_arm(dict(current_device_adapter=dict(arm=arm)))
        self.out = self.root / "experiments/prefix_io_v1/runs/server14-dr-long-u02/details"
        with self.assertRaises(ValueError):
            self.validate()

    def test_entire_request_evaluator_and_cleanup_unchanged(self):
        original = (HERE / "original_scripts/run_concurrent_pilot_p316.py").read_text(encoding="utf-8")
        current = (HERE / "run_concurrent_pilot_p316_single_family_cal_v1.py").read_text(encoding="utf-8")
        builder = ast.parse((HERE / "build_dr_p316_calibration_runner_v1.py").read_text(encoding="utf-8"))
        changes = next(ast.literal_eval(n.value) for n in builder.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "changes" for t in n.targets))
        for before, after in changes:
            self.assertEqual(current.count(after), 1)
            current = current.replace(after, before)
        embedded = (HERE / "dr_p316_calibration_input_v1.py").read_text(encoding="utf-8") + "\n\ndef main():"
        self.assertEqual(current.count(embedded), 1)
        current = current.replace(embedded, "def main():")
        self.assertEqual(current, original)
        original_main = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == "main")
        restored_main = next(n for n in ast.parse(current).body if isinstance(n, ast.FunctionDef) and n.name == "main")
        self.assertEqual(ast.dump(original_main, include_attributes=False),
                         ast.dump(restored_main, include_attributes=False))


if __name__ == "__main__":
    unittest.main()
