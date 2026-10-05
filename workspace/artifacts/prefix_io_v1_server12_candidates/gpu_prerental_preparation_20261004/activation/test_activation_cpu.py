"""CPU arithmetic and rejection cases; no positive native GPU capability fixtures."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ORIGINAL_FROZEN_CONTROL_DIR = HERE.parents[1] / "i_pilot_cpu_preparation_20261004" / "i_bridge" / "frozen" / "control"
PAIR_CONFIG_PATH = HERE.parent / "runner" / "LOCAL_PRIVATE_U_I_CONFIG.json"
sys.path.insert(0, str(HERE / "source"))
from prefix_io_control import gpu_cell_issuer as G
from prefix_io_control.p4_cost_table import CostCell, CostTable, _from_verified_gpu_capability, load_conditional_table
from prefix_io_control.dispatch_budget import ZERO, Amount
from prefix_io_control.p4_policy import P4Policy
from prefix_io_control.p4_types import P4Config


def validation_module():
    name = "_activation_original_raw_validator_cpu"
    path = HERE / "native_conditional_cost.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ActivationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.signature = ("GPU-fixture", "a" * 64, "b" * 64, "c" * 64, 1, 1, 0, 527, 917504)
        self.cell = CostCell(self.signature, ZERO, "ssd_read", 917504, "existing_io_plus_delta", 100, 20, 1)
        self.validation = validation_module()
        self.original = HERE / "source" / "prefix_io_control" / "p4_paired_measurement_verifier.py"
        self.reference = {"path": "source.py", "bytes": 10, "sha256": "a" * 64}

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = data if isinstance(data, bytes) else json.dumps(data, sort_keys=True).encode()
        path.write_bytes(raw)
        return dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    def arguments(self, plan):
        plan_ref = self.put("plan.json", plan)
        raw = self.put("raw.json", {"origin": "CPU_fixture", "production_qualified": True})
        guard = self.put("guard.json", {"GPU_verified": True})
        intent = self.put("intent.json", {"plan_ref": plan_ref})
        return dict(plan_ref=plan_ref, measurements_ref=raw, guard_ref=guard, intent_ref=intent,
                    expected_plan_ref=plan_ref, expected_guard_ref=guard, expected_intent_ref=intent,
                    expected_runtime_domain_sha256="a" * 64, expected_gpu_uuid="GPU-fixture", expected_source_lock_sha256="b" * 64)

    def test_mock_cost_exact_math_stays_production_closed(self):
        table = CostTable((self.cell,), scope="mock_only", source_sha256="d" * 64)
        estimate = table.lookup(self.signature, ZERO, "ssd_read", 917504, execution="cpu_mock")
        self.assertEqual(estimate.total_ns, 121)
        self.assertTrue(estimate.mock_only)
        self.assertFalse(estimate.production_qualified)
        self.assertFalse(table.production_qualified)
        self.assertIsNone(table.lookup(self.signature, ZERO, "ssd_read", 917504, execution="production"))
        self.assertIsNone(G.qualified_identity(table))

    def test_conditional_cost_even_valid_cells_never_production(self):
        table = CostTable((self.cell,), scope="conditional", source_sha256="d" * 64)
        self.assertFalse(table.production_qualified)
        self.assertIsNone(table.lookup(self.signature, ZERO, "ssd_read", 917504, execution="cpu_mock"))
        self.assertIsNone(table.lookup(self.signature, ZERO, "ssd_read", 917504))

    def test_constructor_no_production_flag_or_scope(self):
        with self.assertRaises(ValueError): CostTable((self.cell,), scope="gpu_verified_exact_cells", source_sha256="d" * 64)
        with self.assertRaises(TypeError): CostTable((self.cell,), source_sha256="d" * 64, production_qualified=True)

    def test_naked_private_token_cannot_construct_GPU_table(self):
        with self.assertRaisesRegex(ValueError, "verified raw GPU issuer"):
            _from_verified_gpu_capability(object())
        self.assertEqual(len(G._ISSUED), 0)

    def test_table_immutable_and_self_declared_json_closed(self):
        ref = self.put("declared.json", {"production_qualified": True, "scope": "gpu_verified_exact_cells", "origin": "native_gpu_recording"})
        table = load_conditional_table(self.root / ref["path"])
        self.assertFalse(table.production_qualified)
        with self.assertRaises(AttributeError): table.scope = "gpu_verified_exact_cells"

    def test_P4_exact_CostTable_type_check_is_retained(self):
        class Derived(CostTable): pass
        config = P4Config("interference", 100, 1000, 10000, (1, 2, 4, 8))
        with self.assertRaisesRegex(TypeError, "exact finite"):
            P4Policy("run", config, table=Derived((self.cell,), scope="mock_only", source_sha256="d" * 64))
        policy = P4Policy("run", config, table=CostTable((self.cell,), scope="mock_only", source_sha256="d" * 64))
        self.assertFalse(policy.production_interference_qualified)

    def test_unknown_signature_IO_stage_or_bytes_never_zero_cost(self):
        table = CostTable((self.cell,), scope="mock_only", source_sha256="d" * 64)
        for signature, existing, stage, physical in ((self.signature[:-1] + (917505,), ZERO, "ssd_read", 917504),
                (self.signature, (Amount(1, 1),) + ZERO[1:], "ssd_read", 917504),
                (self.signature, ZERO, "h2d", 917504), (self.signature, ZERO, "ssd_read", 917505)):
            with self.subTest(stage=stage, physical=physical):
                self.assertIsNone(table.lookup(signature, existing, stage, physical, execution="cpu_mock"))

    def test_original_estimator_AST_preserved_and_calibration_only(self):
        result = G.replay_math_audit(self.validation, self.original, calibration_A_ns=[100, 110], calibration_B_ns=[120, 140],
                                   heldout_B_ns=140, reference_source_ref=self.reference)
        self.assertEqual(result["calibration"]["baseline_ns"], 105)
        self.assertEqual(result["calibration"]["incremental_or_joint_ns"], 25)
        self.assertEqual(result["calibration"]["uncertainty_ns"], 10)
        self.assertEqual(result["upper_ns"], 140)
        self.assertTrue(result["heldout_covered"])
        self.assertFalse(result["actual_gpu_execution_proved"])
        self.assertFalse(result["production_qualified"])
        self.assertFalse(result["original_math_proof"]["mathematical_expressions_changed"])

    def test_holdout_underprediction_does_not_inflate_fit(self):
        first = G.replay_math_audit(self.validation, self.original, calibration_A_ns=[100, 110], calibration_B_ns=[120, 140],
                                  heldout_B_ns=140, reference_source_ref=self.reference)
        failed = G.replay_math_audit(self.validation, self.original, calibration_A_ns=[100, 110], calibration_B_ns=[120, 140],
                                   heldout_B_ns=99999, reference_source_ref=self.reference)
        self.assertEqual(first["upper_ns"], failed["upper_ns"])
        self.assertFalse(failed["heldout_covered"])
        self.assertFalse(failed["holdout_used_to_fit"])
        self.assertEqual(len(G._ISSUED), 0)

    def test_bool_or_nonpositive_arithmetic_not_cost_samples(self):
        for value in (True, 0, -1, 1.2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                G.replay_math_audit(self.validation, self.original, calibration_A_ns=[value, 100], calibration_B_ns=[120, 140],
                                   heldout_B_ns=140, reference_source_ref=self.reference)

    def test_original_native_capture_requires_all_128_frames(self):
        with self.assertRaisesRegex(ValueError, "all 128"):
            self.validation.validate_capture({"scope": "server11_full_step_native_capture_v1", "run_id": "fixture", "origin": "native_gpu_recording",
                "valid": True, "frames": [], "event_witnesses": []}, run_id="fixture", request_id="req", output_ids=list(range(128)),
                prompt_tokens=512, measured_offset=16, warmup_offsets=[1], cached_tokens=496)

    def test_original_native_guard_rejects_declared_GPU_boolean(self):
        with self.assertRaisesRegex(ValueError, "guard"):
            self.validation.validate_guard({"GPU_verified": True}, gpu_uuid="GPU-fixture", job_id="job", wrapper_path="wrapper.py")

    def test_CPU_semantic_old_singleton_and_fixture_cannot_issue(self):
        for plan in ({"scope": "p4_cpu_semantic_recomputation", "production_qualified": True},
                     {"scope": "server11_preregistered_native_conditional_cell_v1", "conditional_cost_cell_qualified": True},
                     {"schema": "strong_gpu_exact_cell_prelaunch_plan_v1", "evidence_origin": "native_runtime_preregistered",
                      "cpu_preparation_only": False, "synthetic_fixture": True}):
            with self.subTest(plan=plan), self.assertRaisesRegex(ValueError, "no CPU fixture or old singleton"):
                G.issue_verified_gpu_table(self.root, **self.arguments(plan))
        self.assertEqual(len(G._ISSUED), 0)

    def test_expected_parent_plan_or_guard_cannot_be_derived_from_raw(self):
        args = self.arguments({})
        args["expected_plan_ref"] = dict(args["plan_ref"], sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "independent parent"):
            G.issue_verified_gpu_table(self.root, **args)

    def test_raw_ref_nonhex_size_or_mutation_rejected(self):
        ref = self.put("bytes.json", {"x": 1})
        for bad in (dict(ref, sha256="z" * 64), dict(ref, bytes=ref["bytes"] + 1), dict(ref, sha256="0" * 64)):
            with self.subTest(ref=bad), self.assertRaises(ValueError): G.read_ref(self.root, bad)

    def test_raw_ref_escape_nonfinite_or_duplicate_JSON_rejected(self):
        for path in ("../escape.json", "/escape.json", "a/../b", "a\\b"):
            with self.subTest(path=path), self.assertRaises(ValueError): G.read_ref(self.root, dict(path=path, bytes=1, sha256="a" * 64))
        for raw in (b'{"x": NaN}', b'{"x":1,"x":2}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError): G.json_ref(self.root, self.put("bad.json", raw))

    def test_unchanged_P4_policy_and_options_source_AST(self):
        frozen = ORIGINAL_FROZEN_CONTROL_DIR
        for name in ("p4_policy.py", "p4_options.py", "p4_bridge.py", "p4_paired_measurement_verifier.py"):
            self.assertEqual((HERE / "source" / "prefix_io_control" / name).read_bytes(), (frozen / name).read_bytes())
        tree = ast.parse((HERE / "source" / "prefix_io_control" / "p4_cost_table.py").read_text(encoding="utf-8"))
        self.assertTrue(any(isinstance(node, ast.FunctionDef) and node.name == "_from_verified_gpu_capability" for node in tree.body))

    def test_actual_wrapper_unwrap_is_strict_and_CPU_not_GPU_authority(self):
        path = PAIR_CONFIG_PATH
        document = json.loads(path.read_bytes())
        self.assertEqual(G._unwrap_pair_config(document), document["configurations"])
        for mutation in ("raw_pair", "extra", "schema", "self_qualified"):
            import copy
            changed = copy.deepcopy(document)
            if mutation == "raw_pair": changed = document["configurations"]
            elif mutation == "extra": changed["unknown"] = True
            elif mutation == "schema": changed["schema"] = "other"
            else: changed["native_execution_verified"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): G._unwrap_pair_config(changed)

    def test_layout_SHA_is_actual_compact_model_derived_geometry(self):
        config = self.put("model_config.json", dict(num_hidden_layers=28, num_key_value_heads=4, hidden_size=3584, num_attention_heads=28))
        geometry = dict(model_config_sha256=config["sha256"], num_hidden_layers=28, num_key_value_heads=4, head_dim=128,
                        dtype="bfloat16", dtype_bytes=2, tokens_per_block=16, tensor_parallel_size=1, physical_block_bytes=917504)
        layout_ref = self.put("geometry.json", G.canonical(geometry))
        self.assertEqual(G._canonical_geometry(self.root, dict(model_config_source_ref=config, kv_layout_ref=layout_ref)), geometry)
        for raw in (G.canonical(geometry) + b"\n", json.dumps(geometry, indent=2).encode(), G.canonical({"geometry_sha256": layout_ref["sha256"]})):
            ref = self.put("wrong_geometry.json", raw)
            with self.subTest(raw_size=len(raw)), self.assertRaises(ValueError):
                G._canonical_geometry(self.root, dict(model_config_source_ref=config, kv_layout_ref=ref))


if __name__ == "__main__":
    unittest.main(verbosity=2)
