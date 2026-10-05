"""Explicit CPU fixtures around original reference methods; no GPU qualification."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--source-root", type=Path)
args, remaining = parser.parse_known_args()
SOURCE = args.source_root or HERE.parents[1] / "prefix_io_v1_server08_primary_qualification/contents"

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

M = load("g3_reference_result_cpu_test", HERE / "g3_reference_result.py")
F = load("g3_reference_base_fixture", HERE / "test_g3_calibration_result.py")
backend_attempts = []
class BlockBackend:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "vllm", "py_kvcache", "triton", "cuda", "cupy"}:
            backend_attempts.append(fullname)
            raise RuntimeError("CPU fixture forbids backend import")
sys.meta_path.insert(0, BlockBackend())

def top5(token):
    return [{str(token + i): {"logprob": -float(i + 1), "rank": i + 1} for i in range(5)}]

def fixture(mode, storage=None):
    report, runtime, guard, binding = F.job_fixture(mode)
    config = runtime["frozen_config"]
    config["sampling"]["logprobs"] = 5
    config["native_hot_diagnostic"] = report["native_hot_diagnostic"] = mode == "cold"
    config["cached_reference_logprobs"] = report["cached_reference_logprobs"] = mode == "paired"
    if mode == "cold":
        rows = []
        for cold in report["rows"]:
            hot = copy.deepcopy(cold)
            hot["kind"] = "gpu_hot"; hot["num_cached_tokens"] = 128
            rows.extend([cold, hot])
        report["rows"] = rows
    runtime["purpose"] = M.PURPOSE
    runtime["origin"] = "CPU_FIXTURE_NOT_GPU"
    runtime["production_qualified"] = runtime["performance_claim"] = runtime["cost_qualified"] = False
    runtime["requests"] = []
    for ordinal, row in enumerate(report["rows"]):
        row["request_id"] = str(ordinal)
        row["trace"]["internal_request_id"] = str(ordinal) + "-1234abcd"
        for transfer in row["trace"]["load_transfers"]:
            transfer["req_id"] = row["trace"]["internal_request_id"]
        row["output_logprobs"] = top5(row["output_token_ids"][0])
        runtime["requests"].append({"ordinal": ordinal, "prompt_token_ids": row["prompt_token_ids"],
            "output_token_ids": row["output_token_ids"], "num_cached_tokens": row["num_cached_tokens"],
            "sentinel": False, "request_id": row["request_id"]})
    binding["expected_label"] = guard["label"] = M.LABELS[mode]
    binding["expected_model"] = copy.deepcopy(config["model"])
    binding["expected_alias_target"] = config["alias_target"]
    if storage is not None:
        binding["expected_storage"] = storage
        if mode == "paired":
            config["engine"]["kv_transfer_config"]["kv_connector_extra_config"]["shared_storage_path"] = storage
    elif mode == "cold":
        binding["expected_storage"] = F.STORAGE + "-unused-reference"
    report["storage_preflight"] = {"path": binding["expected_storage"]}
    return report, runtime, guard, binding, config

class JobTests(unittest.TestCase):
    def setUp(self):
        self.inputs = fixture("paired")
    def validate(self, mode="paired"):
        return M.validate_reference_job(mode, *self.inputs)
    def test_cold_six_original_outputs_not_three_raw_request_receipt(self):
        d = M.validate_reference_job("cold", *fixture("cold"))
        self.assertEqual(d["requests"], 6)
        self.assertEqual(d["status"], "PASS_REFERENCE_SIX_REQUEST_PATH_CLOSURE_ONLY")
        self.assertTrue(d["all_warmups_in_numeric_gate"])
        self.assertFalse(d["latency_fit_allowed"])
    def test_paired_six_fixture_outputs_not_GPU_qualification(self):
        d = self.validate()
        for key in M.UNQUALIFIED:
            self.assertIs(d[key], False)
    def test_wrong_source_guard_binding_rejected(self):
        self.inputs[1]["source_lock_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "runtime source"):
            self.validate()
    def test_failed_guard_rejected(self):
        self.inputs[2]["exit"] = 1
        with self.assertRaises(ValueError): self.validate()
    def test_OS_session_not_drained_rejected(self):
        self.inputs[2]["session_drained"] = False
        with self.assertRaises(ValueError): self.validate()
    def test_stale_PID_rejected(self):
        self.inputs[3]["prior_process_identities"] = [self.inputs[1]["process_identity"]]
        with self.assertRaisesRegex(ValueError, "fresh"): self.validate()
    def test_wrong_logprobs_sampling_rejected(self):
        self.inputs[4]["sampling"]["logprobs"] = True
        with self.assertRaises(ValueError): self.validate()
    def test_wrong_reference_flags_rejected(self):
        self.inputs[0]["cached_reference_logprobs"] = False
        with self.assertRaises(ValueError): self.validate()
    def test_wrong_native_label_rejected(self):
        a = fixture("cold"); a[3]["expected_label"] = "old-raw-cold"
        with self.assertRaises(ValueError): M.validate_reference_job("cold", *a)
    def test_model_identity_drift_rejected(self):
        self.inputs[4]["model"]["revision"] = "drift"
        with self.assertRaisesRegex(ValueError, "model identity"): self.validate()
    def test_alias_path_drift_rejected(self):
        self.inputs[4]["alias_target"] = "/wrong/model"
        with self.assertRaises(ValueError): self.validate()
    def test_async_configuration_drift_rejected(self):
        self.inputs[4]["engine"]["async_scheduling"] = True
        with self.assertRaises(ValueError): self.validate()
    def test_curves_not_allowed_in_raw_reference(self):
        self.inputs[4]["engine"]["kv_transfer_config"]["kv_connector_extra_config"]["prefix_cache_break_even_path"] = "old.json"
        with self.assertRaises(ValueError): self.validate()
    def test_wrong_actual_cache_shape_rejected(self):
        self.inputs[0]["rows"][0]["num_cached_tokens"] = 0
        with self.assertRaises(ValueError): self.validate()
    def test_missing_actual_SSD_bytes_rejected(self):
        self.inputs[0]["rows"][0]["trace"]["preload_actual_read_bytes"] = 0
        with self.assertRaises(ValueError): self.validate()
    def test_wrong_staging_cache_medium_rejected(self):
        self.inputs[0]["rows"][1]["trace"]["src_cache"] = 0
        with self.assertRaises(ValueError): self.validate()
    def test_empty_external_final_drain_rejected(self):
        self.inputs[0]["final_drain"]["handlers"] = []
        with self.assertRaises(ValueError): self.validate()
    def test_missing_top5_on_warmup_rejected(self):
        self.inputs[0]["rows"][0]["output_logprobs"] = []
        with self.assertRaisesRegex(ValueError, "logprob"): self.validate()
    def test_nonfinite_logprob_rejected(self):
        values = self.inputs[0]["rows"][0]["output_logprobs"][0]
        next(iter(values.values()))["logprob"] = float("nan")
        with self.assertRaises(ValueError): self.validate()
    def test_Boolean_logprob_rejected(self):
        next(iter(self.inputs[0]["rows"][0]["output_logprobs"][0].values()))["logprob"] = True
        with self.assertRaises(ValueError): self.validate()
    def test_duplicate_top5_rank_rejected(self):
        list(self.inputs[0]["rows"][0]["output_logprobs"][0].values())[1]["rank"] = 1
        with self.assertRaises(ValueError): self.validate()
    def test_request_missing_from_actual_runtime_rejected(self):
        self.inputs[1]["requests"].pop()
        with self.assertRaises(ValueError): self.validate()
    def test_request_output_disagreement_rejected(self):
        self.inputs[1]["requests"][0] = copy.deepcopy(self.inputs[1]["requests"][0])
        self.inputs[1]["requests"][0]["output_token_ids"] = [999]
        with self.assertRaises(ValueError): self.validate()
    def test_self_claim_cost_qualification_rejected(self):
        self.inputs[1]["cost_qualified"] = True
        with self.assertRaises(ValueError): self.validate()


class FixturePath(type(Path())):
    # Explicit test-only compatibility for Python3.8; production uses std Path.
    def is_relative_to(self, other):
        try: self.relative_to(other); return True
        except ValueError: return False

def exact_functions(path, namespace):
    raw = path.read_bytes(); tree = ast.parse(raw)
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(path.resolve()), "exec", dont_inherit=True), namespace)

class PairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="explicit-g3-reference-CPU-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runs = self.root / "experiments/prefix_io_v1/runs"
        self.runs.mkdir(parents=True)
        self.refs = {}
        sources = SOURCE / "experiments/prefix_io_v1/scripts"
        scripts = self.root / "scripts"; scripts.mkdir()
        for key, (name, digest) in M.ORIGINAL_PINS.items():
            path = scripts / name
            shutil.copyfile(sources / name, path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            self.refs[key] = self.ref(path)
        repeated = {"Path": Path, "json": json, "math": math, "hashlib": hashlib}
        exact_functions(scripts / "analyze_repeated_native_costs.py", repeated)
        storage = {"Path": FixturePath, "ROOT": self.root, "json": json,
            "permission": lambda project: {"approved_experiment_root": str(self.runs)},
            "__name__": "ORIGINAL_STORAGE_WITH_EXPLICIT_CPU_FIXTURE_PERMISSION"}
        exact_functions(scripts / "experiment_storage.py", storage)
        # Only the storage permission is a temporary, explicit CPU fixture.
        # All original storage path functions and numerical functions stay exact.
        storage["permission"] = lambda project: {"approved_experiment_root": str(self.runs)}
        self.analyzer = types.ModuleType("original_cached_analyzer_CPU_fixture")
        self.analyzer.__file__ = str((scripts / "analyze_cached_references.py").resolve())
        self.analyzer.__dict__.update(Path=Path, json=json, math=math, hashlib=hashlib,
            require=repeated["require"], check_drains=repeated["check_drains"],
            details_path=storage["details_path"], authorized_path=storage["authorized_path"])
        exact_functions(scripts / "analyze_cached_references.py", self.analyzer.__dict__)
        self.inputs = {}
        for mode in M.LABELS:
            self.inputs[mode] = fixture(mode, str(self.runs / ("unused-native-storage" if mode == "cold" else "published-private-storage")))
            folder = self.runs / M.LABELS[mode] / "details/acquisition"
            folder.mkdir(parents=True)
        self.plan = {"native_label": M.LABELS["cold"], "sizes": [128], "reps": 3, "domain": 1024,
            "gpu_uuid": F.GPU, "kv_budget_bytes": 268435456, "staging_budget_bytes": 134217728,
            "native_storage_path": self.inputs["cold"][3]["expected_storage"],
            "external_groups": [{"label": M.LABELS["paired"], "sizes": [128],
                "storage": "published-private-storage",
                "storage_path": self.inputs["paired"][3]["expected_storage"]}],
            "run_details": {label: str(self.runs / label / "details/acquisition") for label in M.LABELS.values()}}
        self.plan_path = self.root / "reference-plan.json"
    def ref(self, path):
        raw = path.read_bytes()
        return {"path": path.relative_to(self.root).as_posix(), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    def run_pair(self):
        checks = {}
        for mode, inputs in self.inputs.items():
            checks[mode] = M.validate_reference_job(mode, *inputs)
            folder = Path(self.plan["run_details"][M.LABELS[mode]])
            (folder / "result.json").write_text(json.dumps(inputs[0]), encoding="utf-8")
            (folder / "frozen-config.json").write_text(json.dumps(inputs[4]), encoding="utf-8")
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        return M.verify_reference_pair(self.root, self.plan_path, expected_plan_ref=self.ref(self.plan_path),
            native_check=checks["cold"], paired_check=checks["paired"], analyzer_refs=self.refs, original_analyzer=self.analyzer)
    def test_original_analyzer_all_six_cached_comparisons_including_warmup(self):
        result = self.run_pair()
        self.assertEqual(result["status"], "PASSED_EXACT_CACHED_REFERENCE")
        self.assertEqual(len(result["original_analysis"]["comparisons"]), 6)
        self.assertEqual(sum(v["warmup"] for v in result["original_analysis"]["comparisons"]), 2)
        for key in M.UNQUALIFIED: self.assertIs(result[key], False)
        self.assertFalse(result["latency_fit_allowed"])
    def test_original_exact_numeric_difference_returns_failed_not_new_tolerance(self):
        row = self.inputs["paired"][0]["rows"][0]
        row["output_logprobs"][0][str(row["output_token_ids"][0])]["logprob"] -= 1e-6
        result = self.run_pair()
        self.assertEqual(result["status"], "FAILED_EXACT_CACHED_REFERENCE")
        self.assertEqual(result["original_analysis"]["failed_comparisons"], 1)
    def test_native_cold_hot_difference_is_reported_and_not_bit_equality_requirement(self):
        row = self.inputs["cold"][0]["rows"][0]
        row["output_logprobs"][0][str(row["output_token_ids"][0])]["logprob"] -= 1.0
        result = self.run_pair()
        self.assertEqual(result["status"], "PASSED_EXACT_CACHED_REFERENCE")
        self.assertFalse(result["original_analysis"]["cold_hot"][0]["top5_equal"])
    def test_same_PID_SID_rejected(self):
        self.inputs["paired"][1]["process_identity"] = self.inputs["cold"][1]["process_identity"]
        self.inputs["paired"][2]["session_id"] = self.inputs["cold"][2]["session_id"]
        with self.assertRaises(ValueError): self.run_pair()
    def test_original_analyzer_source_changed_rejected(self):
        path = self.root / self.refs["cached"]["path"]
        path.write_bytes(path.read_bytes() + b"\n# drift\n")
        with self.assertRaises(ValueError): self.run_pair()
    def test_fake_module_callable_rejected(self):
        self.analyzer.analyze = lambda root, plan: {"status": "PASSED_EXACT_CACHED_REFERENCE"}
        with self.assertRaisesRegex(ValueError, "callable"): self.run_pair()
    def test_exact_function_code_with_forged_globals_rejected(self):
        original = self.analyzer.analyze
        globals_copy = dict(original.__globals__)
        globals_copy["compare_rows"] = lambda *args: {"output_equal": True, "top5_exact_equal": True}
        self.analyzer.analyze = types.FunctionType(original.__code__, globals_copy, original.__name__,
                original.__defaults__, original.__closure__)
        with self.assertRaisesRegex(ValueError, "substituted global namespace"): self.run_pair()
    def test_wrong_plan_domain_rejected(self):
        self.plan["domain"] = 16384
        with self.assertRaises(ValueError): self.run_pair()
    def test_wrong_plan_GPU_rejected(self):
        self.plan["gpu_uuid"] = "GPU-other"
        with self.assertRaises(ValueError): self.run_pair()
    def test_three_raw_rows_cannot_be_reference_gate(self):
        self.inputs["cold"][0]["rows"] = self.inputs["cold"][0]["rows"][::2]
        with self.assertRaises(ValueError): self.run_pair()
    def test_runtime_source_scope_disagree_between_reference_jobs(self):
        self.inputs["paired"][1]["scope_sha256"] = "8" * 64
        self.inputs["paired"][3]["expected_scope_sha256"] = "8" * 64
        with self.assertRaises(ValueError): self.run_pair()
    def test_no_backend_imports(self):
        self.assertEqual(backend_attempts, [])

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
