"""Synthetic CPU source/128-frame counterexamples, never native measurements."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
NATIVE = Path(os.environ.get("SERVER11_C5_COST_NATIVE_ROOT", str(BASE / "native_cost_v6"))).resolve()
CANDIDATE = Path(os.environ.get("SERVER11_C5_COST_CANDIDATE_ROOT", str(BASE / "p4_single_file_candidate_v5_cpu"))).resolve()
PREPARATION = Path(os.environ.get("SERVER11_C5_COST_PREPARATION_ROOT", str(BASE / "notification_v5_runtime_preparation"))).resolve()
ORIGINAL = Path(os.environ.get("NATIVE_ORIGINAL_ESTIMATOR", str(HERE.parents[1] /
    "prefix_io_v1_server08_p4_cpu_extended_20261001/verified-v3/source/" / 
    "third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_paired_measurement_verifier.py"))).resolve()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


M = module("_c5_binding_under_test", HERE / "c5_cost_binding_preparation.py")
F = module("_c5_frozen_synthetic_fixture", NATIVE / "test_native_conditional_cost.py")
LEGACY_RECEIPT = module("_c5_legacy_receipt_audit_only", CANDIDATE /
    "source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_single_file_receipt.py")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


class C5SourceBindingContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("C5_COST_TEST_TEMP_ROOT", str(HERE)))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.sources = {}
        def copy(source, name):
            target = self.root / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target); self.sources[name] = M.file_ref(self.root, name)
            return self.sources[name]
        def fixture(name):
            target = self.root / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# SYNTHETIC_CPU_SOURCE_OR_ASSET\n", encoding="utf-8")
            self.sources[name] = M.file_ref(self.root, name)
            return self.sources[name]
        native = copy(CANDIDATE / M.NATIVE_SUFFIX, "candidate/" + M.NATIVE_SUFFIX)
        collector = copy(CANDIDATE / "native_full_step_collector.py", "candidate/native_full_step_collector.py")
        scoped = [dict(scope="candidate", path=ref["path"].removeprefix("candidate/"),
            bytes=ref["bytes"], sha256=ref["sha256"]) for ref in (native, collector)]
        for name in M.PREPARATION_REQUIRED:
            value = copy(PREPARATION / name, "preparation/" + name)
            scoped.append(dict(scope="preparation", path=name, bytes=value["bytes"], sha256=value["sha256"]))
        prep_lock = dict(schema="cpu_notification_preparation_source_lock_v1", gpu_qualified=False,
            gpu_launch_allowed=False, native_receipt=None, immutable_parent_sources=True, files=scoped)
        dump(self.root / "preparation/PREPARATION_SOURCE_LOCK.json", prep_lock)
        prep_ref = M.file_ref(self.root, "preparation/PREPARATION_SOURCE_LOCK.json")
        self.sources[prep_ref["path"]] = prep_ref
        original = copy(ORIGINAL, M.ORIGINAL_RELATIVE)
        verifier = copy(NATIVE / "native_conditional_cost.py", "native/native_conditional_cost.py")
        serializer = copy(NATIVE / "prepare_and_verify_native_cost.py", "native/prepare_and_verify_native_cost.py")
        factory = copy(HERE / "c5_cost_binding_preparation.py", "factory/c5_cost_binding_preparation.py")
        model_plan = fixture("model-plan.json"); model_config = fixture("model/config.json")
        event = fixture(".venv/lib/python3.12/site-packages/torch/cuda/streams.py")
        runner = fixture(M.RUNNER_SUFFIX)
        original_plan, _, _, _ = F.fixture()
        contract = {name: deepcopy(original_plan[name]) for name in ("scope", "qualification_rule", "stage",
            "units", "operations", "transfer_quantum_bytes", "cached_prompt_tokens", "prompt_tokens",
            "output_tokens", "measured_offset", "warmup_offsets", "entries")}
        contract.update(transfer_quantum_bytes=917504, external_warmup_output_tokens=128,
            external_flush_output_tokens=1, process_design="six_fresh_original_processes_and_handlers_one_guard")
        self.plan = dict(scope=M.PLAN_SCOPE, schema_version=1, origin="synthetic_cpu_contract",
            source_lock_ref={}, source_lock_files=len(self.sources), preparation_source_lock_ref=prep_ref,
            candidate_relative="candidate", preparation_relative="preparation", reactor_source_ref=native,
            collector_source_ref=collector, wrapper_source_ref=self.sources["preparation/run_p4_single_file_experiment.py"],
            original_estimator_ref=original, verifier_source_ref=verifier, serializer_source_ref=serializer,
            model_plan_ref=model_plan, model_config_ref=model_config, cuda_event_source_ref=event,
            model_runner_ref=runner, common_owner_parameters=dict(max_accepted_parents=8, bridge_is_none=True),
            formula_contract=contract)
        self.lock = dict(schema="c5_cost_binding_cpu_source_lock_v1", gpu_launch_allowed=False,
            native_execution_verified=False, files=list(self.sources.values()))
        common_names = [native["path"], original["path"], verifier["path"], serializer["path"], model_plan["path"],
            model_config["path"], event["path"], runner["path"]]
        self.binding = dict(scope=M.PREPARATION_SCOPE, schema_version=1, origin="synthetic_cpu_contract",
            plan_ref={}, factory_source_ref=factory,
            runtime_common_refs=[self.sources[name] for name in common_names],
            runtime_overlay_refs=[ref for name, ref in self.sources.items() if name not in common_names])

    def prepare(self, **changes):
        dump(self.root / "closure.json", self.lock)
        lock_ref = M.file_ref(self.root, "closure.json")
        self.plan["source_lock_ref"] = lock_ref
        dump(self.root / "plan.json", self.plan)
        plan_ref = M.file_ref(self.root, "plan.json"); self.binding["plan_ref"] = plan_ref
        dump(self.root / "binding.json", self.binding); binding_ref = M.file_ref(self.root, "binding.json")
        args = dict(binding_ref=binding_ref, expected_binding_ref=binding_ref, expected_plan_ref=plan_ref,
            expected_source_lock_ref=lock_ref, expected_factory_ref=self.binding["factory_source_ref"])
        args.update(changes)
        return M.prepare_cost_binding(self.root, **args).document()

    def test_actual_c5_and_preparation_sources_bind_but_issue_no_receipt(self):
        result = self.prepare()
        self.assertTrue(result["source_binding_prepared"])
        self.assertTrue(result["collector_overlay_bound"])
        self.assertTrue(result["complete_preparation_source_bound"])
        for name in ("native_cost_qualified", "native_execution_verified", "conditional_cost_cell_qualified",
            "full_runtime_cost_qualified", "on_observation_cost_measured", "gpu_launch_allowed", "performance_claim"):
            self.assertIs(result[name], False)
        self.assertIsNone(result["valid_native_receipt"])
        self.assertIsNone(result["effective_cost_upper_ns"])
        self.assertNotIn("signature", result)
        self.assertNotIn("cost_upper_ns", result)

    def test_preparation_cannot_use_old_exact_receipt_constructor(self):
        receipt = M.prepare_cost_binding
        with self.assertRaises(ValueError): LEGACY_RECEIPT.ExactSingleFileReceipt(receipt)
        with self.assertRaises(ValueError): M.C5CostBindingPreparation(document={"PASS": True})

    def test_old_c4_v6_binding_refused(self):
        self.binding["scope"] = "p4_single_file_native_binding_v1"
        with self.assertRaisesRegex(ValueError, "C4/v6"): self.prepare()

    def test_native_origin_cannot_upgrade_preparation(self):
        self.binding["origin"] = "native_gpu_recording"
        with self.assertRaisesRegex(ValueError, "native receipt"): self.prepare()

    def test_wrong_expected_binding_ref_refused(self):
        with self.assertRaisesRegex(ValueError, "expected binding"):
            self.prepare(expected_binding_ref=dict(path="not-binding.json", bytes=1, sha256="0" * 64))

    def test_wrong_expected_plan_ref_refused(self):
        with self.assertRaisesRegex(ValueError, "expected plan/factory"):
            self.prepare(expected_plan_ref=dict(path="old-plan.json", bytes=1, sha256="0" * 64))

    def test_wrong_expected_source_lock_ref_refused(self):
        with self.assertRaisesRegex(ValueError, "expected new source lock"):
            self.prepare(expected_source_lock_ref=dict(path="old-lock.json", bytes=1, sha256="0" * 64))

    def test_source_lock_cannot_claim_native_execution(self):
        self.lock["native_execution_verified"] = True
        with self.assertRaisesRegex(ValueError, "CPU-only source lock"): self.prepare()

    def test_collector_in_common_instead_of_overlay_refused(self):
        value = self.plan["collector_source_ref"]
        self.binding["runtime_overlay_refs"].remove(value); self.binding["runtime_common_refs"].append(value)
        with self.assertRaisesRegex(ValueError, "remain runtime overlay"): self.prepare()

    def test_collector_duplicate_common_overlay_refused(self):
        self.binding["runtime_common_refs"].append(self.plan["collector_source_ref"])
        with self.assertRaisesRegex(ValueError, "common-overlay"): self.prepare()

    def test_wrong_plan_collector_refused(self):
        self.plan["collector_source_ref"] = self.plan["reactor_source_ref"]
        with self.assertRaisesRegex(ValueError, "same C5 plan collector"): self.prepare()

    def test_wrong_c4_native_digest_refused(self):
        self.plan["reactor_source_ref"] = dict(self.plan["reactor_source_ref"], sha256="4" * 64)
        with self.assertRaisesRegex(ValueError, "fixed C5 native"): self.prepare()

    def test_runtime_observer_omission_refused(self):
        self.binding["runtime_overlay_refs"] = [ref for ref in self.binding["runtime_overlay_refs"]
            if not ref["path"].endswith("notification_runtime_adapter.py")]
        with self.assertRaisesRegex(ValueError, "complete C5/preparation"): self.prepare()

    def test_factory_omission_refused(self):
        self.lock["files"] = [ref for ref in self.lock["files"] if not ref["path"].startswith("factory/")]
        self.plan["source_lock_files"] = len(self.lock["files"])
        with self.assertRaisesRegex(ValueError, "factory absent"): self.prepare()

    def test_actual_candidate_python_added_without_lock_refused(self):
        (self.root / "candidate/late.py").write_text("# synthetic drift\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "actual C5 Python closure"): self.prepare()

    def test_runtime_source_unlocked_refused(self):
        self.lock["files"] = [ref for ref in self.lock["files"] if not ref["path"].endswith("streams.py")]
        self.plan["source_lock_files"] = len(self.lock["files"])
        with self.assertRaisesRegex(ValueError, "outside independent"): self.prepare()

    def test_changed_preparation_source_bytes_refused(self):
        with (self.root / "preparation/notification_runtime_adapter.py").open("ab") as stream:
            stream.write(b"# synthetic drift\n")
        with self.assertRaisesRegex(ValueError, "evidence size drift"): self.prepare()

    def test_common_source_drift_after_first_read_refused(self):
        original_rows = M._rows
        def drift(root, rows, label):
            result = original_rows(root, rows, label)
            if label == "overlay runtime refs":
                path = self.root / self.plan["cuda_event_source_ref"]["path"]
                with path.open("ab") as stream: stream.write(b"# late synthetic drift\n")
            return result
        with patch.object(M, "_rows", side_effect=drift):
            with self.assertRaisesRegex(ValueError, "evidence size drift"): self.prepare()

    def test_wrong_fixed_workload_and_no_holdout_refit(self):
        changes = (("prompt_tokens", 128), ("cached_prompt_tokens", 112), ("output_tokens", 1),
            ("measured_offset", 32), ("operations", 8), ("transfer_quantum_bytes", 8),
            ("external_warmup_output_tokens", 1), ("warmup_offsets", [0]), ("units", True))
        for name, value in changes:
            with self.subTest(name=name):
                original = deepcopy(self.plan["formula_contract"][name]); self.plan["formula_contract"][name] = value
                with self.assertRaisesRegex(ValueError, "finite workload"): self.prepare()
                self.plan["formula_contract"][name] = original

    def test_changed_holdout_prompt_refused(self):
        self.plan["formula_contract"]["entries"][2]["prompt_token_ids"] = deepcopy(
            self.plan["formula_contract"]["entries"][0]["prompt_token_ids"])
        with self.assertRaisesRegex(ValueError, "fixed prompt"): self.prepare()

    def test_changed_pair_order_refused(self):
        self.plan["formula_contract"]["entries"][1]["arm_order"] = "AB"
        with self.assertRaisesRegex(ValueError, "fixed prompt"): self.prepare()

    def test_changed_owner_bridge_and_parent_refused(self):
        for value in (dict(max_accepted_parents=64, bridge_is_none=True),
            dict(max_accepted_parents=8, bridge_is_none=False),
            dict(max_accepted_parents=8, bridge_is_none=1),
            dict(max_accepted_parents=True, bridge_is_none=True)):
            self.plan["common_owner_parameters"] = value
            with self.assertRaisesRegex(ValueError, "bridge=None"): self.prepare()

    def test_boolean_warmup_offset_refused(self):
        self.plan["formula_contract"]["warmup_offsets"] = [True]
        with self.assertRaisesRegex(ValueError, "integer warmup"): self.prepare()

    def test_reactor_moved_out_of_common_refused(self):
        value = self.plan["reactor_source_ref"]
        self.binding["runtime_common_refs"].remove(value); self.binding["runtime_overlay_refs"].append(value)
        with self.assertRaisesRegex(ValueError, "complete C5/preparation"): self.prepare()

    def test_duplicate_and_traversal_refused(self):
        self.lock["files"].append(deepcopy(self.lock["files"][0]))
        self.plan["source_lock_files"] += 1
        with self.assertRaisesRegex(ValueError, "duplicate source"): self.prepare()
        self.lock["files"].pop(); self.plan["source_lock_files"] -= 1
        self.lock["files"][0]["path"] = "../escape.py"
        with self.assertRaisesRegex(ValueError, "bounded project path"): self.prepare()

    def test_symlink_source_refused(self):
        name = self.plan["collector_source_ref"]["path"]; path = self.root / name
        target = self.root / "symlink-target.py"; path.rename(target)
        try: path.symlink_to(target)
        except OSError as error: self.skipTest("local symlink privilege absent: " + str(error))
        with self.assertRaisesRegex(ValueError, "symlink evidence"): self.prepare()

    def test_duplicate_json_keys_and_nonfinite_refused(self):
        for text in ('{"scope":1,"scope":2}', '{"value":NaN}'):
            path = self.root / "ambiguous.json"; path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError): M.FileRef.from_mapping(M.file_ref(self.root, "ambiguous.json")).json(self.root)

    def test_launch_and_native_issuance_block_before_input_read(self):
        with self.assertRaisesRegex(RuntimeError, M.BLOCKED): M.load_verified_single_file("/does/not/exist", "old.json")
        with self.assertRaisesRegex(RuntimeError, M.BLOCKED): M.launch_gpu(scope={"allowed": True})


class SyntheticFormulaContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("C5_COST_TEST_TEMP_ROOT", str(HERE)))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        shutil.copyfile(NATIVE / "native_conditional_cost.py", self.root / "analyzer.py")
        shutil.copyfile(ORIGINAL, self.root / "original.py")
        self.analyzer_ref = M.file_ref(self.root, "analyzer.py")
        self.original_ref = M.file_ref(self.root, "original.py")
        plan, windows, journal, drain = F.fixture()
        # Preserve the frozen workload and one physical file geometry. Durations
        # and native journal fields remain SYNTHETIC CPU metadata throughout.
        plan["transfer_quantum_bytes"] = 917504
        for window in windows:
            payload = window["independent_payload"]
            if payload is not None: payload["physical_bytes"] = 917504
        for event in journal["events"]:
            event["physical_bytes"] = 917504
            if event["result"] is not None: event["result"] = 917504
        for frame in journal["frames"]:
            for name in ("accepted", "completed", "inflight"):
                for stage in frame[name]: stage["nbytes"] = stage["ops"] * 917504
        for name in ("accepted", "completed"):
            for stage in drain[name]: stage["nbytes"] = stage["ops"] * 917504
        drain["transferred_bytes"] = [3 * 917504, 0, 0, 0]
        self.envelope = dict(origin="synthetic_cpu_contract", plan=plan, windows=windows, journal=journal, drain=drain)

    def analyze(self):
        return M.audit_synthetic_formula(self.root, envelope=self.envelope,
            analyzer_ref=self.analyzer_ref, original_estimator_ref=self.original_ref)

    def test_actual_original_6x128_algorithm_runs_without_native_issuance(self):
        result = self.analyze()
        self.assertTrue(result["synthetic_holdout_covered"])
        self.assertEqual(result["full_output_tokens"], 768)
        self.assertEqual(result["original_estimator_proof"]["source_sha256"], M.ORIGINAL_SHA256)
        self.assertFalse(result["original_estimator_proof"]["mathematical_expressions_changed"])
        self.assertFalse(result["native_execution_verified"])
        self.assertIsNone(result["effective_cost_upper_ns"])
        self.assertIsNone(result["effective_step_budget_ns"])

    def test_real_synthetic_numeric_math_and_a_only_budget_unchanged(self):
        analyzer = M._load_pinned_analyzer(self.root, self.analyzer_ref)
        result = analyzer.analyze_paired(self.envelope["plan"], self.envelope["windows"],
            journal=self.envelope["journal"], drain=self.envelope["drain"], original_source_path=self.root / "original.py")
        self.assertEqual(result["calibration_only_cost"]["baseline_ns"], 100)
        self.assertEqual(result["calibration_only_cost"]["incremental_or_joint_ns"], 15)
        self.assertEqual(result["calibration_only_cost"]["uncertainty_ns"], 5)
        self.assertEqual(result["calibration_predicted_upper_ns"], 120)
        self.assertEqual(LEGACY_RECEIPT._calibration_a_budget(self.envelope["plan"], result), 100)
        self.assertFalse(result["native_execution_verified"])

    def test_holdout_underprediction_remains_failed_without_refit(self):
        self.envelope["windows"][-1]["capture"]["event_witnesses"][16]["gpu_elapsed_ns"] = 130
        result = self.analyze()
        self.assertFalse(result["synthetic_holdout_covered"])
        self.assertFalse(result["holdout_used_to_refit"])
        self.assertFalse(result["conditional_cost_cell_qualified"])

    def test_missing_frame_is_rejected_by_original_algorithm(self):
        self.envelope["windows"][0]["capture"]["frames"].pop()
        with self.assertRaisesRegex(ValueError, "128 original steps"): self.analyze()

    def test_mismatched_outputs_is_rejected_by_original_algorithm(self):
        self.envelope["windows"][0]["output_token_ids"][5] = 99
        with self.assertRaisesRegex(ValueError, "sampled output"): self.analyze()

    def test_added_gpu_sync_is_rejected_by_original_algorithm(self):
        self.envelope["windows"][0]["capture"]["no_added_synchronization"] = False
        with self.assertRaisesRegex(ValueError, "query-only"): self.analyze()

    def test_wrong_io_result_is_rejected_by_original_algorithm(self):
        self.envelope["journal"]["events"][1]["result"] = 8
        with self.assertRaisesRegex(ValueError, "CQE result"): self.analyze()

    def test_native_text_cannot_upgrade_synthetic_envelope(self):
        self.envelope["origin"] = "native_gpu_recording"
        with self.assertRaisesRegex(ValueError, "explicit synthetic CPU"): self.analyze()

    def test_old_eight_byte_synthetic_geometry_refused(self):
        self.envelope["plan"]["transfer_quantum_bytes"] = 8
        with self.assertRaisesRegex(ValueError, "fixed C5 synthetic"): self.analyze()

    def test_frozen_estimator_drift_is_rejected(self):
        with (self.root / "original.py").open("ab") as stream: stream.write(b"# drift\n")
        with self.assertRaisesRegex(ValueError, "size drift"): self.analyze()

    def test_analyzer_digest_substitution_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "frozen analyzer SHA"):
            M.audit_synthetic_formula(self.root, envelope=self.envelope,
                analyzer_ref=dict(self.analyzer_ref, sha256="0" * 64), original_estimator_ref=self.original_ref)


if __name__ == "__main__":
    unittest.main()
