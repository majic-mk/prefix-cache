"""CPU fixtures exercise gates; none is production trace or GPU evidence."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import prerental_protocol as P


SOURCE_DIR = Path(__file__).resolve().parents[2] / "i_pilot_cpu_preparation_20261004" / "source_inputs"
CONTROLLED_SOURCE_DIR = Path(__file__).resolve().parents[1] / "source_inputs"


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.trace = SOURCE_DIR / "shared_storage_trace_replay.py"
        self.common = SOURCE_DIR / "prefix_cache_common.py"
        self.dataset = self.root / "fixture.jsonl"
        self.dataset.write_text('\n'.join(json.dumps({"prompt": text}) for text in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")) + '\n', encoding="utf-8")
        self.inspection = P.inspect_trace(self.dataset, self.trace, self.common)
        self.declaration = {"schema": "natural_trace_declaration_v1", "origin": "prospective_before_any_new_gpu_outcome",
                            "prefix_injection": False, "artificial_throttle": False, "per_request_cache_reset": False,
                            "initial_cache_state": "fresh_equal_namespace_preserved_through_whole_partition",
                            "dataset_sha256": self.inspection["dataset_sha256"], "partition_counts": {"calibration": 2, "development": 2, "evaluation": 2},
                            "output_tokens": 128, "max_model_len": 32768, "max_concurrency": 8, "schedule_seed": 42,
                            "arrival_rate": None, "model_manifest_sha256": "a" * 64}
        self.receipt = {"schema": "cpu_actual_tokenizer_family_receipt_v1", "synthetic_fixture": False,
                        "dataset_sha256": self.inspection["dataset_sha256"], "ordered_prompt_digest": self.inspection["ordered_prompt_digest"],
                        "model_manifest_sha256": "a" * 64, "tokenizer_source_refs": [self.ref(self.common)],
                        "family_proof": "all_sessions_documents_and_public_prefix_families_closed_before_partitioning",
                        "records": [{"request_id": row["request_id"], "prompt_sha256": row["prompt_sha256"],
                                     "prompt_token_ids": [100 + row["request_id"], 101], "prefix_family": f"family{row['request_id']}"} for row in self.inspection["records"]]}
        tokenizer_output = self.root / "fixture_tokenization_result.json"
        tokenizer_output.write_text(json.dumps({"schema": "actual_cpu_tokenizer_result_v1", "exit_code": 0, "synthetic_fixture": False,
                                   "dataset_sha256": self.inspection["dataset_sha256"], "ordered_prompt_digest": self.inspection["ordered_prompt_digest"],
                                   "tokenized_records_sha256": P.digest(self.receipt["records"])}), encoding="utf-8")
        self.receipt["tokenization_result_ref"] = self.ref(tokenizer_output)
        authority = self.root / "fixture_authority.json"
        authority.write_text(json.dumps({"schema": "independent_deadline_authority_v1", "origin": "independent_requirement_before_development",
                            "full_control_window_deadline_ns": 1000, "service_SLO": None}), encoding="utf-8")
        scope = {"host_id": "fixture_cpu_host", "boot_id": "00000000-0000-0000-0000-000000000001", "clock": "CLOCK_MONOTONIC"}
        self.deadline = {"schema": "independent_deadline_declaration_v1", "origin": "independent_requirement_before_development",
                         "full_control_window_deadline_ns": 1000, "authority_ref": self.ref(authority),
                         "declared_monotonic_ns": 10, "service_SLO": None, "clock_scope": scope}
        self.reserve = {"schema": "actual_development_control_reserve_v1", "actual_native_gpu_run": True,
                        "synthetic_fixture": False, "deadline_declaration_sha256": P.digest(self.deadline),
                        "first_development_monotonic_ns": 20, "phase": "development", "first_on_has_run": False,
                        "clock": "monotonic_ns_host_control_intervals", "all_control_categories": ["scheduler", "sampling", "output", "controller"],
                        "per_step_control_intervals": [[[30, 40], [35, 50], [80, 100]], [[200, 250]]], "reserve_uncertainty_ns": 5,
                        "clock_scope": scope}
        native = self.root / "fixture_native_reserve_source.json"
        native.write_text(json.dumps({"schema": "actual_development_reserve_source_v1", "actual_native_gpu_run": True,
                          "synthetic_fixture": False, "phase": "development", "control_intervals_sha256": P.digest(self.reserve["per_step_control_intervals"])}), encoding="utf-8")
        self.reserve["native_result_ref"] = self.ref(native)

    def ref(self, path):
        return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def update_tokenizer_result(self, receipt):
        path = Path(receipt["tokenization_result_ref"]["path"])
        value = json.loads(path.read_text(encoding="utf-8"))
        value.update(dataset_sha256=receipt["dataset_sha256"], ordered_prompt_digest=receipt["ordered_prompt_digest"], tokenized_records_sha256=P.digest(receipt["records"]))
        path.write_text(json.dumps(value), encoding="utf-8")
        receipt["tokenization_result_ref"] = self.ref(path)

    def tearDown(self):
        self.tmp.cleanup()

    def freeze(self, declaration=None, receipt=None):
        return P.freeze_trace(self.dataset, self.trace, self.common, declaration or self.declaration, receipt or self.receipt)

    def test_original_author_jsonl_sharegpt_parser_no_clients(self):
        self.dataset.write_text('\n' + json.dumps({"conversations": [{"from": "human", "value": " first "}, {"from": "gpt", "value": "ignored"}, {"from": "user", "value": "second"}]}) + '\n' + json.dumps({"prompt": " third "}) + '\n', encoding="utf-8")
        namespace = P.author_cpu_namespace(self.trace, self.common)
        self.assertEqual(namespace["load_trace_prompts"](str(self.dataset), None), ["first\n\nsecond", "third"])
        self.assertNotIn("AsyncOpenAI", namespace)
        self.assertNotIn("launch_cluster", namespace)

    def test_profiler_trace_is_not_natural_prompt_data(self):
        profile = self.root / "trace.json"
        profile.write_text(json.dumps({"traceEvents": [{"name": "execute_model"}]}), encoding="utf-8")
        self.assertEqual(P.inspect_trace(profile, self.trace, self.common)["accepted_prompt_count"], 0)

    def test_original_author_source_drift_is_rejected(self):
        bad = self.root / "author.py"
        bad.write_bytes(self.trace.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "source drift"):
            P.inspect_trace(self.dataset, bad, self.common)

    def test_dataset_expected_sha_drift_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dataset source drift"):
            P.inspect_trace(self.dataset, self.trace, self.common, "0" * 64)

    def test_all_requests_original_order_and_author_schedule_retained(self):
        before = self.dataset.read_bytes()
        manifest = self.freeze()
        self.assertEqual([r["request_id"] for r in manifest["records"]], list(range(6)))
        self.assertEqual([r["split"] for r in manifest["records"]], ["calibration"] * 2 + ["development"] * 2 + ["evaluation"] * 2)
        self.assertTrue(all(r["scheduled_ns"] == 0 for r in manifest["records"]))
        self.assertFalse(manifest["recorded_natural_arrival_claim"])
        self.assertEqual(self.dataset.read_bytes(), before)
        self.assertFalse(manifest["gpu_effect_qualified"])

    def test_poisson_schedule_is_exact_original_author_replay(self):
        declaration = copy.deepcopy(self.declaration)
        declaration["arrival_rate"] = 3.0
        manifest = self.freeze(declaration)
        namespace = P.author_cpu_namespace(self.trace, self.common)
        import math, random
        specs = namespace["build_global_specs"](6, 3.0, random.Random(42))
        self.assertEqual([r["scheduled_ns"] for r in manifest["records"]], [math.ceil(s.scheduled_time * 1e9) for s in specs])

    def test_prefix_family_leak_is_rejected_without_reordering(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["records"][2]["prefix_family"] = receipt["records"][0]["prefix_family"]
        self.update_tokenizer_result(receipt)
        with self.assertRaisesRegex(ValueError, "family leaks"):
            self.freeze(receipt=receipt)

    def test_identical_prompt_cross_partition_is_rejected(self):
        self.dataset.write_text('\n'.join(json.dumps({"prompt": t}) for t in ["alpha", "beta", "alpha", "delta", "epsilon", "zeta"]), encoding="utf-8")
        inspection = P.inspect_trace(self.dataset, self.trace, self.common)
        self.declaration["dataset_sha256"] = inspection["dataset_sha256"]
        self.receipt["dataset_sha256"] = inspection["dataset_sha256"]
        self.receipt["ordered_prompt_digest"] = inspection["ordered_prompt_digest"]
        for row, token in zip(inspection["records"], self.receipt["records"]):
            token["prompt_sha256"] = row["prompt_sha256"]
        self.update_tokenizer_result(self.receipt)
        with self.assertRaisesRegex(ValueError, "identical prompt leaks"):
            self.freeze()

    def test_injection_throttle_reset_or_posthoc_changes_are_rejected(self):
        for field, value in (("prefix_injection", True), ("artificial_throttle", True), ("per_request_cache_reset", True), ("origin", "after_evaluation")):
            declaration = copy.deepcopy(self.declaration)
            declaration[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.freeze(declaration)

    def test_dropped_reordered_or_untokenized_requests_are_rejected(self):
        for mutation in ("drop", "reorder", "missing_ids", "fixture", "missing_family_proof"):
            receipt = copy.deepcopy(self.receipt)
            if mutation == "drop": receipt["records"].pop()
            elif mutation == "reorder": receipt["records"].reverse()
            elif mutation == "missing_ids": receipt["records"][0]["prompt_token_ids"] = None
            elif mutation == "fixture": receipt["synthetic_fixture"] = True
            else: receipt["family_proof"] = "exact_text_hash_only"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.freeze(receipt=receipt)

    def test_context_overflow_single_token_and_bool_values_are_rejected(self):
        for field, value in (("max_model_len", 2), ("output_tokens", 1), ("max_concurrency", True)):
            declaration = copy.deepcopy(self.declaration)
            declaration[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.freeze(declaration)

    def test_qualification_text_manifest_is_bounded_and_not_effect_data(self):
        declaration = dict(self.declaration, schema="natural_off_qualification_declaration_v1", max_prompts=4)
        result = P.freeze_qualification_trace(self.dataset, self.trace, self.common, declaration)
        self.assertEqual([r["prompt"] for r in result["records"]], ["alpha", "beta", "gamma", "delta"])
        self.assertEqual(result["unselected_tail_count"], 2)
        self.assertTrue(all(r["prefix_family"] is None and r["prompt_token_ids"] is None for r in result["records"]))
        self.assertFalse(result["fit_or_evaluation_input_allowed"])
        self.assertFalse(result["gpu_effect_qualified"])

    def test_qualification_limits_and_empty_profiler_refuse(self):
        declaration = dict(self.declaration, schema="natural_off_qualification_declaration_v1", max_prompts=33)
        with self.assertRaises(ValueError): P.freeze_qualification_trace(self.dataset, self.trace, self.common, declaration)
        declaration["max_prompts"] = 6
        declaration["max_concurrency"] = 9
        with self.assertRaises(ValueError): P.freeze_qualification_trace(self.dataset, self.trace, self.common, declaration)

    def test_control_interval_union_avoids_double_reserve_accounting(self):
        self.assertEqual(P.union_ns([[0, 10], [5, 20], [30, 35], [20, 22]]), 27)

    def test_independent_budget_subtracts_measured_reserve_without_SLO_claim(self):
        budget = P.freeze_development_budget(self.deadline, self.reserve)
        self.assertEqual(budget["non_gpu_reserve_ns"], 55)
        self.assertEqual(budget["internal_step_budget_ns"], 945)
        self.assertFalse(budget["formal_goodput_allowed"])
        self.assertFalse(budget["derived_from_A_max"])

    def test_Amax_posthoc_deadline_cannot_be_relabelled(self):
        for field, value in (("origin", "observed_A_max"), ("declared_monotonic_ns", 21), ("full_control_window_deadline_ns", None)):
            deadline = copy.deepcopy(self.deadline)
            deadline[field] = value
            reserve = dict(self.reserve, deadline_declaration_sha256=P.digest(deadline))
            with self.subTest(field=field), self.assertRaises(ValueError): P.freeze_development_budget(deadline, reserve)

    def test_wrong_clock_phase_fixture_partial_control_and_post_on_rejected(self):
        for field, value in (("clock", "CUDA_Event_and_wall_time_subtraction"), ("phase", "evaluation"), ("synthetic_fixture", True),
                             ("first_on_has_run", True), ("actual_native_gpu_run", False), ("all_control_categories", ["scheduler"])):
            reserve = copy.deepcopy(self.reserve)
            reserve[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): P.freeze_development_budget(self.deadline, reserve)

    def test_no_reserve_or_no_budget_cannot_be_ordinary_admission(self):
        for value in (None, [], [[[20, 2000]]]):
            reserve = dict(self.reserve, per_step_control_intervals=value)
            with self.subTest(value=value), self.assertRaises(ValueError): P.freeze_development_budget(self.deadline, reserve)

    def test_rental_is_diagnostic_only_when_all_CPU_inputs_bound(self):
        decision = P.rental_decision(workload_bound=True, strong_runner_cpu_ready=True, source_lock_verified=True,
                                     guarded_off_max_seconds=320, remaining_seconds=P.REMAINING_SNAPSHOT_SECONDS)
        self.assertEqual(decision["status"], "BOUNDED_STRONG_U_OFF_QUALIFICATION_HAS_DIAGNOSTIC_VALUE")
        self.assertEqual(len(decision["later_effect_gate_missing"]), 3)
        self.assertFalse(decision["benefit_proved"])
        self.assertEqual(decision["investment_gain_target_fraction"], 0.10)

    def test_old_planner_off_cost_and_missing_workload_block_rental(self):
        decision = P.rental_decision(workload_bound=False, strong_runner_cpu_ready=True, source_lock_verified=True,
                                     guarded_off_max_seconds=320, remaining_seconds=P.REMAINING_SNAPSHOT_SECONDS, old_planner_off_cost_requested=True)
        self.assertEqual(decision["status"], "BLOCK_RENTAL_UNTIL_CPU_INPUTS_BOUND")
        self.assertIn("planner_off_cal01_does_not_cover_strong_U_domain", decision["prerental_blockers"])

    def test_bounded_off_reservation_does_not_promise_remaining_stages(self):
        result = P.budget_plan(P.REMAINING_SNAPSHOT_SECONDS, [{"id": "strong-u-off01", "execution_seconds": 300, "cleanup_seconds": 20}])
        self.assertEqual(result["maximum_reserved_seconds"], 320)
        self.assertTrue(result["fits_snapshot"])
        self.assertFalse(result["actual_reservation_performed"])
        self.assertFalse(result["complete_effect_result_promised"])

    def test_oversized_budget_duplicate_jobs_and_overwrite_rejected(self):
        result = P.budget_plan(100, [{"id": "off", "execution_seconds": 300, "cleanup_seconds": 20}])
        self.assertFalse(result["fits_snapshot"])
        with self.assertRaises(ValueError): P.budget_plan(100, [{"id": "off", "execution_seconds": 1, "cleanup_seconds": 1}] * 2)
        target = self.root / "evidence.json"
        P.append_json(target, {"x": 1})
        with self.assertRaises(FileExistsError): P.append_json(target, {"x": 2})

    def test_nonfinite_bool_negative_or_expanded_remaining_budget_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf"), True, -1, 28801):
            with self.subTest(value=value), self.assertRaises(ValueError):
                P.rental_decision(workload_bound=True, strong_runner_cpu_ready=True, source_lock_verified=True,
                                  guarded_off_max_seconds=320, remaining_seconds=value)

    def test_nonhex_or_unjoined_authority_cannot_freeze_budget(self):
        for mutation in ("nonhex", "wrong_byte_hash", "missing_file"):
            declaration = copy.deepcopy(self.deadline)
            if mutation == "nonhex": declaration["authority_ref"]["sha256"] = "z" * 64
            elif mutation == "wrong_byte_hash": declaration["authority_ref"]["sha256"] = "0" * 64
            else: declaration["authority_ref"]["path"] = str(self.root / "missing.json")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): P.freeze_development_budget(declaration, self.reserve)

    def test_unvalidated_SLO_and_wrong_clock_scope_rejected(self):
        for value in ({"unvalidated": "fixture"}, {}, {"schema": "independent_service_SLO_v1", "origin": "independent_requirement_before_development", "TTFT_ns": True, "request_ITL_P95_ns": 1}):
            declaration = dict(self.deadline, service_SLO=value)
            with self.subTest(SLO=value), self.assertRaises(ValueError): P.freeze_development_budget(declaration, self.reserve)
        reserve = copy.deepcopy(self.reserve)
        reserve["clock_scope"]["boot_id"] = "00000000-0000-0000-0000-000000000002"
        with self.assertRaisesRegex(ValueError, "same host and boot"): P.freeze_development_budget(self.deadline, reserve)

    def test_nonempty_self_reported_tokenizer_refs_do_not_qualify(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["tokenizer_source_refs"] = [{"sha256": "b" * 64}]
        with self.assertRaises(ValueError): self.freeze(receipt=receipt)

    def test_controlled_P3_off_branch_never_claims_natural_or_effect(self):
        decision = P.rental_decision(workload_bound=True, strong_runner_cpu_ready=True, source_lock_verified=True,
                                    guarded_off_max_seconds=320, remaining_seconds=P.REMAINING_SNAPSHOT_SECONDS,
                                    workload_kind="controlled_original_P3_mechanism")
        self.assertEqual(decision["status"], "BOUNDED_STRONG_U_OFF_QUALIFICATION_HAS_DIAGNOSTIC_VALUE")
        self.assertFalse(decision["natural_problem_claim_allowed"])
        self.assertFalse(decision["ordinary_interference_or_effect_qualified"])

    def test_actual_original_P3_artifact_freezes_exact_tokens_specs_and_arrivals(self):
        dataset = CONTROLLED_SOURCE_DIR / "low_contention.json"
        index = CONTROLLED_SOURCE_DIR / "index.json"
        refs = CONTROLLED_SOURCE_DIR / "P3_CONTROLLED_AND_CLOSURE_SOURCE_INPUTS.json"
        before = dataset.read_bytes()
        manifest = P.freeze_original_p3_qualification(dataset, index, refs, self.trace, self.common, "a" * 64)
        original = json.loads(before)
        self.assertEqual(len(manifest["records"]), 12)
        for request, source in zip(manifest["records"], original["requests"]):
            self.assertEqual(request["prompt_token_ids"], source["prompt_token_ids"])
            self.assertEqual(request["scheduled_time_s"], source["scheduled_time"])
            self.assertEqual(request["original_request_spec"]["reuse_source_id"], source["reuse_source_id"])
            self.assertEqual(request["max_tokens"], 128)
        self.assertEqual(dataset.read_bytes(), before)
        self.assertFalse(manifest["natural_trace_bound"])
        self.assertFalse(manifest["fit_or_evaluation_input_allowed"])
        self.assertFalse(manifest["gpu_effect_qualified"])

    def test_original_P3_byte_drift_and_wrong_index_reject(self):
        dataset = CONTROLLED_SOURCE_DIR / "low_contention.json"
        index = CONTROLLED_SOURCE_DIR / "index.json"
        refs = CONTROLLED_SOURCE_DIR / "P3_CONTROLLED_AND_CLOSURE_SOURCE_INPUTS.json"
        changed = self.root / "changed.json"
        changed.write_bytes(dataset.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "request bytes drift"):
            P.freeze_original_p3_qualification(changed, index, refs, self.trace, self.common, "a" * 64)
        with self.assertRaisesRegex(ValueError, "index bytes drift"):
            P.freeze_original_p3_qualification(dataset, changed, refs, self.trace, self.common, "a" * 64)

    def test_original_P3_actual_source_sizes_and_path_binding_reject(self):
        dataset = CONTROLLED_SOURCE_DIR / "low_contention.json"
        index = CONTROLLED_SOURCE_DIR / "index.json"
        source = json.loads((CONTROLLED_SOURCE_DIR / "P3_CONTROLLED_AND_CLOSURE_SOURCE_INPUTS.json").read_text(encoding="utf-8"))
        for field, value in (("bytes", 94012), ("path", "/another_project/different.json")):
            refs = copy.deepcopy(source)
            next(row for row in refs["files"] if row["sha256"] == P.ORIGINAL_P3_LOW_CONTENTION_SHA256)[field] = value
            path = self.root / f"changed_ref_{field}.json"
            path.write_text(json.dumps(refs), encoding="utf-8")
            with self.subTest(field=field), self.assertRaises(ValueError):
                P.freeze_original_p3_qualification(dataset, index, path, self.trace, self.common, "a" * 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
