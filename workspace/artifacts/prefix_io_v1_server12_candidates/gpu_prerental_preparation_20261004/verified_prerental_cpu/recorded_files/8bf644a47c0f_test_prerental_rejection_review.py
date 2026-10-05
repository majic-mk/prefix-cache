"""Independent CPU rejection tests: fixtures never constitute runtime evidence."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "protocol/prerental_protocol.py"
AUTHOR_INPUT = HERE.parents[1] / "i_pilot_cpu_preparation_20261004/source_inputs"
spec = importlib.util.spec_from_file_location("_independent_prerent_protocol_review", SOURCE)
P = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = P
spec.loader.exec_module(P)


def fixture_ref(path):
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def development_fixture(root):
    authority = root / "CPU_FIXTURE_authority.json"
    authority.write_text(json.dumps({"schema": "independent_deadline_authority_v1", "origin": "independent_requirement_before_development",
                                     "full_control_window_deadline_ns": 1000, "service_SLO": None}), encoding="utf-8")
    scope = {"host_id": "CPU_FIXTURE_HOST_A", "boot_id": "00000000-0000-0000-0000-000000000001", "clock": "CLOCK_MONOTONIC"}
    deadline = {"schema": "independent_deadline_declaration_v1", "origin": "independent_requirement_before_development",
                "full_control_window_deadline_ns": 1000, "authority_ref": fixture_ref(authority),
                "declared_monotonic_ns": 10, "service_SLO": None,
                "clock_scope": scope}
    reserve = {"schema": "actual_development_control_reserve_v1", "actual_native_gpu_run": True,
               "synthetic_fixture": False, "deadline_declaration_sha256": P.digest(deadline),
               "first_development_monotonic_ns": 20, "phase": "development", "first_on_has_run": False,
               "clock": "monotonic_ns_host_control_intervals", "all_control_categories": ["scheduler", "sampling", "output", "controller"],
               "per_step_control_intervals": [[[30, 40], [35, 50], [80, 100]], [[200, 250]]], "reserve_uncertainty_ns": 5,
               "clock_scope": copy.deepcopy(deadline["clock_scope"])}
    native = root / "CPU_FIXTURE_native_reserve_source.json"
    native.write_text(json.dumps({"schema": "actual_development_reserve_source_v1", "actual_native_gpu_run": True,
                                 "synthetic_fixture": False, "phase": "development",
                                 "control_intervals_sha256": P.digest(reserve["per_step_control_intervals"])}), encoding="utf-8")
    reserve["native_result_ref"] = fixture_ref(native)
    return deadline, reserve


class IndependentRejections(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prerent_review_fixture_")
        self.deadline, self.reserve = development_fixture(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def test_valid_math_fixture_is_a_positive_control_without_runtime_promotion(self):
        result = P.freeze_development_budget(self.deadline, self.reserve)
        self.assertEqual(result["internal_step_budget_ns"], 945)
        self.assertFalse(result["formal_goodput_allowed"])
        self.assertTrue(result["native_qualification_verifier_required_separately"])

    def test_rental_rejects_nonfinite_or_bool_budget(self):
        for value in (float("nan"), float("inf"), -float("inf"), True):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                P.rental_decision(workload_bound=True, strong_runner_cpu_ready=True, source_lock_verified=True,
                                  guarded_off_max_seconds=320, remaining_seconds=value)

    def test_rental_rejects_above_original_cumulative_limit(self):
        with self.assertRaises(ValueError):
            P.rental_decision(workload_bound=True, strong_runner_cpu_ready=True, source_lock_verified=True,
                              guarded_off_max_seconds=320, remaining_seconds=30000)

    def test_independent_authority_digest_requires_hexadecimal(self):
        deadline, reserve = copy.deepcopy(self.deadline), copy.deepcopy(self.reserve)
        deadline["authority_ref"]["sha256"] = "z" * 64
        reserve["deadline_declaration_sha256"] = P.digest(deadline)
        with self.assertRaises(ValueError):
            P.freeze_development_budget(deadline, reserve)

    def test_arbitrary_SLO_cannot_become_formal_goodput_permission(self):
        deadline, reserve = copy.deepcopy(self.deadline), copy.deepcopy(self.reserve)
        deadline["service_SLO"] = {"unvalidated": "CPU_FIXTURE"}
        reserve["deadline_declaration_sha256"] = P.digest(deadline)
        try:
            result = P.freeze_development_budget(deadline, reserve)
        except ValueError:
            return
        self.assertFalse(result["formal_goodput_allowed"], "unvalidated SLO cannot establish formal goodput eligibility")

    def test_migration_monotonic_clock_scope_mismatch_is_rejected(self):
        deadline, reserve = copy.deepcopy(self.deadline), copy.deepcopy(self.reserve)
        reserve["clock_scope"] = {"host_id": "CPU_FIXTURE_HOST_B", "boot_id": "00000000-0000-0000-0000-000000000002", "clock": "CLOCK_MONOTONIC"}
        with self.assertRaises(ValueError):
            P.freeze_development_budget(deadline, reserve)

    def test_qualification_rejects_short_output_for_this_frozen_route(self):
        with tempfile.TemporaryDirectory(prefix="prerent_review_fixture_") as temporary:
            dataset = Path(temporary) / "fixture.jsonl"
            dataset.write_text(json.dumps({"prompt": "CPU fixture prompt, not a natural dataset"}) + "\n", encoding="utf-8")
            declaration = {"schema": "natural_off_qualification_declaration_v1", "origin": "prospective_before_any_new_gpu_outcome",
                           "prefix_injection": False, "artificial_throttle": False, "per_request_cache_reset": False,
                           "initial_cache_state": "fresh_equal_namespace_preserved_through_whole_partition",
                           "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(), "max_prompts": 1,
                           "output_tokens": 2, "max_concurrency": 1, "schedule_seed": 42, "arrival_rate": None,
                           "model_manifest_sha256": "a" * 64}
            with self.assertRaises(ValueError):
                P.freeze_qualification_trace(dataset, AUTHOR_INPUT / "shared_storage_trace_replay.py", AUTHOR_INPUT / "prefix_cache_common.py", declaration)


def main():
    global AUTHOR_INPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=AUTHOR_INPUT,
                        help="Explicit author parser/helper source directory; local default is preserved.")
    args = parser.parse_args()
    AUTHOR_INPUT = args.source_dir.resolve(strict=True)
    if not AUTHOR_INPUT.is_dir() or AUTHOR_INPUT.is_symlink():
        raise ValueError("actual author CPU source directory required")
    author_sources = [AUTHOR_INPUT / name for name in ("shared_storage_trace_replay.py", "prefix_cache_common.py")]
    author_before = [fixture_ref(path) for path in author_sources]
    source_before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentRejections))
    source_after = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    author_after = [fixture_ref(path) for path in author_sources]
    forbidden = sorted(name for name in sys.modules if name.split(".")[0] in {"torch", "vllm", "py_kvcache", "openai"})
    okay = result.wasSuccessful() and source_before == source_after and author_before == author_after and not forbidden
    record = {"schema": "gpu_prerent_independent_rejection_tests_v1", "status": "PASS" if okay else "FAIL",
              "tests_run": result.testsRun, "failures": [{"test": str(test), "traceback": trace} for test, trace in result.failures],
              "errors": [{"test": str(test), "traceback": trace} for test, trace in result.errors], "skips": len(result.skipped),
              "source_sha256_before": source_before, "source_sha256_after": source_after, "source_unchanged": source_before == source_after,
              "actual_author_source_dir": str(AUTHOR_INPUT), "author_source_refs_before": author_before,
              "author_source_refs_after": author_after, "author_sources_unchanged": author_before == author_after,
              "elapsed_seconds": time.monotonic() - started, "forbidden_modules_imported": forbidden,
              "GPU_operations": 0, "RPC_operations": 0, "fixtures_are_actual_GPU_or_natural_dataset_evidence": False}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": record["status"], "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "GPU_operations": 0}))
    return 0 if okay else 1


if __name__ == "__main__":
    raise SystemExit(main())
