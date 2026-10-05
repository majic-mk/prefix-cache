"""CPU negative/fixture tests only. No successful GPU qualification fixture."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

import host_control_observer as h
import reserve_join as r


def metadata_boundary(value=None):
    return value


def failing_boundary(error):
    raise error


def execute_model():
    return "FORBIDDEN_WHOLE_GPU_METHOD"


class ObserverCPUTests(unittest.TestCase):
    def observer(self, *, clock=None, max_steps=4096, max_intervals=32):
        observer = h.HostControlObserver("CPU_test_only", _fixture_clock=clock,
                                         max_steps=max_steps, max_intervals_per_step=max_intervals)
        for category in h.CATEGORIES:
            observer.bind(category, category, metadata_boundary, source_ref=h.closed_ref(__file__),
                          boundary_kind=h.BOUNDARIES[category])
        return observer

    def complete_fixture(self):
        observer = self.observer(clock=h.FixtureClock([100, 110, 120, 130, 140, 150, 160, 170]))
        for category in h.CATEGORIES:
            self.assertEqual(observer.call(category, 7, category, metadata_boundary, category), category)
        return observer.export()

    def test_actual_host_clock_is_typed_and_never_GPU_qualified(self):
        observer = self.observer()
        for category in h.CATEGORIES:
            observer.call(category, 0, category, metadata_boundary, category)
        record = observer.export()
        self.assertTrue(record["valid"])
        self.assertIs(record["synthetic_fixture"], False)
        self.assertEqual(record["origin"], "actual_host_method_observation")
        for row in record["per_step"][0]["intervals"]:
            self.assertIs(type(row["start_ns"]), int)
            self.assertGreater(row["end_ns"], row["start_ns"])
        checked = h.checked_intervals(record, require_production_clock=False)
        self.assertEqual(checked["ordinals"], [0])
        self.assertIs(record["actual_native_gpu_run"], False)
        self.assertIs(record["formal_goodput_allowed"], False)
        self.assertIs(record["pure_CPU_compute_claim"], False)

    def test_fixture_is_explicit_and_not_native_reserve(self):
        record = self.complete_fixture()
        self.assertEqual(h.checked_intervals(record, require_production_clock=False)["ordinals"], [7])
        with self.assertRaisesRegex(ValueError, "actual Linux"):
            h.checked_intervals(record)

    def test_missing_actual_controller_interval_rejected(self):
        record = self.complete_fixture()
        record["per_step"][0]["intervals"].pop()
        with self.assertRaisesRegex(ValueError, "missing|all four"):
            h.checked_intervals(record, require_production_clock=False)

    def test_empty_interval_list_is_unknown_not_zero(self):
        record = self.complete_fixture()
        record["per_step"][0]["intervals"] = []
        with self.assertRaisesRegex(ValueError, "missing"):
            h.checked_intervals(record, require_production_clock=False)

    def test_exact_mixed_cuda_clock_rejected(self):
        record = self.complete_fixture()
        record["per_step"][0]["intervals"][1]["clock_scope"]["clock"] = "torch.cuda.Event.elapsed_time"
        with self.assertRaisesRegex(ValueError, "mixed host boot or CUDA"):
            h.checked_intervals(record, require_production_clock=False)

    def test_exact_other_host_and_boot_rejected(self):
        for field, value in (("host_id", "other-host"), ("boot_id", "00000000-0000-0000-0000-000000000002")):
            record = self.complete_fixture()
            record["per_step"][0]["intervals"][0]["clock_scope"][field] = value
            with self.assertRaisesRegex(ValueError, "mixed host"):
                h.checked_intervals(record, require_production_clock=False)

    def test_GPU_subtraction_flag_rejected(self):
        record = self.complete_fixture()
        record["GPU_delta_subtracted"] = True
        with self.assertRaisesRegex(ValueError, "no mixed CUDA"):
            h.checked_intervals(record, require_production_clock=False)

    def test_whole_GPU_or_engine_method_not_wrapped(self):
        observer = h.HostControlObserver("CPU_test_only")
        with self.assertRaisesRegex(ValueError, "whole GPU"):
            observer.bind("bad", "sampling", execute_model, source_ref=h.closed_ref(__file__),
                          boundary_kind=h.BOUNDARIES["sampling"])

    def test_original_return_and_exception_are_preserved(self):
        clock = h.FixtureClock([100, 110, 120, 130])
        observer = self.observer(clock=clock)
        marker = object()
        self.assertIs(observer.call("scheduler", 0, "scheduler", metadata_boundary, marker), marker)
        observer.bind("raises", "output", failing_boundary, source_ref=h.closed_ref(__file__),
                      boundary_kind=h.BOUNDARIES["output"])
        error = RuntimeError("original failure")
        with self.assertRaises(RuntimeError) as caught:
            observer.call("output", 0, "raises", failing_boundary, error)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(observer.export()["per_step"][0]["intervals"]), 2)

    def test_failed_clock_does_not_replace_original_exception(self):
        observer = self.observer(clock=h.FixtureClock([100, 90]))
        observer.bind("raises", "output", failing_boundary, source_ref=h.closed_ref(__file__),
                      boundary_kind=h.BOUNDARIES["output"])
        error = RuntimeError("original failure wins")
        with self.assertRaises(RuntimeError) as caught:
            observer.call("output", 0, "raises", failing_boundary, error)
        self.assertIs(caught.exception, error)
        record = observer.export()
        self.assertIs(record["valid"], False)
        self.assertTrue(any("backwards" in value for value in record["failures"]))

    def test_unfinished_interval_rejected(self):
        observer = self.observer(clock=h.FixtureClock([100]))
        observer.begin("scheduler", 0, "scheduler")
        record = observer.export()
        with self.assertRaisesRegex(ValueError, "unfinished"):
            h.checked_intervals(record, require_production_clock=False)

    def test_cross_thread_token_rejected(self):
        observer = self.observer(clock=h.FixtureClock([100, 110]))
        token = observer.begin("controller", 0, "controller")
        thread = threading.Thread(target=observer.end, args=(token,))
        thread.start()
        thread.join()
        self.assertFalse(observer.export()["valid"])

    def test_duplicate_token_cannot_count_twice(self):
        observer = self.observer(clock=h.FixtureClock([100, 110]))
        token = observer.begin("scheduler", 0, "scheduler")
        observer.end(token)
        observer.end(token)
        record = observer.export()
        self.assertFalse(record["valid"])
        self.assertEqual(len(record["per_step"][0]["intervals"]), 1)

    def test_nested_categories_keep_all_intervals_without_wall_delta(self):
        observer = self.observer(clock=h.FixtureClock([100, 110, 120, 130, 140, 150, 160, 170]))
        with observer.observe("scheduler", 3, "scheduler"):
            for category in ("sampling", "output", "controller"):
                observer.call(category, 3, category, metadata_boundary)
        record = observer.export()
        checked = h.checked_intervals(record, require_production_clock=False)
        intervals = checked["per_step_control_intervals"][0]
        self.assertEqual(intervals[0], [100, 170])
        protocol_path = Path(__file__).resolve().parents[2] / "protocol" / "prerental_protocol.py"
        protocol = r._load_pinned(h.closed_ref(protocol_path), r.PROTOCOL_SHA, "CPU_protocol_math")
        self.assertEqual(protocol.union_ns(intervals), 70)

    def test_boolean_nan_and_absent_span_are_not_timestamps(self):
        for bad in (True, float("nan"), None):
            record = self.complete_fixture()
            record["per_step"][0]["intervals"][0]["start_ns"] = bad
            with self.assertRaises(ValueError):
                h.checked_intervals(record, require_production_clock=False)

    def test_overflow_invalidates_observation_and_calls_original(self):
        observer = self.observer(clock=h.FixtureClock([100, 110]), max_steps=1)
        self.assertEqual(observer.call("scheduler", 0, "scheduler", metadata_boundary, "one"), "one")
        self.assertEqual(observer.call("scheduler", 1, "scheduler", metadata_boundary, "two"), "two")
        record = observer.export()
        self.assertFalse(record["valid"])
        self.assertEqual(len(record["per_step"]), 1)

    def test_source_bytes_and_hex_pin_are_enforced(self):
        for sha in ("z" * 64, "0" * 64):
            ref = h.closed_ref(__file__)
            ref["sha256"] = sha
            with self.assertRaises(ValueError):
                h.read_ref(ref)

    def test_deadline_missing_closed_even_for_complete_fixture(self):
        with self.assertRaisesRegex(ValueError, "independent deadline missing"):
            h.reserve_candidate(self.complete_fixture(), declaration=None, protocol=None)

    def test_native_join_independent_expected_anchors_required(self):
        with self.assertRaisesRegex(ValueError, "independent prelaunch"):
            r.join_completed_development(Path(__file__).parent, plan_ref={}, expected_plan_ref=None,
                guard_ref={}, expected_guard_ref=None, observation_ref={}, native_result_ref={},
                output_directory=Path(__file__).parent)

    def test_off_bookkeeping_can_never_issue_I_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "CPU_fixture_off_plan.json"
            path.write_text(json.dumps(dict(schema="guarded_host_control_development_plan_v1", phase="development",
                first_on_has_run=False, evaluation_used=False, project_root=root.as_posix(), mode="off", arm="U",
                native_phase="qualification")), encoding="utf-8")
            ref = h.closed_ref(path)
            with self.assertRaisesRegex(ValueError, "off bookkeeping"):
                r.join_completed_development(root, plan_ref=ref, expected_plan_ref=ref,
                    guard_ref=ref, expected_guard_ref=ref, observation_ref={}, native_result_ref={}, output_directory=root)

    def test_existing_reserve_read_rejection_writes_zero_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "independent prelaunch"):
                r.verify_existing_native_reserve(root, reserve_source_ref={}, reserve_receipt_ref={}, native_verification_ref={},
                    plan_ref={}, expected_plan_ref=None, guard_ref={}, expected_guard_ref=None,
                    observation_ref={}, native_result_ref={})
            self.assertEqual(list(root.iterdir()), [])

    def test_U_shadow_cannot_replace_actual_I_controller_development(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "CPU_fixture_U_shadow_plan.json"
            path.write_text(json.dumps(dict(schema="guarded_host_control_development_plan_v1", phase="development",
                first_on_has_run=False, evaluation_used=False, project_root=root.as_posix(), mode="shadow", arm="U",
                native_phase="shadow")), encoding="utf-8")
            ref = h.closed_ref(path)
            with self.assertRaisesRegex(ValueError, "qualified I"):
                r.verify_development_inputs(root, plan_ref=ref, expected_plan_ref=ref,
                    guard_ref=ref, expected_guard_ref=ref, observation_ref={}, native_result_ref={})

    def _CPU_negative_coverage_fixture(self):
        cell = ("GPU-fixture", "domain-fixture", "model-fixture", "layout-fixture", 1, 1, 0, 144, 917504)
        identity = SimpleNamespace(cells=(cell,), source_lock_sha256="fixture-source-lock")
        plan = {"coverage_contract": {"schema": "finite_natural_eligible_host_reserve_v1",
                "selection": "all_actual_original_ordinary_preload_preview_windows_in_exact_issued_cells",
                "required_signatures": [list(cell)], "unknown_domain_fallback": "U",
                "synthetic_candidates_allowed": False, "condition_max_age_ns": 100}}
        native = {"host_controller_coverage": {"kind": "same_qualified_I_lookup_binding_path_observed_only_no_issue",
                "native_step_ordinals": [0], "ordinary_IO_issued_by_observer": False,
                "qualified_identity_source_lock_sha256": identity.source_lock_sha256, "qualified_cells": [list(cell)],
                "overflow": False, "unknown_coverage": False, "preview_windows": [], "attempt_count": 0}}
        return plan, native, {}, {"native_step_ordinals": [0]}, identity

    def test_no_real_eligible_preview_stays_blocked(self):
        with self.assertRaisesRegex(ValueError, "complete actual ordinary preview"):
            r._eligible_coverage(*self._CPU_negative_coverage_fixture(), table=None)

    def test_unknown_or_overflow_eligible_coverage_stays_blocked(self):
        for field in ("unknown_coverage", "overflow"):
            values = self._CPU_negative_coverage_fixture()
            values[1]["host_controller_coverage"][field] = True
            with self.assertRaisesRegex(ValueError, "missing/unknown/overflow"):
                r._eligible_coverage(*values, table=None)

    def test_post_selection_of_fast_eligible_states_rejected(self):
        values = self._CPU_negative_coverage_fixture()
        values[0]["coverage_contract"]["selection"] = "select_fastest_after_results"
        with self.assertRaisesRegex(ValueError, "prospective finite natural"):
            r._eligible_coverage(*values, table=None)

    def sparse_CPU_fixture(self):
        observer = self.observer(clock=h.FixtureClock(list(range(100, 280, 10))))
        observer.bind("qualified_finite_controller_preview", "controller", metadata_boundary,
                      source_ref=h.closed_ref(__file__), boundary_kind=h.BOUNDARIES["controller"])
        for ordinal in (0, 1):
            for category in h.CATEGORIES:
                observer.call(category, ordinal, category, metadata_boundary)
            if ordinal == 0:
                observer.call("controller", ordinal, "qualified_finite_controller_preview", metadata_boundary)
        record = observer.export()
        return record, [dict(attempt_id=1, native_step_ordinal=0, begin_ns=181, end_ns=189, clock_scope=record["clock_scope"])]

    def test_sparse_actual_preview_interval_shape_accepts_CPU_fixture_without_promotion(self):
        record, windows = self.sparse_CPU_fixture()
        self.assertEqual(h.checked_intervals(record, require_production_clock=False)["ordinals"], [0, 1])
        result = r.match_preview_interval_shape(record, windows, clock_scope=record["clock_scope"])
        self.assertEqual(result["associated_attempts"], 1)
        self.assertEqual(result["full_native_step_ordinals"], [0, 1])
        self.assertIs(result["actual_native_gpu_run"], False)
        self.assertIs(result["formal_goodput_allowed"], False)

    def test_sparse_real_preview_omission_is_rejected(self):
        record, _ = self.sparse_CPU_fixture()
        with self.assertRaisesRegex(ValueError, "all actual preview"):
            r.match_preview_interval_shape(record, [], clock_scope=record["clock_scope"])

    def test_extra_scheduler_step_is_not_deleted_to_match_GPU_capture(self):
        capture = dict(run_id="CPU_test_only", origin="native_gpu_recording", valid=True,
            scope="bounded_original_full_step_stream_v1", failures=[], pending_event_pairs=0, open_event_pair=False,
            no_added_synchronization=True, cross_clock_absolute_mapping=False, frames=[{}]*128, event_witnesses=[{}]*128)
        with self.assertRaisesRegex(ValueError, "all actual native steps"):
            r._capture(capture, {}, run_id="CPU_test_only", expected_ordinals=list(range(129)), source_ref={})

    def test_empty_prefill_output_is_explicitly_unsupported_not_performance_success(self):
        frame = dict(native_step_ordinal=0, start_ns=10, end_ns=20, gpu_elapsed_ns=None, existing_io=None,
            new_io=None, intended_timing_scope="full_decode_step", outputs=[],
            prepared=dict(native_step_ordinal=0, batch=1, active_decode=0, prefill_tokens=128, context_length=0))
        witness = dict(native_step_ordinal=0, start_record_before_ns=1, start_record_after_ns=5,
            start_completed_query_ns=5, end_record_before_ns=30, end_record_after_ns=40, end_completed_query_ns=50,
            event_elapsed_source="torch.cuda.Event.elapsed_time", gpu_elapsed_ns=20)
        capture = dict(run_id="CPU_test_only", origin="native_gpu_recording", valid=True,
            scope="bounded_original_full_step_stream_v1", failures=[], pending_event_pairs=0, open_event_pair=False,
            no_added_synchronization=True, cross_clock_absolute_mapping=False, frames=[frame]*128,
            event_witnesses=[witness]*128, selected_offsets=[], actions=[])
        with self.assertRaisesRegex(ValueError, "actual original sampled outputs"):
            r._capture(capture, {}, run_id="CPU_test_only", expected_ordinals=list(range(128)), source_ref={})


if __name__ == "__main__":
    unittest.main()
