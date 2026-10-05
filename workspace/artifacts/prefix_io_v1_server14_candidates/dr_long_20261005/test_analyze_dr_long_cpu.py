"""Local CPU arithmetic/binding tests; no passing current GPU run fixture."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("_dr_result_analysis_cpu", HERE / "analyze_dr_long_results.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def cpu_order_fixture():
    """Only two scalar work IDs for testing the join; not a model/GPU record."""
    before, after = [[2, 0, "ssd_read"], [1, 2, "ssd_read"]], [[1, 2, "ssd_read"], [2, 0, "ssd_read"]]
    change = dict(sequence=1, at_ns=10, owner_epoch=3, action="selected", reason="CPU_TEST_ONLY",
        target_id="CPU_TEST_ONLY", dependency_parent_ids=[1], before_work_ids=before, after_work_ids=after,
        actual_native_submission_proved_here=False)
    window = dict(sequence=1, at_ns=11, before_work_ids=before, after_work_ids=after,
        parent_facts=[[1,3,2,1,3,False,False], [2,10,0,1,1,False,False]], h2d_ready_count=0)
    queued, accepted = [], []
    for index, (work, uid) in enumerate(zip(after, (7, 8)), 1):
        queued.append(dict(sequence=index, at_ns=11+index, work_id=work, user_data=uid, fd=6,
            slot_index=index-1, boundary="original_queue_read_returned_then_original_stage_accepted"))
        accepted.append(dict(sequence=index, at_ns=20+index, user_data=uid, kind="read", fd=6,
            nbytes=917504, boundary="original_linux_io_submit_successful_prefix"))
    journal = dict(schema="bounded_original_restore_order_journal_v1", run_id="CPU_TEST_ONLY",
        valid=True, fault=None, overflow=False, new_work_queues=0, held_native_owners=False,
        actual_order_changes=[change], common_native_ready_windows=[window],
        actual_accepted_read_submission_order=queued, first_parent_reason_windows=[],
        reason_counts={"CPU_TEST_ONLY":1}, native_ready_calls=1)
    kernel = dict(valid=True, fault=None, overflow=False, successful_operation_count=2, records=accepted)
    return journal, kernel


class TargetedAnalysisTests(unittest.TestCase):
    def test_actual_absolute_guard_command_metadata_only(self):
        config = json.loads((HERE / "actual_server14/CONFIG_U04.json").read_bytes())
        run = HERE / "actual_server14/runs/server14-dr-long-u01"
        guard = json.loads((run / "result.json").read_bytes())
        adapter = json.loads((run / "current-device-adapter-evidence.json").read_bytes())
        M.validate_frozen_guard_command(guard["command"], config["runner_ref"], adapter["config_ref"])
        wrong = deepcopy(guard["command"])
        wrong[2] = "/wrong-server-root/" + config["runner_ref"]["path"]
        with self.assertRaisesRegex(ValueError, "exact known-server"):
            M.validate_frozen_guard_command(wrong, config["runner_ref"], adapter["config_ref"])

    def test_actual_frozen_fixed8_configuration_only(self):
        config = json.loads((HERE / "actual_server14/CONFIG_CAL04.json").read_bytes())
        fixed = config["engine_controls"]["prefix_io_p4_policy"]["fixed_stage_policy"]
        M.validate_original_fixed8(fixed, config["run_id"])
        wrong = deepcopy(fixed)
        wrong["inflight"]["ssd_read"] = dict(ops=8, bytes=8*M.Q)
        with self.assertRaisesRegex(ValueError, "preserved original P316 fixed8-control"):
            M.validate_original_fixed8(wrong, config["run_id"])

    def test_existing_historical_rows_scheduled_clock_math_only(self):
        path = PROJECT / "artifacts/prefix_io_v1_server08_primary_qualification/contents/experiments/prefix_io_v1/runs/server08-p3-14-model-off-01/details/result.json"
        report = json.loads(path.read_bytes())
        metrics, outputs = M.request_metrics(report, token_clock_verified=True)
        by_id = {r["request_id"]: r for r in report["rows"]}
        self.assertEqual(len(outputs), 10)
        for measured in metrics:
            raw = by_id[measured["request_id"]]
            self.assertEqual(measured["completion_from_scheduled_arrival_seconds"],
                raw["response_end_seconds"] - raw["scheduled_arrival_seconds"])
            self.assertEqual(measured["ttft_from_scheduled_arrival_seconds"],
                raw["engine_token_timestamps"][0] - report["cohort_start_monotonic"] - raw["scheduled_arrival_seconds"])
            self.assertEqual(measured["in_request_itl_p95_seconds"], raw["itl_p95_seconds"])
        unverified, _ = M.request_metrics(report, token_clock_verified=False)
        self.assertTrue(all(r["ttft_from_scheduled_arrival_seconds"] is None for r in unverified))

    def test_scalar_inversion_needs_both_actual_successful_prefix_receipts(self):
        journal, kernel = cpu_order_fixture()
        result = M.mechanism_evidence(journal, kernel)
        self.assertEqual(result["kernel_confirmed_order_changes"], 1)
        self.assertEqual([r["user_data"] for r in result["confirmed_changes"][0]["successful_kernel_receipts"]], [7,8])
        self.assertFalse(result["resource_release_credit"])

    def test_advice_without_second_kernel_acceptance_cannot_prove_change(self):
        journal, kernel = cpu_order_fixture()
        kernel["records"] = kernel["records"][:1]
        kernel["successful_operation_count"] = 1
        result = M.mechanism_evidence(journal, kernel)
        self.assertEqual(result["kernel_confirmed_order_changes"], 0)
        self.assertEqual(result["classification"], "NOT_EXERCISED")
        self.assertEqual(len(result["unconfirmed_changes"]), 1)

    def test_missing_startup_result_stays_failed_and_outside_effects(self):
        result = M.analyze_runs(PROJECT, [dict(label="CAL01", arm="CAL")],
            expected_gpu_uuid="CPU_TEST_ONLY", expected_common_domain_sha256="a"*64)
        self.assertEqual(len(result["run_rows"]), 1)
        self.assertFalse(result["run_rows"][0]["valid_performance_sample"])
        self.assertIsNone(result["run_rows"][0]["metrics"])
        self.assertEqual(result["valid_pair_count"], 0)
        self.assertFalse(result["CAL_in_effect_estimate"])
        self.assertFalse(result["all_cross_pairs_computed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
