"""CPU-only gates for the locked P3 descriptive analysis.

All run summaries below are fabricated scalar fixtures, never GPU results.
The tests import only the standard-library analysis module and write only
TemporaryDirectory payloads; no model/native modules or server corpus are read.
"""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_PATH = ROOT / "experiments/prefix_io_v1/scripts/analyze_decode_interference_calibration.py"
SPEC = importlib.util.spec_from_file_location("_p3_evidence_analysis_gates", ANALYSIS_PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def fixture_runs():
    """Scalar outputs of audit_run, for aggregate semantics only."""
    shared = dict(
        engine={"max_num_seqs": 2}, sampling={"max_tokens": 128},
        owned={"owned_bytes": 29360128}, io={"iodepth": 8},
        cells=copy.deepcopy(analysis.CELLS), sequence=["A", "B", "B", "A"],
        pulse={"max_pulses": 12}, active_decode_requests=1,
        table_domain={"engine_conditional": True, "p4_connected": False},
        gpu_uuid="GPU-fixture-only", native_worktree="fixture-no-runtime",
        source_sha256={"fixture-only": "a" * 64},
    )
    runs = []
    for name, partition in (("cal-a", "calibration"), ("cal-b", "calibration"),
                            ("val", "validation")):
        spec = copy.deepcopy(shared)
        spec.update(
            prompt_token_ids=[31000 if partition == "calibration" else 32000],
            prompt_family="calibration-fixture" if partition == "calibration" else "validation-fixture",
            reference_results=[],
        )
        path = str((ROOT / "experiments/prefix_io_v1/runs" / name / "result.json").resolve())
        runs.append(dict(
            path=path, sha256={"cal-a": "1", "cal-b": "2", "val": "3"}[name] * 64,
            session_id=name, partition=partition, spec=spec, model={"identity": "fixture"},
            cells=[dict(
                loaded_itl_median_seconds=1.0, baseline_first_last_relative_drift=0.0,
                delta_itl_seconds=0.0, delta_relative_to_baseline=0.0,
            ) for _ in analysis.CELLS],
            windows=[], requests=28, output_tokens=3584,
        ))
    runs[2]["spec"]["reference_results"] = [
        dict(path=str(Path(r["path"]).relative_to(ROOT)), sha256=r["sha256"])
        for r in runs[:2]
    ]
    return runs


def aggregate(runs):
    return analysis.aggregate(runs[:2], runs[2], copy.deepcopy(analysis.CRITERION))


def model_rows():
    development = [
        dict(arm=arm, source="dev-" + str(i), status="PASS_FULL_MODEL_SIMPLE_STAGE",
             cohort_seconds_including_drain={"U": 12.0, "F": 10.0, "P": 11.0}[arm])
        for i, arm in enumerate(("U", "F", "P", "P", "F", "U"))
    ]
    allhit = [
        dict(arm=arm, source="hit-" + str(i), status="PASS_FULL_MODEL_SIMPLE_STAGE",
             cohort_seconds_including_drain=100.0 if arm == "U" else 101.0,
             actual_read_bytes=0, actual_write_bytes=0)
        for i, arm in enumerate(("U", "B", "B", "U"))
    ]
    return development, allhit


class ConditionalTableGates(unittest.TestCase):
    def test_clean_scalar_fixture_remains_conditional(self):
        result = aggregate(fixture_runs())
        self.assertEqual(result["status"], "PASS_COMPLETE_CONDITIONAL_TABLE")
        self.assertTrue(result["complete_conditional_table"])
        self.assertFalse(result["production_state_table_complete"])
        self.assertFalse(result["formal_performance_claim"])
        self.assertFalse(result["p4_connected"])
        self.assertIsNone(result["confidence_interval"])
        self.assertIsNone(result["production_state_fallback"])
        self.assertEqual(result["inputs"][0]["requests"], 28)

    def test_invalid_input_cannot_export_favorable_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            failure = dict(
                status="FAILED", scope=analysis.SCOPE,
                spec={"scope": analysis.SCOPE, "partition": "calibration"},
                windows=[{"metrics": {"itl_median_seconds": 1e-12}}],
            )
            paths = []
            for i in range(3):
                path = tmp / ("run-" + str(i) + ".json")
                path.write_text(json.dumps(failure))
                paths.append(path)
            criterion = tmp / "criterion.json"
            criterion.write_text(json.dumps(analysis.CRITERION))
            out = tmp / "analysis.json"
            argv = ["analysis", "--calibration", str(paths[0]), str(paths[1]),
                    "--validation", str(paths[2]), "--criterion", str(criterion),
                    "--output", str(out)]
            with mock.patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                code = analysis.main()
            actual = json.loads(out.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(actual["status"], "FAILED_INPUT_AUDIT")
            self.assertNotIn("cells", actual)
            self.assertNotIn("complete_conditional_table", actual)
            self.assertFalse(actual["formal_performance_claim"])
            self.assertIsNone(actual["production_state_fallback"])

    def test_favorable_output_still_requires_shutdown_witness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps(dict(
                status="PASSED_SPARSE_DECODE_INTERFERENCE", scope=analysis.SCOPE,
                spec={"scope": analysis.SCOPE, "partition": "calibration"},
                engine_shutdown="failed", inputs_preserved=True,
                windows=[{"metrics": {"itl_median_seconds": 1e-12}}],
            )))
            with self.assertRaisesRegex(ValueError, "shutdown/input preservation"):
                analysis.audit_run(path, "calibration")

    def test_unsupported_cell_exports_none_while_other_cell_stays_supported(self):
        runs = fixture_runs()
        runs[2]["cells"][0]["loaded_itl_median_seconds"] = 0.5
        table = aggregate(runs)
        self.assertEqual(table["status"], "INCOMPLETE_CONDITIONAL_TABLE")
        self.assertFalse(table["complete_conditional_table"])
        self.assertIsNone(table["cells"][0]["conditional_loaded_itl_seconds"])
        self.assertIsNone(table["cells"][0]["production_value"])
        self.assertIsNone(analysis.lookup(table, table["domain"], analysis.CELLS[0]))
        self.assertTrue(table["cells"][1]["supported"])
        self.assertEqual(analysis.lookup(table, table["domain"], analysis.CELLS[1]), 1.0)

    def test_lookup_rejects_float_for_integer_domain(self):
        table = aggregate(fixture_runs())
        domain = copy.deepcopy(table["domain"])
        domain["engine"]["max_num_seqs"] = 2.0
        self.assertIsNone(analysis.lookup(table, domain, analysis.CELLS[0]))

    def test_lookup_rejects_boolean_for_integer_domain(self):
        table = aggregate(fixture_runs())
        domain = copy.deepcopy(table["domain"])
        domain["active_decode_requests"] = True
        self.assertIsNone(analysis.lookup(table, domain, analysis.CELLS[0]))

    def test_lookup_rejects_float_units_and_unmeasured_cell(self):
        table = aggregate(fixture_runs())
        cell = copy.deepcopy(analysis.CELLS[0])
        cell["units"] = 8.0
        self.assertIsNone(analysis.lookup(table, table["domain"], cell))
        self.assertIsNone(analysis.lookup(table, table["domain"], {"anchor": "warm_h2d", "units": 4}))

    def test_lookup_rejects_extra_domain_field(self):
        table = aggregate(fixture_runs())
        domain = copy.deepcopy(table["domain"])
        domain["production_state"] = "invented"
        self.assertIsNone(analysis.lookup(table, domain, analysis.CELLS[0]))

    def test_validation_error_uses_observed_denominator_and_inclusive_boundary(self):
        runs = fixture_runs()
        for run in runs[:2]:
            run["cells"][0]["loaded_itl_median_seconds"] = 1.25
        result = aggregate(runs)["cells"][0]
        self.assertEqual(result["validation_loaded_relative_error"], 0.25)
        self.assertTrue(result["supported"])
        for run in runs[:2]:
            run["cells"][0]["loaded_itl_median_seconds"] = 1.250001
        result = aggregate(runs)["cells"][0]
        self.assertEqual(result["unsupported_reasons"], ["validation_loaded_error"])
        self.assertIsNone(result["conditional_loaded_itl_seconds"])

    def test_calibration_spread_uses_equal_run_median_and_inclusive_boundary(self):
        runs = fixture_runs()
        runs[0]["cells"][0]["loaded_itl_median_seconds"] = 0.875
        runs[1]["cells"][0]["loaded_itl_median_seconds"] = 1.125
        result = aggregate(runs)["cells"][0]
        self.assertEqual(result["calibration_loaded_relative_spread"], 0.25)
        self.assertEqual(result["conditional_loaded_itl_seconds"], 1.0)
        runs[0]["cells"][0]["loaded_itl_median_seconds"] = 0.87
        runs[1]["cells"][0]["loaded_itl_median_seconds"] = 1.13
        result = aggregate(runs)["cells"][0]
        self.assertEqual(result["unsupported_reasons"], ["calibration_run_spread"])
        self.assertIsNone(result["conditional_loaded_itl_seconds"])

    def test_each_independent_run_baseline_drift_can_block_cell(self):
        for index in range(3):
            with self.subTest(run=index):
                runs = fixture_runs()
                runs[index]["cells"][0]["baseline_first_last_relative_drift"] = 0.250001
                result = aggregate(runs)["cells"][0]
                self.assertEqual(result["unsupported_reasons"], ["within_run_baseline_drift"])
                self.assertIsNone(result["conditional_loaded_itl_seconds"])

    def test_baseline_drift_inclusive_boundary(self):
        runs = fixture_runs()
        for run in runs:
            run["cells"][0]["baseline_first_last_relative_drift"] = 0.25
        self.assertTrue(aggregate(runs)["cells"][0]["supported"])

    def test_duplicate_independent_session_and_path_rejected(self):
        for key in ("session_id", "path"):
            with self.subTest(key=key):
                runs = fixture_runs()
                runs[1][key] = runs[0][key]
                with self.assertRaisesRegex(ValueError, "independent sessions/paths"):
                    aggregate(runs)

    def test_validation_must_freeze_exact_calibration_hashes(self):
        runs = fixture_runs()
        runs[2]["spec"]["reference_results"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "exact calibration results"):
            aggregate(runs)

    def test_cross_run_domain_type_change_rejected(self):
        runs = fixture_runs()
        runs[2]["spec"]["engine"]["max_num_seqs"] = 2.0
        with self.assertRaisesRegex(ValueError, "domain differs"):
            aggregate(runs)

    def test_criterion_cannot_loosen_a_threshold(self):
        runs = fixture_runs()
        criterion = copy.deepcopy(analysis.CRITERION)
        criterion["validation_loaded_itl_relative_error_max"] = 1.0
        with self.assertRaisesRegex(ValueError, "frozen P3 contract"):
            analysis.aggregate(runs[:2], runs[2], criterion)


class ModelComparisonGates(unittest.TestCase):
    def test_valid_sequence_selects_fixed_descriptively(self):
        development, allhit = model_rows()
        result = analysis.summarize_model_runs(development, allhit)
        self.assertEqual(result["strongest_simple_baseline"], "F")
        self.assertTrue(result["allhit_control"]["descriptive_target_met"])
        self.assertFalse(result["formal_goodput"])
        self.assertIsNone(result["slo"])
        self.assertIsNone(result["confidence_interval"])

    def test_fixed_wins_exact_duration_tie(self):
        development, allhit = model_rows()
        for row in development:
            if row["arm"] == "P":
                row["cohort_seconds_including_drain"] = 10.0
        self.assertEqual(analysis.summarize_model_runs(development, allhit)["strongest_simple_baseline"], "F")

    def test_development_arm_sequence_cannot_be_reordered(self):
        development, allhit = model_rows()
        development[1], development[2] = development[2], development[1]
        with self.assertRaisesRegex(ValueError, "independent run sequence"):
            analysis.summarize_model_runs(development, allhit)

    def test_failed_model_run_cannot_contribute_fast_duration(self):
        development, allhit = model_rows()
        development[1].update(status="FAILED", cohort_seconds_including_drain=1e-12)
        with self.assertRaisesRegex(ValueError, "unaudited/failed model run"):
            analysis.summarize_model_runs(development, allhit)

    def test_duplicate_model_source_cannot_count_as_an_independent_run(self):
        development, allhit = model_rows()
        allhit[0]["source"] = development[0]["source"]
        with self.assertRaisesRegex(ValueError, "duplicated model runs"):
            analysis.summarize_model_runs(development, allhit)

    def test_allhit_must_have_original_abba_sequence(self):
        development, allhit = model_rows()
        allhit[0], allhit[1] = allhit[1], allhit[0]
        with self.assertRaisesRegex(ValueError, "all-hit ABBA sequence"):
            analysis.summarize_model_runs(development, allhit)

    def test_allhit_rejects_either_real_ssd_direction(self):
        for key in ("actual_read_bytes", "actual_write_bytes"):
            with self.subTest(direction=key):
                development, allhit = model_rows()
                allhit[1][key] = 1
                with self.assertRaisesRegex(ValueError, "measured SSD I/O"):
                    analysis.summarize_model_runs(development, allhit)

    def test_allhit_above_target_is_reported_without_success_claim(self):
        development, allhit = model_rows()
        for row in allhit:
            if row["arm"] == "B":
                row["cohort_seconds_including_drain"] = 103.0
        result = analysis.summarize_model_runs(development, allhit)
        self.assertFalse(result["allhit_control"]["descriptive_target_met"])
        self.assertEqual(result["status"], "DESCRIPTIVE_DEVELOPMENT_ONLY")
        self.assertFalse(result["formal_goodput"])

    def test_nonfinite_duration_rejected(self):
        for value in (float("nan"), float("inf"), 0.0, -1.0, True):
            with self.subTest(value=value):
                development, allhit = model_rows()
                development[0]["cohort_seconds_including_drain"] = value
                with self.assertRaisesRegex(ValueError, "invalid model run duration"):
                    analysis.summarize_model_runs(development, allhit)


if __name__ == "__main__":
    unittest.main()
