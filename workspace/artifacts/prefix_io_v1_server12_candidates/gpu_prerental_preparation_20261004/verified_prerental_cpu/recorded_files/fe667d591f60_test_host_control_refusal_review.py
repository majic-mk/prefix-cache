"""Independent host-interval CPU refusal tests; no successful GPU issuance."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
H = None


def original_small_boundary(value):
    return value + 1


def step(value):
    return value + 1


async def original_async_boundary(value):
    return value + 1


def observer(max_intervals=4):
    result = H.HostControlObserver("explicit-cpu-refusal", max_steps=1,
        max_intervals_per_step=max_intervals, _fixture_clock=H.FixtureClock(range(1, 100)))
    for category in H.CATEGORIES:
        result.bind(category, category, original_small_boundary,
            source_ref=H.closed_ref(__file__), boundary_kind=H.BOUNDARIES[category])
    return result


class HostControlReview(unittest.TestCase):
    def test_four_boundaries_are_cpu_observation_without_gpu_authority(self):
        value = observer()
        for category in H.CATEGORIES:
            self.assertEqual(value.call(category, 0, category, original_small_boundary, 40), 41)
        result = value.export()
        self.assertIs(result["valid"], True)
        self.assertIs(result["actual_native_gpu_run"], False)
        self.assertIs(result["production_qualified"], False)
        self.assertIs(result["reserve_source_issued"], False)
        self.assertEqual(result["origin"], "explicit_CPU_fixture")
        with self.assertRaisesRegex(ValueError, "actual Linux host clock"):
            H.checked_intervals(result)
        checked = H.checked_intervals(result, require_production_clock=False)
        self.assertEqual(checked["per_step_control_intervals"], [[[1, 2], [3, 4], [5, 6], [7, 8]]])

    def test_missing_category_is_unknown_not_zero(self):
        value = observer()
        for category in H.CATEGORIES[:-1]:
            value.call(category, 0, category, original_small_boundary, 10)
        with self.assertRaisesRegex(ValueError, "missing/overflowed host intervals"):
            H.checked_intervals(value.export(), require_production_clock=False)

    def test_whole_engine_step_and_async_method_wrappers_are_refused(self):
        value = H.HostControlObserver("cpu", _fixture_clock=H.FixtureClock(range(1, 100)))
        with self.assertRaisesRegex(ValueError, "whole GPU/executor method"):
            value.bind("step", "scheduler", step, source_ref=H.closed_ref(__file__),
                       boundary_kind=H.BOUNDARIES["scheduler"])
        with self.assertRaisesRegex(ValueError, "async boundary"):
            value.bind("async", "output", original_async_boundary, source_ref=H.closed_ref(__file__),
                       boundary_kind=H.BOUNDARIES["output"])

    def test_cross_thread_end_invalidates_observation(self):
        value = observer()
        token = value.begin("scheduler", 0, "scheduler")
        worker = threading.Thread(target=value.end, args=(token,))
        worker.start()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        result = value.export()
        self.assertIs(result["valid"], False)
        self.assertTrue(any("crossed original host thread" in text for text in result["failures"]))

    def test_overflow_records_failure_while_original_call_still_completes(self):
        value = observer()
        for category in H.CATEGORIES:
            value.call(category, 0, category, original_small_boundary, 10)
        self.assertEqual(value.call("scheduler", 0, "scheduler", original_small_boundary, 98), 99)
        result = value.export()
        self.assertIs(result["valid"], False)
        self.assertTrue(any("interval overflow" in text for text in result["failures"]))
        self.assertEqual(len(result["per_step"][0]["intervals"]), 4)

    def test_missing_deadline_cannot_be_inferred_from_observation(self):
        value = observer()
        for category in H.CATEGORIES:
            value.call(category, 0, category, original_small_boundary, 10)
        with self.assertRaisesRegex(ValueError, "independent deadline missing"):
            H.reserve_candidate(value.export(), declaration=None, protocol=SimpleNamespace())


def main():
    global H
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observer-source", type=Path,
        default=HERE.parent / "activation/control_observation/host_control_observer.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.observer_source.resolve(strict=True)
    raw = path.read_bytes()
    spec = importlib.util.spec_from_file_location("_independent_host_refusal_CPU", path)
    H = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = H
    spec.loader.exec_module(H)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HostControlReview))
    unchanged = raw == path.read_bytes()
    proof = dict(schema="independent_cpu_host_control_refusal_review_v1",
        status="PASS" if result.wasSuccessful() and unchanged else "FAIL", tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        source_ref=dict(path=path.as_posix(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
        source_unchanged=unchanged, actual_gpu_runs=0, RPC_calls=0, native_framework_imports=0,
        genuine_native_reserve_receipts_issued=0, real_GPU_positive_acceptance_tested=False,
        failure_details=[dict(test=str(test), traceback=trace) for test, trace in result.failures + result.errors])
    with args.output.open("xb") as stream:
        stream.write((json.dumps(proof, indent=2, sort_keys=True) + "\n").encode())
    return 0 if result.wasSuccessful() and unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
