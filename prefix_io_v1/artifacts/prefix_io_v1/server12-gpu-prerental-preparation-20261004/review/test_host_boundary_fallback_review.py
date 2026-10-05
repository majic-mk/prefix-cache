"""CPU interface refusal: optional ordinal failure cannot skip original calls."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import unittest
import weakref
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
H = B = None


class CPUOriginalHost:
    def __init__(self):
        self.calls = 0

    def schedule(self, value):
        self.calls += 1
        return value + 1


class CPURunner:
    _profile_step = None


def frontend_progress(rows, step):
    rows.append(step)
    return step + 1


def boundary_fixture():
    runner = CPURunner()
    value = B.HostBoundaryBinding.__new__(B.HostBoundaryBinding)
    value.observer = H.HostControlObserver("cpu-ordinal-refusal", _fixture_clock=H.FixtureClock(range(1, 100)))
    value._runner = weakref.ref(runner)
    value.bindings = []
    value.controller_id = "frontend_progress_bookkeeping"
    value.failure = None
    # The adapter targets Linux and emits str(root / relative). Keep this CPU
    # fixture's root in the actual portable forward-slash reference spelling.
    root = PurePosixPath(HERE.parents[3].as_posix())
    source = H.closed_ref(__file__)
    relative = Path(source["path"]).relative_to(root).as_posix()
    refs = {relative: dict(source, path=relative)}
    driver = SimpleNamespace(require=H.require, check_ref=lambda project, row: None)
    return value, runner, root, refs, driver


class HostBoundaryFallbackReview(unittest.TestCase):
    def test_optional_invalid_ordinal_still_calls_original_method(self):
        boundary, runner, root, refs, driver = boundary_fixture()
        target = CPUOriginalHost()
        boundary.attach(target, "schedule", "scheduler", "actual_cpu_source_schedule", False, root, refs, driver)
        try:
            self.assertEqual(target.schedule(41), 42)
            self.assertEqual(target.calls, 1)
            self.assertIs(boundary.observer.export()["valid"], False)
        finally:
            self.assertIs(boundary.detach(), True)
        self.assertNotIn("schedule", vars(target))

    def test_optional_invalid_ordinal_still_calls_frontend_bookkeeping(self):
        boundary, runner, root, refs, driver = boundary_fixture()
        rows = []
        wrapped = boundary.bind_frontend_progress(frontend_progress, root=root, refs=refs, driver=driver)
        self.assertEqual(wrapped(rows, 15), 16)
        self.assertEqual(rows, [15])
        self.assertIs(boundary.observer.export()["valid"], False)


def main():
    global H, B
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding-source", type=Path, default=HERE.parent / "runner/host_boundary_binding.py")
    parser.add_argument("--observer-source", type=Path, default=HERE.parent / "activation/control_observation/host_control_observer.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = []
    for name, path in (("_independent_host_boundary_observer", args.observer_source),
                       ("_independent_host_boundary_adapter", args.binding_source)):
        path = path.resolve(strict=True)
        raw = path.read_bytes()
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if H is None:
            H = module
        else:
            B = module
        sources.append(dict(path=path.as_posix(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HostBoundaryFallbackReview))
    unchanged = all(hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"] for row in sources)
    proof = dict(schema="independent_cpu_optional_host_boundary_fallback_review_v1",
        status="PASS" if result.wasSuccessful() and unchanged else "FAIL", tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped), source_refs=sources,
        source_unchanged=unchanged, actual_gpu_runs=0, RPC_calls=0, native_framework_imports=0,
        genuine_native_reserve_receipts_issued=0, real_GPU_positive_acceptance_tested=False,
        fixture_kind="explicit_CPU_original_method_source_bound_calls_with_invalid_runtime_ordinal",
        failure_details=[dict(test=str(test), traceback=trace) for test, trace in result.failures + result.errors])
    with args.output.open("xb") as stream:
        stream.write((json.dumps(proof, indent=2, sort_keys=True) + "\n").encode())
    return 0 if result.wasSuccessful() and unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
