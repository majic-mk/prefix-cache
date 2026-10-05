"""Verify actual CPU rejection/math results; never mint any GPU capability."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest


def refs(paths):
    return [{"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(paths)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-frozen-control", "--original-control-root", dest="original_frozen_control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-config", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    original = sorted(args.original_frozen_control.glob("*.py"))
    paths = sorted(here.rglob("*.py")) + original + [args.pair_config]
    before = refs(paths)
    spec = importlib.util.spec_from_file_location("_activation_tests_actual_cpu", here / "test_activation_cpu.py")
    tests = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tests
    spec.loader.exec_module(tests)
    tests.ORIGINAL_FROZEN_CONTROL_DIR = args.original_frozen_control
    tests.PAIR_CONFIG_PATH = args.pair_config
    start = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    after = refs(paths)
    forbidden = sorted(name for name in sys.modules if name.split(".")[0] in {"torch", "vllm", "py_kvcache", "openai", "numpy"})
    issuer = sys.modules["prefix_io_control.gpu_cell_issuer"]
    issued = len(issuer._ISSUED)
    passed = result.wasSuccessful() and before == after and not forbidden and issued == 0
    report = dict(schema="finite_activation_actual_cpu_result_v1", status="PASS_FINITE_CPU_REJECTION_AND_ORIGINAL_MATH" if passed else "FAIL_FINITE_ACTIVATION_CPU",
                  tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skips=len(result.skipped),
                  actual_GPU_operations=0, actual_GPU_qualified_cells=0, private_capabilities_issued=issued,
                  input_files_before=before, input_files_after=after, input_bytes_unchanged=before == after,
                  forbidden_native_or_GPU_modules=forbidden, elapsed_seconds=time.monotonic() - start,
                  original_estimator_AST_used=True, holdout_never_fitted=True, CPU_fixtures_are_native_runtime_proof=False,
                  positive_GPU_qualification_test_performed=False, actual_strong_domain_GPU_raw_data_available=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "tests_run": result.testsRun, "qualified_GPU_cells": 0, "GPU_operations": 0}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
