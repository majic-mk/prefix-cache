"""Frozen CPU checks for the normal native entry; never starts a GPU job."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent
TEST_NAMES = ("test_normal_native_controller.py", "test_normal_native_runtime.py", "test_historical_calibration.py")
FORBIDDEN = ("torch", "vllm", "py_kvcache")

def require(value, reason):
    if not value:
        raise ValueError(reason)

def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))

def safe(root, name):
    require(type(name) is str and name and ":" not in name and "\\" not in name
        and not name.startswith("/") and all(p not in ("", ".", "..") for p in name.split("/")),
        "project-relative source reference")
    path = root
    for part in name.split("/"):
        path /= part
        require(not path.is_symlink(), "source symlink refused")
    require(path.resolve().is_relative_to(root), "source outside project")
    return path

def ref(root, name):
    path = safe(root, name)
    require(path.is_file(), "regular source required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"path": name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}

def verify(root, rows):
    require(type(rows) is list and 1 <= len(rows) <= 10000, "bounded real source rows")
    require(len({r["path"] for r in rows}) == len(rows), "unique source references")
    actual = []
    for row in rows:
        require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "exact source reference")
        require(ref(root, row["path"]) == row, "source drift: " + row["path"])
        actual.append(row)
    return hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def put(path, document):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--source-lock", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.project_root.resolve(strict=True)
    lock_path = args.source_lock.resolve(strict=True)
    require(lock_path.is_relative_to(root) and not args.source_lock.is_symlink(), "source lock in project")
    output = args.output_dir.absolute()
    require(output.resolve().is_relative_to(root) and not output.exists(), "new project output required")
    require(not any(p.is_symlink() for p in (output, *output.parents)), "output symlink refused")
    lock_bytes = lock_path.read_bytes()
    lock = read_json(lock_path)
    rows = lock["files"]
    for name in (*TEST_NAMES, Path(__file__).name):
        rel = (HERE / name).relative_to(root).as_posix()
        require(sum(r["path"] == rel for r in rows) == 1, "test/runner not frozen: " + name)
    ledger = root / "experiments/prefix_io_v1/gpu-budget-ledger.json"
    before_ledger_bytes = ledger.read_bytes()
    before_ledger = read_json(ledger)
    require(before_ledger["active_reservation"] is None, "active GPU job")
    started = time.monotonic()
    before = verify(root, rows)
    output.mkdir(parents=True, exist_ok=False)
    fixture_tmp = output / "fixture-tmp"
    fixture_tmp.mkdir()
    os.environ.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
        C5_NORMAL_PROJECT_ROOT=str(root), TMPDIR=str(fixture_tmp), TEMP=str(fixture_tmp), TMP=str(fixture_tmp))
    suite = unittest.TestSuite()
    test_modules = []
    for index, name in enumerate(TEST_NAMES):
        spec = importlib.util.spec_from_file_location("_normal_cpu_test_" + str(index), HERE / name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        test_modules.append(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    with (output / "TEST_STDOUT.log").open("x", encoding="utf-8", newline="\n") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    after = verify(root, rows)
    imports = sorted(name for name in sys.modules
        if any(name == p or name.startswith(p + ".") for p in FORBIDDEN))
    attempted_imports = [name for module in test_modules
        for name in getattr(module, "ATTEMPTS", [])]
    require(imports == [], "GPU/model/native backend imported: " + repr(imports))
    require(attempted_imports == [], "forbidden model/GPU import attempted: " + repr(attempted_imports))
    require(before_ledger_bytes == ledger.read_bytes(), "GPU budget changed during CPU checks")
    require(lock_bytes == lock_path.read_bytes(), "common source lock changed during CPU checks")
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    success = result.wasSuccessful() and skipped == 0 and before == after
    document = {
        "schema": "c5_normal_native_server_cpu_result_v1",
        "status": "PASS_NORMAL_NATIVE_CPU_CHECKS" if success else "FAIL_NORMAL_NATIVE_CPU_CHECKS",
        "location": "server_cpu",
        "tests": result.testsRun,
        "passed": result.testsRun - failures - errors - skipped,
        "failed": failures, "errors": errors, "skipped": skipped,
        "source_count": len(rows), "source_before": before, "source_after": after,
        "source_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "GPU_jobs": 0, "GPU_seconds": 0, "actual_model_processes_started": 0,
        "gpu_ledger_unchanged": True, "gpu_ledger_sha256": hashlib.sha256(before_ledger_bytes).hexdigest(),
        "gpu_seconds_used": before_ledger["gpu_wall_seconds"],
        "remaining_GPU_seconds": 28800 - before_ledger["gpu_wall_seconds"],
        "torch_vllm_native_backend_imports": imports,
        "forbidden_model_GPU_import_attempts": attempted_imports,
        "normal_native_execution_verified": False, "actual_on_installed": False,
        "normal_runtime_condition_qualified": False, "full_runtime_cost_qualified": False,
        "strategy_effect_verified": False, "P4_completed": False,
        "old_tests_or_CPU_benchmark_rerun": False,
        "elapsed_seconds": time.monotonic() - started}
    put(output / "CPU_RESULT.json", document)
    print(json.dumps(document, sort_keys=True))
    return 0 if success else 1

if __name__ == "__main__":
    raise SystemExit(main())

