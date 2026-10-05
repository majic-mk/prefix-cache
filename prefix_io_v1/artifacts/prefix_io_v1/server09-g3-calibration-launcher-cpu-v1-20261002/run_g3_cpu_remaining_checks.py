"""Finish the CPU closure with explicit Linux source roots; no GPU operations.

The preserved first CPU runner omitted the result test's --source-root. Its
already passing 31 plan and 18 runtime tests remain evidence; this entry runs
the failed result suite and previously unexecuted integration/import/preflight.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--source-lock", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    root = a.project.resolve(strict=True)
    here = Path(__file__).resolve().parent
    out = a.output_dir.resolve()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert out.parent == here and not out.exists()
    out.mkdir()
    ledger = root / "experiments/prefix_io_v1/gpu-budget-ledger.json"
    before = sha(ledger)
    commands = [
        ("test_g3_calibration_result.py", [sys.executable, "-B", "-I", "-S",
            str(here / "test_g3_calibration_result.py"), "--source-root", str(root)], 0, 29),
        ("test_run_g3_calibration_pilot.py", [sys.executable, "-B", "-I", "-S",
            str(here / "test_run_g3_calibration_pilot.py")], 0, 8),
        ("actual-support-imports", [sys.executable, "-B", str(here / "replay_g3_cpu_support_imports.py"),
            "--project", str(root), "--source-lock", a.source_lock], 0, 0),
        ("actual-full-source-preflight", [sys.executable, "-B", "-I", "-S",
            str(here / "run_g3_calibration_pilot.py"), "--project", str(root),
            "--source-lock", a.source_lock, "--preflight"], 0, 0),
        ("actual-no-scope-launch-rejected", [sys.executable, "-B", "-I", "-S",
            str(here / "run_g3_calibration_pilot.py"), "--project", str(root),
            "--source-lock", a.source_lock, "--launch"], 78, 0)]
    receipts = []
    for name, command, expected_exit, tests in commands:
        start = time.monotonic()
        run = subprocess.run(command, cwd=root, env=dict(os.environ,
            CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True, text=True, timeout=240)
        (out / (name + ".log")).write_text(run.stdout + "\n" + run.stderr, encoding="utf-8")
        passed = run.returncode == expected_exit
        item = dict(name=name, command=command, exit=run.returncode, expected_exit=expected_exit,
                    elapsed_seconds=time.monotonic()-start, unique_tests=tests, passed=passed)
        if tests:
            matches = re.findall(r"Ran (\d+) tests? in ", run.stderr)
            item["passed"] = passed and matches == [str(tests)] and re.search(r"\nOK\s*$", run.stderr) is not None
        else:
            try:
                item["output"] = json.loads(run.stdout)
            except ValueError:
                item["passed"] = False
        receipts.append(item)
        if not item["passed"]:
            break
    after = sha(ledger)
    passed = len(receipts) == 5 and all(r["passed"] for r in receipts) and before == after
    report = dict(status="PASS_ACTUAL_SERVER_REMAINING_CPU_CLOSURE" if passed else "FAILED_ACTUAL_SERVER_REMAINING_CPU_CLOSURE",
        commands=receipts, ledger_sha256_before=before, ledger_sha256_after=after,
        GPU_operations=0, GPU_jobs_launched=0, model_loaded=False,
        unique_tests_in_this_remaining_run=37, source_lock=a.source_lock,
        production_qualified=False, performance_claim=False)
    (out / "ACTUAL_SERVER_REMAINING_CPU_RESULT.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], commands=len(receipts), ledger_unchanged=before == after,
                         GPU_operations=0, output=str(out))))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
