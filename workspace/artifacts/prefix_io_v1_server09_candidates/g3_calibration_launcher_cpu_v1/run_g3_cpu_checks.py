"""CPU test and real support-import closure; never reserves or queries a GPU."""
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
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--source-lock", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    root = a.project.resolve(strict=True)
    here = Path(__file__).resolve().parent
    out = a.output_dir.absolute()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert out.is_relative_to(root / "artifacts/prefix_io_v1") and not out.exists()
    out.mkdir()
    ledger = root / "experiments/prefix_io_v1/gpu-budget-ledger.json"
    before = sha(ledger)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
        PREFIX_G3_CALIBRATION_BASELINE_PATH=str(root / "artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/gpu-source-lock-candidate.json"))
    commands = []
    for name in ("test_g3_calibration_plan.py", "test_g3_calibration_runtime.py",
                 "test_g3_calibration_result.py", "test_run_g3_calibration_pilot.py"):
        command = [sys.executable, "-B", "-I", "-S", str(here / name)]
        if name == "test_g3_calibration_runtime.py":
            command.extend(["--source-root", str(root)])
        commands.append((name, command, True))
    commands.extend([
        ("actual-support-imports", [sys.executable, "-B", str(here / "replay_g3_cpu_support_imports.py"),
            "--project", str(root), "--source-lock", a.source_lock], False),
        ("actual-full-source-preflight", [sys.executable, "-B", "-I", "-S", str(here / "run_g3_calibration_pilot.py"),
            "--project", str(root), "--source-lock", a.source_lock, "--preflight"], False),
        ("actual-no-scope-launch-rejected", [sys.executable, "-B", "-I", "-S", str(here / "run_g3_calibration_pilot.py"),
            "--project", str(root), "--source-lock", a.source_lock, "--launch"], False)])
    receipts = []
    total = 0
    for name, command, is_tests in commands:
        start = time.monotonic()
        run = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, timeout=180)
        text = run.stdout + "\n" + run.stderr
        (out / (name + ".log")).write_text(text, encoding="utf-8")
        expected = 78 if name == "actual-no-scope-launch-rejected" else 0
        passed = run.returncode == expected
        item = dict(name=name, command=command, exit=run.returncode, expected_exit=expected,
                    elapsed_seconds=time.monotonic()-start, passed=passed)
        if is_tests:
            matches = re.findall(r"Ran (\d+) tests? in ", run.stderr)
            passed = passed and len(matches) == 1 and re.search(r"\nOK\s*$", run.stderr) is not None
            item.update(passed=passed, unique_tests=int(matches[0]) if len(matches) == 1 else 0)
            total += item["unique_tests"]
        else:
            try:
                item["output"] = json.loads(run.stdout)
            except ValueError:
                item["passed"] = False
        receipts.append(item)
        if not item["passed"]:
            break
    after = sha(ledger)
    passed = len(receipts) == len(commands) and all(r["passed"] for r in receipts) and before == after
    report = dict(status="PASS_ACTUAL_SERVER_G3_CPU_CLOSURE" if passed else "FAILED_ACTUAL_SERVER_G3_CPU_CLOSURE",
        commands=receipts, unique_tests=total, ledger_sha256_before=before, ledger_sha256_after=after,
        GPU_operations=0, GPU_jobs_launched=0, model_loaded=False, source_lock=a.source_lock,
        production_qualified=False, effect_verified=False, performance_claim=False)
    (out / "ACTUAL_SERVER_CPU_RESULT.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], unique_tests=total, commands=len(receipts),
                         ledger_unchanged=before == after, GPU_operations=0, output=str(out))))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
