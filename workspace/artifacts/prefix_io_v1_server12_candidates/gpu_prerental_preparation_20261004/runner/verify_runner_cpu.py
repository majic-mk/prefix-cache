"""Run actual standard-library CPU checks and retain source/command evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
TESTS = ("test_strong_trace_runner_cpu.py", "test_finite_current_binding_cpu.py", "test_runner_activation_cpu.py")


def refs():
    return [dict(path=path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            for path in sorted(HERE.glob("*.py")) for raw in (path.read_bytes(),)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = refs()
    commands, passed = [], True
    for name in TESTS:
        command = [sys.executable, "-B", "-I", "-S", str(HERE / name)]
        started = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        passed = passed and result.returncode == 0
        commands.append(dict(command=command, exit_code=result.returncode, wall_seconds=time.monotonic() - started,
                             stdout=result.stdout, stderr=result.stderr))
    after = refs()
    passed = passed and before == after
    report = dict(schema="gpu_prerental_runner_actual_CPU_verification_v1",
        status="PASS_CPU_RUNNER_PREPARATION" if passed else "FAIL_CPU_RUNNER_PREPARATION",
        commands=commands, source_before=before, source_after=after, source_bytes_unchanged=before == after,
        expected_test_count=47, actual_GPU_runs=0, original_model_initialized=False,
        cost_cells_qualified=0, formal_effect_qualified=False, origin="actual_CPU_tests_API_and_source_fixtures_only",
        full_server_model_SDK_source_and_raw_template_closure_required_separately=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=report["status"], CPU_test_count=47, actual_GPU_runs=0, evidence=str(args.output))))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
