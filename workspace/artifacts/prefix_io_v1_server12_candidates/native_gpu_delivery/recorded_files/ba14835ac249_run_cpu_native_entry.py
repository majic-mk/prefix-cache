"""Run only new native-entry rejection/preservation tests under a CPU source lock."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def check(root, path):
    lock = json.loads(path.read_bytes())
    if lock.get("schema") != "c5_gpu_entry_cpu_source_lock_v1" or lock.get("gpu_uuid") is not None:
        raise ValueError("explicit CPU source closure required")
    rows = lock["files"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 512:
        raise ValueError("bounded CPU source closure")
    seen = set()
    for row in rows:
        relative = row["path"]
        if set(row) != {"path", "bytes", "sha256"} or relative in seen or "\\" in relative or ":" in relative \
                or any(p in ("", ".", "..") for p in relative.split("/")):
            raise ValueError("exact unique safe CPU source")
        seen.add(relative)
        source = root
        for component in relative.split("/"):
            source /= component
            if source.is_symlink():
                raise ValueError("symlink source refused")
        if root not in source.resolve().parents or not source.is_file() or source.stat().st_size != row["bytes"]:
            raise ValueError("CPU source size/path drift")
        raw = source.read_bytes()
        if sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("CPU source SHA drift")
    return dict(files_verified=len(rows), source_lock_sha256=sha256(path.read_bytes()).hexdigest(), failed=[])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--source-lock", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--location", required=True, choices=("local_cpu", "server_cpu"))
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    before = check(root, args.source_lock)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", "-I", "-S", str(Path(__file__).with_name("test_native_entry.py")), "-v"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    with (args.output_dir / "TEST_STDOUT.log").open("xb") as stdout, \
            (args.output_dir / "TEST_STDERR.log").open("xb") as stderr:
        result = subprocess.run(command, env=env, cwd=root, stdout=stdout, stderr=stderr, timeout=60)
    after = check(root, args.source_lock)
    log = (args.output_dir / "TEST_STDERR.log").read_text(encoding="utf-8", errors="replace")
    summary = re.search(r"Ran (\d+) tests? in", log)
    tests = int(summary.group(1)) if summary else 0
    outcomes = re.findall(r"^test_.* \.\.\. (ok|FAIL|ERROR|skipped[^\n]*)$", log, flags=re.M)
    failed = outcomes.count("FAIL")
    errors = outcomes.count("ERROR")
    skipped = sum(value.startswith("skipped") for value in outcomes)
    passed = outcomes.count("ok")
    consistent = tests == len(outcomes) == 23 and tests == passed + failed + errors + skipped
    audit = json.loads((args.output_dir / "TEST_STDOUT.log").read_bytes())
    audit_consistent = (audit.get("scope") == "CPU_NATIVE_ENTRY_IMPORT_AUDIT" and
        audit.get("status") == "PASS" and audit.get("tests") == tests and
        audit.get("failures") == failed and audit.get("errors") == errors and audit.get("skipped") == skipped and
        type(audit.get("forbidden_import_attempts")) is list and
        type(audit.get("loaded_forbidden_modules")) is list)
    forbidden = sorted(set(audit.get("forbidden_import_attempts", []) + audit.get("loaded_forbidden_modules", [])))
    success = result.returncode == 0 and consistent and audit_consistent and not forbidden and failed == errors == skipped == 0
    document = dict(status="PASS" if success else "FAILED",
        location=args.location, command=command, exit=result.returncode,
        tests=tests, passed=passed, failed=failed, errors=errors, skipped=skipped,
        source_before=before, source_after=after, source_lock_sha256=before["source_lock_sha256"],
        gpu_runs=0, actual_gpu_runs=0, actual_gpu_seconds=0, gpu_uuid=None,
        forbidden_imports=forbidden, import_audit=audit,
        native_execution_verified=False, native_cost_qualified=False,
        gpu_launch_allowed=False, valid_native_receipt=None, full_runtime_cost_qualified=False,
        effective_cost_upper_ns=None, effective_step_budget_ns=None,
        on_observation_cost_measured=False, production_qualified=False)
    (args.output_dir / "CPU_NATIVE_ENTRY_RESULT.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
