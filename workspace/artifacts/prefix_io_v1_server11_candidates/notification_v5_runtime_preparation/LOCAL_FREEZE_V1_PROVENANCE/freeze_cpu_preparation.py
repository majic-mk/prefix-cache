"""Build a CPU source manifest only. Does not create native receipts or scopes."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ref(path, root, scope):
    data = path.read_bytes()
    return dict(scope=scope, path=path.relative_to(root).as_posix(), bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest())


def put(name, value):
    target = HERE / name
    if target.exists():
        raise ValueError("CPU source lock is append-only: " + name)
    target.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--runtime-v4-root", type=Path, required=True)
    args = parser.parse_args()
    candidate, parent = args.candidate_root.resolve(strict=True), args.runtime_v4_root.resolve(strict=True)
    baseline = []
    for path in sorted(parent.iterdir()):
        if path.is_file():
            row = ref(path, parent, "parent_runtime_v4")
            current = HERE / path.name
            row["inherited_unchanged"] = current.read_bytes() == path.read_bytes()
            baseline.append(row)
    put("BASELINE_INHERITANCE.json", dict(parent_artifact="server11-p4-single-file-runtime-v4-20261003",
        new_artifact="server11-c5-runtime-preparation-cpu-20261004", files=baseline,
        old_source_modified=False, old_gpu_qualification_inherited=False))
    put("PREPARATION_MANIFEST.json", dict(status="CPU_PREPARATION_ONLY_GPU_BLOCKED",
        source_scope="new directory only; original C5/C4/v4/v6/author unmodified",
        candidate_artifact="server11-p4-notification-candidate-v5-cpu-20261003",
        job_namespace="server11-c5-notification-{off,shadow,on}-preparation01",
        gpu_jobs_created=0, gpu_launch_allowed=False, gpu_qualified=False,
        actual_native_execution_verified=False, cpu_timing_qualification=False,
        new_native_cost_receipt=None, old_c4_v6_receipt_reusable=False,
        permission_scope_created=False, inherited_test_files_executed=False,
        server_cpu_tests=None, local_cpu_tests=None,
        test_entry="test_notification_runtime_preparation.py", runner="run_cpu_preparation.py",
        event_origin="SYNTHETIC_CPU_METADATA", performance_claim=False))
    files = [ref(path, HERE, "preparation") for path in sorted(HERE.iterdir())
             if path.is_file() and path.suffix in (".py", ".md", ".json")]
    files.extend(ref(path, candidate, "candidate") for path in sorted(candidate.rglob("*.py"))
                 if "__pycache__" not in path.parts)
    put("PREPARATION_SOURCE_LOCK.json", dict(schema="cpu_notification_preparation_source_lock_v1",
        gpu_qualified=False, gpu_launch_allowed=False, native_receipt=None,
        immutable_parent_sources=True, files=files))
    lock = HERE / "PREPARATION_SOURCE_LOCK.json"
    print(json.dumps(dict(files=len(files), source_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
                         gpu_qualified=False, gpu_launch_allowed=False), sort_keys=True))


if __name__ == "__main__":
    main()
