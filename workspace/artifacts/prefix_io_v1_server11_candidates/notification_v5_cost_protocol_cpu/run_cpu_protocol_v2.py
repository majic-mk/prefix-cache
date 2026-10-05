"""Run source AST/protocol checks and symbolic tests; never native/GPU work."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
PROTOCOL_FILES = (
    "run_cpu_protocol.py", "check_cost_protocol.py", "test_cost_protocol.py",
    "PROTOCOL.json", "SOURCE_PINS.json", "README.md",
)


def protocol_source_files():
    # A RunLogged COMMAND.json/result/log may exist before this process starts.
    # Such run evidence is never a new source dependency or a self-hash input.
    return [HERE / name for name in PROTOCOL_FILES] + [Path(__file__).resolve()]


def outer_lock(project, lock, required):
    if project not in lock.resolve().parents: raise ValueError("outer lock must be inside project")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError("duplicate outer lock key")
            result[key] = value
        return result
    raw = lock.read_bytes()
    document = json.loads(raw, object_pairs_hook=pairs)
    if (document.get("schema") != "c5_cost_binding_cpu_source_lock_v1" or
            document.get("gpu_launch_allowed") is not False or document.get("native_execution_verified") is not False):
        raise ValueError("exact CPU-only outer source lock required")
    rows = document.get("files")
    if type(rows) is not list or not 1 <= len(rows) <= 8192: raise ValueError("bounded outer source closure")
    seen = set()
    for row in rows:
        if type(row) is not dict or set(row) != {"path", "bytes", "sha256"}: raise ValueError("exact outer source ref")
        name = row["path"]
        if (type(name) is not str or not name or any(c in name for c in ("\\", ":", "\0")) or
                any(part in ("", ".", "..") for part in name.split("/")) or name in seen):
            raise ValueError("unique project-relative source ref")
        seen.add(name); target = project
        for part in name.split("/"):
            target /= part
            if target.is_symlink(): raise ValueError("source symlink refused")
        if project not in target.resolve().parents or type(row["bytes"]) is not int or not 0 < row["bytes"] <= 32 * 1024**2:
            raise ValueError("bounded source within project")
        data = target.read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("outer source changed: " + name)
    for path in required:
        if project not in path.resolve().parents or path.resolve().relative_to(project).as_posix() not in seen:
            raise ValueError("actual protocol/source file missing from outer lock: " + str(path))
    return dict(files=len(rows), source_lock_sha256=hashlib.sha256(raw).hexdigest())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-root", "native-root", "preparation-root", "runtime-v4-root", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args()
    if (args.source_lock is None) != (args.project_root is None):
        parser.error("--source-lock and --project-root must be supplied together")
    roots = {key: getattr(args, key + "_root").resolve(strict=True) for key in ("candidate", "native", "preparation", "runtime_v4")}
    pins = json.loads((HERE / "SOURCE_PINS.json").read_text(encoding="utf-8"))
    required = protocol_source_files()
    required += [roots[row["scope"]] / row["path"] for row in pins["files"]]
    outer_before = (outer_lock(args.project_root.resolve(strict=True), args.source_lock.resolve(strict=True), required)
                    if args.source_lock is not None else None)
    C = load(HERE / "check_cost_protocol.py", "_checked_cost_protocol")
    before = C.audit_sources(pins, roots)
    checked = C.check_protocol(json.loads((HERE / "PROTOCOL.json").read_text(encoding="utf-8")))
    module = load(HERE / "test_cost_protocol.py", "_cpu_cost_protocol_tests")
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    after = C.audit_sources(pins, roots)
    outer_after = (outer_lock(args.project_root.resolve(strict=True), args.source_lock.resolve(strict=True), required)
                   if args.source_lock is not None else None)
    if outer_after != outer_before: raise ValueError("outer source lock changed during CPU protocol checks")
    before["outer_source_lock"] = outer_before; after["outer_source_lock"] = outer_after
    record = dict(status="PASS_CPU_PROTOCOL_ONLY" if result.wasSuccessful() else "FAIL_CPU_PROTOCOL",
        tests=result.testsRun, failed=len(result.failures), failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        source_lock_sha256=outer_before["source_lock_sha256"] if outer_before is not None else None,
        source_before=before, source_after=after, protocol=checked,
        symbolic_tests_only=True, gpu_runs=0, GPU_runs=0, formal_benchmark=0, jobs_created=0, cost_values_emitted=False,
        native_execution_verified=False, receipt_issued=False, performance_claim=False,
        failure_provenance=dict(prior_runner="run_cpu_protocol.py",
            prior_failure_phase="pre_test_required_source_validation", prior_tests_executed=0,
            prior_failure_artifact="SERVER_CPU_COST_PROTOCOL_COMMAND.json",
            cause="broad directory enumeration treated a run command log as a frozen source dependency",
            original_source_and_failure_record_preserved=True,
            repair_scope="v2 fixed original six protocol files plus actual v2 runner source only"),
        commands=sys.argv, local_artifact_hashes={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in protocol_source_files()})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "CPU_PROTOCOL_TESTS.log").write_text(log.getvalue(), encoding="utf-8")
    (args.output_dir / "CPU_PROTOCOL_RESULT.json").write_text(json.dumps(record, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
