"""Run append-only CPU contract tests with explicit roots and a caller's lock."""
import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
FORBIDDEN = ("torch", "vllm", "py_kvcache", "cupy", "cuda")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pairs(items):
    result = {}
    for name, value in items:
        require(name not in result, "duplicate JSON key"); result[name] = value
    return result


def load(path):
    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def lock_check(project, source_lock):
    value = load(source_lock)
    require(value.get("schema") == "c5_cost_binding_cpu_source_lock_v1" and
        value.get("gpu_launch_allowed") is False and value.get("native_execution_verified") is False,
        "caller frozen CPU-only source lock required")
    files = value.get("files")
    require(type(files) is list and 1 <= len(files) <= 8192, "bounded complete source refs")
    seen = set()
    for ref in files:
        require(type(ref) is dict and set(ref) == {"path", "bytes", "sha256"}, "exact lock ref")
        name = ref["path"]
        require(type(name) is str and name and not name.startswith("/") and
            not any(c in name for c in ("\\", ":", "\0")) and
            all(part not in ("", ".", "..") for part in name.split("/")) and name not in seen,
            "unique bounded source path")
        seen.add(name); path = project
        for part in name.split("/"):
            path /= part; require(not path.is_symlink(), "source symlink refused")
        require(path.is_file() and project in path.resolve().parents and type(ref["bytes"]) is int and
            0 < ref["bytes"] <= 32 * 1024**2 and path.stat().st_size == ref["bytes"], "source size drift")
        data = path.read_bytes()
        require(sha256(data).hexdigest() == ref["sha256"], "source SHA drift: " + name)
    return len(files), seen


def child_tests():
    # This branch is selected by the parent process's internal subcommand only.
    import unittest
    spec = importlib.util.spec_from_file_location("_c5_cost_cpu_test_suite",
        HERE / "test_c5_cost_binding_preparation.py")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    forbidden = [name for name in sys.modules if any(name == prefix or name.startswith(prefix + ".")
        for prefix in FORBIDDEN)]
    document = dict(status="PASS_CPU_COST_BINDING_CONTRACT_ONLY" if result.wasSuccessful() and not forbidden
        else "FAIL_CPU_COST_BINDING_CONTRACT", tests=result.testsRun, failures=len(result.failures),
        errors=len(result.errors), skipped=len(result.skipped), forbidden_modules_imported=forbidden,
        origin="synthetic_cpu_contract", actual_gpu_runs=0, native_execution_verified=False,
        native_cost_qualified=False, full_runtime_cost_qualified=False, on_observation_cost_measured=False,
        valid_native_receipt=None, effective_cost_upper_ns=None, effective_step_budget_ns=None,
        gpu_launch_allowed=False, permission_scope_created=False, performance_claim=False)
    print(json.dumps(document, sort_keys=True))
    return 0 if document["status"].startswith("PASS_") else 1


def main():
    if sys.argv[1:] == ["--internal-test-process"]:
        return child_tests()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-root", "preparation-root", "native-root", "original-estimator",
        "source-lock", "project-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--location", choices=("local_cpu", "server_cpu"), required=True)
    args = parser.parse_args()
    project = args.project_root.resolve(strict=True); lock = args.source_lock.resolve(strict=True)
    candidate = args.candidate_root.resolve(strict=True); prep = args.preparation_root.resolve(strict=True)
    native = args.native_root.resolve(strict=True); original = args.original_estimator.resolve(strict=True)
    require(project in lock.parents, "source lock must be in independent project root")
    count, source_paths = lock_check(project, lock)
    required = [HERE / name for name in ("run_cpu_cost_binding.py", "c5_cost_binding_preparation.py",
        "test_c5_cost_binding_preparation.py")]
    required.extend([candidate / "native_full_step_collector.py", candidate / "source/third_party/work/"
        "py-kvcache-p4-02-cpu/py_kvcache/reactor.py", prep / "PREPARATION_SOURCE_LOCK.json",
        native / "native_conditional_cost.py", native / "prepare_and_verify_native_cost.py",
        native / "test_native_conditional_cost.py", original])
    require(all(project in path.resolve().parents and path.resolve().relative_to(project).as_posix()
        in source_paths for path in required), "all actual test/adapter/fixture/numerical sources must be frozen")
    raw_lock = lock.read_bytes(); before_sha = sha256(raw_lock).hexdigest()
    output = args.output_dir.resolve()
    require(HERE in output.parents and not output.exists(), "fresh output directory within new factory artifact")
    output.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", SERVER11_C5_COST_CANDIDATE_ROOT=str(candidate),
        SERVER11_C5_COST_PREPARATION_ROOT=str(prep), SERVER11_C5_COST_NATIVE_ROOT=str(native),
        NATIVE_ORIGINAL_ESTIMATOR=str(original), C5_COST_TEST_TEMP_ROOT=str(output))
    command = [sys.executable, "-B", "-I", "-S", str(HERE / "run_cpu_cost_binding.py"), "--internal-test-process"]
    started = time.monotonic()
    completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=120)
    elapsed = time.monotonic() - started
    (output / "TEST_STDOUT.log").write_text(completed.stdout, encoding="utf-8")
    (output / "TEST_STDERR.log").write_text(completed.stderr, encoding="utf-8")
    after_count, _ = lock_check(project, lock)
    require(lock.read_bytes() == raw_lock and after_count == count, "frozen closure changed during CPU tests")
    child = json.loads(completed.stdout) if completed.stdout.strip() else {}
    document = dict(location=args.location, interpreter=sys.executable, python_version=platform.python_version(),
        platform=platform.platform(), command=command, elapsed_wall_seconds=elapsed, exit=completed.returncode,
        source_lock_sha256=before_sha, source_files_verified_before=count, source_files_verified_after=after_count,
        explicit_roots=dict(project=str(project), candidate=str(candidate), preparation=str(prep),
            native=str(native), original_estimator=str(original)), result=child)
    with (output / "CPU_RESULT.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    print(json.dumps(document, sort_keys=True))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
