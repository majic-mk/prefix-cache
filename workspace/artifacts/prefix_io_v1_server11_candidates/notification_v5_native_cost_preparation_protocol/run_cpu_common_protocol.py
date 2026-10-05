"""Finite C5 common collector/namespace source audit; no GPU/native execution."""
import argparse
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
FILES = ("run_cpu_common_protocol.py", "test_common_collector_boundary.py", "PATCH_MAP.md")
FALSE_FLAGS = ("gpu_launch_allowed", "native_execution_verified", "native_cost_qualified",
               "full_runtime_cost_qualified", "on_observation_cost_measured")


def require(ok, message):
    if not ok: raise ValueError(message)


def check_lock(project, path, required):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate lock key"); value[key] = item
        return value
    raw = path.read_bytes(); lock = json.loads(raw, object_pairs_hook=pairs)
    require(lock.get("schema") == "c5_native_cost_preparation_cpu_source_lock_v1" and
        all(lock.get(name) is False for name in FALSE_FLAGS) and lock.get("gpu_uuid") is None,
        "new CPU-only native preparation outer lock")
    refs = lock.get("files")
    require(type(refs) is list and 1 <= len(refs) <= 8192, "bounded source list")
    seen = set()
    for row in refs:
        require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "exact source ref")
        name = row["path"]
        require(type(name) is str and name and name not in seen and not any(c in name for c in ("\\", ":", "\0")) and
            all(p not in ("", ".", "..") for p in name.split("/")), "unique project-relative source")
        seen.add(name); target = project
        for part in name.split("/"):
            target /= part; require(not target.is_symlink(), "source symlink refused")
        require(project in target.resolve().parents and type(row["bytes"]) is int and 0 <= row["bytes"] <= 32 * 1024**2,
                "bounded source file")
        data = target.read_bytes()
        require(len(data) == row["bytes"] and sha256(data).hexdigest() == row["sha256"], "source drift: " + name)
    for source in required:
        source = source.resolve(strict=True)
        require(project in source.parents and source.relative_to(project).as_posix() in seen, "actual source missing from outer lock: " + str(source))
    return dict(files=len(refs), sha256=sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preparation-root", "candidate-root", "native-root", "output-dir"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args()
    require((args.source_lock is None) == (args.project_root is None), "source-lock/project-root supplied together")
    prep = args.preparation_root.resolve(strict=True); candidate = args.candidate_root.resolve(strict=True)
    native = args.native_root.resolve(strict=True)
    reactor_relative = "source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py"
    required = [HERE/name for name in FILES] + [candidate/"native_full_step_collector.py",
        candidate/reactor_relative, prep/"common_candidate/native_full_step_collector.py",
        prep/"common_candidate"/reactor_relative]
    for root in (prep, native):
        required.extend(root/name for name in ("run_native_cost_experiment.py", "prepare_and_verify_native_cost.py", "native_conditional_cost.py"))
    project = args.project_root.resolve(strict=True) if args.project_root is not None else None
    source_before = check_lock(project, args.source_lock.resolve(strict=True), required) if project is not None else None
    source_hashes_before = {str(path): sha256(path.read_bytes()).hexdigest() for path in required}
    os.environ.update(CUDA_VISIBLE_DEVICES="", C5_NATIVE_PREPARATION_ROOT=str(prep),
        C5_NATIVE_PROTOCOL_CANDIDATE_ROOT=str(candidate), C5_NATIVE_PROTOCOL_BASELINE_ROOT=str(native))
    spec = importlib.util.spec_from_file_location("_common_collector_protocol_cpu", HERE / "test_common_collector_boundary.py")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    source_after = check_lock(project, args.source_lock.resolve(strict=True), required) if project is not None else None
    source_hashes_after = {str(path): sha256(path.read_bytes()).hexdigest() for path in required}
    require(source_before == source_after and source_hashes_before == source_hashes_after, "source changed during protocol audit")
    forbidden = [name for name in sys.modules if any(name == base or name.startswith(base+".") for base in ("torch", "vllm", "cuda", "cupy"))]
    result = dict(status="PASS_COMMON_COLLECTOR_PROTOCOL_CPU_ONLY" if tests.wasSuccessful() and not forbidden else "FAIL_COMMON_COLLECTOR_PROTOCOL_CPU",
        tests=tests.testsRun, passed=tests.testsRun-len(tests.failures)-len(tests.errors)-len(tests.skipped),
        failed=len(tests.failures), errors=len(tests.errors), skipped=len(tests.skipped),
        origin="synthetic_cpu_api_and_readonly_source_audit", actual_cuda_event=False, actual_model=False,
        actual_worker_connector=False, actual_collector_install_api=True,
        actual_gpu_runs=0, GPU_runs=0, formal_benchmark=0, performance_claim=False,
        native_execution_verified=False, native_cost_qualified=False, full_runtime_cost_qualified=False,
        on_observation_cost_measured=False, gpu_launch_allowed=False, gpu_uuid=None,
        source_lock_sha256=source_before["sha256"] if source_before is not None else None,
        source_before=source_before, source_after=source_after, actual_source_hashes=source_hashes_after,
        forbidden_imports=forbidden, command=sys.argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "CPU_PROTOCOL_TESTS.log").write_text(log.getvalue(), encoding="utf-8")
    (args.output_dir / "CPU_PROTOCOL_RESULT.json").write_text(json.dumps(result, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
