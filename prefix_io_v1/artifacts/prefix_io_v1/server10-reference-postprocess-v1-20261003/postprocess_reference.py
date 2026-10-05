"""Read completed server10 GPU references with a CPU-only callable-pin repair.

The original numerical analyzer and frozen GPU source remain unchanged. Python
3.12 knows imported modules when compiling a complete module, so isolated AST
function compilation is not a valid bytecode reference for those functions.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT = Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
D = "artifacts/prefix_io_v1/server10-reference-migration-v1-20261003"
POST = "artifacts/prefix_io_v1/server10-reference-postprocess-v1-20261003"
ENTRY_REF = dict(path=D + "/run_server10_reference.py", bytes=14507,
    sha256="7db73e900c344a3f44e9dd60d8583125794e8291728d3adca3871267959c157d")
LOCK = D + "/gpu-source-lock-server10-reference.json"
SCOPE = D + "/SERVER10_REFERENCE_AUTHORIZED_SCOPE.json"
LEDGER = "experiments/prefix_io_v1/gpu-budget-ledger.json"
NAMES = ("server10-g3-reference-native-01", "server10-g3-reference-paired-01")
BACKENDS = ("torch", "vllm", "py_kvcache", "cupy", "triton", "pynvml")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def original_function_from_module(function, path, raw, name):
    """Compile all pinned source, but execute none of it, to preserve imports."""
    require(type(function) is types.FunctionType and
        Path(function.__code__.co_filename).resolve() == path,
        "original analyzer callable filename: " + name)
    require(type(raw) is bytes and len(raw) <= 4 * 1024 ** 2,
        "bounded pinned analyzer bytes")
    module_code = compile(raw, str(path), "exec", dont_inherit=True)
    candidates = [code for code in module_code.co_consts
        if isinstance(code, types.CodeType) and code.co_name == name]
    require(len(candidates) == 1, "one original top-level analyzer function: " + name)
    require(function.__code__ == candidates[0], "original analyzer callable code drift: " + name)


@contextmanager
def corrected_callable_check(checker):
    original = checker._original_function
    checker._original_function = original_function_from_module
    try:
        yield
    finally:
        checker._original_function = original


def safe(root, relative):
    require(type(relative) is str and relative and not Path(relative).is_absolute()
        and "\\" not in relative and all(p not in ("", ".", "..") for p in relative.split("/")),
        "bounded relative evidence path")
    path = root / relative
    cursor = path
    while cursor != root:
        require(not cursor.is_symlink(), "evidence symlink refused")
        cursor = cursor.parent
    require(path.resolve().is_relative_to(root), "evidence must remain in project")
    return path


def reference(root, relative):
    path = safe(root, relative)
    require(path.is_file() and path.stat().st_size <= 8 * 1024 ** 2, "bounded actual CPU source/evidence")
    raw = path.read_bytes()
    return dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def raw_result_paths():
    paths = []
    suffixes = ("result.json", "details/post-source-verification.json",
        "details/server10-migration-runtime-state.json", "details/reference-runtime-result.json",
        "details/calibration-runtime-result.json", "details/acquisition/result.json",
        "details/acquisition/frozen-config.json")
    for name in NAMES:
        prefix = "experiments/prefix_io_v1/runs/" + name + "/"
        paths.extend(prefix + suffix for suffix in suffixes)
        paths.extend(prefix + "details/original-generate-%02d.json" % i for i in range(6))
    return paths


class CPUImportBlocker:
    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in BACKENDS):
            self.attempts.append(fullname)
            raise ImportError("CPU postprocess refuses backend import: " + fullname)
        return None


def process(root):
    require(root == ROOT and root.resolve(strict=True) == root, "fixed actual cloned project root")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU postprocess requires hidden GPU")
    require(not any(n == b or n.startswith(b + ".") for n in sys.modules for b in BACKENDS),
        "fresh CPU-only process required")
    result = dict(schema_version=1, status="FAILED_CPU_REFERENCE_POSTPROCESS",
        actual_GPU_runs_this_process=0, GPU_initialized=False, old_GPU_results_modified=False,
        frozen_GPU_source_modified=False, numerical_comparator_changed=False, numerical_tolerance_changed=False,
        performance_claim=False, P4_complete=False, source_lock_path=LOCK, scope_record_path=SCOPE,
        correction="whole pinned module compile instead of isolated AST function compile; no module exec",
        original_parent_exit78_retained=True)
    ledger_before = reference(root, LEDGER)
    raw_before = [reference(root, path) for path in raw_result_paths()]
    scope_before = reference(root, SCOPE)
    lock_before = reference(root, LOCK)
    result.update(ledger_before=ledger_before, original_GPU_result_refs=raw_before,
        original_scope_ref=scope_before, original_source_lock_ref=lock_before,
        postprocessor_ref=reference(root, POST + "/postprocess_reference.py"))
    entry_pin = reference(root, ENTRY_REF["path"])
    require(entry_pin == ENTRY_REF, "unchanged server10 entry source")
    before_modules = set(sys.modules)
    blocker = CPUImportBlocker()
    sys.meta_path.insert(0, blocker)
    restore = None
    checker = None
    checker_before = None
    started = time.perf_counter()
    try:
        path = safe(root, ENTRY_REF["path"])
        migration = types.ModuleType("_cpu_postprocess_migration_entry")
        migration.__file__ = str(path)
        exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), migration.__dict__)
        entry, restore = migration.build_facade(root, LOCK)
        plan, refs, lock_ref = entry.full_sources(root, LOCK)
        require(len(refs) == 4104 and lock_ref == lock_before, "actual original 4104-reference source closure")
        scope = entry.read_json(safe(root, SCOPE))
        require(scope["source_lock"] == lock_ref, "same original GPU scope/source lock")
        plan.validate_scope(root, scope, refs)
        scope["_actual_scope_path"] = SCOPE
        ledger = entry.read_json(safe(root, LEDGER))
        require(ledger.get("active_reservation") is None, "completed GPU jobs must have no active reservation")
        receipts, checks = entry.prior_results(root, plan, refs, scope,
            scope_before["sha256"], ledger, LOCK)
        require(len(receipts) == 2 and plan.next_job(receipts) is None,
            "both original GPU jobs must succeed, close and drain")
        checker = entry.load_module(root, entry.RESULT_MODULE, refs[entry.RESULT_MODULE])
        checker_before = checker._original_function
        with corrected_callable_check(checker):
            analysis = entry.analyze_pair(root, refs, checks)
        require(checker._original_function is checker_before, "original checker must restore")
        require(analysis["status"] in ("PASSED_EXACT_CACHED_REFERENCE", "FAILED_EXACT_CACHED_REFERENCE"),
            "only original numerical verdict accepted")
        result.update(status=analysis["status"], original_analysis_result=analysis,
            verified_original_job_receipts=receipts, verified_reference_path_checks=checks,
            source_count=len(refs), original_reference_requests=12,
            cumulative_gpu_wall_seconds=ledger["gpu_wall_seconds"],
            remaining_original_gpu_budget_seconds=28800 - ledger["gpu_wall_seconds"],
            callable_check_restored=True)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        if checker is not None and checker_before is not None:
            result["callable_check_restored"] = checker._original_function is checker_before
        if restore is not None:
            restore()
        if blocker in sys.meta_path:
            sys.meta_path.remove(blocker)
        for name in set(sys.modules) - before_modules:
            if name.startswith("_server10_reference_"):
                sys.modules.pop(name, None)
        result["CPU_seconds"] = time.perf_counter() - started
        result["blocked_backend_import_attempts"] = list(blocker.attempts)
        result["ledger_after"] = reference(root, LEDGER)
        result["ledger_unchanged"] = result["ledger_after"] == ledger_before
        result["original_GPU_inputs_unchanged"] = all(reference(root, row["path"]) == row for row in raw_before)
        result["original_scope_and_lock_unchanged"] = (reference(root, SCOPE) == scope_before
            and reference(root, LOCK) == lock_before)
        result["entry_source_unchanged"] = reference(root, ENTRY_REF["path"]) == ENTRY_REF
        require(result["ledger_unchanged"] and result["original_GPU_inputs_unchanged"] and
            result["original_scope_and_lock_unchanged"] and result["entry_source_unchanged"],
            "CPU postprocess input or ledger changed")
        require(not blocker.attempts, "unexpected backend import attempted")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.project.resolve()
    output = safe(root, POST + "/REFERENCE_CPU_POSTPROCESS_RESULT.json")
    require(output.parent.is_dir() and not output.exists(), "new standalone CPU result path required")
    result = process(root)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], evidence=str(output),
        actual_GPU_runs=0, ledger_unchanged=result["ledger_unchanged"],
        original_GPU_inputs_unchanged=result["original_GPU_inputs_unchanged"],
        error=result.get("error")), ensure_ascii=False))
    return 0 if result["status"] == "PASSED_EXACT_CACHED_REFERENCE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
