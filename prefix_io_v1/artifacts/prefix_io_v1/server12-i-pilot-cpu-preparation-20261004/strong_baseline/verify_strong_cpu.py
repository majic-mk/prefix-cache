"""Actual CPU command/result recorder; never creates a GPU effect receipt."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("_verify_strong_replay_values", HERE / "original_value_replay.py")
    replay = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = replay
    spec.loader.exec_module(replay)
    entries = replay.manifest_entries()
    engine_pin = json.loads((replay.INPUTS / "STRONG_COMMON_ENGINE_SOURCE_INPUT.json").read_text(encoding="utf-8"))
    entries.append((engine_pin, replay.INPUTS / engine_pin["local_basename"]))
    paths = [replay.source(name) for name in replay.HASHES]
    for row, path in entries:
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError("actual author/engine source mismatch")
        paths.append(path)
    control = replay.source("vllm").parents[2] / "prefix-io-p4-02-cpu/src/prefix_io_control"
    # Pure parser closure may be imported by the CPU options replay; record all
    # source inputs in this already-frozen control directory without execution.
    paths.extend(sorted(control.glob("*.py")))
    paths = list(dict.fromkeys(paths))
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    names = {"from_extra_config", "_build_planner", "plan", "plan_candidates", "generate_request_schedule",
             "build_parser", "load_trace_prompts", "build_global_specs", "parse_p4_options", "compute_slot_count"}
    methods = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in names:
                methods.append(dict(source=str(path), source_sha256=before[str(path)], symbol=node.name,
                                    line=node.lineno, end_line=node.end_lineno,
                                    ast_sha256=hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()))
    command = [sys.executable, "-B", "-I", "-S", "-m", "unittest", "discover", "-s", str(HERE),
               "-p", "test_strong_baseline.py", "-v"]
    started = time.monotonic()
    run = subprocess.run(command, cwd=replay.PROJECT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    success = run.returncode == 0 and before == after
    record = dict(schema="strong_native_u_i_cpu_preparation_result_v1",
                  status="PASS_CPU_ENTRY_PREPARATION_EFFECT_BLOCKED" if success else "FAIL_CPU_PREPARATION",
                  command=command, exit_code=run.returncode, elapsed_seconds=time.monotonic()-started,
                  stdout=run.stdout, stderr=run.stderr, inputs_before=before, inputs_after=after,
                  input_file_count=len(paths), input_bytes_unchanged=before == after, original_source_bindings=methods,
                  actual_gpu_runs=0, valid_native_receipt=None, ordinary_interference_qualified=False,
                  strong_u_effect_run_qualified=False, cache_model_executor_changed=False,
                  preserved_original_features=["exact_prefix_cache", "LoadPlanner_and_cost_admission", "preload",
                                                "shared_staging", "copy_fusion", "async_pipeline"],
                  required_gpu_gates=["independent_prospective_development_deadline", "frozen_normal_request_stream",
                                      "planner_on_domain_calibration_holdout", "on_lifecycle", "existing_original_budget_guard"],
                  prior_singlefile_planner_off_receipt_reusable_for_strong_effect=False,
                  CPU_fixtures_are_GPU_observations=False)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: record[key] for key in ("status", "exit_code", "input_file_count", "input_bytes_unchanged", "actual_gpu_runs")}))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
