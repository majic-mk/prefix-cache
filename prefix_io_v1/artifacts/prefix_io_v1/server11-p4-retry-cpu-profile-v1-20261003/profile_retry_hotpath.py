"""Bounded CPU-only diagnosis of frozen v3 source-extracted retry paths.

Uses the existing CPU fixture and real reactor AST, bridge, policy and immutable
value constructors. No candidate source is changed. Measurements are CPU fixture
costs, not GPU costs or an explanation of measured GPU latency by themselves.
The fixture replaces hardware and the live capture accessor with scalar values.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time


RETRIES = 155


def sha_ref(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def summarize(values):
    middle = statistics.median(values)
    return {"count": len(values), "min": min(values), "median": middle,
            "max": max(values),
            "median_absolute_deviation": statistics.median(abs(x - middle) for x in values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", default=os.environ.get("CANDIDATE_ROOT"))
    parser.add_argument("--author-source-root", default=os.environ.get("SERVER11_AUTHOR_SOURCE_ROOT"))
    parser.add_argument("--groups", type=int, default=41)
    parser.add_argument("--warmup-groups", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()
    require(args.candidate_root, "explicit frozen CANDIDATE_ROOT required")
    require(5 <= args.groups <= 101 and 0 <= args.warmup_groups <= 10,
            "bounded CPU group counts required")
    candidate = Path(args.candidate_root).resolve(strict=True)
    if args.author_source_root:
        os.environ["SERVER11_AUTHOR_SOURCE_ROOT"] = str(Path(args.author_source_root).resolve(strict=True))
    fixture_path = candidate / "test_single_file_retry_observation.py"
    spec = importlib.util.spec_from_file_location("_cpu_retry_profile_fixture", fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    require("torch" not in sys.modules and "numpy" not in sys.modules,
            "pure CPU AST fixture must not import hardware packages")

    def group(profile=False):
        # Construct a fresh exact existing fixture outside the timed region.
        owner = fixture.Owner(max_wait=10**12)
        owner_tid = threading.get_ident()
        prof = cProfile.Profile() if profile else None
        if prof is not None:
            prof.enable()
        wall0, cpu0 = time.perf_counter_ns(), time.thread_time_ns()
        for _ in range(RETRIES):
            require(owner.repeat() is False, "all bounded retries must defer")
        cpu_ns, wall_ns = time.thread_time_ns() - cpu0, time.perf_counter_ns() - wall0
        if prof is not None:
            prof.disable()
        require(threading.get_ident() == owner_tid, "same CPU owner thread required")
        facts = {"retries": RETRIES, "collects": owner.collects,
                 "previews": owner.bridge.single_file_previews,
                 "actual_blocked_attempts": owner.bridge.single_file_blocked_attempts,
                 "observation_reuses": owner.bridge.single_file_observation_reuses,
                 "reserve_calls": owner.staging_pool.reserves,
                 "release_calls": owner.staging_pool.releases,
                 "final_free_slots": owner.staging_pool.free_count,
                 "original_open_start_ns": owner.ready.open_start_ns,
                 "cached_snapshot_monotonic_ns": owner._prefix_single_file_retry[1].monotonic_ns,
                 "cached_native_captured_ns": owner._prefix_single_file_retry[1].native_state.captured_ns,
                 "submitted_count": len(owner.submitted)}
        require(facts == {"retries":155, "collects":1, "previews":155,
            "actual_blocked_attempts":155, "observation_reuses":154,
            "reserve_calls":155, "release_calls":155, "final_free_slots":16,
            "original_open_start_ns":100, "cached_snapshot_monotonic_ns":1000,
            "cached_native_captured_ns":1000, "submitted_count":0},
            "original bounded retry semantics changed")
        return {"thread_cpu_ns":cpu_ns, "wall_ns":wall_ns}, facts, prof

    for _ in range(args.warmup_groups):
        group()
    timings = []
    for _ in range(args.groups):
        timing, facts, _ = group()
        timings.append(timing)
    prof_timing, prof_facts, prof = group(profile=True)
    prof.create_stats()
    rows = []
    for (filename, line, name), (primitive, total, self_seconds, cumulative_seconds, callers) in prof.stats.items():
        rows.append({"file":filename, "line":line, "function":name,
                     "primitive_calls":primitive, "total_calls":total,
                     "self_ns":round(self_seconds*1e9),
                     "cumulative_ns":round(cumulative_seconds*1e9)})
    rows.sort(key=lambda row: (-row["cumulative_ns"], row["file"], row["line"]))
    sources = [sha_ref(fixture_path), sha_ref(fixture.SOURCE),
               sha_ref(Path(sys.modules["prefix_io_control.p4_bridge"].__file__)),
               sha_ref(Path(sys.modules["prefix_io_control.p4_policy"].__file__)),
               sha_ref(Path(sys.modules["prefix_io_control.p4_types"].__file__)),
               sha_ref(Path(sys.modules["prefix_io_control.dispatch_shadow"].__file__)),
               sha_ref(Path(sys.modules["prefix_io_control.dispatch_budget"].__file__))]
    result = {"schema_version":1, "scope":"cpu_fixture_hotpath_diagnosis_only",
              "candidate_root":str(candidate), "python":sys.version,
              "thread_time_clock":time.get_clock_info("thread_time").implementation,
              "retries_per_group":RETRIES, "warmup_groups":args.warmup_groups,
              "unprofiled_groups":timings,
              "unprofiled_thread_cpu_ns":summarize([x["thread_cpu_ns"] for x in timings]),
              "unprofiled_wall_ns":summarize([x["wall_ns"] for x in timings]),
              "profiled_group_timing":prof_timing, "profiled_function_rows":rows,
              "semantic_facts":facts, "profile_semantic_facts":prof_facts,
              "source_refs":sources,
              "limitations":["CPU fixture costs do not measure GPU overhead or establish a GPU latency cause",
                  "cProfile changes execution costs; its times are not merged with unprofiled timing",
                  "cumulative function times overlap and must not be summed",
                  "existing fixture substitutes native slot, file, clock and live capture accessor",
                  "no scheduling, kernel, sleep, switch interval, workload or threshold was changed"],
              "torch_imported":False, "gpu_workloads_run":0,
              "candidate_files_changed":False, "performance_effect_verified":False}
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        with Path(args.output).open("x", encoding="utf-8") as out:
            out.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
