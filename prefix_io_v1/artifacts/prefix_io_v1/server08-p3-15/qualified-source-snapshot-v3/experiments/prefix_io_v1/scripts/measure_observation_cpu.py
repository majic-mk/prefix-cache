"""CPU-only observer microbenchmark; synthetic counters, no device I/O or model.

Measures the optional hook and publisher only, NOT reactor throughput, decode,
end-to-end overhead, GPU latency, or the 2% engineering target.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import threading
import time
from types import SimpleNamespace as NS

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["RUN_E2E_TESTS"] = ""
from py_kvcache.reactor import IoReactor
from prefix_io_control.publication import SnapshotPublisher


def reactor(sink):
    r = IoReactor.__new__(IoReactor)  # No native constructor / backend allocation.
    r._worker = NS(ident=threading.get_ident())
    r.actual_staging_bytes = 64 * 4096 + 4095  # Synthetic metadata, not allocated bytes.
    r.staging_pool = NS(free_count=0)
    r._active = [NS(job_id=i, total_files=4, done_files=1, inflight_files=2,
                   future_set=False, failed=None) for i in range(32)]
    r._inflight = {i: NS(op_kind="read" if i % 2 else "write") for i in range(64)}
    r._pending_copies = [NS(nbytes=4096, is_store=bool(i % 2)) for i in range(64)]
    r._observation_sink = sink
    r._observation_failures = 0
    return r


def call_hook(r):
    # Same optional branch used after the native pump; the pump is NOT measured.
    if getattr(r, "_observation_sink", None) is not None:
        r._observe_once()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if not (100 <= args.iterations <= 100000 and 3 <= args.repeats <= 20):
        parser.error("CPU diagnostic bounds: iterations 100..100000, repeats 3..20")
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if not output.is_relative_to(root / "artifacts" / "prefix_io_v1"):
        parser.error("output must be inside project artifacts/prefix_io_v1")
    if output.exists():
        parser.error("refusing to overwrite prior evidence")
    cases = {
        "off_hook": reactor(None),
        "enabled_sample_not_due": reactor(SnapshotPublisher(run_id="cpu-bench-skip", interval_ns=10**18)),
        "periodic_1ms_diagnostic": reactor(SnapshotPublisher(run_id="cpu-bench-periodic", interval_ns=1_000_000)),
        "capture_every_call_worst_case": reactor(SnapshotPublisher(run_id="cpu-bench-all", interval_ns=1)),
    }
    for r in cases.values():
        for _ in range(100):
            call_hook(r)
    raw = []
    rng = random.Random(0)
    for repeat in range(args.repeats):
        order = list(cases)
        rng.shuffle(order)
        for name in order:
            r = cases[name]
            sink = r._observation_sink
            before_count = sink.published_count if sink else 0
            wall = time.perf_counter_ns()
            cpu = time.process_time_ns()
            for _ in range(args.iterations):
                call_hook(r)
            cpu_elapsed = time.process_time_ns() - cpu
            wall_elapsed = time.perf_counter_ns() - wall
            assert r._observation_failures == 0
            raw.append({"case": name, "repeat": repeat,
                        "wall_ns": wall_elapsed, "cpu_ns": cpu_elapsed,
                        "wall_ns_per_call": wall_elapsed / args.iterations,
                        "cpu_ns_per_call": cpu_elapsed / args.iterations,
                        "snapshots_published": (sink.published_count if sink else 0) - before_count})
    summary = {}
    for name in cases:
        rows = [row for row in raw if row["case"] == name]
        summary[name] = {
            "median_wall_ns_per_call": statistics.median(row["wall_ns_per_call"] for row in rows),
            "median_cpu_ns_per_call": statistics.median(row["cpu_ns_per_call"] for row in rows),
            "max_repeat_mean_wall_ns_per_call": max(row["wall_ns_per_call"] for row in rows),
            "snapshots_published_in_timed_calls": sum(row["snapshots_published"] for row in rows),
        }
    source_paths = [root / "src/prefix_io_control/observation.py",
                    root / "src/prefix_io_control/publication.py",
                    root / "src/prefix_io_control/dependencies.py",
                    Path(sys.modules["py_kvcache.reactor"].__file__).resolve()]
    result = {
        "evidence_type": "CPU_microbenchmark_synthetic_metadata_real_observer_hook",
        "python": sys.version, "platform": platform.platform(),
        "iterations_per_repeat": args.iterations, "repeats": args.repeats,
        "order_seed": 0, "synthetic_shape": {"parents": 32, "ring_ops": 64, "copies": 64},
        "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
        "cgroup_cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        "source_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "summary": summary, "raw_repeats": raw,
        "gpu_executed": False, "end_to_end_measured": False,
        "interference_calibration": False, "engineering_2pct_target_verified": False,
        "limitations": [
            "Hook and snapshot construction only; no I/O, model, GPU or native pump measured.",
            "1 ms is a diagnostic sampling point, not a frozen production parameter.",
            "Synthetic staging byte fields describe fixture metadata, not actual GPU/pinned allocation.",
            "Reported medians are across repeat means, not individual-call latency percentiles.",
            "Container scheduling and benchmark harness overhead are included.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
