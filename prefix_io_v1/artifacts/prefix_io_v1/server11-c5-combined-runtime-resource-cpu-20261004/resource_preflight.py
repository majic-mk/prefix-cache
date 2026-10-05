"""One read-only CPU resource prerequisite check; never starts a measurement."""
from __future__ import annotations
import argparse
from fractions import Fraction
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sys

SCHEMA = "c5_combined_runtime_cpu_source_lock_v1"
SERVER_SCOPE = "artifacts/prefix_io_v1/server11-c5-combined-runtime-resource-cpu-20261004"
FILES = ("resource_preflight.py", "test_resource_preflight.py", "run_cpu_resource.py", "COMPLETE_ENTRY_CONTRACT.md")
FLAGS = ("gpu_launch_allowed", "native_execution_verified", "native_cost_qualified",
         "full_runtime_cost_qualified", "on_observation_cost_measured")
OLD_PROTOCOL_SHA256 = "6c00cf687f4903988a363e18a178ccca413076ced81920271bf9c962abde1218"
OLD_PROTOCOL_RELATIVE = "artifacts/prefix_io_v1/server11-p4-notification-benchmark-v5-cpu-20261003/CPU_COMPARISON_PROTOCOL.json"
MAX_METADATA = 10 * 1024**2
HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate JSON key")
        value[key] = item
    return value


def read_json(raw):
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def bounded_read(path, maximum=MAX_METADATA):
    with Path(path).open("rb") as stream:
        value = stream.read(maximum + 1)
    require(len(value) <= maximum, "bounded metadata/source")
    return value


def safe_source(root, name):
    require(type(name) is str and name and not name.startswith("/") and
        not any(c in name for c in ("\\", ":", "\0")) and
        all(part not in ("", ".", "..") for part in name.split("/")), "project-relative source required")
    path = root
    for part in name.split("/"):
        path /= part
        require(not path.is_symlink(), "source symlink refused")
    require(path.resolve().is_relative_to(root), "source escapes project")
    return path


def verify_lock(root, lockpath, required_files):
    require(lockpath.is_file() and not lockpath.is_symlink() and
        lockpath.resolve().is_relative_to(root), "project source lock required")
    raw = bounded_read(lockpath)
    lock = read_json(raw)
    require(type(lock) is dict and lock.get("schema") == SCHEMA and lock.get("gpu_uuid") is None and
        all(lock.get(name) is False for name in FLAGS), "CPU-only combined runtime source lock")
    rows = lock.get("files")
    require(type(rows) is list and 1 <= len(rows) <= 512, "bounded source rows")
    seen = set()
    for row in rows:
        require(type(row) is dict and set(row) == {"path", "bytes", "sha256"} and
            type(row["bytes"]) is int and 0 <= row["bytes"] <= MAX_METADATA and
            type(row["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", row["sha256"]), "exact bounded source ref")
        require(row["path"] not in seen, "duplicate source path")
        path = safe_source(root, row["path"]); seen.add(row["path"])
        require(path.is_file(), "source missing")
        data = bounded_read(path)
        require(len(data) == row["bytes"] and sha256(data).hexdigest() == row["sha256"], "source drift")
    for path in required_files:
        path = Path(path).resolve(strict=True)
        require(path.is_relative_to(root) and path.relative_to(root).as_posix() in seen, "own source missing from lock")
    return dict(source_count=len(rows), source_lock_sha256=sha256(raw).hexdigest(),
        source_rows_sha256=sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest())


def closed_protocol(root, supplied=None):
    path = Path(os.path.abspath(supplied)) if supplied is not None else root / OLD_PROTOCOL_RELATIVE
    require(path.is_relative_to(root), "frozen protocol outside project")
    path = safe_source(root, path.relative_to(root).as_posix())
    require(sha256(bounded_read(path)).hexdigest() == OLD_PROTOCOL_SHA256, "frozen CPU protocol SHA drift")
    return path


def parse_quota(text):
    require(type(text) is str, "cpu.max text required")
    words = text.split()
    require(len(words) == 2 and re.fullmatch("[0-9]+", words[1]) and 0 < int(words[1]) <= 2**63-1, "unknown cpu.max period")
    if words[0] == "max":
        return dict(kind="unbounded", cores=None, quota_us=None, period_us=int(words[1]))
    require(re.fullmatch("[0-9]+", words[0]) and 0 < int(words[0]) <= 2**63-1, "unknown cpu.max quota")
    quota, period = int(words[0]), int(words[1])
    return dict(kind="finite", cores=float(Fraction(quota, period)), quota_us=quota, period_us=period)


def parse_cpu_stat(text):
    require(type(text) is str, "cpu.stat text required")
    result = {}
    for line in text.splitlines():
        words = line.split()
        require(len(words) == 2 and re.fullmatch("[a-z_][a-z_0-9.]*", words[0]) and
            re.fullmatch("[0-9]+", words[1]) and words[0] not in result, "unknown or duplicated cpu.stat counter")
        result[words[0]] = int(words[1])
    require(all(name in result for name in ("usage_usec", "nr_periods", "nr_throttled", "throttled_usec")),
        "required cgroup counters unavailable")
    return result


def resource_decision(quota, counters, affinity, *, ledger_active=False):
    """Known adequate quota is a readiness condition, never old qualification."""
    affinity_known = (type(affinity) in (list, tuple) and bool(affinity) and
        all(type(cpu) is int and cpu >= 0 for cpu in affinity) and len(set(affinity)) == len(affinity))
    stat_known = type(counters) is dict and all(type(counters.get(name)) is int and counters[name] >= 0
        for name in ("usage_usec", "nr_periods", "nr_throttled", "throttled_usec"))
    period_known = (type(quota) is dict and type(quota.get("period_us")) is int and
        0 < quota["period_us"] <= 2**63-1)
    unlimited = period_known and quota.get("kind") == "unbounded" and quota.get("quota_us") is None
    finite = (period_known and quota.get("kind") == "finite" and
        type(quota.get("quota_us")) is int and 0 < quota["quota_us"] <= 2**63-1)
    quota_known = unlimited or finite
    adequate_quota = unlimited or (finite and quota["quota_us"] >= quota["period_us"])
    ready = quota_known and stat_known and affinity_known and adequate_quota and not ledger_active
    if not (quota_known and stat_known and affinity_known):
        status = "RESOURCE_UNKNOWN"
    elif ledger_active:
        status = "RESOURCE_BUSY_LEDGER_ACTIVE"
    elif not ready:
        status = "RESOURCE_LIMITED"
    else:
        status = "CPU_RESOURCE_PREREQUISITE_MET"
    return dict(status=status, resource_ready=bool(ready), cpu_resource_prerequisite_met=bool(ready), resource_readiness_only=True,
        finite_quota_minimum_cores=1, cpu_qualified=False, old_paired_qualification_changed=False,
        old_protocol_sha256=OLD_PROTOCOL_SHA256, old_scored_trials=84, old_total_trials=132,
        formal_benchmark_started=False, formal_benchmark_runs=0, measurement_iterations=0,
        performance_comparisons=0, actual_gpu_runs=0, gpu_launch_allowed=False, gpu_uuid=None,
        native_execution_verified=False, native_cost_qualified=False, full_runtime_cost_qualified=False,
        on_observation_cost_measured=False, valid_native_receipt=None, effective_cost_upper_ns=None,
        effective_step_budget_ns=None, production_qualified=False, performance_claim=False,
        native_attach_qualified=False, full_entry_qualified=False, on_installed_cost_qualified=False)


def resource_snapshot():
    facts = dict(cpu_max=None, cpu_stat=None, cpu_affinity=None, memory_max=None,
        GPU_nodes=sorted(str(path) for path in Path("/dev").glob("nvidia*")), probe_errors=[])
    parsers = (("cpu_max", "/sys/fs/cgroup/cpu.max", parse_quota),
               ("cpu_stat", "/sys/fs/cgroup/cpu.stat", parse_cpu_stat))
    for key, path, parse in parsers:
        try:
            raw = bounded_read(path, 4096).decode("ascii")
            facts[key] = dict(raw=raw.strip(), parsed=parse(raw))
        except (OSError, UnicodeError, ValueError) as exc:
            facts["probe_errors"].append(dict(probe=key, error_type=type(exc).__name__))
    try:
        facts["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        facts["probe_errors"].append(dict(probe="cpu_affinity", error_type=type(exc).__name__))
    try:
        facts["memory_max"] = bounded_read("/sys/fs/cgroup/memory.max", 4096).decode("ascii").strip()
    except (OSError, UnicodeError) as exc:
        facts["probe_errors"].append(dict(probe="memory_max", error_type=type(exc).__name__))
    return facts


def run(args):
    root = args.project_root.resolve(strict=True)
    lockpath = Path(os.path.abspath(args.source_lock))
    require(lockpath.is_relative_to(root), "source lock outside project")
    lockpath = safe_source(root, lockpath.relative_to(root).as_posix())
    output = args.output_dir.resolve()
    require(output.is_relative_to(root) and not output.exists(), "new project output directory required")
    protocol = closed_protocol(root, getattr(args, "old_protocol", None))
    required = [HERE / name for name in FILES] + [protocol]
    source_before = verify_lock(root, lockpath, required)
    ledgerpath = safe_source(root, "experiments/prefix_io_v1/gpu-budget-ledger.json")
    before_ledger = bounded_read(ledgerpath)
    ledger = read_json(before_ledger)
    require(type(ledger) is dict and type(ledger.get("gpu_wall_seconds")) in (int, float) and
        math.isfinite(ledger["gpu_wall_seconds"]) and ledger["gpu_wall_seconds"] >= 0,
        "actual bounded GPU budget ledger required")
    facts = resource_snapshot()
    decision = resource_decision(facts["cpu_max"]["parsed"] if facts["cpu_max"] else None,
        facts["cpu_stat"]["parsed"] if facts["cpu_stat"] else None, facts["cpu_affinity"],
        ledger_active=ledger.get("active_reservation") is not None)
    source_after = verify_lock(root, lockpath, required)
    require(source_before == source_after and bounded_read(ledgerpath) == before_ledger,
        "source/ledger changed during one resource snapshot")
    imports = sorted(name for name in sys.modules if name == "torch" or name.startswith("torch.") or
        name == "vllm" or name.startswith("vllm.") or name == "cupy" or name.startswith("cupy."))
    require(not imports, "resource preflight imported GPU/model modules")
    result = dict(decision, origin="actual_readonly_resource_preflight", location="server_cpu" if os.name == "posix" else "local_cpu",
        resource_facts=facts, source_before=source_before, source_after=source_after,
        source_count=source_before["source_count"], source_lock_sha256=source_before["source_lock_sha256"],
        gpu_ledger_sha256=sha256(before_ledger).hexdigest(), gpu_wall_seconds=ledger["gpu_wall_seconds"],
        verified_frozen_protocol_path=protocol.relative_to(root).as_posix(),
        active_reservation=ledger.get("active_reservation"), gpu_ledger_unchanged=True,
        forbidden_imports=imports, actual_model_processes_started=0, configs_created=0, jobs_created=0,
        command=[sys.executable] + sys.argv)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "RESOURCE_PREFLIGHT.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "source-lock", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--old-protocol", type=Path, help="local archived path; must have the same frozen SHA")
    result = run(parser.parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
