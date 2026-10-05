"""CPU fail-fast environment diagnosis; preserves the frozen GPU entry and lock.

Early checks can only reject or request the original complete CPU preflight.
They never authorize, launch, or mark a GPU/model/source qualification complete.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import time

ROOT = "/root/autodl-tmp/prefix-io-v1-handoff/project"
BASE = "artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002"
LOCK_REF = dict(path=BASE + "/gpu-source-lock-reference-v1.json", bytes=910072,
    sha256="8443f45dd4dd6c1c64595199051f5e88449268af347bcbd836c927b754eabe30")
ENTRY_REF = dict(path=BASE + "/run_g3_reference_pilot.py", bytes=22725,
    sha256="1bd502be997fd1a784ab49aa0262d45dabeca2374a53312d4c96dbdc504ea7a6")
INVENTORY_REF = dict(path="artifacts/prefix_io_v1/server09-cuda13-cpu-20261001/CUDA13_SOURCE_INVENTORY.json",
    bytes=692329, sha256="2ed079701c53d3e37005cf33432937e6c442f06cad16c5aded06efffcc7c36e0")
PROOF_REF = dict(path="artifacts/prefix_io_v1/server09-cuda13-cpu-20261001/CPU_COMPILE_LINK_RESULT.json",
    bytes=5589, sha256="3c3f9b092aea3ca89e07f31530a2ae3b38ab65cf0a8868821031f1d7f8d14e3f")
SDK_REF = dict(path="artifacts/prefix_io_v1/server09-cuda13-toolchain-v2-20261001/cuda13_sdk_overlay.py",
    bytes=16929, sha256="ed846361788e9bdde853e6e0aa16e361329e3f2dffbfbe2c4f396acf9e9d1ecf")
DRIVER_REF = dict(path="/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05", bytes=91501576,
    sha256="76e0d9678d41cf6b6ae71d18549d88a963d3d434f08108baa12457eb53ca88d6")
FALSE_FLAGS = dict(actual_gpu_runs=0, GPU_metadata_queries=0, GPU_initialized=False,
    framework_imported=False, full_source_bytes_verified=False, runtime_assets_fully_verified=False,
    numerical_reference_verified=False, production_qualified=False, performance_claim=False,
    new_GPU_authorization_created=False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe(root, relative):
    require(type(relative) is str and relative and "\\" not in relative
        and not Path(relative).is_absolute() and all(p not in ("", ".", "..") for p in relative.split("/")),
        "canonical project-relative pin required")
    path = root / relative
    cursor = path
    while cursor != root:
        require(not cursor.is_symlink(), "symlink in source pin path")
        cursor = cursor.parent
    require(path.resolve().is_relative_to(root), "source pin outside project")
    return path


def read_verified(root, reference, profile):
    require(type(reference) is dict and set(reference) == {"path", "bytes", "sha256"}
        and type(reference["bytes"]) is int and 0 < reference["bytes"] <= 2 * 1024**2,
        "bounded exact CPU source pin")
    path = safe(root, reference["path"])
    require(path.is_file() and path.stat().st_size == reference["bytes"], "CPU pin size drift: " + reference["path"])
    raw = path.read_bytes()
    profile["small_pin_bytes_read"] += len(raw)
    require(len(raw) == reference["bytes"] and hashlib.sha256(raw).hexdigest() == reference["sha256"],
        "CPU pin SHA drift: " + reference["path"])
    return raw


def parse_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON: " + value)))


def driver_metadata(reference):
    """Check only canonical metadata, never load the driver or infer readiness."""
    require(type(reference) is dict and set(reference) == {"path", "bytes", "sha256"}, "exact driver pin")
    require(type(reference["path"]) is str and Path(reference["path"]).is_absolute()
        and type(reference["bytes"]) is int and reference["bytes"] > 0, "canonical driver metadata pin")
    path = Path(reference["path"])
    actual = dict(path=str(path), exists=path.exists(), bytes=None, canonical=False, regular=False)
    try:
        observed = path.lstat()
        actual.update(bytes=observed.st_size, canonical=path.resolve(strict=True) == path,
            regular=stat.S_ISREG(observed.st_mode), symlink=stat.S_ISLNK(observed.st_mode))
        ready = actual["canonical"] and actual["regular"] and not actual["symlink"] and actual["bytes"] == reference["bytes"]
    except OSError as exc:
        actual["error"] = type(exc).__name__
        ready = False
    return dict(status="CONTINUE_ORIGINAL_FULL_PREFLIGHT_REQUIRED" if ready else "BLOCKED_DRIVER_ASSET_GATE",
        reason=None if ready else "canonical frozen driver is missing, redirected, nonregular or wrong size",
        driver_expected=dict(reference), driver_observed=actual, driver_SHA_verified=False, **FALSE_FLAGS)


def resource_snapshot():
    out = dict(origin="read_only_cgroup", cpu_quota_cores=None, memory_limit_bytes=None)
    cpu = Path("/sys/fs/cgroup/cpu.max")
    memory = Path("/sys/fs/cgroup/memory.max")
    try:
        if cpu.is_file():
            quota, period = cpu.read_text().strip().split()
            if quota != "max" and int(period) > 0:
                out["cpu_quota_cores"] = int(quota) / int(period)
        if memory.is_file():
            limit = memory.read_text().strip()
            if limit != "max":
                out["memory_limit_bytes"] = int(limit)
    except (OSError, ValueError):
        out["unavailable"] = True
    return out


def early_driver_gate(root):
    profile = dict(small_pin_bytes_read=0, model_bytes_read_by_early_gate=0, full_source_checks_called=0)
    started = time.monotonic()
    manifest = parse_json(read_verified(root, LOCK_REF, profile))
    rows = manifest.get("files")
    require(type(rows) is list and len(rows) == 4088, "unchanged complete reference lock declaration")
    refs = {}
    for row in rows:
        require(type(row) is dict and set(row) == {"path", "bytes", "sha256"} and row["path"] not in refs,
            "unique frozen reference rows")
        refs[row["path"]] = row
    require(manifest.get("allow_gpu_runs") is False and manifest.get("allow_gpu_initialization") is False,
        "source candidate is not authority")
    for row in (ENTRY_REF, INVENTORY_REF, PROOF_REF, SDK_REF):
        require(refs.get(row["path"]) == row, "exact original entry/SDK pins must remain in frozen lock")
    read_verified(root, SDK_REF, profile)  # Verify bytes; do not execute its tree reader yet.
    inventory = parse_json(read_verified(root, INVENTORY_REF, profile))
    proof = parse_json(read_verified(root, PROOF_REF, profile))
    require(inventory.get("status") == "CPU_ONLY_EXISTING_CUDA13_SOURCE_INVENTORY"
        and inventory.get("driver") == DRIVER_REF and proof.get("driver_ref") == DRIVER_REF
        and proof.get("status") == "PASS_CPU_CUDA13_SM120F_COMPILE_AND_HOST_LINK_ONLY",
        "original CPU inventory/proof/exact driver binding")
    result = driver_metadata(DRIVER_REF)
    result.update(original_source_lock_ref=dict(LOCK_REF), profile=profile,
        early_elapsed_seconds=time.monotonic() - started, resources=resource_snapshot())
    return result


def original_full_preflight(root, profile):
    raw = read_verified(root, ENTRY_REF, profile)
    path = safe(root, ENTRY_REF["path"])
    name = "_fast_reference_original_cpu_entry"
    require(name not in sys.modules, "fresh CPU wrapper module required")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
        stdout = io.StringIO()
        profile["full_source_checks_called"] += 1
        with contextlib.redirect_stdout(stdout):
            code = module.main(["--project", str(root), "--preflight"])
        data = parse_json(stdout.getvalue())
        require(type(code) is int and type(data) is dict, "original CPU preflight result required")
        if code == 0:
            require(data.get("status") == "CPU_READY_NO_NEW_REFERENCE_HUMAN_SCOPE"
                and data.get("source_lock") == LOCK_REF and data.get("source_count") == 4088
                and data.get("actual_gpu_runs") == 0 and data.get("GPU_initialized") is False,
                "only original complete preflight can establish CPU readiness")
        return code, data
    finally:
        sys.modules.pop(name, None)


def diagnose(root, quick_only=False):
    started = time.monotonic()
    result = early_driver_gate(root)
    if result["status"] == "BLOCKED_DRIVER_ASSET_GATE":
        result["total_elapsed_seconds"] = time.monotonic() - started
        return 78, result
    if quick_only:
        result["total_elapsed_seconds"] = time.monotonic() - started
        return 0, result  # Not PASS/ready; source/assets remain explicitly unverified.
    full_started = time.monotonic()
    code, original = original_full_preflight(root, result["profile"])
    result.update(status=original["status"], original_full_preflight=original,
        full_elapsed_seconds=time.monotonic() - full_started,
        total_elapsed_seconds=time.monotonic() - started)
    if code == 0:
        result["full_source_bytes_verified"] = True
        result["runtime_assets_fully_verified"] = True
        result["driver_SHA_verified"] = True
    return code, result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path(ROOT))
    p.add_argument("--quick-only", action="store_true",
        help="metadata diagnosis only; a match requires original full preflight")
    args = p.parse_args(argv)
    started = time.monotonic()
    try:
        code, result = diagnose(args.project.resolve(), args.quick_only)
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        code, result = 78, dict(status="BLOCKED_FAST_CPU_PIN_OR_PATH_GATE", reason=str(exc),
            total_elapsed_seconds=time.monotonic() - started, **FALSE_FLAGS)
    print(json.dumps(result, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
