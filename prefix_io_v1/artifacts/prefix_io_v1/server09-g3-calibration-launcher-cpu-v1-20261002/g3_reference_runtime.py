"""Thin, source-bound reuse of the original guarded reference acquisition.

Import/preflight are stdlib CPU work. The parent must validate a NEW human scope,
full source lock and active original guard reservation before calling execute.
No model executor, cache, transfer, timing or numerical implementation is added.
"""

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


DELIVERY = "artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002"
SOURCE = DELIVERY + "/g3_reference_runtime.py"
DELEGATE = DELIVERY + "/g3_calibration_runtime_metrics_v2.py"
DELEGATE_REF = dict(path=DELEGATE, bytes=29978,
    sha256="0c04287325fd84c31aaea35dc6ff7431229511a04f3ab71bf9aa96a2ce83f1cc")
PURPOSE = "CURRENT_CONTEXT_CACHED_NUMERICAL_REFERENCE_ONLY"
GPU_UUID = "GPU-54ace1eb-dc9d-3fee-d720-2051f5d337a2"
MODES = ("cold", "paired")
JOB_NAMES = {"cold": "server09-g3-reference-native-01", "paired": "server09-g3-reference-paired-01"}
STORAGE = {
    "cold": "experiments/prefix_io_v1/runs/server09-g3-reference-native-01-unused-storage",
    "paired": "experiments/prefix_io_v1/runs/server09-g3-calibration-02-private-storage",
}
RECEIPT_FILE = "reference-runtime-result.json"
LEGACY_RECEIPT_FILE = "calibration-runtime-result.json"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _source(root, refs, relative, expected=None):
    require(type(refs) is dict and type(relative) is str and relative in refs, "reference source ref required")
    row = refs[relative]
    require(type(row) is dict and set(row) == {"path", "bytes", "sha256"} and row["path"] == relative,
            "exact reference source ref shape")
    require(type(row["bytes"]) is int and 0 < row["bytes"] <= 4*1024**2
            and type(row["sha256"]) is str and len(row["sha256"]) == 64,
            "bounded reference source ref")
    if expected is not None:
        require(row == expected, "frozen delegate pin changed")
    path = root / relative
    cursor = path
    while cursor != root:
        require(not cursor.is_symlink(), "reference source path symlink")
        cursor = cursor.parent
    require(path.resolve().is_relative_to(root) and path.is_file() and path.stat().st_size == row["bytes"],
            "reference source path/bytes drift")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == row["sha256"], "reference source SHA drift")
    return path, raw


def _load_delegate(root, refs):
    own, _ = _source(root, refs, SOURCE)
    require(Path(__file__).resolve() == own.resolve(), "reference runtime must be loaded from its locked project source")
    path, raw = _source(root, refs, DELEGATE, DELEGATE_REF)
    name = "_g3_reference_delegate_" + hashlib.sha256(str(path).encode()).hexdigest()[:20]
    require(name not in sys.modules, "fresh private reference delegate required")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _unload(delegate):
    if sys.modules.get(delegate.__name__) is delegate:
        sys.modules.pop(delegate.__name__)
        return True
    return False


def validate_guard(witness, mode):
    """Witness is a parent receipt, never an independent authorization API."""
    require(type(mode) is str and mode in MODES and type(witness) is dict, "fixed reference mode/guard witness")
    keys = {"schema_version", "purpose", "gpu_uuid", "label", "source_lock_sha256", "scope_sha256",
            "permissions_ref", "session_id", "guard_command", "seconds_limit", "reserved_seconds",
            "authorized_scope_verified", "active_reservation_verified"}
    require(set(witness) == keys and type(witness["schema_version"]) is int and witness["schema_version"] == 1,
            "exact new reference guard witness")
    require(all(type(witness[k]) is str for k in ("purpose", "gpu_uuid", "label"))
            and witness["purpose"] == PURPOSE and witness["gpu_uuid"] == GPU_UUID
            and witness["label"] == JOB_NAMES[mode], "new reference identity; old calibration scope cannot be reused")
    require(witness["authorized_scope_verified"] is True and witness["active_reservation_verified"] is True,
            "parent must verify new human scope and active original reservation")
    require(type(witness["seconds_limit"]) is int and witness["seconds_limit"] == 300
            and type(witness["reserved_seconds"]) is int and witness["reserved_seconds"] == 320,
            "fixed reference guard limit/reserve")
    for key in ("source_lock_sha256", "scope_sha256"):
        value = witness[key]
        require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "exact reference evidence SHA")
    ref = witness["permissions_ref"]
    require(type(ref) is dict and set(ref) == {"path", "bytes", "sha256"}
            and type(ref["path"]) is str and type(ref["bytes"]) is int
            and type(ref["sha256"]) is str, "parent effective permission ref")
    command = witness["guard_command"]
    require(type(command) is list and command and all(type(v) is str and len(v) <= 4096 for v in command),
            "actual reference child command")
    require(os.name == "posix" and type(witness["session_id"]) is int
            and witness["session_id"] == os.getsid(0) and os.environ.get("CUDA_VISIBLE_DEVICES") == GPU_UUID,
            "actual guarded reference child SID/UUID")


def fixed_argv(mode, output, storage, model, model_plan):
    require(type(mode) is str and mode in MODES, "fixed numerical reference mode only")
    flags = ["--native-hot-diagnostic", "--diagnostic-logprobs"] if mode == "cold" else ["--cached-reference-logprobs"]
    return ["acquire_native_aio_costs.py", "--output-dir", str(output), "--storage", str(storage),
            "--model-dir", str(model), "--model-plan", str(model_plan), "--mode", mode,
            "--domain", "1024", "--sizes", "128", "--reps", "3", "--max-num-seqs", "1", "--iodepth", "4", *flags]


def _paths(root, mode, out, storage):
    require(type(mode) is str and mode in MODES, "fixed reference path mode")
    out, storage = Path(out).absolute(), Path(storage).absolute()
    expected_out = root / "experiments/prefix_io_v1/runs" / JOB_NAMES[mode] / "details"
    require(out == expected_out and storage == root / STORAGE[mode], "exact frozen reference output/storage paths")
    for path in (out, storage):
        cursor = path
        while cursor != root:
            require(not cursor.is_symlink(), "reference evidence/storage symlink")
            cursor = cursor.parent
        require(path.resolve().is_relative_to(root), "reference path outside root")
    require(out.is_dir(), "parent must create new approved details directory")
    require(not (out / "acquisition").exists() and not (out / RECEIPT_FILE).exists()
            and not (out / LEGACY_RECEIPT_FILE).exists(), "append-new reference acquisition/evidence only")
    require(not storage.exists() if mode == "cold" else storage.is_dir(), "cold unused storage / paired published storage required")
    return out, storage


def _json_ref(path, root):
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 4*1024**2,
            "bounded original JSON evidence required")
    raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path.relative_to(root)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def reference_rows(report, receipt, mode):
    """Align all real original rows to real tap IDs; no cross-run comparison."""
    require(type(report) is dict and report.get("mode") == mode and
            report.get("status") == "PASSED_NATIVE_" + mode.upper() + "_ACQUISITION", "original reference acquisition failed")
    require(report.get("native_hot_diagnostic") is (mode == "cold")
            and report.get("cached_reference_logprobs") is (mode == "paired"), "exact original diagnostic flags required")
    rows, requests = report.get("rows"), receipt.get("requests")
    require(type(rows) is list and type(requests) is list and len(rows) == len(requests) == 6,
            "complete six original diagnostic rows and output taps required")
    kinds = ("f", "gpu_hot") if mode == "cold" else ("g_ssd", "g_mem")
    copied = []
    for ordinal, (row, tap) in enumerate(zip(rows, requests)):
        require(type(row) is dict and type(tap) is dict and row.get("kind") == kinds[ordinal % 2]
                and type(row.get("rep")) is int and row["rep"] == ordinal // 2
                and row.get("prefix_tokens") == 128 and row.get("warmup") is (ordinal // 2 == 0),
                "original reference order/rep/axis")
        require(type(tap.get("ordinal")) is int and tap["ordinal"] == ordinal and tap.get("sentinel") is False,
                "continuous actual diagnostic output tap")
        for key in ("request_id", "prompt_token_ids", "output_token_ids", "num_cached_tokens"):
            require(row.get(key) == tap.get(key), "original diagnostic row/output tap mismatch: " + key)
        require(type(row["prompt_token_ids"]) is list and len(row["prompt_token_ids"]) == 129
                and type(row["output_token_ids"]) is list and len(row["output_token_ids"]) == 1,
                "complete original prompt and one sampled output required")
        require(type(row["num_cached_tokens"]) is int and row["num_cached_tokens"] == (0 if row["kind"] == "f" else 128),
                "original cached reference shape")
        values = row.get("output_logprobs")
        require(type(values) is list and len(values) == 1 and type(values[0]) is dict and 5 <= len(values[0]) <= 6
                and str(row["output_token_ids"][0]) in values[0], "complete original top5 and selected token required")
        for token, item in values[0].items():
            require(type(token) is str and token.isdecimal() and type(item) is dict and set(item) == {"logprob", "rank"},
                    "original logprob record shape")
            require(type(item["logprob"]) in (int, float) and math.isfinite(item["logprob"])
                    and type(item["rank"]) is int and item["rank"] > 0, "finite original diagnostic values")
        copied.append({key: row[key] for key in ("kind", "prefix_tokens", "rep", "warmup", "request_id",
                      "prompt_token_ids", "output_token_ids", "num_cached_tokens", "output_logprobs")})
    return copied


def preflight_runtime(root, refs):
    root = Path(root).resolve(strict=True)
    delegate = _load_delegate(root, refs)
    try:
        result = delegate.preflight_runtime(root, refs)
        require(type(result) is dict, "original CPU runtime preflight receipt")
        return dict(result, status="CPU_REFERENCE_RUNTIME_SOURCE_ASSETS_ONLY", purpose=PURPOSE,
                    delegate_runtime_ref=DELEGATE_REF, new_GPU_authorization=False,
                    numerical_reference_verified=False, cost_qualified=False, production_qualified=False)
    finally:
        _unload(delegate)


def execute_original_acquisition(root, mode, out, storage, refs, guard_witness):
    validate_guard(guard_witness, mode)  # Before source/assets/backend/filesystem work.
    root = Path(root).resolve(strict=True)
    out, storage = _paths(root, mode, out, storage)
    delegate = _load_delegate(root, refs)
    saved = {key: getattr(delegate, key) for key in ("PURPOSE", "MODES", "validate_guard", "fixed_argv")}
    restored = False
    try:
        # Only this fresh private module is changed; no frozen source or author
        # module is edited. Reused common runtime still owns all model/IO work.
        delegate.PURPOSE, delegate.MODES = PURPOSE, MODES
        delegate.validate_guard, delegate.fixed_argv = validate_guard, fixed_argv
        outcome = delegate.execute_original_acquisition(root, mode, out, storage, refs, guard_witness)
    finally:
        for key, value in saved.items():
            setattr(delegate, key, value)
        restored = all(getattr(delegate, key) is value for key, value in saved.items()) and _unload(delegate)
    require(type(outcome) is dict and set(outcome) == {"original_exit_code", "runtime_receipt"}
            and type(outcome["original_exit_code"]) is int, "original acquisition outcome contract")
    old, old_ref = _json_ref(out / LEGACY_RECEIPT_FILE, root)
    require(old == delegate.scalar_copy(outcome["runtime_receipt"]), "original delegate receipt changed or lost fields")
    receipt = dict(old)
    receipt.update(purpose=PURPOSE, raw_calibration_only=False, numerical_reference_only=True,
                   diagnostic_logprobs=5, diagnostic_latency_excluded_from_fit=True,
                   original_delegate_receipt_ref=old_ref, delegate_runtime_ref=DELEGATE_REF,
                   reference_delegate_overrides_restored=restored, expected_original_request_count=6,
                   numerical_reference_verified=False, GPU_qualified=False,
                   full_vocabulary_verified=False, KV_byte_consistency_verified=False,
                   cost_qualified=False, production_qualified=False, performance_claim=False)
    receipt["reference_diagnostic_rows"] = None
    receipt["original_report_ref"] = None
    receipt["reference_metadata_fields"] = [key for key in receipt if key not in old] + ["raw_calibration_only", "reference_metadata_fields"]
    if outcome["original_exit_code"] == 0:
        report, report_ref = _json_ref(out / "acquisition/result.json", root)
        receipt["reference_diagnostic_rows"] = reference_rows(report, receipt, mode)
        receipt["original_report_ref"] = report_ref
        config = receipt.get("frozen_config")
        require(type(config) is dict and config.get("sampling", {}).get("logprobs") == 5,
                "actual original diagnostic sampling required")
    # The original full report/logs/IDs/logprobs and original receipt remain.
    receipt = delegate.scalar_copy(receipt)
    with (out / RECEIPT_FILE).open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return dict(original_exit_code=outcome["original_exit_code"], runtime_receipt=receipt)
