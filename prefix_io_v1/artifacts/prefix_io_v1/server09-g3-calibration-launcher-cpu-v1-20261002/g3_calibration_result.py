"""Finite original-acquirer result gates; no framework, GPU or curve export.

The checks can gate cold -> populate -> fresh paired within an externally
authorized pilot. They do not authenticate arbitrary JSON as physical GPU/owner
evidence, and never grant GPU, production, release, clock mapping or P4 effects.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

DOMAIN = 1024
PREFIX_TOKENS = 128
REPS = 3
BLOCK_TOKENS = 16
BYTES_PER_TOKEN = 57344
FILE_BYTES = 917504
FILES = 24
READ_BYTES = PREFIX_TOKENS * BYTES_PER_TOKEN
MODEL_ALIAS = "Qwen/Qwen2.5-7B-Instruct"
STORAGE_PREFIX = (MODEL_ALIAS + "/block_size_16_blocks_per_file_1/"
                  "tp_1_pp_size_1_pcp_size_1/rank_0/auto/")
TIMING = "original metrics.first_token_latency seconds; author frontend wall-clock TTFT"
UNQUALIFIED = {
    "GPU_qualified": False, "production_qualified": False, "effect_verified": False,
    "physical_release_verified": False, "KV_byte_consistency_verified": False,
    "GPU_clock_mapping_verified": False, "runtime_curve_exported": False,
    "P4_performance_verified": False,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, name, minimum=0, maximum=10**12):
    require(type(value) is int and minimum <= value <= maximum, "exact integer: " + name)
    return value


def text(value, name, maximum=4096):
    require(type(value) is str and 0 < len(value) <= maximum, "exact text: " + name)
    return value


def scalar_time(value, name):
    require(type(value) in (float, int) and math.isfinite(value) and value > 0,
            "positive finite original time: " + name)
    return value


def sha(value, name):
    require(type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None,
            "exact SHA256: " + name)
    return value


def expected_prompt(rep):
    integer(rep, "rep", 0, REPS - 1)
    return [2000 + PREFIX_TOKENS + rep * 1500] + [1000 + i % 500 for i in range(PREFIX_TOKENS - 1)] + [777]


def _tokens(values, length, name):
    require(type(values) is list and len(values) == length, "exact token length: " + name)
    for value in values:
        integer(value, name, 0, 1000000)


def validate_frozen_config(mode, config):
    require(type(mode) is str and mode in ("cold", "populate", "paired"), "fixed acquisition mode")
    require(type(config) is dict, "original frozen-config required")
    require(config.get("sizes") == [PREFIX_TOKENS] and type(config.get("sizes")) is list
            and all(type(v) is int for v in config["sizes"])
            and type(config.get("reps")) is int and config["reps"] == REPS
            and type(config.get("acquisition_domain")) is int and config["acquisition_domain"] == DOMAIN,
            "fixed domain1024 size128 reps3")
    require(config.get("model_alias") == MODEL_ALIAS and type(config.get("model_alias")) is str
            and type(config.get("model")) is dict and config["model"].get("model_id") == MODEL_ALIAS,
            "original model identity/alias")
    text(config.get("alias_target"), "alias_target")
    text(config.get("gpu_uuid"), "GPU UUID", 128)
    require(config["gpu_uuid"].startswith("GPU-"), "original UUID")
    require(config.get("pythonhashseed") == "0" and type(config.get("pythonhashseed")) is str
            and config.get("planned_request_batch") == 1 and type(config.get("planned_request_batch")) is int
            and config.get("cost_acquisition_only") is True and config.get("prompt_manifest") is None
            and config.get("native_hot_diagnostic") is False
            and config.get("cached_reference_logprobs") is False, "raw finite acquisition; no diagnostic/planned path")
    sampling = config.get("sampling")
    require(type(sampling) is dict and type(sampling.get("temperature")) is float
            and sampling["temperature"] == 0.0 and type(sampling.get("seed")) is int
            and sampling["seed"] == 0 and sampling.get("ignore_eos") is True
            and sampling.get("detokenize") is False and sampling.get("logprobs") is None,
            "original deterministic one-token sampling")
    for key in ("max_tokens", "min_tokens"):
        require(type(sampling.get(key)) is int and sampling[key] == 1, "one output token")
    engine = config.get("engine")
    require(type(engine) is dict, "original engine config")
    fixed = {"max_model_len": 1040, "max_num_batched_tokens": 1040, "max_num_seqs": 1,
             "block_size": 16, "kv_cache_memory_bytes": 268435456, "tensor_parallel_size": 1,
             "compilation_config": 0, "seed": 0}
    for key, expected in fixed.items():
        require(type(engine.get(key)) is int and engine[key] == expected, "fixed engine: " + key)
    require(engine.get("dtype") == "bfloat16" and engine.get("kv_cache_dtype") == "auto"
            and engine.get("quantization") is None and engine.get("enable_prefix_caching") is True
            and engine.get("enable_chunked_prefill") is True and engine.get("enforce_eager") is True
            and engine.get("async_scheduling") is False and engine.get("prefix_caching_hash_algo") == "sha256"
            and engine.get("distributed_executor_backend") == "uni"
            and engine.get("attention_config") == {"backend": "TRITON_ATTN"}, "fixed original engine path")
    connector = engine.get("kv_transfer_config")
    if mode == "cold":
        require(connector is None, "cold must have no external connector")
    else:
        require(type(connector) is dict and connector.get("kv_connector") == "OffloadingConnector"
                and connector.get("kv_role") == "kv_both", "original native connector")
        extra = connector.get("kv_connector_extra_config")
        require(type(extra) is dict, "original connector extras")
        values = {"spec_name": "PyKvCacheOffloadingSpec", "spec_module_path": "py_kvcache.vllm",
                  "block_size": 16, "iodepth": 4, "preload_lookahead_requests": 1,
                  "staging_cache": "lru", "load_planner": "off", "io_backend": "linux_aio"}
        for key, expected in values.items():
            require(type(extra.get(key)) is type(expected) and extra[key] == expected, "fixed connector: " + key)
        require(type(extra.get("staging_mem")) is float and extra["staging_mem"] == 0.125
                and extra.get("sync_on_store") is False and extra.get("enable_preload") is True
                and extra.get("preload_share_staging") is True
                and extra.get("prefix_cache_break_even_path") is None,
                "128MiB shared staging and raw load path without curves")
        text(extra.get("shared_storage_path"), "shared storage")
    return engine


def _drain(drain, external, profile=False):
    require(type(drain) is dict, "original drain snapshot")
    if profile:
        require(drain.get("profile_flushed") is True, "original profiler flushed")
    handlers = drain.get("handlers", [])
    require(type(handlers) is list and len(handlers) == (1 if external else 0), "actual native handler count")
    if external:
        integer(drain.get("queued_stores_submitted"), "queued stores")
        waited = drain.get("waited_job_ids")
        require(type(waited) is list and len(waited) <= 128, "bounded waited job IDs")
        for job_id in waited:
            integer(job_id, "waited job ID")
        for handler in handlers:
            require(type(handler) is dict and handler.get("pinned") is True, "original pinned staging")
            actual = integer(handler.get("staging_bytes"), "actual staging bytes", 1)
            budget = integer(handler.get("staging_budget"), "staging budget", 1)
            require(actual <= budget == 134217728, "128MiB staging actual/budget")
            require(type(handler.get("io_size")) is int and type(handler.get("storage_block_bytes")) is int
                    and handler["io_size"] == handler["storage_block_bytes"] == FILE_BYTES, "actual canonical storage geometry")
            aio = handler.get("aio")
            require(type(aio) is dict, "actual native AIO snapshot")
            for key in ("accepted", "completed", "reaped", "outstanding", "pending", "ready", "unreaped"):
                integer(aio.get(key), "AIO " + key)
            require(aio["accepted"] == aio["completed"] == aio["reaped"]
                    and all(aio[k] == 0 for k in ("outstanding", "pending", "ready", "unreaped"))
                    and aio.get("fatal") is None, "original pre-shutdown AIO unsettled")
            # closed/drained may be false here: this is the original pre-shutdown snapshot.
    return handlers


def _trace(row, kind):
    trace = row.get("trace")
    require(type(trace) is dict, "original trace summary")
    internal = text(trace.get("internal_request_id"), "internal request ID", 128)
    request_id = text(row.get("request_id"), "frontend request ID", 128)
    require(internal == request_id or re.fullmatch(re.escape(request_id) + "-[0-9a-f]{8}", internal) is not None,
            "original frontend/internal trace request association")
    integer(trace.get("event_count"), "event count", 1, 1000000)
    transfers = trace.get("load_transfers")
    require(type(transfers) is list and len(transfers) <= 16, "bounded original load transfers")
    require(trace.get("transfer_success") is True, "successful original transfer")
    for key in ("foreground_logical_read_bytes", "preload_actual_read_bytes", "src_cache", "src_file", "src_preload", "h2d_events"):
        integer(trace.get(key), "trace " + key)
    planners = trace.get("planner_events")
    require(type(planners) is list and not planners, "raw baseline must not invoke planner")
    read_bytes = trace["foreground_logical_read_bytes"] + trace["preload_actual_read_bytes"]
    if kind in ("f", "store"):
        require(not transfers and read_bytes == 0 and trace["h2d_events"] == 0
                and trace["src_cache"] == trace["src_file"] == trace["src_preload"] == 0,
                "cold/store must not restore external prefix")
    else:
        require(transfers and trace["h2d_events"] > 0, "native load/H2D absent")
        sums = {key: 0 for key in ("num_bytes", "src_cache", "src_file", "src_preload")}
        for transfer in transfers:
            require(type(transfer) is dict and transfer.get("req_id") == internal
                    and type(transfer.get("req_id")) is str
                    and transfer.get("direction") == "storage_to_gpu"
                    and type(transfer.get("direction")) is str and transfer.get("success") is True,
                    "native transfer request/direction/success")
            for key in sums:
                sums[key] += integer(transfer.get(key), "transfer " + key)
        require(sums["num_bytes"] == READ_BYTES, "complete original load byte coverage")
        require(all(sums[key] == trace[key] for key in ("src_cache", "src_file", "src_preload"))
                and sum(sums[key] for key in ("src_cache", "src_file", "src_preload")) == PREFIX_TOKENS // BLOCK_TOKENS,
                "native load source coverage")
        if kind == "g_ssd":
            require(read_bytes == READ_BYTES and trace["src_cache"] == 0,
                    "fresh paired SSD foreground plus preload coverage")
        else:
            require(read_bytes == 0 and trace["src_cache"] == PREFIX_TOKENS // BLOCK_TOKENS,
                    "staging cache H2D must read zero SSD bytes")
    return read_bytes


def verify_acquisition_report(mode, report, config):
    """Validate exact original report fields, never invent missing sentinel rows."""
    validate_frozen_config(mode, config)
    require(type(report) is dict and type(report.get("mode")) is str and report["mode"] == mode
            and report.get("status") == "PASSED_NATIVE_" + mode.upper() + "_ACQUISITION",
            "failed/wrong original acquisition")
    require(report.get("performance_claim") is False and report.get("full_P1_pass") is False
            and report.get("native_hot_diagnostic") is False
            and report.get("cached_reference_logprobs") is False and report.get("model_loaded") is True,
            "original raw calibration report")
    require(report.get("engine_shutdown") == "completed" and type(report.get("engine_shutdown")) is str
            and report.get("error") is None and report.get("shutdown_error") is None
            and report.get("storage_budget_error") is None, "original shutdown/result failure")
    kinds = {"cold": ("f",), "populate": ("store", "g_mem"), "paired": ("g_ssd", "g_mem")}[mode]
    rows = report.get("rows")
    require(type(rows) is list and len(rows) == REPS * len(kinds), "exact original reported row count")
    samples = {kind: [] for kind in kinds}
    outputs, requests, ssd_bytes = [], set(), 0
    for index, row in enumerate(rows):
        require(type(row) is dict, "original row")
        rep, kind = index // len(kinds), kinds[index % len(kinds)]
        require(type(row.get("kind")) is str and row["kind"] == kind
                and type(row.get("rep")) is int and row["rep"] == rep
                and type(row.get("prefix_tokens")) is int and row["prefix_tokens"] == PREFIX_TOKENS
                and type(row.get("warmup")) is bool and row["warmup"] is (rep == 0),
                "original row order/axis/warmup")
        _tokens(row.get("prompt_token_ids"), PREFIX_TOKENS + 1, "prompt")
        _tokens(row.get("output_token_ids"), 1, "output")
        require(row["prompt_token_ids"] == expected_prompt(rep), "original deterministic prompt identity")
        expected_cached = 0 if kind in ("f", "store") else PREFIX_TOKENS
        require(type(row.get("num_cached_tokens")) is int and row["num_cached_tokens"] == expected_cached,
                "cold/store zero or complete external cache coverage")
        request_id = text(row.get("request_id"), "request ID", 128)
        require(request_id not in requests, "duplicate reported request")
        requests.add(request_id)
        metrics = row.get("metrics")
        require(type(metrics) is dict and metrics.get("is_corrupted") is False
                and type(metrics.get("num_generation_tokens")) is int and metrics["num_generation_tokens"] == 1,
                "original metric result integrity")
        ttft = scalar_time(metrics.get("first_token_latency"), "TTFT seconds")
        scalar_time(metrics.get("arrival_time"), "frontend wall-clock arrival")
        monotonic = [scalar_time(metrics.get(k), "engine-core monotonic " + k)
                     for k in ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")]
        require(monotonic == sorted(monotonic), "engine-core monotonic order")
        scalar_time(row.get("wall_seconds"), "generate wall diagnostic")
        # Never subtract wall-clock arrival_time from monotonic first_token_ts.
        if rep != 0:
            samples[kind].append(ttft)
        _drain(row.get("drain"), mode != "cold", profile=True)
        ssd_bytes += _trace(row, kind)
        outputs.append(tuple(row["output_token_ids"]))
        if kind == "g_mem" and mode == "paired":
            require(row["output_token_ids"] == rows[index - 1]["output_token_ids"], "cached pair output mismatch")
    _drain(report.get("final_drain"), mode != "cold")
    require(all(len(values) == 2 for values in samples.values()), "two non-warmup points per path")
    return {"status": "PASS_ORIGINAL_REPORT_STRUCTURE_AND_PATH_FIELDS_ONLY", "mode": mode,
            "reported_rows": len(rows), "measured_samples_seconds": samples,
            "warmup_reps": [0], "ssd_read_bytes": ssd_bytes, "timing": TIMING,
            "generate_wall_seconds_role": "diagnostic perf_counter only, excluded from TTFT",
            "unreported_original_sentinels": 3 if mode == "populate" else 0,
            "native_owner_physical_evidence": "UNKNOWN", **UNQUALIFIED}


def validate_job_result(mode, report, runtime_receipt, guard_event, binding, *, config=None):
    """Gate a completed bounded path job using parent-held expected identities.

    guard_event is the actual original guard result/final ledger event schema,
    not a newly invented GPU receipt. The parent must source it from that guard.
    Passing fixture-shaped JSON tests this contract; it grants no GPU authority.
    """
    require(type(runtime_receipt) is dict and type(guard_event) is dict and type(binding) is dict,
            "actual runtime/guard/binding required; missing evidence UNKNOWN")
    if config is None:
        config = runtime_receipt.get("frozen_config")
    analysis = verify_acquisition_report(mode, report, config)
    for key in ("expected_source_lock_sha256", "expected_scope_sha256"):
        sha(binding.get(key), key)
    for key in ("expected_label", "expected_gpu_uuid", "expected_storage"):
        text(binding.get(key), key)
    command = binding.get("expected_command")
    require(type(command) is list and 1 <= len(command) <= 64
            and all(type(v) is str and 0 < len(v) <= 4096 for v in command), "exact frozen child argv")
    require(type(runtime_receipt.get("mode")) is str and runtime_receipt["mode"] == mode
            and runtime_receipt.get("source_lock_sha256") == binding["expected_source_lock_sha256"]
            and type(runtime_receipt.get("source_lock_sha256")) is str
            and runtime_receipt.get("scope_sha256") == binding["expected_scope_sha256"]
            and type(runtime_receipt.get("scope_sha256")) is str,
            "runtime source/scope/mode identity")
    require(runtime_receipt.get("original_shutdown_completed") is True
            and runtime_receipt.get("source_unchanged") is True
            and runtime_receipt.get("common_helpers_restored") is True, "original shutdown/source/helper restore closure")
    require(type(runtime_receipt.get("original_exit_code")) is int and runtime_receipt["original_exit_code"] == 0,
            "original acquirer exit zero")
    require(type(runtime_receipt.get("engine_shutdown_calls")) is int
            and runtime_receipt["engine_shutdown_calls"] == 1
            and type(runtime_receipt.get("handler_shutdown_calls")) is int
            and runtime_receipt["handler_shutdown_calls"] == (0 if mode == "cold" else 1),
            "original shutdown methods once")
    for key in ("exit", "child_exit"):
        require(type(guard_event.get(key)) is int and guard_event[key] == 0, "original guard/child exit zero")
    require(guard_event.get("gpu_job_attempted") is True and guard_event.get("timed_out") is False
            and guard_event.get("session_drained") is True and guard_event.get("interrupted_signal") is None
            and guard_event.get("error") is None and guard_event.get("session_members_after_cleanup") == []
            and type(guard_event.get("session_members_after_cleanup")) is list
            and guard_event.get("session_members_before_cleanup") == []
            and type(guard_event.get("session_members_before_cleanup")) is list,
            "original guard OS session drain")
    require(type(guard_event.get("command")) is list and guard_event["command"] == command
            and all(type(v) is str for v in guard_event["command"])
            and guard_event.get("label") == binding["expected_label"] and type(guard_event.get("label")) is str
            and guard_event.get("gpu_uuid") == binding["expected_gpu_uuid"] and type(guard_event.get("gpu_uuid")) is str,
            "original guard argv/label/GPU binding")
    scalar_time(guard_event.get("elapsed_seconds"), "actual guard billed elapsed seconds")
    sid = integer(guard_event.get("session_id"), "original guard session ID", 1)
    identity = runtime_receipt.get("process_identity")
    require(type(identity) is dict, "actual fresh process identity UNKNOWN")
    pid = integer(identity.get("pid"), "actual child PID", 1)
    require(type(identity.get("sid")) is int and identity["sid"] == sid, "child/original guard session identity")
    prior = binding.get("prior_process_identities", [])
    require(type(prior) is list and len(prior) <= 2, "bounded prior process identities")
    if mode == "paired":
        require(len(prior) >= 1, "fresh paired process evidence UNKNOWN")
    for old in prior:
        require(type(old) is dict, "prior process identity")
        old_pid = integer(old.get("pid"), "prior PID", 1)
        old_sid = integer(old.get("sid"), "prior SID", 1)
        require(pid != old_pid and sid != old_sid, "paired/acquisition process must be fresh")
    require(config["gpu_uuid"] == binding["expected_gpu_uuid"], "config GPU binding")
    if mode != "cold":
        require(config["engine"]["kv_transfer_config"]["kv_connector_extra_config"]["shared_storage_path"]
                == binding["expected_storage"], "exact shared published storage target")
    if mode == "populate":
        sentinels = runtime_receipt.get("sentinel_observations")
        require(type(sentinels) is list and len(sentinels) == REPS,
                "three original unreported sentinel observations UNKNOWN")
        for rep, sentinel in enumerate(sentinels):
            require(type(sentinel) is dict and type(sentinel.get("rep")) is int and sentinel["rep"] == rep,
                    "original sentinel repetition")
            _tokens(sentinel.get("prompt_token_ids"), 1, "sentinel prompt")
            _tokens(sentinel.get("output_token_ids"), 1, "sentinel output")
            require(sentinel["prompt_token_ids"] == [30000 + rep]
                    and type(sentinel.get("ordinal")) is int and sentinel["ordinal"] == rep * 3 + 1
                    and sentinel.get("sentinel") is True
                    and type(sentinel.get("num_cached_tokens")) is int and sentinel["num_cached_tokens"] == 0,
                    "original store -> sentinel -> drain -> staging sequence")
            text(sentinel.get("request_id"), "original sentinel request ID", 128)
    requests = runtime_receipt.get("requests")
    count = {"cold": 3, "populate": 9, "paired": 6}[mode]
    require(type(requests) is list and len(requests) == count, "exact actual original generate requests")
    for ordinal, request in enumerate(requests):
        require(type(request) is dict and type(request.get("ordinal")) is int
                and request["ordinal"] == ordinal
                and set(request) == {"ordinal", "prompt_token_ids", "output_token_ids", "num_cached_tokens",
                                     "sentinel", "request_id"}, "actual original generate order/schema")
        is_sentinel = mode == "populate" and ordinal % 3 == 1
        require(request.get("sentinel") is is_sentinel, "original sentinel classification")
        _tokens(request.get("output_token_ids"), 1, "actual original output")
        if is_sentinel:
            observation = dict(runtime_receipt["sentinel_observations"][ordinal // 3])
            del observation["rep"]  # Only the derived side receipt has this validated metadata.
            require(request == observation, "same observed original sentinel")
        else:
            row_index = ordinal if mode != "populate" else ordinal - (ordinal + 1) // 3
            row = report["rows"][row_index]
            require(request.get("prompt_token_ids") == row["prompt_token_ids"]
                    and type(request.get("prompt_token_ids")) is list
                    and request["output_token_ids"] == row["output_token_ids"]
                    and request.get("request_id") == row["request_id"] and type(request.get("request_id")) is str
                    and type(request.get("num_cached_tokens")) is int
                    and request["num_cached_tokens"] == row["num_cached_tokens"], "runtime/original row identity")
    faults = runtime_receipt.get("observer_fault_types")
    require(type(faults) is list and len(faults) <= 8 and all(type(v) is str and 0 < len(v) <= 128 for v in faults),
            "bounded runtime observer diagnostics")
    return {**analysis, "status": "PASS_BOUNDED_ORIGINAL_ACQUISITION_PATH_CHECKS",
            "original_shutdown_and_OS_drain_checked": True,
            "guard_billed_elapsed_seconds": guard_event["elapsed_seconds"],
            "process_identity": {"pid": pid, "sid": sid},
            "scope_sha256": binding["expected_scope_sha256"],
            "source_lock_sha256": binding["expected_source_lock_sha256"],
            "next_same_authorized_scope_job_eligible": True,
            "observer_fault_types": list(faults),
            "native_owner_physical_evidence": "UNKNOWN; separate actual owner/backing capture required"}


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")


def _storage_paths(storage):
    root = Path(storage)
    require(root.is_absolute() and root.is_dir() and not root.is_symlink()
            and root.resolve() == root.absolute(), "actual canonical shared storage directory")
    paths, queue, directories = [], [root], 0
    while queue:
        directory = queue.pop()
        directories += 1
        require(directories <= 128, "bounded published storage directory count")
        with os.scandir(directory) as scan:
            entries = list(scan)
        require(len(entries) <= 128, "bounded storage directory entries")
        for entry in entries:
            require(not entry.is_symlink(), "published symlink rejected")
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                queue.append(path)
            else:
                require(entry.is_file(follow_symlinks=False), "published nonregular object")
                relative = path.relative_to(root).as_posix()
                require(relative.startswith(STORAGE_PREFIX), "original mapper/model/layout path")
                tail = relative[len(STORAGE_PREFIX):].split("/")
                require(len(tail) == 3 and re.fullmatch("[0-9a-f]{64}\\.bin", tail[2]) is not None,
                        "canonical SHA256 block hash filename")
                filename_hash = tail[2][:-4]
                require(tail[0] == filename_hash[:3] and tail[1] == filename_hash[3:5], "original mapper hash shards")
                require(path.stat().st_size == FILE_BYTES, "published canonical file size")
                paths.append((relative, path))
                require(len(paths) <= FILES, "bounded published file count")
    require(len(paths) == FILES, "exact24 published files")
    return root, sorted(paths)


def publish_storage_manifest(storage):
    """Read-only 24-file size/SHA publication; caller saves it after populate gate."""
    root, paths = _storage_paths(storage)
    rows = []
    for relative, path in paths:
        named_before = path.stat()
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            require(stat.S_ISREG(before.st_mode) and before.st_size == FILE_BYTES, "regular published descriptor")
            digest, count = hashlib.sha256(), 0
            while True:
                raw = handle.read(1024 * 1024)
                if not raw:
                    break
                count += len(raw)
                require(count <= FILE_BYTES, "bounded published payload read")
                digest.update(raw)
            after = os.fstat(handle.fileno())
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        portable_identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        named_after = path.stat()
        # Windows 3.12 path.stat ctime is birthtime while fstat ctime can be
        # last-write time. Each API must remain stable; compare common identity
        # fields across APIs instead of equating their distinct ctime semantics.
        require(count == FILE_BYTES and identity(before) == identity(after)
                and identity(named_before) == identity(named_after)
                and portable_identity(named_before) == portable_identity(before)
                and portable_identity(named_after) == portable_identity(after), "published file changed during snapshot")
        rows.append({"path": relative, "bytes": count, "sha256": digest.hexdigest()})
    _, final_paths = _storage_paths(root)
    require([p for p, _ in paths] == [p for p, _ in final_paths], "published inventory changed")
    return {"schema_version": 1, "status": "PASS_BOUNDED_PUBLICATION_BYTES_ONLY",
            "storage": str(root), "files": rows, "file_count": FILES,
            "total_bytes": FILES * FILE_BYTES, "rows_sha256": hashlib.sha256(_canonical_json(rows)).hexdigest(),
            "logical_equals_physical_file_bytes": True, **UNQUALIFIED}


def verify_storage_manifest(storage, manifest):
    """Re-read identical full target before a fresh paired launch; no publication edit."""
    require(type(manifest) is dict and manifest.get("schema_version") == 1
            and type(manifest.get("schema_version")) is int
            and manifest.get("status") == "PASS_BOUNDED_PUBLICATION_BYTES_ONLY"
            and type(manifest.get("storage")) is str, "actual prior publication manifest required")
    require(type(manifest.get("files")) is list and len(manifest["files"]) == FILES, "exact published manifest rows")
    previous = ""
    for row in manifest["files"]:
        require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "exact published row")
        name = text(row["path"], "published relative path")
        require(name > previous, "unique sorted published rows")
        previous = name
        require(type(row["bytes"]) is int and row["bytes"] == FILE_BYTES, "published row size")
        sha(row["sha256"], "published payload")
    require(type(manifest.get("file_count")) is int and manifest["file_count"] == FILES
            and type(manifest.get("total_bytes")) is int and manifest["total_bytes"] == FILES * FILE_BYTES
            and manifest.get("rows_sha256") == hashlib.sha256(_canonical_json(manifest["files"])).hexdigest()
            and manifest.get("logical_equals_physical_file_bytes") is True,
            "published manifest byte/hash integrity")
    for key in UNQUALIFIED:
        require(manifest.get(key) is False, "publication has no experimental qualification")
    actual = publish_storage_manifest(storage)
    require(actual["storage"] == manifest["storage"] and actual["files"] == manifest["files"]
            and actual["rows_sha256"] == manifest["rows_sha256"], "published target size/content changed")
    return actual


# Parent launcher names. Both are read-only; publication is performed only after
# its separately verified original guard shutdown/drain, never inside the child.
def build_storage_publication(storage):
    return publish_storage_manifest(storage)


def verify_storage_publication(storage, publication):
    return verify_storage_manifest(storage, publication)


def verify_point_pilot(cold, populate, paired, *, configs, job_checks):
    """Three actual gates and original reports only; not a curve or P4 comparison."""
    require(type(configs) is dict and type(job_checks) is dict, "three config/job checks")
    reports = {"cold": cold, "populate": populate, "paired": paired}
    analyses = {mode: verify_acquisition_report(mode, reports[mode], configs.get(mode)) for mode in reports}
    for mode in reports:
        check = job_checks.get(mode)
        require(type(check) is dict and check.get("status") == "PASS_BOUNDED_ORIGINAL_ACQUISITION_PATH_CHECKS"
                and check.get("mode") == mode and check.get("original_shutdown_and_OS_drain_checked") is True,
                "three original completed guarded path jobs")
    # Comparison uses unchanged original alias/model/config, never a relabeled old curve.
    common = ("model", "model_alias", "alias_target", "gpu_uuid", "sampling", "sizes", "reps",
              "pythonhashseed", "acquisition_domain")
    require(all(configs[mode][key] == configs["cold"][key] for mode in reports for key in common),
            "three-session original measurement identity")
    base_engine = {k: v for k, v in configs["cold"]["engine"].items() if k != "kv_transfer_config"}
    require(all({k: v for k, v in configs[mode]["engine"].items() if k != "kv_transfer_config"} == base_engine
                for mode in reports), "same original engine except connector")
    require(configs["populate"]["engine"]["kv_transfer_config"] == configs["paired"]["engine"]["kv_transfer_config"],
            "same original populate/paired connector and storage")
    for rep in range(REPS):
        cached = paired["rows"][rep * 2:rep * 2 + 2]
        require(cached[0]["output_token_ids"] == cached[1]["output_token_ids"],
                "original cached output consistency including warmup")
        # Cold versus cached numerical equivalence is deliberately not assumed.
    identities = [job_checks[mode].get("process_identity") for mode in reports]
    require(all(type(x) is dict for x in identities)
            and len({x.get("pid") for x in identities}) == 3
            and len({x.get("sid") for x in identities}) == 3, "three fresh original acquisition processes")
    return {"status": "PASS_BOUNDED_NATIVE_PATH_POINT_PILOT_ONLY", "prefix_tokens": PREFIX_TOKENS,
            "domain": DOMAIN, "request_count": 18, "reported_rows": 15,
            "sentinel_requests": 3, "output_tokens_per_request": 1,
            "measured_samples_seconds": {"f": analyses["cold"]["measured_samples_seconds"]["f"],
                                         "g_ssd": analyses["paired"]["measured_samples_seconds"]["g_ssd"],
                                         "g_mem": analyses["paired"]["measured_samples_seconds"]["g_mem"]},
            "timing": TIMING, "cold_cached_numerical_reference": "NOT_PERFORMED",
            "limitations": ["two measured samples per path, no fitting or statistical performance conclusion",
                            "TTFT is end-to-end original frontend wall-clock, not isolated I/O/CUDA cost",
                            "original snapshots/guard closure do not prove physical release or full KV bytes",
                            "raw JSON/result gates grant no GPU authorization or production qualification"],
            **UNQUALIFIED}
