"""CPU preparation for the original two-process production-KV SSD pilot.

This imports no GPU backend and grants no runtime, timing or production
qualification.  Pricing and the admission decision execute byte-pinned original
PCHIP/cost-table/LoadPlanner code.  Only profiling is replaced by an explicit
CPU no-op; the liburing module is not executed, only its pinned alignment literal.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import types

SOURCE_FILES = ("_pchip.py", "break_even.py", "cost_model.py", "load_planner.py",
                "staging.py", "liburing_file.py", "profiling.py")
BACKENDS = ("torch", "vllm", "numpy", "triton", "cupy", "py_kvcache")
_STANDARD = {"__future__", "bisect", "collections", "dataclasses", "enum",
             "json", "logging", "math", "time", "typing", "_bootlocale"}


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, name, minimum=1):
    require(type(value) is int and value >= minimum, name + " must be an exact integer")
    return value


def text(value, name):
    require(type(value) is str and 0 < len(value) <= 4096, name + " must be bounded text")
    return value


def _checked_bytes(path, ref):
    path = Path(path)
    require(type(ref) is dict and {"bytes", "sha256"} <= set(ref), "explicit byte reference required")
    integer(ref["bytes"], "reference bytes")
    digest = ref["sha256"]
    require(type(digest) is str and len(digest) == 64 and
            all(c in "0123456789abcdef" for c in digest), "reference SHA-256 required")
    require(not path.is_symlink() and path.is_file(), "regular nonsymlink source/curve required")
    require(path.stat().st_size <= 524288, "bounded source/curve file required")
    raw = path.read_bytes()
    require(len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == digest,
            "source/curve bytes changed")
    return raw


class BoundOriginalCPUProbe:
    """Execute only the original scalar admission algorithm, in a private namespace.

    expected_source_refs maps each SOURCE_FILES basename to its independently
    frozen {path?, bytes, sha256}. native_root contains py_kvcache/. No package
    initializer, ctypes/liburing or optional profiler is imported.
    """

    def __init__(self, native_root, expected_source_refs):
        require(type(expected_source_refs) is dict and
                set(expected_source_refs) == set(SOURCE_FILES), "all original CPU source refs required")
        self.root = Path(native_root).resolve(strict=True)
        source = self.root / "py_kvcache"
        require(source.is_dir() and not source.is_symlink(), "original source directory required")
        self.raw = {name: _checked_bytes(source / name, expected_source_refs[name])
                    for name in SOURCE_FILES}
        self.source_refs = {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                            for name, raw in self.raw.items()}
        self.namespace = "_g3_original_cpu_" + hashlib.sha256(
            repr(sorted(self.source_refs.items())).encode()).hexdigest()[:16] + "_" + str(id(self))
        self.modules = {}
        self.closed = False
        tree = ast.parse(self.raw["liburing_file.py"])
        alignments = [ast.literal_eval(node.value) for node in tree.body
                      if isinstance(node, ast.Assign) and any(
                          isinstance(t, ast.Name) and t.id == "DIRECT_IO_ALIGNMENT" for t in node.targets)]
        require(len(alignments) == 1, "one original alignment literal required")
        self.alignment = integer(alignments[0], "original direct I/O alignment")
        require(self.alignment <= 1048576 and self.alignment & (self.alignment - 1) == 0,
                "power-of-two original alignment required")
        self._shim("liburing_file", DIRECT_IO_ALIGNMENT=self.alignment)
        self._shim("profiling", now_ns=time.perf_counter_ns, add_event=lambda *a, **k: None)
        try:
            for name in ("_pchip", "break_even", "cost_model", "load_planner", "staging"):
                self._load(name)
        except BaseException:
            self.close()
            raise

    def _shim(self, name, **values):
        module = types.ModuleType(self.namespace + "." + name)
        module.__dict__.update(values)
        self.modules[name] = module
        sys.modules[module.__name__] = module
        return module

    def _import(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1:
            require(name in self.modules or name + ".py" in SOURCE_FILES,
                    "unexpected original relative dependency")
            return self._load(name)
        if name.startswith("py_kvcache."):
            require(fromlist and name.count(".") == 1, "bounded original scalar dependency required")
            return self._load(name.split(".")[1])
        require(level == 0 and name.split(".")[0] in _STANDARD,
                "non-stdlib/backend import rejected before import: " + name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    def _load(self, name):
        if name in self.modules:
            return self.modules[name]
        require(name + ".py" in self.raw, "unfrozen original dependency")
        module = self._shim(name)
        module.__package__ = self.namespace
        module.__file__ = str(self.root / "py_kvcache" / (name + ".py"))
        private_builtins = dict(vars(builtins), __import__=self._import)
        module.__dict__["__builtins__"] = private_builtins
        exec(compile(self.raw[name + ".py"], module.__file__, "exec"), module.__dict__)
        return module

    def close(self):
        for module in self.modules.values():
            if sys.modules.get(module.__name__) is module:
                del sys.modules[module.__name__]
        self.closed = True


def validate_layout(engine, extra, layout, probe):
    """Validate copied/planned scalar geometry; this does not verify a live GPU."""
    require(type(probe) is BoundOriginalCPUProbe and not probe.closed, "bound original CPU probe required")
    require(type(layout) is dict and set(layout) == {
        "origin", "group_count", "gpu_block_tokens", "storage_block_tokens",
        "canonical_page_bytes", "num_gpu_blocks", "actual_kv_backing_bytes"}, "explicit layout fields required")
    require(layout["origin"] in ("cpu_planned", "native_scalar_candidate"), "layout origin must stay unqualified")
    require(layout["group_count"] == 1 and type(layout["group_count"]) is int, "one actual KV group only")
    gpu = integer(layout["gpu_block_tokens"], "GPU block tokens")
    storage = integer(layout["storage_block_tokens"], "storage block tokens")
    require(storage % gpu == 0 and engine["block_size"] == gpu and extra["block_size"] == storage,
            "engine/connector/layout block geometry differs")
    pages = layout["canonical_page_bytes"]
    require(type(pages) is list and 1 <= len(pages) <= 64, "bounded actual canonical pages required")
    page_bytes = sum(integer(n, "canonical page bytes") for n in pages)
    require(page_bytes % gpu == 0, "integral actual KV bytes per token required")
    blocks = integer(layout["num_gpu_blocks"], "actual/planned GPU blocks")
    backing = layout["actual_kv_backing_bytes"]
    if layout["origin"] == "native_scalar_candidate":
        require(integer(backing, "actual unique KV backing bytes") == blocks * page_bytes,
                "canonical pages disagree with actual unique KV backing")
    else:
        require(backing is None, "CPU planned layout cannot declare actual allocation")
    budget = integer(engine["kv_cache_memory_bytes"], "KV budget")
    require(blocks * page_bytes <= budget, "canonical KV backing exceeds frozen budget")
    require(blocks * gpu >= engine["max_model_len"], "KV pool cannot hold the one-request context")
    payload = page_bytes * (storage // gpu)
    io_size = (payload + probe.alignment - 1) // probe.alignment * probe.alignment
    slots = probe.modules["staging"].StagingPool.compute_slot_count(
        staging_mem_gib=extra["staging_mem"], io_size=io_size,
        min_slots=extra["iodepth"] + max(4, extra["iodepth"] // 2))
    return dict(status="CPU_LAYOUT_GEOMETRY_VALID_ONLY", origin=layout["origin"],
                gpu_block_tokens=gpu, storage_block_tokens=storage,
                canonical_operand_count=len(pages), kv_bytes_per_token=page_bytes // gpu,
                payload_bytes=payload, io_size=io_size, staging_slots=slots,
                max_preload_slots=max(0, slots - extra["iodepth"]),
                production_qualified=False, GPU_verified=False)


def _parse_curve(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate curve JSON key")
            result[key] = value
        return result
    data = json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite curve")))
    require(type(data) is dict and data.get("schema_version") == 2 and
            type(data.get("schema_version")) is int, "original v2 curve required")
    require(type(data.get("curves")) is dict and set(data["curves"]) == {"f", "g_ssd", "g_mem"},
            "all three original curves required")
    for curve in data["curves"].values():
        require(type(curve) is dict and {"floor", "knots"} <= set(curve), "curve shape required")
        require(type(curve["knots"]) is dict and curve["knots"], "actual measured knots required")
        for value in (curve["floor"], *curve["knots"].values()):
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    "finite nonnegative original curve values required")
    return data


def probe_cold_consumer(curve_path, curve_ref, *, model_name, kv_dtype,
                        prompt_tokens, max_model_len, layout_summary, probe):
    """One fresh original LoadPlanner call, with all prefix files present/cold.

    Synthetic identities describe only CPU pricing geometry, not real KV hashes.
    ADMIT or DEFER predicts eligibility; real jobs/SSD acceptance are still required.
    """
    require(type(probe) is BoundOriginalCPUProbe and not probe.closed, "bound original CPU probe required")
    integer(prompt_tokens, "complete prompt tokens")
    integer(max_model_len, "frozen max model length")
    raw = _checked_bytes(curve_path, curve_ref)
    data = _parse_curve(raw)
    require(data.get("model_name") == model_name and data.get("kv_dtype") == kv_dtype,
            "original model/dtype curve binding differs")
    require(integer(data.get("kv_bytes_per_token"), "curve KV bytes per token") ==
            layout_summary["kv_bytes_per_token"], "curve/actual layout bytes per token differs")
    block = layout_summary["storage_block_tokens"]
    load_blocks = prompt_tokens // block
    require(load_blocks > 0, "no complete offload prefix block")
    scored_tokens = (prompt_tokens + block - 1) // block * block
    ranges = [(min(int(k) for k in c["knots"]), max(int(k) for k in c["knots"]))
              for c in data["curves"].values()]
    require(all(low <= load_blocks * block and scored_tokens <= high for low, high in ranges),
            "consumer query outside original measured curve support")
    require(prompt_tokens <= max_model_len, "prompt exceeds frozen model context")
    be = probe.modules["break_even"]
    curves = be.load_curves(str(curve_path), model_name=model_name, kv_dtype=kv_dtype)
    thresholds = be.load_break_even(str(curve_path), model_name=model_name, kv_dtype=kv_dtype)
    worker_gate = be.should_load(load_blocks * block, thresholds, ram_resident=False)
    # The fresh consumer begins cold, but a preserved native preload may finish
    # before the foreground load. Its original RAM gate must also admit the work.
    preload_worker_gate = be.should_load(load_blocks * block, thresholds, ram_resident=True)
    tables = probe.modules["cost_model"].build_cost_tables(
        curves, block_tokens=block, max_model_len=max_model_len)
    native = probe.modules["load_planner"]
    planner = native.LoadPlanner(tables, max_preload_slots=layout_summary["max_preload_slots"],
                                 defer_tolerance=2.0, defer_deadline_max_s=2.0,
                                 time_source=lambda: 0.0)
    candidate = native.CandidateCostInput(0, "cpu-geometry-only", prompt_tokens, load_blocks,
        tuple(hashlib.sha256(("CPU-GEOMETRY-" + str(n)).encode()).digest() for n in range(load_blocks)),
        False, False)
    decision = planner.plan([candidate], outstanding_load_blocks=())[candidate.req_id]
    require(_checked_bytes(curve_path, curve_ref) == raw, "curve changed during original pricing")
    require(worker_gate and preload_worker_gate and decision.decision != native.LoadDecision.DECLINE,
            "original worker gate or LoadPlanner declines this cold consumer")
    return dict(status="CPU_ORIGINAL_COLD_ADMISSION_PREDICTED_ONLY",
                curve_ref=dict(curve_ref), prompt_tokens=prompt_tokens,
                planned_preload_blocks=load_blocks, prefix_tokens=load_blocks * block,
                worker_ssd_threshold_tokens=thresholds.ssd_tokens,
                worker_preloaded_mem_threshold_tokens=thresholds.mem_tokens,
                decision=decision.decision.value, preload_blocks=decision.preload_blocks,
                original_source_refs=probe.source_refs, profiler="CPU_NOOP_ONLY",
                real_KV_hashes_verified=False, actual_SSD_IO_verified=False,
                GPU_verified=False, production_qualified=False)


def validate_two_process_plan(plan, *, curve_path, curve_ref, probe):
    """Validate a finite proposal, never producer receipts or GPU authorization."""
    require(type(plan) is dict and set(plan) == {
        "schema_version", "mode", "pair_id", "shared_storage_path", "engine", "sampling",
        "kv_transfer_config", "seed_token_ids", "flush_token_ids", "consumer_token_ids", "layout",
        "distinct_original_processes", "producer_shutdown_before_consumer", "reset_connector"},
        "exact two-process CPU proposal fields required")
    require(type(plan["schema_version"]) is int and plan["schema_version"] == 1, "pilot schema required")
    require(plan["mode"] in ("off", "shadow"), "G3 off/shadow only")
    run = text(plan["pair_id"], "pair id")
    require(plan["distinct_original_processes"] is True and plan["producer_shutdown_before_consumer"] is True,
            "fresh consumer after original producer shutdown required")
    require(plan["reset_connector"] is False, "connector/cache reset forbidden in this two-process plan")
    storage = Path(text(plan["shared_storage_path"], "private SSD root"))
    require(storage.is_absolute(), "absolute private SSD root required")
    engine = plan["engine"]
    require(type(engine) is dict, "frozen original engine required")
    expected = dict(dtype="bfloat16", kv_cache_dtype="auto", quantization=None,
                    tensor_parallel_size=1, distributed_executor_backend="uni", max_num_seqs=1,
                    enable_prefix_caching=True, enforce_eager=True, async_scheduling=False,
                    cpu_offload_gb=0, offload_group_size=0, kv_offloading_size=None)
    require(all(key in engine and type(engine[key]) is type(value) and engine[key] == value
                for key, value in expected.items()), "ordinary original BF16 single-request engine required")
    text(engine.get("model"), "actual engine model string")
    for key in ("max_model_len", "max_num_batched_tokens", "block_size", "kv_cache_memory_bytes"):
        integer(engine.get(key), key)
    require(engine["max_model_len"] <= 16400, "bounded G3 model context required")
    sampling = plan["sampling"]
    require(type(sampling) is dict and all(key in sampling for key in
            ("temperature", "seed", "max_tokens", "min_tokens", "ignore_eos", "detokenize")),
            "full original sampling fields required")
    require(type(sampling["temperature"]) is float and sampling["temperature"] == 0.0 and
            type(sampling["seed"]) is int and sampling["seed"] == 0 and
            type(sampling["max_tokens"]) is int and sampling["max_tokens"] == 128 and
            type(sampling["min_tokens"]) is int and sampling["min_tokens"] == 128 and
            sampling["ignore_eos"] is True and sampling["detokenize"] is False,
            "complete deterministic 128 outputs required")
    tokens = []
    for key in ("seed_token_ids", "flush_token_ids", "consumer_token_ids"):
        values = plan[key]
        require(type(values) is list and 16 <= len(values) <= 16272, "bounded complete prompt IDs required")
        for value in values:
            integer(value, "original prompt token ID", 0)
        require(len(values) + 128 <= engine["max_model_len"], "full output/context budget does not fit")
        tokens.append(values)
    require(tokens[0] == tokens[2], "consumer must use the exact complete producer seed")
    require(tokens[0][0] != tokens[1][0], "flush request must be prefix-disjoint")
    kt = plan["kv_transfer_config"]
    require(type(kt) is dict and set(kt) == {"kv_connector", "kv_role", "kv_connector_extra_config"} and
            kt["kv_connector"] == "OffloadingConnector" and kt["kv_role"] == "kv_both",
            "original OffloadingConnector/kv_both required")
    extra = kt["kv_connector_extra_config"]
    keys = {"spec_name", "spec_module_path", "shared_storage_path", "block_size", "io_backend",
            "iodepth", "open_lookahead", "staging_mem", "sync_on_store", "enable_preload",
            "preload_share_staging", "preload_lookahead_requests", "staging_cache", "load_planner",
            "prefix_cache_break_even_path", "prefix_io_parent_admission", "prefix_io_p4_policy"}
    require(type(extra) is dict and set(extra) == keys, "bounded original connector fields required")
    require(extra["spec_name"] == "PyKvCacheOffloadingSpec" and extra["spec_module_path"] == "py_kvcache.vllm" and
            extra["shared_storage_path"] == str(storage) and extra["io_backend"] == "linux_aio" and
            extra["load_planner"] == "on" and extra["enable_preload"] is True and
            extra["preload_share_staging"] is True and extra["sync_on_store"] is False and
            extra["staging_cache"] == "off", "original planner/shared staging/asynchronous SSD path required")
    integer(extra["iodepth"], "I/O depth")
    require(extra["iodepth"] <= 4 and extra["open_lookahead"] == extra["iodepth"] and
            type(extra["open_lookahead"]) is int and type(extra["preload_lookahead_requests"]) is int and
            extra["preload_lookahead_requests"] == 1, "bounded original I/O/preload window required")
    require(type(extra["staging_mem"]) in (int, float) and math.isfinite(extra["staging_mem"]) and
            0 < extra["staging_mem"] <= 1.0, "bounded explicit GiB staging budget required")
    require(extra["prefix_cache_break_even_path"] == str(Path(curve_path)), "same unchanged curve file required")
    parent = extra["prefix_io_parent_admission"]
    require(type(parent) is dict and parent == dict(schema_version=1, run_id=run, max_accepted_parents=2) and
            all(type(parent[k]) is int for k in ("schema_version", "max_accepted_parents")),
            "same common two-parent admission required")
    policy = extra["prefix_io_p4_policy"]
    if plan["mode"] == "off":
        require(type(policy) is dict and policy == {"mode": "off"}, "off must contain no strategy state")
    else:
        require(type(policy) is dict and set(policy) == {"schema_version", "run_id", "mode", "sample_max_age_ns",
                "max_wait_ns", "internal_step_budget_ns", "candidate_batches", "fixed_stage_policy", "cost_table"},
                "exact shadow fields required")
        require(type(policy["schema_version"]) is int and policy["schema_version"] == 1 and
                policy["run_id"] == run and policy["mode"] == "shadow" and
                policy["fixed_stage_policy"] is None and policy["cost_table"] is None and
                policy["internal_step_budget_ns"] is None and policy["candidate_batches"] == [1, 2, 4, 8] and
                all(type(n) is int for n in policy["candidate_batches"]), "shadow cannot activate caps/cost strategy")
        integer(policy["sample_max_age_ns"], "sample max age")
        integer(policy["max_wait_ns"], "max wait")
    geometry = validate_layout(engine, extra, plan["layout"], probe)
    admission = probe_cold_consumer(curve_path, curve_ref, model_name=engine["model"],
        kv_dtype=engine["kv_cache_dtype"], prompt_tokens=len(tokens[2]),
        max_model_len=engine["max_model_len"], layout_summary=geometry, probe=probe)
    return dict(status="CPU_TWO_ORIGINAL_PROCESS_PILOT_PREPARED_ONLY", mode=plan["mode"],
                pair_id=run, layout=geometry, cold_admission=admission,
                planned_guarded_jobs_for_both_modes=4, planned_requests_per_pair=3,
                runtime_required=["target stores actually submitted before producer shutdown",
                    "original producer shutdown and OS session empty before consumer",
                    "same frozen producer SSD payload manifest; consumer staging starts empty",
                    "actual original jobs/bytes/byte-exact restore and final native drain"],
                gpu_authorized=False, GPU_verified=False, production_qualified=False,
                performance_claim=False)
