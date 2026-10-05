"""Exact finite current-step binding to the existing NativeP4Bridge.

This module has no owner, resource queue, backend, model execution or release
credit. Actual activation requires the private GPU-cell issuer's real evidence;
CPU fixture arithmetic cannot attach to a production bridge.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import types
import uuid
import weakref

_TOKEN = object()
KERNEL = "eager"
MAX_COUNTER_KEYS = 16


def require(value, reason):
    if not value:
        raise ValueError("FINITE_CURRENT_BINDING_REJECTED: " + reason)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def normalize_uuid(value):
    if type(value) is bytes:
        require(len(value) == 16, "GPU UUID byte count")
        value = str(uuid.UUID(bytes=value))
    return "GPU-" + str(uuid.UUID(str(value).removeprefix("GPU-"))).lower()


def identity_runtime_ref(item):
    if type(item) is tuple:
        require(len(item) == 3 and type(item[0]) is str and type(item[1]) is int and
                type(item[2]) is str, "exact issuer scalar runtime reference")
        return dict(path=item[0], bytes=item[1], sha256=item[2])
    require(type(getattr(item, "path", None)) is str and type(getattr(item, "bytes", None)) is int and
            type(getattr(item, "sha256", None)) is str, "typed issuer runtime reference")
    return dict(path=item.path, bytes=item.bytes, sha256=item.sha256)


def frozen(path, root, refs):
    path = Path(path).resolve(strict=True)
    relative = path.relative_to(root).as_posix()
    row = refs.get(relative)
    require(type(row) is dict and set(row) == {"path", "bytes", "sha256"} and row["path"] == relative,
            "actual executed source in independent runtime closure")
    raw = path.read_bytes()
    require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], "actual source byte drift")
    return dict(row)


@dataclass(frozen=True, init=False)
class RuntimeFiniteIdentity:
    signature_prefix: tuple
    source_refs: tuple
    calibration_source_lock_sha256: str
    gpu_uuid: str
    geometry_sha256: str
    collector_source_sha256: str
    cuda_event_source_sha256: str
    _runner_ref: object

    def __init__(self, *, _token=None, signature_prefix, source_refs, calibration_source_lock_sha256,
                 gpu_uuid, geometry_sha256, collector_source_sha256, cuda_event_source_sha256, runner):
        require(_token is _TOKEN, "identity comes only from actual source/device/model/layout verification")
        for key, value in dict(signature_prefix=tuple(signature_prefix), source_refs=tuple(source_refs),
                               calibration_source_lock_sha256=calibration_source_lock_sha256, gpu_uuid=gpu_uuid,
                               geometry_sha256=geometry_sha256, collector_source_sha256=collector_source_sha256,
                               cuda_event_source_sha256=cuda_event_source_sha256, _runner_ref=weakref.ref(runner)).items():
            object.__setattr__(self, key, value)

    def matches_runner(self, runner):
        return runner is not None and self._runner_ref() is runner


def verify_running_identity(worker, handler, capture, table, *, issuer, root, runtime_refs,
                            common_runtime_domain_sha256, model_identity, calibration_lock_ref):
    """Inspect already constructed original runtime; never import/initialize CUDA."""
    root = Path(root).resolve(strict=True)
    proof = issuer.qualified_identity(table)
    require(proof is not None and table.production_qualified is True,
            "exact private issuer-qualified table required; CPU/mock/conditional cannot attach")
    frozen(__file__, root, runtime_refs)
    frozen(issuer.__file__, root, runtime_refs)
    require(type(proof).__module__ == issuer.__name__, "original private issuer identity type")
    torch = sys.modules.get("torch")
    require(torch is not None and "vllm" in sys.modules and "py_kvcache.vllm" in sys.modules,
            "only already running original GPU process")
    runner, reactor = worker.model_runner, handler.coordinator.reactor
    require(capture.runner_ref() is runner and capture.origin == "native_gpu_recording" and capture.valid is True,
            "actual same live collector/runner")
    sources = {}
    for owner in (worker, runner, runner.model, handler, handler.coordinator, reactor, reactor.layout, capture):
        path = inspect.getsourcefile(type(owner))
        require(path is not None, "original runtime Python type source")
        row = frozen(path, root, runtime_refs)
        sources[row["path"]] = tuple(sorted(row.items()))
    event_ref = frozen(inspect.getsourcefile(torch.cuda.Event), root, runtime_refs)
    require(torch.cuda.Event.__module__ == "torch.cuda.streams" and
            event_ref["sha256"] == proof.cuda_event_source_sha256 == capture.event_source["sha256"],
            "same actual original CUDA Event source as paired measurement")
    collector_ref = frozen(inspect.getsourcefile(type(capture)), root, runtime_refs)
    native_ref = frozen(inspect.getsourcefile(type(reactor)), root, runtime_refs)
    require(collector_ref["sha256"] == proof.collector_source_sha256 and
            native_ref["sha256"] == proof.native_source_sha256, "same original collector/native sources")
    config = worker.vllm_config
    require(runner.vllm_config is config and runner.model_config is config.model_config,
            "one actual original model configuration")
    model, cache, parallel = config.model_config, config.cache_config, config.parallel_config
    hf = model.hf_config
    require(model.dtype == torch.bfloat16 and model.quantization is None and model.enforce_eager is True and
            config.speculative_config is None and config.scheduler_config.async_scheduling is False,
            "actual deterministic eager original full-step domain")
    require(getattr(config.compilation_config.mode, "value", config.compilation_config.mode) == 0 and
            getattr(config.compilation_config.cudagraph_mode, "value", config.compilation_config.cudagraph_mode) == 0,
            "no compile/graph domain substitution")
    require(parallel.tensor_parallel_size == parallel.pipeline_parallel_size == parallel.data_parallel_size == 1 and
            parallel.distributed_executor_backend == "uni", "same original single-device executor")
    require(cache.block_size == 16 and cache.enable_prefix_caching is True and cache.cache_dtype == "auto" and
            cache.kv_cache_memory_bytes == 268435456, "original common KV geometry")
    require(hf.num_hidden_layers == 28 and hf.num_key_value_heads == 4 and hf.hidden_size == 3584 and
            hf.num_attention_heads == 28 and hf.model_type == "qwen2", "fixed Qwen architecture")
    require(reactor.config.load_planner == "on" and reactor.config.enable_preload is True and
            reactor.config.preload_share_staging is True and reactor.config.io_backend == "linux_aio" and
            reactor.config.sync_on_store is False and reactor.config.iodepth == reactor.iodepth == 4 and
            reactor.config.staging_cache == "lru" and reactor.staging_budget_bytes == 134217728 and
            0 < reactor.actual_staging_bytes <= reactor.staging_budget_bytes,
            "retained actual strong original planner/preload/shared staging/backend")
    layout = handler.coordinator.layout
    require(reactor.layout is layout and layout.storage_block_size_factor == 1 and
            layout.storage_block_bytes == reactor.file_store.io_size == 917504 and
            sum(layout.bytes_per_kernel_block) == 917504 and len(layout.gpu_tensors) > 0 and
            len(layout.gpu_tensors) == len(layout.bytes_per_kernel_block), "actual original canonical physical KV block")
    for tensor, width in zip(layout.gpu_tensors, layout.bytes_per_kernel_block):
        require(tensor.is_cuda and tensor.dtype == torch.int8 and tensor.ndim == 2 and
                tensor.device == runner.device and tensor.element_size() * tensor.stride(0) == width,
                "actual canonical GPU byte tensor metadata")
    model_config_ref = frozen(Path(model_identity["model_directory"]) / "config.json", root, runtime_refs)
    geometry = dict(model_config_sha256=model_config_ref["sha256"], num_hidden_layers=hf.num_hidden_layers,
                    num_key_value_heads=hf.num_key_value_heads, head_dim=hf.hidden_size // hf.num_attention_heads,
                    dtype="bfloat16", dtype_bytes=2, tokens_per_block=cache.block_size,
                    tensor_parallel_size=1, physical_block_bytes=reactor.file_store.io_size)
    require(geometry["physical_block_bytes"] == 28 * 2 * 4 * 128 * 2 * 16, "model-derived physical KV quantum")
    require(Path(model.model).resolve() == Path(model_identity["model_directory"]).resolve() and
            model_identity["manifest_sha256"] == proof.model_sha256, "actual offline model matches qualified cell")
    require(torch.cuda.device_count() == 1 and runner.device.type == "cuda" and runner.device.index == 0 and
            worker.device == runner.device, "one actual visible original GPU")
    actual_uuid = normalize_uuid(torch.cuda.get_device_properties(runner.device).uuid)
    require(actual_uuid == normalize_uuid(proof.gpu_uuid) and common_runtime_domain_sha256 == proof.common_runtime_domain_sha256,
            "actual hardware/common strong configuration matches paired cell")
    require(canonical_sha(geometry) == proof.kv_layout_sha256 and proof.kernel_mode == KERNEL,
            "actual layout/kernel matches paired exact cell")
    groups = runner.attn_groups
    require(type(groups) is list and len(groups) == 1 and 0 < len(groups[0]) <= 32,
            "actual original single KV attention group")
    layers = []
    for group in groups[0]:
        require(group.backend.get_name() == "TRITON_ATTN", "actual original Triton attention backend")
        frozen(inspect.getsourcefile(group.backend), root, runtime_refs)
        layers.extend(group.layer_names)
    require(len(layers) == len(set(layers)) == hf.num_hidden_layers,
            "actual Triton attention covers every model layer once")
    require(calibration_lock_ref["sha256"] == proof.source_lock_sha256, "exact calibration source-lock identity")
    calibration_path = root / calibration_lock_ref["path"]
    row = frozen(calibration_path, root, runtime_refs)
    require(row == calibration_lock_ref, "calibration source-lock frozen in new runtime closure")
    original_lock = json.loads(calibration_path.read_bytes())
    require(type(original_lock.get("files")) is list and all(runtime_refs.get(item["path"]) == item
            for item in original_lock["files"]), "all calibration source rows inherited without drift")
    for item in proof.runtime_refs:
        reference = identity_runtime_ref(item)
        require(runtime_refs.get(reference["path"]) == reference and frozen(root / reference["path"], root, runtime_refs) == reference,
                "qualified table required runtime source inherited exactly")
    prefix = (actual_uuid, common_runtime_domain_sha256, proof.model_sha256, proof.kv_layout_sha256)
    require(proof.cells and len(proof.cells) <= 8 and all(tuple(cell[:4]) == prefix for cell in proof.cells),
            "finite cell signatures share actual source/model/layout/hardware domain")
    return RuntimeFiniteIdentity(_token=_TOKEN, signature_prefix=prefix,
        source_refs=tuple(sources[name] for name in sorted(sources)), calibration_source_lock_sha256=proof.source_lock_sha256,
        gpu_uuid=actual_uuid, geometry_sha256=proof.kv_layout_sha256, collector_source_sha256=proof.collector_source_sha256,
        cuda_event_source_sha256=proof.cuda_event_source_sha256, runner=runner)


def current_step(capture, identity, *, now_ns, max_age_ns):
    if (type(identity) is not RuntimeFiniteIdentity or not identity.matches_runner(capture.runner_ref()) or
            capture.origin != "native_gpu_recording" or capture.valid is not True):
        return None
    state = capture.current_single_file_step()
    if (type(state) is not tuple or len(state) != 7 or any(type(value) is not int for value in state) or
            not 0 < state[1] <= state[2] <= now_ns or not 0 <= now_ns - state[2] <= max_age_ns or
            state[3] != 1 or state[4] != 1 or state[5] != 0 or state[6] < 1):
        return None
    return state


def covered_subset(cells, supplied, *, observation_only):
    """Validate a private caller's measured subset; never issue qualification."""
    all_cells = frozenset(tuple(cell) for cell in cells)
    if observation_only:
        require(supplied is None, "development observes all issued cells before reserve measurement")
        return all_cells
    require(type(supplied) is tuple and supplied and
            all(type(cell) is tuple and cell in all_cells for cell in supplied) and
            len(set(supplied)) == len(supplied), "effect requires the real native-reserve covered subset")
    return frozenset(supplied)


class CurrentFiniteBinding:
    """Weak, bounded scalar attachment; delegates all advice to original policy."""
    def __init__(self, bridge, capture, identity, *, issuer, observation_only=False, host_observer=None,
                 reserve_covered_cell_signatures=None):
        require(type(identity) is RuntimeFiniteIdentity and identity.matches_runner(capture.runner_ref()), "actual weak runtime binding")
        proof = issuer.qualified_identity(bridge.policy.table)
        require(proof is not None and bridge.policy.table.production_qualified is True and
                tuple(proof.cells) and all(tuple(cell[:4]) == identity.signature_prefix for cell in proof.cells),
                "only privately issued real finite cells may reach native original bridge")
        from prefix_io_control.p4_bridge import NativeP4Bridge
        require(type(bridge) is NativeP4Bridge, "same active exact original native bridge type")
        bridge._owner()  # This installation is performed only by original owner callback.
        require(bridge.policy.single_file is None, "separate exact finite table, no old singleton receipt")
        self._bridge_ref, self._capture_ref = weakref.ref(bridge), weakref.ref(capture)
        self.identity = identity
        require(type(observation_only) is bool, "explicit development shadow semantics")
        self.observation_only = observation_only
        self.covered_cell_signatures = covered_subset(proof.cells, reserve_covered_cell_signatures,
                                                     observation_only=observation_only)
        self.host_observer = host_observer
        if host_observer is not None:
            source = Path(__file__).resolve()
            raw = source.read_bytes()
            host_observer.bind("qualified_finite_controller_preview", "controller", self._preview_impl,
                source_ref=dict(path=source.as_posix(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
                boundary_kind="controller_bookkeeping")
        self.counters = {}
        self.eligible_windows, self.eligible_log_lost = {}, 0
        self.preview_windows, self.preview_log_lost = [], 0
        self.original_preview, self.original_batch = bridge.preview_issue, bridge.batch_prefix
        require("preview_issue" not in vars(bridge) and "batch_prefix" not in vars(bridge), "original bridge functions unmodified")
        self.preview_wrapper = types.MethodType(self._preview, bridge)
        self.batch_wrapper = types.MethodType(self._batch, bridge)
        bridge.preview_issue = self.preview_wrapper
        bridge.batch_prefix = self.batch_wrapper
        self._last_batch_state = None

    def count(self, reason):
        key = reason if reason in self.counters or len(self.counters) < MAX_COUNTER_KEYS - 1 else "other"
        self.counters[key] = self.counters.get(key, 0) + 1

    def snapshot(self, original, work, *, now_ns):
        capture = self._capture_ref()
        bridge = self._bridge_ref()
        if capture is None or bridge is None or bridge.fault is not None:
            self.count("detached_or_fault")
            return None, None
        state = current_step(capture, self.identity, now_ns=now_ns, max_age_ns=bridge.policy.config.sample_max_age_ns)
        native = original.native_state
        if (state is None or native is None or native.inflight is None or
                not 0 <= now_ns - native.captured_ns <= bridge.policy.config.sample_max_age_ns):
            self.count("unknown_or_stale_condition")
            return None, None
        signature = self.identity.signature_prefix + state[3:] + (work.nbytes,)
        if signature not in self.covered_cell_signatures:
            self.count("native_reserve_cell_not_covered")
            return None, None
        if bridge.policy.table.lookup(signature, native.inflight, work.stage, work.nbytes, execution="production") is None:
            self.count("unknown_exact_cell")
            return None, None
        enriched = replace(original, load_signature=signature,
                           capabilities=original.capabilities | frozenset(("production_gpu_load_state",)))
        if current_step(capture, self.identity, now_ns=now_ns, max_age_ns=bridge.policy.config.sample_max_age_ns) != state:
            self.count("condition_changed_before_advice")
            return None, None
        return enriched, state

    def _preview(self, bridge, work, original, *, now_ns):
        if self.host_observer is None:
            return self._preview_impl(bridge, work, original, now_ns=now_ns)
        runner = self.identity._runner_ref()
        ordinal = getattr(runner, "_profile_step", None)
        if type(ordinal) is not int or ordinal < 1:
            # No qualified native GPU step exists yet. Preserve native U;
            # this absence cannot qualify a controller reserve interval.
            from prefix_io_control.p4_types import IssuePreview
            return IssuePreview("native_fallback", "finite_controller_step_unknown")
        token = self.host_observer.begin("controller", ordinal - 1, "qualified_finite_controller_preview")
        attempt = dict(native_step_ordinal=ordinal - 1, verdict="native_fallback", reason="unknown_live_condition")
        try:
            return self._preview_impl(bridge, work, original, now_ns=now_ns, attempt=attempt)
        finally:
            self.host_observer.end(token)
            if token is None:
                self.preview_log_lost += 1
            else:
                with self.host_observer._lock:
                    intervals = self.host_observer._intervals.get(ordinal - 1, ())
                    span = next((item for item in reversed(intervals) if item["boundary_id"] == "qualified_finite_controller_preview" and
                                 item["start_ns"] == token.start_ns and item["thread_id"] == token.thread_id), None)
                if span is None or len(self.preview_windows) >= 4096:
                    self.preview_log_lost += 1
                else:
                    attempt.update(attempt_id=len(self.preview_windows) + 1, begin_ns=span["start_ns"], end_ns=span["end_ns"],
                                   boundary_id=span["boundary_id"], clock_scope=span["clock_scope"], thread_id=span["thread_id"])
                    self.preview_windows.append(attempt)

    def _preview_impl(self, bridge, work, original, *, now_ns, attempt=None):
        bridge._owner()
        if work.progress is not None or now_ns - work.created_ns >= bridge.policy.config.max_wait_ns:
            self.count("native_mandatory_or_age_override")
            if attempt is not None:
                attempt["reason"] = "native_mandatory_or_max_wait_override"
            return self.original_preview(work, original, now_ns=now_ns)
        enriched, state = self.snapshot(original, work, now_ns=now_ns)
        if enriched is None:
            if attempt is not None:
                attempt["reason"] = "finite_current_condition_unknown"
            from prefix_io_control.p4_types import IssuePreview
            return IssuePreview("native_fallback", "finite_current_condition_unknown")
        if attempt is not None:
            attempt.update(signature=list(enriched.load_signature), state_before=list(state), stage=work.stage,
                physical_bytes=work.nbytes, snapshot_captured_ns=original.native_state.captured_ns,
                existing_io=[dict(ops=amount.ops, nbytes=amount.nbytes) for amount in original.native_state.inflight],
                existing_io_origin="actual_owner_StageAccounting_snapshot", start_query_complete=True,
                end_record_started=False, mandatory_or_max_wait_override=False,
                ready_candidate_id=str(work.work_id)[:256], native_created_ns=work.created_ns, injected_candidate=False)
        key = (state[0], enriched.load_signature)
        if key not in self.eligible_windows:
            if len(self.eligible_windows) >= 512:
                self.eligible_log_lost += 1
            else:
                self.eligible_windows[key] = dict(native_step_ordinal=state[0], signature=list(enriched.load_signature),
                    existing_io=[dict(ops=amount.ops, bytes=amount.nbytes) for amount in original.native_state.inflight],
                    stage=work.stage, physical_bytes=work.nbytes, first_native_eligible_ns=now_ns,
                    last_native_eligible_ns=now_ns, actual_eligible_calls=0,
                    ready_candidate_id=str(work.work_id)[:256], native_created_ns=work.created_ns,
                    private_qualified_exact_cell=True, injected_candidate=False, observation_only=self.observation_only)
        if key in self.eligible_windows:
            self.eligible_windows[key]["actual_eligible_calls"] += 1
            self.eligible_windows[key]["last_native_eligible_ns"] = now_ns
        advice = self.original_preview(work, enriched, now_ns=now_ns)
        capture = self._capture_ref()
        after = None if capture is None else current_step(capture, self.identity, now_ns=now_ns,
                max_age_ns=bridge.policy.config.sample_max_age_ns)
        if capture is None or after != state:
            if attempt is not None:
                attempt["reason"] = "finite_current_condition_changed"
            self.count("condition_changed_after_advice")
            from prefix_io_control.p4_types import IssuePreview
            return IssuePreview("native_fallback", "finite_current_condition_changed")
        if attempt is not None:
            attempt.update(verdict="eligible", reason="real_original_ordinary_candidate_exact_private_cell",
                           state_after=list(after), observed_original_proposal=advice.action)
        self.count(advice.action)
        if self.observation_only:
            from prefix_io_control.p4_types import IssuePreview
            self.count("shadow_observed_" + advice.action)
            return IssuePreview("native_fallback", "finite_development_shadow_observation_only")
        return advice

    def _batch(self, bridge, original, works, *, now_ns):
        bridge._owner()
        # The existing production batch gate has no real qualification. Keep
        # original fusion and queue order; this initial finite port only acts
        # on one legal ordinary issue/deferral candidate at a time.
        self.count("native_fusion_batch_not_activated")
        return None

    def detach(self):
        bridge = self._bridge_ref()
        if bridge is None:
            return True
        okay = vars(bridge).get("preview_issue") is self.preview_wrapper and vars(bridge).get("batch_prefix") is self.batch_wrapper
        if vars(bridge).get("preview_issue") is self.preview_wrapper:
            del bridge.preview_issue
        if vars(bridge).get("batch_prefix") is self.batch_wrapper:
            del bridge.batch_prefix
        self._capture_ref = lambda: None
        return okay

    def evidence(self):
        return dict(schema="finite_current_original_bridge_binding_v1", condition_counters=dict(self.counters),
                    actual_eligible_preview_windows=list(self.eligible_windows.values()), eligible_preview_log_lost=self.eligible_log_lost,
                    preview_windows=list(self.preview_windows), attempt_count=len(self.preview_windows),
                    overflow=self.preview_log_lost > 0, unknown_coverage=self.preview_log_lost > 0,
                    eligibility_capture_rule="all_actual_native_candidates_with_current_privately_qualified_exact_cell",
                    observation_only=self.observation_only, production_batch_activation=False,
                    reserve_covered_cell_signatures=[list(cell) for cell in sorted(self.covered_cell_signatures)],
                    cells_max=8, unmeasured_cells_fallback_native=True, new_resource_owner=False,
                    resource_release_credit=False, mandatory_and_max_wait_original=True,
                    mock_promoted_to_GPU=False, performance_gain_proved=False)
