"""Native single-request CacheBlend bridge, shared by Mistral and Qwen.

No fixture, fixed block table, altered prompt, or qualification bypass is used.
Importing this module never loads a model or starts CUDA.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import asdict
import inspect
import math
import time
from functools import lru_cache

from .cacheblend_v6_online_engine import CacheBlendV6OnlineEngine
from .model_adapters import PinnedCacheBlendResumableAdapter
from .v8_schema10_native_blocks import NativeBlockRequest
from .v8_schema10_inventory import native_segment_inventory, mandatory_suffix_positions
from .v8_schema10_execution import digest_json
from .v8_schema10_cost_provider import RequestExecutionShape
from .v8_schema10_measured_costs import MeasuredRequestCostProvider
from .v8_schema6_hbm import HBMReservationKind
from .v8_schema6_contracts import PlannerSnapshot
from .v8_schema6_planner import JointTimelineContext
from .v7_contracts import SourceVariantIdentity
from .p0_diagnostic_context_v2 import diagnostic_context_v2, reject_diagnostic_publication_v2


def dispatch_depths(selection_path, model_spec):
    if selection_path == "d1_only":
        return (1,)
    if selection_path == "d1_d2_rescue":
        return (1, 2)
    if selection_path == "legacy_multicheckpoint":
        # model_spec must be the schema6+ spec, not historical v6 MISTRAL_SPEC.
        return tuple(model_spec.checkpoints)
    raise ValueError("unconnected native dispatch")


def _defer_layer_timing_capability(method):
    # Never retain a bound method in the global cache: it owns the model.
    # Replacing the implementation naturally produces a different cache key.
    return _defer_layer_timing_implementation(getattr(method, "__func__", method))


@lru_cache(maxsize=128)
def _defer_layer_timing_implementation(method):
    """Fail-closed patch capability check, cached per implementation object.

    The audited patch is immutable for a running adapter, so repeating
    ``inspect.getsource`` on every request only adds host setup overhead.
    Keeping this as a pure cached function preserves the existing rejection
    behavior while making the check a one-time adapter capability probe.
    """
    try:
        source = inspect.getsource(method)
    except (OSError, TypeError) as exc:
        raise RuntimeError("deferred timing capability cannot be audited") from exc
    return "probekv_defer_layer_timing" in source


_defer_layer_timing_capability.cache_clear = _defer_layer_timing_implementation.cache_clear
_defer_layer_timing_capability.cache_info = _defer_layer_timing_implementation.cache_info


def validate_native_sampling_request(request):
    from .v8_schema10_qa import bounded_answer
    bounded_answer('', request)
    count = request.get("max_new_tokens", 32)
    if type(count) is not int or count < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if "teacher_token_ids" in request:
        tokens = request["teacher_token_ids"]
        if (not request.get("capture_logits") or not isinstance(tokens, (list, tuple))
                or len(tokens) != count - 1
                or any(type(t) is not int or t < 0 for t in tokens)
                or request.get("capture_original_full_prefill")):
            raise ValueError("teacher-forced diagnostic requires exactly max_new_tokens-1 input tokens and logits")


class NativeOnlineAdapter:
    def __init__(self, *, llm, model_spec, selection_path, loader, hbm, shadow_store,
                 store_provider, provenance, cost_provider, shared_runtime_state=None):
        import torch
        self.torch, self.llm, self.spec = torch, llm, model_spec
        self.worker = llm.llm_engine.model_executor.driver_worker
        self.runner, self.outer = self.worker.model_runner, self.worker.model_runner.model
        self.inner, self.kv = self.outer.model, self.worker.cache_engine.gpu_cache
        self.scheduler = llm.llm_engine.scheduler[0] if isinstance(llm.llm_engine.scheduler, list) else llm.llm_engine.scheduler
        self.loader, self.hbm, self.shadows = loader, hbm, shadow_store
        self.store_provider, self.provenance, self.costs = store_provider, provenance, cost_provider
        self.path, self.depths = selection_path, dispatch_depths(selection_path, model_spec)
        self.capabilities = {"native_prefix_block_allocator": True, "production_dispatch": selection_path,
                             "snapshot_accepts_retention": True}
        self.retained_shadows = {}
        self.hot_layer_cache = {}
        self.hot_reservations = {}
        self.shared = shared_runtime_state if shared_runtime_state is not None else {"active": None, "warm_history": [], "generation": 1}
        self.deadline = math.inf
        self.projection = PinnedCacheBlendResumableAdapter(self.inner, self.spec)
        params = inspect.signature(self.runner.prepare_input_tensors).parameters
        self.prepare_accepts_cache = len(params) >= 2  # pinned CacheBlend adds kv_caches

    @property
    def persistent_hot_hbm_bytes(self):
        return sum(r.bytes for r in self.hot_reservations.values() if not r.released)

    @property
    def active(self):
        return self.shared["active"]

    @active.setter
    def active(self, value):
        self.shared["active"] = value

    @property
    def warm_history(self):
        return self.shared["warm_history"]

    @warm_history.setter
    def warm_history(self, value):
        self.shared["warm_history"] = value

    @property
    def generation(self):
        return self.shared["generation"]

    @generation.setter
    def generation(self, value):
        self.shared["generation"] = value

    def prepare(self, metadata, *, caches=None):
        caches = self.kv if caches is None else caches
        if self.prepare_accepts_cache:
            return self.runner.prepare_input_tensors([metadata], caches)
        return self.runner.prepare_input_tensors([metadata])

    def check_deadline(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("session deadline reached before next layer")

    def reset(self):
        if self.active is not None:
            raise RuntimeError("cannot reset active native request")
        self.torch.cuda.synchronize()
        manager = self.scheduler.block_manager
        if manager.block_tables:
            raise RuntimeError("native allocator is owned by another request")
        if manager.block_sliding_window is not None:
            raise RuntimeError("frozen native Prefix stack cannot silently disable sliding window")
        self.scheduler.block_manager = type(manager)(block_size=manager.block_size,
            num_gpu_blocks=manager.num_total_gpu_blocks, num_cpu_blocks=manager.num_total_cpu_blocks,
            watermark=manager.watermark, sliding_window=None, enable_caching=True)
        self.shadows.clear()
        self.warm_history = []
        self.generation += 1

    def snapshot(self, *, retain=False):
        if self.active is not None:
            raise RuntimeError("native snapshot requires quiescence")
        row = {"native_prefix_rebuild": list(self.warm_history), "selection_path": self.path,
                "model_signature": self.provenance["model_signature"],
                "prefix_shadows": self.shadows.descriptor(),
                "gpu_pointers_serialized": False, "ssd_page_cache_controlled": False}
        if retain:
            key = self.path + ":" + digest_json(row)
            self.shadows.retain(key)
            self.retained_shadows[digest_json(row)] = key
        return row

    def release_snapshot(self, snapshot):
        self.shadows.release_retained(self.retained_shadows.pop(digest_json(snapshot)))

    def restore(self, snapshot):
        if snapshot["selection_path"] != self.path or snapshot["model_signature"] != self.provenance["model_signature"]:
            raise ValueError("native snapshot dispatch/model mismatch")
        key = digest_json(snapshot)
        if key not in self.retained_shadows:
            raise ValueError("native restoration requires an explicitly retained snapshot")
        self.reset()
        # Re-execute the same dense warm history through native allocation.
        # No CUDA pointer or block index is deserialized.
        for request in snapshot["native_prefix_rebuild"]:
            with self.open_request(request, arrival_ns=time.perf_counter_ns()) as ctx:
                ctx.finish(lambda: None)
        self.shadows.restore_retained(self.retained_shadows[key])
        if self.snapshot() != snapshot:
            raise RuntimeError("native Prefix warm reconstruction differs")

    @contextmanager
    def open_request(self, request, *, arrival_ns):
        initialization_landmarks = [("adapter_open_enter", time.perf_counter_ns())]
        from vllm import SamplingParams
        from .v8_schema10_numerical_policy import assert_numerical_policy
        assert_numerical_policy(self.torch, getattr(self, "expected_numerical_execution_policy", None))
        mandatory_suffix_positions(request)
        validate_native_sampling_request(request)
        if self.active is not None:
            raise RuntimeError("max_integrated_concurrency=1")
        self.check_deadline()
        self.active = request["request_id"]
        holder = {}
        try:
            native = NativeBlockRequest(block_manager=self.scheduler.block_manager,
                request_id=request["request_id"], prompt_token_ids=request["token_ids"],
                sampling_params=SamplingParams(temperature=0, max_tokens=int(request.get("max_new_tokens", 32))),
                synchronize=self.torch.cuda.synchronize,
                apply_block_copies=self.worker.cache_engine.copy,
                prefix_shadow_provider=lambda tokens, ids: self.shadows.lookup(tokens, ids, native_request=holder["native"]))
        except Exception:
            self.active = None
            raise
        holder["native"] = native
        initialization_landmarks.append(("native_request_constructed", time.perf_counter_ns()))
        native.allow_missing_shadow = True
        ctx = None
        try:
            with self.torch.inference_mode(), native:
                initialization_landmarks.append(("native_blocks_and_shadow_acquired", time.perf_counter_ns()))
                ctx = NativeRequestContext(self, native, request, arrival_ns)
                ctx.initialization_landmarks = initialization_landmarks + ctx.initialization_landmarks
                try:
                    yield ctx
                finally:
                    ctx.close()
        finally:
            self.active = None

    def build_exact_dense_source(self, request, segment_id):
        from .v8_schema10_canonical import capture_exact_dense_source
        if self.active is not None:
            raise RuntimeError("independent canonical build must run after online request")
        self.check_deadline()
        return capture_exact_dense_source(self, request, segment_id)


class NativeRequestContext:
    evidence_origin = "real_cuda_execution"
    selection_backing_tier = "pinned_cpu"
    actual_repair_check_sunk_ms = 0.0

    def __init__(self, adapter, native, request, arrival_ns):
        self.adapter, self.native, self.request = adapter, native, request
        self.arrival_ns = arrival_ns
        self.segments = {s["segment_id"]: dict(s) for s in request["segments"]}
        from .v8_schema10_canonical import request_occurrences
        occurrences, targets, _ = request_occurrences(request)
        for sid, segment in self.segments.items():
            segment["prefix_occurrences"] = [asdict(o) for o in occurrences
                if o.provenance_position < targets[sid].provenance_position]
        self.cached_prefix_tokens = native.cached_prefix_tokens
        self.prefix_cache_mode = "native_exact_blocks"
        self.sampling_signature = {"temperature": 0, "max_new_tokens": int(request.get("max_new_tokens", 32))}
        self.execution_inventory = native_segment_inventory(self.segments,
            prompt_tokens=len(request["token_ids"]), cached_prefix_tokens=self.cached_prefix_tokens)
        self.probe_fallback_reason = "native_prefix_shadow_unavailable" if native.shadow_missing else None
        self.prepared, self.replica_reservations, self.supports, self.committed = {}, {}, {}, {}
        self.hot_replicas, self.hot_leases = {}, ExitStack()
        # Opt-in diagnostic cache.  Production requests never retain these
        # tensors unless an explicit hot-cache lease is held by the backend.
        self.frozen, self.selection_closed = {}, False
        self.repair_ratio = float(request.get("correctness_repair_ratio", .15))
        if self.repair_ratio not in {.15, 1.0}:
            raise ValueError("native fixed15 path only accepts .15 or correctness r=1")
        self.engine = None
        self._prefix_transfer_buffers = None
        self.workspace = None
        self.capture_reservation = None
        self.capture_collector = None
        self.canonical_exports = {}
        self.source_capture_v2 = None
        self.source_capture_workspace_v2 = None
        self.target_candidates_v2 = {}
        self.source_consumption_v2 = None
        self.p0_diagnostic_v2 = None
        self.p0_diagnostic_resources_v2 = None
        self.closed = self.finished = False
        self.generation = 1
        self._observation = {}
        self._setup_events = []
        self.finish_timing_landmarks = {}
        self.transfer_diagnostics = {}
        self.initialization_landmarks = [("request_inventory_built", time.perf_counter_ns())]
        with self._setup_span("native_inputs_and_sampling"):
            self._prepared_inputs = adapter.prepare(native.metadata(is_prompt=True))
        self.initialization_landmarks.append(("native_inputs_prepared", time.perf_counter_ns()))
        self.attention, self.sampling = self._prepared_inputs[2:4]

    @contextmanager
    def _setup_span(self, name):
        if not self.request.get("component_timing", False):
            yield
            return
        torch = self.adapter.torch
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        host = time.perf_counter_ns()
        begin.record()
        with torch.profiler.record_function("probekv.setup." + name):
            yield
        end.record()
        self._setup_events.append((name, host, time.perf_counter_ns(), begin, end))

    def setup_observations(self):
        if any(not end.query() for _, _, _, _, end in self._setup_events):
            raise RuntimeError("setup observations require completed events")
        return [dict(name=name, host_enqueue_ms=(stop-start)/1e6,
                     cuda_envelope_ms=float(begin.elapsed_time(end)),
                     timing_semantics="diagnostic_envelope_not_additive_ttft")
                for name, start, stop, begin, end in self._setup_events]

    @property
    def current_completed_depth(self):
        return self.engine.session.current_layer if self.engine is not None else 0

    @property
    def actual_sunk_ms(self):
        return (time.perf_counter_ns() - self.arrival_ns) / 1e6

    def assert_dispatch(self, depths):
        if tuple(depths) != self.adapter.depths:
            raise RuntimeError("native FAST/legacy checkpoint mismatch")

    def _begin(self):
        if self.engine is not None:
            return
        reuse_observation = self.request.get("reuse_current_kv_observation", True)
        if not isinstance(reuse_observation, bool):
            raise ValueError("reuse_current_kv_observation must be boolean")
        if self.probe_fallback_reason:
            raise RuntimeError("missing Prefix shadow must use native dense fallback")
        a = self.adapter
        n = len(self.request["token_ids"])
        defer_timing = bool(self.request.get("defer_layer_timing", False))
        if defer_timing:
            with self._setup_span("patch_capability_check"):
                if not _defer_layer_timing_capability(a.inner.probekv_advance_prefill):
                    raise RuntimeError("deferred timing requires the independently audited 0013 patch")
        a.inner.cache_fuse_metadata["probekv_defer_layer_timing"] = defer_timing
        a.inner.cache_fuse_metadata["probekv_host_position_validation"] = bool(
            self.request.get("host_position_validation", False))
        a.inner.cache_fuse_metadata["probekv_position_validation_audit"] = {}
        # Full request working composite, not just per-winner rows. This was
        # previously an unaccounted HBM allocation inside the engine.
        layer = a.inner.layers[0].self_attn
        # The engine retains both the prefix input shadow and the full working
        # composite. They are distinct allocations, even for an exact hit.
        size = (n + self.cached_prefix_tokens) * a.spec.num_layers * layer.num_kv_heads * layer.head_dim * 4
        self.workspace = a.hbm.reserve_batch(owner_request_id=self.request["request_id"],
            rows=(("request_working_kv", size, HBMReservationKind.COMMITTED_EXECUTION),))[0]
        try:
            self._enable_original_capture()
            prefix_cpu = self.native.prefix_shadow or ()
            if prefix_cpu:
                # Keep the established per-layer transfer contract.  The
                # engine retains layer-local views and the shadow store owns
                # pinned immutable inputs; a batched stack can introduce a
                # second allocation and erase the intended transfer benefit.
                with self._setup_span("prefix_shadow_to_gpu"):
                    if self.request.get("prefix_shadow_transfer_mode", "layerwise") == "batched":
                        # One pinned host staging tensor and one H2D per K/V
                        # stream. Unbind views preserve the per-layer API and
                        # exact BF16 values while avoiding 2*num_layers copy
                        # submissions. The base tensors stay alive through the
                        # returned views; no synchronization is introduced.
                        layers, rows = len(prefix_cpu), prefix_cpu[0][0].shape
                        k_host = a.torch.empty((layers, *rows), dtype=prefix_cpu[0][0].dtype,
                                                device="cpu", pin_memory=True)
                        v_host = a.torch.empty((layers, *rows), dtype=prefix_cpu[0][1].dtype,
                                                device="cpu", pin_memory=True)
                        for index, (key, value) in enumerate(prefix_cpu):
                            k_host[index].copy_(key)
                            v_host[index].copy_(value)
                        k_gpu = k_host.to(a.runner.device, non_blocking=True)
                        v_gpu = v_host.to(a.runner.device, non_blocking=True)
                        self._prefix_transfer_buffers = (k_host, v_host, k_gpu, v_gpu)
                        shadows = tuple((k_gpu[index], v_gpu[index]) for index in range(layers))
                        self.transfer_diagnostics["prefix_shadow_transfer_mode"] = "batched"
                    else:
                        shadows = tuple(tuple(t.to(a.runner.device, non_blocking=t.is_pinned())
                                              for t in pair) for pair in prefix_cpu)
                        self.transfer_diagnostics["prefix_shadow_transfer_mode"] = "layerwise"
            else:
                shadows = ()
            self.engine = CacheBlendV6OnlineEngine(inner_model=a.inner, model_spec=a.spec,
                source_loader=a.loader,
                prefetch_window=int(self.request.get("prefetch_window", 0)),
                kv_layout_mode=self.request.get("kv_layout_mode", "legacy"),
                contiguous_source_rows=bool(self.request.get("contiguous_source_rows", False)),
                component_timing=bool(self.request.get("component_timing", False)))
            self.engine.begin_prefill(model_signature=a.provenance["model_signature"],
                token_ids=tuple(self.request["token_ids"][self.cached_prefix_tokens:]),
                absolute_positions=tuple(range(self.cached_prefix_tokens, n)),
                exact_prefix_tokens=self.cached_prefix_tokens, exact_prefix_layers=shadows,
                attention_metadata=self.attention, working_kv=a.kv)
            self.engine.session.reuse_current_kv_observation = reuse_observation
        except Exception:
            a.torch.cuda.synchronize()
            self.engine = None
            self._prefix_transfer_buffers = None
            shadows = ()
            a.inner.old_kvs = [[None, None] for _ in range(a.spec.num_layers)]
            a.hbm.release(self.workspace.reservation_id)
            self.workspace = None
            raise

    def _enable_original_capture(self):
        reject_diagnostic_publication_v2(self)
        # Explicit preregistered capture task only: no surprise CFO work on
        # normal TTFT requests, no claim that extra materialization is free.
        if (not self.request.get("capture_original_full_prefill", False)
                or self.cached_prefix_tokens or self.capture_reservation is not None):
            return
        from .v8_schema10_canonical import request_occurrences
        _, _, ids = request_occurrences(self.request)
        a = self.adapter
        attn = a.inner.layers[0].self_attn
        size = len(ids) * a.spec.num_layers * attn.num_kv_heads * attn.head_dim * 4
        self.capture_reservation = a.hbm.reserve_batch(owner_request_id=self.request["request_id"],
            rows=(("original_full_prefill_capture", size, HBMReservationKind.COMMITTED_EXECUTION),))[0]
        a.inner.cache_fuse_metadata.update(collect=True, probekv_cfo_collector=None)

    def configure_target_capture_v2(self, *, target_ids, registry,
                                    authorization_domain, host_budget_bytes):
        """Reserve same-execution target capture before the first layer.

        Explicit P0 operation, not a user request flag or production readiness
        bypass. Native Prefix requires an independently verified proof adapter
        before this path can support it. No second forward is ever attempted.
        """
        reject_diagnostic_publication_v2(self)
        from .native_source_capture_v2 import NativeTargetCaptureV2
        if (self.finished or self.current_completed_depth or self.source_capture_v2 is not None
                or self.request.get('capture_original_full_prefill', False)):
            raise ValueError('target capture must be reserved from d0 without legacy whole capture')
        if self.cached_prefix_tokens or self.probe_fallback_reason:
            raise ValueError('v2 capture requires a qualified exact Prefix proof adapter for Prefix hits')
        ids=tuple(target_ids)
        if not ids or len(ids)!=len(set(ids)) or not set(ids)<=set(self.segments):
            raise ValueError('explicit distinct actual target Segment IDs required')
        a=self.adapter
        attn=a.inner.layers[0].self_attn
        capture=NativeTargetCaptureV2(request_id=self.request['request_id'],
            token_ids=self.request['token_ids'],model_signature=a.provenance['model_signature'],
            authorization_domain=authorization_domain,num_layers=a.spec.num_layers,
            targets={sid:self.segments[sid]['positions'] for sid in ids},
            selection_depths=a.spec.checkpoints,host_budget_bytes=host_budget_bytes,
            kv_heads=attn.num_kv_heads,head_dim=attn.head_dim,registry=registry)
        self.source_capture_v2=capture
        if capture._failed:
            return capture.audit()
        try:
            self.source_capture_workspace_v2=a.hbm.reserve_batch(owner_request_id=self.request['request_id'],
                rows=(('target_capture_packet_v2',capture.layer_packet_bytes,HBMReservationKind.COMMITTED_EXECUTION),))[0]
            self._begin()
            self.engine.session.provenance_capture=capture
        except (ValueError,MemoryError) as exc:
            capture.disable_capture('capture_reservation_failed:'+str(exc))
            if self.source_capture_workspace_v2:
                a.hbm.release(self.source_capture_workspace_v2.reservation_id)
                self.source_capture_workspace_v2=None
        return capture.audit()

    def bind_capture_parent_v2(self, parent):
        reject_diagnostic_publication_v2(self)
        if getattr(self, 'source_consumption_v2', None) is not None:
            return self.source_consumption_v2.bind_capture_parent(parent)
        if self.source_capture_v2 is None:
            raise RuntimeError('target capture must be reserved first')
        from .source_manifest_v2 import ParentSourceMetadata
        from .segment_capture_v2 import ManifestReference
        if not isinstance(parent, ParentSourceMetadata):
            raise ValueError('typed parent provenance required')
        capture = self.source_capture_v2
        store = self.adapter.store_provider()
        # A supplied source_id/G flag is not evidence. Bind to the exact frozen
        # Artifact being prepared while the logical/physical leases are held.
        with store.pool.mutation_lock:
            sids = [sid for sid, ticket in self.prepared.items()
                    if ticket.source_id == parent.source_id]
            if not sids:
                raise ValueError('parent must have an actual frozen preparation ticket')
            obj = store.objects[parent.source_id]
            for sid in sids:
                ticket = self.prepared[sid]
                row = store.pool._get(self.adapter.provenance['model_signature'],
                    self.segments[sid]['content_key'], parent.source_id)
                reservation = self.replica_reservations.get(sid)
                if (self.frozen.get(sid) != parent.source_id or reservation is None
                        or reservation.released or not row.runtime_available
                        or not store.pool.logical_lease_counts.get(parent.source_id)
                        or not any(replica.busy for replica in row.healthy_backing_replicas)
                        or ticket.transfer_failed or ticket.preparation_cancelled
                        or tuple(ticket.segment_positions) != tuple(self.segments[sid]['positions'])
                        or ticket.expected_artifact_digest != parent.artifact_digest
                        or row.canonical_source_state_digest != parent.artifact_digest):
                    raise ValueError('parent does not match leased prepared Artifact')
            # Old exact-only pool labels do not establish new v2 ancestry.
            # This metadata must be installed by a verified v2 publisher; until
            # that integration exists, legacy artifacts fail closed here.
            metadata = obj.metadata.get('source_provenance_v2')
            if not isinstance(metadata, dict):
                raise ValueError('parent lacks verified v2 publication provenance')
            try:
                reference = ManifestReference(**metadata['manifest_reference'])
                capture.registry.validate_capture_reference(reference,
                    model_signature=parent.model_signature,
                    authorization_domain=parent.authorization_domain,
                    target_token_ids=obj.metadata['token_ids'],
                    target_positions=metadata['birth_target_positions'])
                manifest = capture.registry.export_manifest(reference.manifest_id,
                    model_signature=parent.model_signature,
                    authorization_domain=parent.authorization_domain)
                record = next(o for o in manifest['occurrences']
                              if o['occurrence_id'] == reference.target_occurrence)
            except (KeyError, TypeError, StopIteration) as exc:
                raise ValueError('incomplete parent publication provenance') from exc
            if (record['proof']['origin'] != parent.origin.value
                    or record['proof']['generation'] != parent.generation
                    or record['proof_digest'] != metadata.get('proof_digest')):
                raise ValueError('parent origin/generation disagrees with executed manifest')
            capture.bind_parent(parent)

    def configure_source_consumption_v2(self, comparison_session):
        """Explicit diagnostic bridge; never convert the legacy exact pool."""
        reject_diagnostic_publication_v2(self)
        from .native_consumption_v2 import NativeV2ConsumptionSession
        if getattr(self, 'source_consumption_v2', None) is not None:
            raise RuntimeError('v2 Source consumption already configured')
        bridge = NativeV2ConsumptionSession(self, comparison_session)
        self.source_consumption_v2 = bridge
        return bridge.audit()

    def configure_cuda_comparison_v2(self, *, capacity_bytes):
        """Explicit P0 resource binding, not a Profile or GPU authorization."""
        from .cuda_comparison_v2 import CudaComparisonWorkspaceV2
        return CudaComparisonWorkspaceV2(self,capacity_bytes=capacity_bytes)

    def prepare_selected_v2(self, receipt):
        bridge = getattr(self, 'source_consumption_v2', None)
        if bridge is None:
            raise RuntimeError('issued comparison session must be configured first')
        return bridge.prepare_selected(receipt)

    def export_target_candidates_v2(self):
        reject_diagnostic_publication_v2(self)
        if not self.finished:
            raise RuntimeError('candidate export requires completed birth request')
        # Separate typed provenance, never feed this to the old DENSE_EXACT API.
        return dict(self.target_candidates_v2)

    def publish_target_candidates_v2(self, store, snapshot, scope_by_target):
        """Opt-in post-answer P0 SSD publication, never a legacy build fallback."""
        reject_diagnostic_publication_v2(self)
        from .native_publication_v2 import publish_target_candidates_v2
        return publish_target_candidates_v2(self, store, snapshot, scope_by_target)

    def publish_compared_targets_v2(self, comparison_session, receipts_by_target):
        """Post-answer publication with actual issued K-only comparison receipts."""
        reject_diagnostic_publication_v2(self)
        from .native_publication_v2 import publish_compared_targets_v2
        return publish_compared_targets_v2(self, comparison_session, receipts_by_target)

    def advance_to_depth(self, depth):
        self._begin()
        while self.current_completed_depth < depth:
            self.adapter.check_deadline()
            layer = self.current_completed_depth + 1
            # Layerwise source preparation is owned by the engine pipeline.
            # It submits the current block first and then schedules the next
            # H2D on the copy stream.  Do not prefetch or wait here: doing so
            # serializes the copy before the current compute and destroys the
            # intended load/compute overlap.  The engine's
            # _install_ready_source_rows() inserts the non-blocking CUDA event
            # dependency for the consumer layer.
            self.engine.advance_to_layer(layer)
            self.generation += 1

    def synchronize(self):
        self.adapter.torch.cuda.synchronize()

    def register_ready_hot_replicas(self):
        if getattr(self, 'source_consumption_v2', None) is not None:
            # V2 tickets are request-owned temporary GPU handles. They never
            # enter the legacy exact-only Replica namespace.
            return
        from .contracts import KVLocation
        pool = self.adapter.store_provider().pool
        model = self.adapter.provenance["model_signature"]
        for sid, ticket in self.prepared.items():
            if sid in self.hot_replicas or not ticket.fully_ready() or ticket.integrity_verification_pending:
                continue
            source = pool._get(model, self.segments[sid]["content_key"], ticket.source_id)
            backing = source.healthy_backing_replicas[0]
            with pool.mutation_lock:
                shared = next((pair for pair in self.hot_replicas.values() if pair[0].source_variant_id == ticket.source_id), None)
                if shared is not None:
                    self.hot_replicas[sid] = shared
                    continue
                replica = pool.attach_replica(model, self.segments[sid]["content_key"], ticket.source_id,
                    tier=KVLocation.GPU, locator_value="request-hot:" + self.request["request_id"] + ":" + sid,
                    layout_signature="layer-contiguous-bf16",
                    bytes_digest=ticket.destination_digest or "expected-copy:" + ticket.expected_artifact_digest,
                    size_bytes=ticket.requested_bytes, derived_from_replica_id=backing.replica_id, is_backing=False)
                self.hot_replicas[sid] = (source, replica)
                self.hot_leases.enter_context(pool.lease_replica(model, self.segments[sid]["content_key"], ticket.source_id, replica.replica_id))
            if self.request.get("retain_gpu_hot_cache"):
                self.adapter.hot_layer_cache[ticket.source_id] = dict(ticket.layer_tensors)

    def observe_current_k(self, sid, depth):
        if not self.execution_inventory[sid].comparison_eligible:
            raise RuntimeError("Prefix/tail cannot enter Source comparison")
        if depth not in self._observation:
            self._observation.clear()
            self._observation[depth] = self.engine.session.observe_pre_rope_k(depth)
        indices = {p: i for i, p in enumerate(self.engine.session.active_positions)}
        return self._observation[depth][[indices[p] for p in self.segments[sid]["positions"]]]

    def source_measurement_shape(self, sid, source_id, depth):
        bridge = getattr(self, 'source_consumption_v2', None)
        if bridge is not None:
            from .contracts import KVLocation
            v2_metadata = bridge.metadata_for_shape(sid, source_id)
            tier = KVLocation.SSD.value
        else:
            obj = self.adapter.store_provider().objects[source_id]
            tier = obj.tier.value
        a = self.adapter.inner.layers[0].self_attn
        n = len(self.segments[sid]["positions"])
        shape = {"prompt_tokens": len(self.request["token_ids"]), "prefix_tokens": self.cached_prefix_tokens,
            "positions": list(self.segments[sid]["positions"]), "completed_depth": depth,
            "first_reuse_layer": depth + 1, "num_layers": self.adapter.spec.num_layers,
            "dtype": "bfloat16", "kv_heads": a.num_kv_heads, "head_dim": a.head_dim,
            "tier": tier, "bytes": n * a.num_kv_heads * a.head_dim * self.adapter.spec.num_layers * 4,
            "layout": "pre_rope_k_raw_v", "repair_ratio": self.repair_ratio,
            "timing_scope": "source_local_boundary_future"}
        metric = getattr(self.adapter, "native_repair_metric", "normalized_v_legacy")
        if metric != "normalized_v_legacy":
            shape["repair_metric"] = metric
        if bridge is not None:
            # Old exact measurements must not silently support a new origin
            # policy merely because target tensor dimensions are the same.
            shape.update(source_origin=v2_metadata['origin'], source_generation=v2_metadata['generation'],
                         provenance_policy=bridge.store.config['policy'], source_contract='target_source_v2')
        return shape

    def prepare_winner(self, sid, source_id, layers, reservation):
        if getattr(self, 'source_consumption_v2', None) is not None:
            raise RuntimeError('v2 preparation requires an issued comparison receipt')
        self.frozen[sid] = source_id
        self.replica_reservations[sid] = reservation
        store = self.adapter.store_provider()
        obj = store.objects[source_id]
        row = store.pool._get(self.adapter.provenance["model_signature"], self.segments[sid]["content_key"], source_id)
        if reservation.released or not any(p.busy for p in row.healthy_backing_replicas):
            raise RuntimeError("transfer without physical backing lease and HBM reservation")
        resident_layers = (
            self.adapter.hot_layer_cache.get(source_id)
            if self.request.get("use_gpu_hot_cache") else None
        )
        ticket = self.engine.start_winner_prefetch(segment_id=sid, source_id=source_id,
            canonical_layers=layers, segment_positions=self.segments[sid]["positions"],
            expected_artifact_digest=row.canonical_source_state_digest,
            request_id=self.request["request_id"], replica_id=row.healthy_backing_replicas[0].replica_id,
            resident_layers=resident_layers)
        self.prepared[sid] = ticket
        self.generation += 1
        return ticket

    def finish_selection(self, frozen, prepared):
        if getattr(self, 'source_consumption_v2', None) is not None:
            self.source_consumption_v2.validate_selection_closure(frozen, prepared)
        self.frozen, self.selection_closed = dict(frozen), True
        self.generation += 1

    def ready_for_final_commit(self, prepared):
        if not self.selection_closed:
            raise RuntimeError("dense-clean selection closure required")
        if not prepared:
            return {}, digest_json([])
        start = time.perf_counter_ns()
        depth = self.current_completed_depth
        current_k, current_v = self.engine.session.observe_repair_check_pre_rope_kv(depth)
        local = {p: i for i, p in enumerate(self.engine.session.active_positions)}
        ready = {}
        for sid, ticket in prepared.items():
            if depth + 1 not in ticket.layer_events:
                self.engine.source_loader.prefetch_pending(ticket, depth + 1)
            ticket.layer_events[depth + 1].synchronize()
            positions = tuple(self.segments[sid]["positions"])
            # Winner repair metric is independent of Source-score trimming.
            # New manifests select normalized K/V explicitly. Missing metric
            # remains V-only solely for historical manifest compatibility.
            metric = getattr(self.adapter, "native_repair_metric", "normalized_v_legacy")
            if metric == "normalized_kv_deviation":
                from .source_policy_development import rank_winner_kv_positions
                source_k, source_v = ticket.layer_tensors[depth + 1]
                order = rank_winner_kv_positions(
                    current_k[[local[p] for p in positions]], source_k,
                    current_v[[local[p] for p in positions]], source_v,
                    positions)
            else:
                from .source_policy_development import rank_winner_v_positions
                order = rank_winner_v_positions(current_v[[local[p] for p in positions]],
                    ticket.layer_tensors[depth + 1][1], positions, metric=metric)
            count = min(len(positions), math.ceil(len(positions) * self.repair_ratio))
            support = tuple(sorted(order[:count]))
            self.supports[sid] = {l: support for l in range(depth + 1, self.adapter.spec.num_layers + 1)}
            ready[sid] = depth + 1
        self.actual_repair_check_sunk_ms += (time.perf_counter_ns() - start) / 1e6
        self.register_ready_hot_replicas()
        self.generation += 1
        return ready, digest_json(self.supports)

    def planner_snapshot(self, epoch):
        ready = {sid: [l for l, event in ticket.layer_events.items() if event.query()]
                 for sid, ticket in self.prepared.items()}
        return PlannerSnapshot(self.generation, 1, digest_json([self.native.sequence.seq_id,
                               self.generation, self.current_completed_depth, ready]),
                               epoch, self.adapter.costs.sha)

    def settle_preparation_for_replan(self):
        """Fence only existing winner copies after a readiness-snapshot race.

        This starts no transfer and changes no Source. The backend includes
        the wait in actual sunk time before its next admission attempt.
        """
        before = {sid: [l for l, event in ticket.layer_events.items() if event.query()]
                  for sid, ticket in self.prepared.items()}
        started = time.perf_counter_ns()
        for ticket in self.prepared.values():
            for event in ticket.layer_events.values():
                event.synchronize()
        self.register_ready_hot_replicas()
        after = {sid: [l for l, event in ticket.layer_events.items() if event.query()]
                 for sid, ticket in self.prepared.items()}
        return {"ready_layers_before": before, "ready_layers_after": after,
                "host_wait_ms": (time.perf_counter_ns() - started) / 1e6}

    def execution_shape(self):
        physical = {}
        for sid, source_id in self.frozen.items():
            shape = self.source_measurement_shape(sid, source_id, self.current_completed_depth)
            ticket = self.prepared.get(sid)
            physical[sid] = {k: shape[k] for k in ("tier", "bytes", "layout")}
            if "repair_metric" in shape:
                physical[sid]["repair_metric"] = shape["repair_metric"]
            for field in ('source_origin','source_generation','provenance_policy','source_contract'):
                if field in shape:
                    physical[sid][field] = shape[field]
            physical[sid]["ready_layers"] = [l for l in ticket.layer_events if ticket.layer_ready(l)] if ticket else []
            physical[sid]["copy_in_flight"] = bool(ticket and not ticket.fully_ready())
        return RequestExecutionShape(len(self.request["token_ids"]), self.cached_prefix_tokens,
            self.adapter.spec.num_layers, self.current_completed_depth,
            {sid: o.remaining_positions for sid, o in self.execution_inventory.items()}, self.supports,
            self.committed, physical, MeasuredRequestCostProvider.identity(self))

    def dense_fallback_joint_context(self):
        ids = tuple(self.segments)
        return JointTimelineContext(ids, (), tuple(s for s in ids if s not in self.committed),
            tuple(self.committed), {}, digest_json(self.supports), self.planner_snapshot(self.adapter.hbm.epoch).scheduler_snapshot_id)

    def commit_reuse(self, decision):
        decision.planner_snapshot.assert_current(self.planner_snapshot(self.adapter.hbm.epoch))
        if getattr(self, 'source_consumption_v2', None) is not None:
            for sid in decision.accepted_ready_segment_ids:
                self.source_consumption_v2.assert_can_commit(sid)
        for sid in decision.accepted_ready_segment_ids:
            # An original capture that becomes selective is ineligible. Stop
            # collecting it rather than promoting locally dense pieces.
            self.adapter.inner.cache_fuse_metadata.update(collect=False, probekv_cfo_collector=None)
            boundary = self.current_completed_depth + 1
            self.engine.commit_ready_segment(segment_id=sid, boundary=boundary,
                segment_positions=self.segments[sid]["positions"], repair_positions=self.supports[sid][boundary],
                scheduler_boundary=boundary)
            self.adapter.hbm.promote(self.replica_reservations[sid].reservation_id,
                expected=HBMReservationKind.WINNER_PREFETCH, target=HBMReservationKind.COMMITTED_EXECUTION)
            self.committed[sid] = boundary
        self.generation += 1

    def cancel_uncommitted_preparation(self):
        """Final online rejection: stop future copies, not in-flight ones."""
        cancelled = []
        for sid, ticket in self.prepared.items():
            if sid not in self.committed:
                cancelled.append({"segment_id": sid, **ticket.cancel_pending()})
        if cancelled:
            self.generation += 1
        # close() still fences all in-flight work before releasing leases.
        return cancelled

    def finish(self, on_first_token):
        if diagnostic_context_v2(self) is not None:
            raise ValueError('P0 mixed diagnostic requires its explicit recipe prefill endpoint')
        if self.finished:
            raise RuntimeError("request finish is not repeatable")
        a, torch = self.adapter, self.adapter.torch
        self.finish_timing_landmarks["finish_enter"] = time.perf_counter_ns()
        if self.engine is None:
            ids, pos = self._prepared_inputs[:2]
            a.inner.cache_fuse_metadata.update({"check": False, "collect": False, "probekv_cfo_collector": None})
            self._enable_original_capture()
            hidden = a.outer(input_ids=ids, positions=pos, kv_caches=a.kv, attn_metadata=self.attention)
        else:
            if (self.request.get("native_dense_continuation", False)
                    and not self.committed and self.capture_reservation is None):
                self._advance_native_dense_remaining()
            else:
                self.advance_to_depth(a.spec.num_layers)
            hidden = self.engine.finish_prefill()
            if self.engine.session.active_positions[-1] != len(self.request["token_ids"]) - 1:
                raise RuntimeError("native sampling lost the mandatory suffix row")
        self.finish_timing_landmarks["remaining_prefill_submitted"] = time.perf_counter_ns()
        if self.capture_reservation is not None and not self.committed and not self.cached_prefix_tokens:
            from .v8_schema10_canonical import export_original_full_prefill
            self.canonical_exports = export_original_full_prefill(a, self.request, self.capture_collector)
            # Prefix-on diagnostics must explicitly request their independent
            # shadow feature. Ordinary Source export does not warm Prefix.
            if self.request.get('publish_exact_prefix_shadow', False):
                from .v8_schema10_canonical import publish_exact_prefix_shadow
                shadow = publish_exact_prefix_shadow(a, self.request, origin='exact_dense_full_prefill')
                for row in self.canonical_exports.values():
                    row['capture_audit']['explicit_prefix_shadow'] = shadow
        result = self.finish_from_prefill_hidden(hidden, on_first_token)
        capture = getattr(self, 'source_capture_v2', None)
        if capture is not None:
            started=time.perf_counter_ns()
            # Completion/capture costs occur after the first-token endpoint but
            # remain part of total service time and must be reported separately.
            self.synchronize()
            self.target_candidates_v2=capture.finalize(request_completed=self.finished,
                                                       actual_token_ids=self.request['token_ids'])
            result['source_capture_v2']={**capture.audit(),
                'post_answer_finalize_host_ms':(time.perf_counter_ns()-started)/1e6,
                'candidate_ids':list(self.target_candidates_v2)}
        if getattr(self, 'source_consumption_v2', None) is not None:
            result['source_consumption_v2'] = self.source_consumption_v2.audit()
        return result

    def _advance_native_dense_remaining(self):
        """Continue current hidden/residual on native Prefix attention.

        Only legal before any selective commit. Preserve the original paged
        blocks and sampling metadata; never re-embed or replay earlier layers.
        Opt-in until the matched dense/teacher controls have passed on GPU.
        """
        a, session = self.adapter, self.engine.session
        expected = tuple(range(self.cached_prefix_tokens, len(self.request["token_ids"])))
        if (self.committed or session.commits or tuple(session.active_positions) != expected
                or session._pending_target_positions is not None or session._pending_reuse_commit):
            raise RuntimeError("native dense continuation requires unmodified complete active rows")
        # Shape alone cannot distinguish a composite from native paged storage.
        # Continue only against the exact cache blocks and attention metadata
        # allocated for this request; reject before submitting any layer.
        if (len(session.working_kv) != a.spec.num_layers
                or len(a.kv) != a.spec.num_layers
                or any(actual is not native for actual, native in zip(session.working_kv, a.kv))
                or session.attention_metadata is not self.attention):
            raise RuntimeError("native dense continuation requires original paged KV and attention ownership")
        positions = self._prepared_inputs[1]
        if len(positions) != len(expected) or session.hidden_states.shape[0] != len(expected):
            raise RuntimeError("native dense continuation input geometry mismatch")
        metadata = a.inner.cache_fuse_metadata
        metadata.update(probekv_resumable=False, check=False, collect=False,
                        probekv_cfo_collector=None, reuse_active=False)
        mask_digest = digest_json(expected)
        for index in range(session.current_layer, a.spec.num_layers):
            a.check_deadline()
            begin, end = (a.torch.cuda.Event(enable_timing=True) for _ in range(2))
            host_start = time.perf_counter_ns()
            begin.record()
            def native_layer():
                return a.inner.layers[index](positions, session.hidden_states,
                    session.working_kv[index], session.attention_metadata, session.residual,
                    0, metadata, (None, None))
            if getattr(session,'provenance_capture',None) is None:
                hidden, residual = native_layer()
            else:
                from .resumable_prefill import LayerAdvanceResult
                capture=session.provenance_capture
                capture.prepare_step(session,index+1,expected,expected)
                def captured_native_layer():
                    h,r=native_layer()
                    return LayerAdvanceResult(h,r,session.working_kv)
                try:
                    captured=capture.run_actual_layer(a.inner,
                        dict(layer=index+1,active_positions=expected,target_active_positions=expected),
                        captured_native_layer)
                except Exception:
                    capture.abort_execution('native_continuation_failed')
                    raise
                hidden,residual=captured.hidden_states,captured.residual
            end.record()
            session.hidden_states, session.residual = hidden, residual
            session.current_layer = index + 1
            session.pending_timing_events[index + 1] = (begin, end)
            session.layer_audit.append({"layer": index + 1, "active_before": expected,
                "active_after": expected, "gpu_ms": None,
                "host_ms": (time.perf_counter_ns() - host_start) / 1e6,
                "union_mask_digest": mask_digest,
                "runtime_debug": {"status": 0, "native_dense_continuation": True,
                    "prefix_active": bool(self.cached_prefix_tokens)}})
            self.generation += 1

    def finish_from_prefill_hidden(self, hidden, on_first_token):
        """Common sampling/decode endpoint, also used by isolated loop diagnostics.

        The caller must have actually executed this request's prefill against
        its native blocks. This is not a configuration-level admission bypass.
        """
        if self.finished:
            raise RuntimeError("request finish is not repeatable")
        diagnostic_v2 = diagnostic_context_v2(self)
        a, torch = self.adapter, self.adapter.torch
        # No decode call may append to the full-prefill CFO capture.
        a.inner.cache_fuse_metadata.update(collect=False, probekv_cfo_collector=None)
        exact_prefix_publish = not self.committed
        capture_v2 = getattr(self, 'source_capture_v2', None)
        consumption_v2 = getattr(self, 'source_consumption_v2', None)
        if diagnostic_v2 is not None:
            # A diagnostic marker only restricts provenance. Even a separately
            # injected exact-looking capture ledger cannot override it.
            exact_prefix_publish = False
            self.v2_prefix_publication_audit = {
                'exact_prefix_publication_allowed': False,
                'reason': 'p0_explicit_mixed_reference_not_exact',
            }
        elif capture_v2 is not None:
            # P0 diagnostics can call the engine/session directly. A missing
            # context commit entry must never relabel its mixed paged KV as an
            # exact native Prefix. Require a complete full-request G0 proof,
            # not merely a successful target capture or a caller's boolean.
            exact_prefix_publish = False
            reason = 'unverified_v2_whole_request'
            try:
                from .source_provenance_v2 import RequestExecutionLedger, SourceOrigin
                from .source_manifest_v2 import request_input_digest
                session = self.engine.session if self.engine is not None else None
                ledger = capture_v2.ledger
                tokens = tuple(self.request['token_ids'])
                positions = tuple(range(len(tokens)))
                if self.committed or session is None or session.commits:
                    reason = 'context_or_session_reuse_commit'
                elif getattr(capture_v2, '_failed', True) is not False:
                    reason = 'capture_execution_proof_unavailable'
                elif self.cached_prefix_tokens or session.exact_prefix_tokens:
                    reason = 'v2_exact_prefix_proof_adapter_unqualified'
                elif (not isinstance(ledger, RequestExecutionLedger)
                      or ledger.request_id != self.request['request_id']
                      or ledger.model_signature != a.provenance['model_signature']
                      or ledger.num_layers != a.spec.num_layers
                      or ledger.input_digest != request_input_digest(tokens, positions)
                      or tuple(session.token_ids) != tokens
                      or tuple(session.absolute_positions) != positions
                      or session.model_signature != ledger.model_signature):
                    reason = 'v2_request_identity_mismatch'
                else:
                    proof = ledger.target_proof(positions)
                    exact_prefix_publish = (proof.origin is SourceOrigin.EXACT
                                            and proof.generation == 0)
                    reason = ('complete_known_G0_request' if exact_prefix_publish
                              else 'whole_request_not_complete_known_G0')
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                # Missing or malformed provenance declines Prefix publication,
                # not the successfully computed answer. CUDA/model failures
                # from the actual native finish below are never swallowed.
                reason = 'invalid_v2_execution_proof:' + type(exc).__name__
            self.v2_prefix_publication_audit = {
                'exact_prefix_publication_allowed': exact_prefix_publish,
                'reason': reason,
            }
        elif consumption_v2 is not None:
            # Preparation-only diagnostics lack a complete execution ledger.
            # Neither a missing context commit nor a dense-looking endpoint
            # proves that imported historical KV is a reusable exact Prefix.
            exact_prefix_publish = False
            self.v2_prefix_publication_audit = {
                'exact_prefix_publication_allowed': False,
                'reason': 'v2_consumption_without_whole_request_proof',
            }
        self.native.finish_prefill(exact_dense=exact_prefix_publish)
        self.finish_timing_landmarks["native_prefill_bookkeeping_done"] = time.perf_counter_ns()
        selected = self.sampling.selected_token_indices.clone()
        try:
            self.sampling.selected_token_indices[0] = hidden.shape[0] - 1
            logits = a.outer.compute_logits(hidden, self.sampling)
        finally:
            self.sampling.selected_token_indices.copy_(selected)
        self.finish_timing_landmarks["logits_submitted"] = time.perf_counter_ns()
        predicted = [int(logits.argmax().item())]
        self.finish_timing_landmarks["first_token_host_ready"] = time.perf_counter_ns()
        on_first_token()
        self.logit_trace = [logits.detach().float().cpu()] if self.request.get("capture_logits") else []
        decode_recorder = None
        if self.request.get("capture_logits"):
            from .p0_decode_evidence_v2 import DecodeInputRecorderV2
            decode_recorder = DecodeInputRecorderV2(
                {**self.request, 'max_new_tokens': self.sampling_signature['max_new_tokens']},
                cached_prefix_tokens=self.cached_prefix_tokens)
            decode_recorder.append(predicted[0])
        teachers = self.request.get("teacher_token_ids", ())
        teacher_forced = "teacher_token_ids" in self.request
        eos = a.llm.get_tokenizer().eos_token_id
        for _ in range(1, self.sampling_signature["max_new_tokens"]):
            if not teacher_forced and predicted[-1] == eos:
                break
            if not teacher_forced and self.request.get("answer_boundary_contract"):
                from .v8_schema10_qa import bounded_answer
                _, matched_stop = bounded_answer(a.llm.get_tokenizer().decode(
                    predicted, skip_special_tokens=True), self.request)
                if matched_stop is not None:
                    break
            a.check_deadline()
            feed_token = int(teachers[len(predicted) - 1]) if teacher_forced else predicted[-1]
            metadata = self.native.append_for_decode(feed_token)
            ids, pos, attention, sample = a.prepare(metadata)[:4]
            hidden = a.outer(input_ids=ids, positions=pos, kv_caches=a.kv, attn_metadata=attention)
            logits = a.outer.compute_logits(hidden, sample)
            predicted.append(int(logits.argmax().item()))
            if self.request.get("capture_logits"):
                self.logit_trace.append(logits.detach().float().cpu())
            self.native.finish_decode_step()
            if decode_recorder is not None:
                decode_recorder.append(predicted[-1], fed_token=feed_token)
        self.decode_input_trace_v2 = (decode_recorder.finish(logit_rows=len(self.logit_trace))
                                     if decode_recorder is not None else None)
        self.finished = True
        if self.engine:
            self.engine.session.resolve_completed_layer_timings()
        origin = "selective_reuse" if self.committed else "native_prefix_dense_remaining" if self.cached_prefix_tokens else "exact_dense_full_prefill"
        warm_exact = not self.committed
        v2_audit = {}
        if self.decode_input_trace_v2 is not None:
            v2_audit['decode_input_trace_v2'] = self.decode_input_trace_v2
        if capture_v2 is not None or consumption_v2 is not None:
            committed = bool(self.committed or (self.engine and self.engine.session.commits))
            origin = ('exact_dense_full_prefill' if exact_prefix_publish else
                      'selective_reuse' if committed else 'unverified_v2_execution')
            warm_exact = exact_prefix_publish
            v2_audit['v2_prefix_publication_audit'] = dict(self.v2_prefix_publication_audit)
        if diagnostic_v2 is not None:
            origin = 'p0_' + diagnostic_v2.kind
            warm_exact = False
            v2_audit['p0_diagnostic_context_v2'] = diagnostic_v2.audit()
            v2_audit['v2_prefix_publication_audit'] = dict(self.v2_prefix_publication_audit)
        # Only dense requests rebuild exact Prefix state during paired replay.
        if warm_exact and not teacher_forced:
            a.warm_history.append(self.request)
        if teacher_forced:
            return {"token_ids": predicted, "answer": None, "quality_passed": None,
                "qa_evidence": None, "generation_mode": "teacher_forced_logit_diagnostic", **v2_audit,
                "whole_request_origin": origin, "cached_prefix_tokens": self.cached_prefix_tokens,
                "prefix_shadow_audit": getattr(self.native, "shadow_lookup_audit", {}),
                "layer_audit": self.engine.session.layer_audit if self.engine else [],
                "overlap_trace": self.engine.overlap_trace() if self.engine else []}
        from .v8_schema10_qa import answer_evidence
        evidence = answer_evidence(predicted, tokenizer=a.llm.get_tokenizer(), request=self.request)
        return {**evidence, **v2_audit, "whole_request_origin": origin,
                "position_validation_audit": (dict(a.inner.cache_fuse_metadata.get("probekv_position_validation_audit", {}))
                                              if self.engine else {}),
                "cached_prefix_tokens": self.cached_prefix_tokens,
                "prefix_shadow_audit": getattr(self.native, "shadow_lookup_audit", {}),
                "layer_audit": self.engine.session.layer_audit if self.engine else [],
                "overlap_trace": self.engine.overlap_trace() if self.engine else []}

    def export_exact_dense(self):
        # Ordinary requests do not run the expensive full attention/CFO capture.
        # Independent builds are admitted and measured after request completion.
        if (diagnostic_context_v2(self) is not None or self.committed or self.cached_prefix_tokens
                or getattr(self,'source_capture_v2',None) is not None
                or getattr(self,'source_consumption_v2',None) is not None):
            return {}
        return self.canonical_exports

    def deferred_canonical_builders(self):
        if (diagnostic_context_v2(self) is not None or getattr(self,'source_capture_v2',None) is not None
                or getattr(self,'source_consumption_v2',None) is not None):
            return {}  # Never fill missing early v2 capture with another forward.
        return {sid: (lambda sid=sid: self.adapter.build_exact_dense_source(self.request, sid))
                for sid, owner in self.execution_inventory.items() if owner.comparison_eligible}

    def materialization_metadata(self):
        if (diagnostic_context_v2(self) is not None or getattr(self,'source_capture_v2',None) is not None
                or getattr(self,'source_consumption_v2',None) is not None):
            return {}  # Mixed candidates require their own atomic publisher.
        result = {}
        for sid, s in self.segments.items():
            positions = tuple(s["positions"])
            query = {"prompt_tokens": len(self.request["token_ids"]), "positions": list(positions),
                     "num_layers": self.adapter.spec.num_layers, "origin": "current_request_capture" if sid in self.canonical_exports
                     else "independent_exact_dense_full_prefill"}
            category = "canonical_capture_write" if sid in self.canonical_exports else "canonical_build_and_write"
            row = self.adapter.costs._lookup(category, query)
            result[sid] = {"identity": SourceVariantIdentity(s["content_key"], digest_json(self.request["token_ids"][:positions[0]]),
                digest_json(positions), self.request["request_id"] + ":" + sid, self.adapter.provenance["model_signature"]),
                "estimated_materialization_ms": max(row["samples_ms"]) if row else None,
                "source_metadata": None}
        return result

    def close(self):
        if not self.closed:
            # If a CUDA fence fails, keep reservations/replicas quarantined.
            # The enclosing backend is poisoned and cannot admit another job.
            bridge = getattr(self, 'source_consumption_v2', None)
            diagnostic_resources = getattr(self, 'p0_diagnostic_resources_v2', None)
            try:
                self.synchronize()
            except Exception:
                if diagnostic_resources is not None:
                    diagnostic_resources.quarantined = True
                if bridge is not None:
                    bridge.quarantined = True
                comparison_workspace = getattr(self, 'comparison_workspace_v2', None)
                if comparison_workspace is not None:
                    comparison_workspace.lease.quarantined = True
                raise
            if diagnostic_resources is not None:
                diagnostic_resources.close_after_fence()
            if bridge is not None:
                bridge.close(already_fenced=True)
            comparison_workspace = getattr(self, 'comparison_workspace_v2', None)
            if comparison_workspace is not None:
                # Cached K/V projections may alias a whole QKV allocation;
                # clear them only after their final GPU consumers complete.
                comparison_workspace.close()
            if getattr(self,'source_capture_v2',None) is not None:
                self.source_capture_v2.close()
            self.hot_leases.close()
            if bridge is None and self.hot_replicas:
                pool = self.adapter.store_provider().pool
                with pool.mutation_lock:
                    unique = {replica.replica_id: (source, replica) for source, replica in self.hot_replicas.values()}
                    for source, replica in unique.values():
                        pool._delete_replica(source, replica, "single_request_execution_complete")
            self.engine = None
            self._prefix_transfer_buffers = None
            self.prepared.clear()
            self._observation.clear()
            self.adapter.inner.cache_fuse_metadata.pop("probekv_position_workspace", None)
            self.adapter.inner.old_kvs = [[None, None] for _ in range(self.adapter.spec.num_layers)]
            for block in self.adapter.inner.layers:
                block.self_attn.hack_kv = []
            self.adapter.inner.cache_fuse_metadata.update({"check": False, "probekv_resumable": False,
                "collect": False, "probekv_cfo_collector": None, "exact_prefix_tokens": 0})
            if self.workspace:
                self.adapter.hbm.release(self.workspace.reservation_id)
            if self.capture_reservation:
                self.adapter.hbm.release(self.capture_reservation.reservation_id)
            if getattr(self,'source_capture_workspace_v2',None):
                self.adapter.hbm.release(self.source_capture_workspace_v2.reservation_id)
            self.closed = True


class FastNativeOnlineAdapter(NativeOnlineAdapter):
    """Candidate shallow dispatch. Does not certify d1/d2 policy quality."""
    def __init__(self, **kwargs):
        if kwargs.get("selection_path") not in {"d1_only", "d1_d2_rescue"}:
            raise ValueError("FAST adapter cannot impersonate legacy")
        super().__init__(**kwargs)


class LegacyNativeOnlineAdapter(NativeOnlineAdapter):
    """Dense-clean full-checkpoint dispatch; no forced admission switch.

    The shared native primitive advances every Transformer block between the
    legacy checkpoints. This is the schema10 single-request barrier adapter,
    not a claim that all historical A/C deployments have been requalified.
    """
    def __init__(self, **kwargs):
        if kwargs.get("selection_path") != "legacy_multicheckpoint":
            raise ValueError("legacy adapter requires the complete checkpoint policy")
        super().__init__(**kwargs)
        if self.depths != tuple(self.spec.checkpoints):
            raise RuntimeError("incomplete legacy depth implementation")
