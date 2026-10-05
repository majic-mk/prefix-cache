"""Isolated fixed-recipe sparse arm for P0 mixed-context correctness.

This uses the existing CacheBlend resumable engine, not the full-query reference
hook. It confers no production admission and never publishes a Source. CPU tests
exercise the engine protocol; only an audited model/GPU run can qualify T21.
"""
from __future__ import annotations

import hashlib

from .native_capture_hook_v2 import CurrentLayerCapture
from .v8_schema10_staging import PhysicalLayerwiseSourceLoader


class _ObservedRemoval:
    def __init__(self, handle, observer):
        self.handle, self.observer = handle, observer

    def remove(self):
        try:
            self.handle.remove()
        except BaseException as error:
            self.observer.cleanup_failure = type(error).__name__+': '+str(error)
            raise


def execute_mixed_sparse_control(context, store, snapshot, job, *,
                                 host_reference_bytes, cuda_reference_bytes):
    from .p0_mixed_control_v2 import _execute_mixed_control
    return _execute_mixed_control(context, store, snapshot, job,
        host_reference_bytes=host_reference_bytes,
        cuda_reference_bytes=cuda_reference_bytes, sparse=True)


class SparseMixedExecutionObserver:
    """Observe actual QKV and self-attention calls of each completed block.

    At the first selective boundary QKV still projects ``active_before``;
    attention and the rest of the block retain ``active_after``. Later blocks
    project only retained queries. Recording both is essential for T21.
    """
    def __init__(self, context, engine, source_layers, plan, job, *,
                 host_target_bytes, transient_device_bytes, target_r1_layers=()):
        self.context, self.engine = context, engine
        self.source_layers = source_layers
        self.plan, self.job = plan, job
        self.target_r1 = job.get('target_execution','FULL_ALL_LAYERS') == 'R1_ALL_LAYERS'
        self.target_r1_layers = target_r1_layers
        self.failure = self.cleanup_failure = None
        self._current_capture = self._attention_handle = self._kernel_handle = None
        self._pending_step = None
        self._completed = self._closed = self._released = False
        self._trace, self._targets = [], []
        self._target_digest = None
        self._attention_outputs = []
        self._r1_writeback_calls = 0
        self.layer_count = context.adapter.spec.num_layers
        self.token_count = len(context.request['token_ids'])
        attn = context.adapter.inner.layers[0].self_attn
        self.geometry = (attn.num_kv_heads, attn.head_dim)
        self.layer_geometry = [tuple(getattr(block.self_attn,name) for name in
            ('q_size','kv_size','num_heads','num_kv_heads','head_dim'))
            for block in context.adapter.inner.layers]
        self.target_layer_bytes = 4 * len(plan['target_positions']) * attn.kv_size
        self.working_bytes = 4 * self.layer_count * self.token_count * attn.kv_size
        # Per-layer compact target copies and finite-test scratch. Native model
        # activations have their own allocator budget; the diagnostic adds no
        # second full-QKV clone. All Source tensors are reserved by the owner.
        self.transient_bytes = self.working_bytes + 2*self.target_layer_bytes + len(plan['target_positions'])*attn.kv_size + 1
        if (host_target_bytes < self.layer_count*self.target_layer_bytes
                or transient_device_bytes < self.transient_bytes):
            raise MemoryError('sparse diagnostic working KV/target capture exceeds explicit bound')

    def prepare_step(self, session, layer, before, after):
        if self._closed or self._pending_step is not None or layer != len(self._trace)+1:
            raise RuntimeError('sparse diagnostic requires one ordered actual block')
        expected = tuple(range(self.token_count))
        if layer >= self.job['first_reuse_layer']:
            repair = set(self.plan['repair_positions_by_layer'][layer])
            inactive = set(self.plan['upstream_positions']) - repair
            expected = tuple(p for p in expected if p not in inactive)
        if (tuple(after) != expected or not set(self.plan['target_positions']) <= set(before)
                or not set(self.plan['target_positions']) <= set(after)
                or after[-1] != self.token_count-1):
            raise ValueError('sparse diagnostic dropped target/suffix or used a different mask')
        self._pending_step = (layer, tuple(before), tuple(after))

    def _attention_output(self, module, args, output):
        import torch
        tensor = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(tensor) or tensor.ndim != 2:
            raise ValueError('actual native self-attention must return 2D query rows')
        self._attention_outputs.append(int(tensor.shape[0]))

    def _target_r1_writeback(self, module, args, output):
        """Observe post-RoPE inputs to the actual attention backend.

        After that call, all target rows in working composite must equal its
        freshly projected current K/V, not the target historical ticket.
        This hook makes no new model/RoPE call and allocates no full tensor.
        """
        import torch
        layer,before,after = self._pending_step
        if len(args) < 3:
            raise ValueError('target r1 needs actual attention query/key/value arguments')
        _, key,value = args[:3]
        kv_size = self.geometry[0]*self.geometry[1]
        if any(not torch.is_tensor(t) or t.dtype != torch.bfloat16
               or t.shape[0] != len(before) or t.numel() != len(before)*kv_size
               for t in (key,value)):
            raise ValueError('target r1 attention inputs differ from active native KV geometry')
        for absolute in self.plan['target_positions']:
            local=before.index(absolute)
            for current,working in zip((key,value),self.context.adapter.inner.old_kvs[layer-1]):
                if not torch.equal(working[absolute],current[local].reshape(self.geometry)):
                    raise ValueError('target r1 failed actual current K/V overwrite after attention')
        self._r1_writeback_calls += 1

    def run_actual_layer(self, inner, arguments, run):
        import torch
        step = self._pending_step
        if step is None or inner is not self.context.adapter.inner:
            raise RuntimeError('unprepared or foreign native sparse execution')
        layer, before, after = step
        if (arguments['layer'] != layer or tuple(arguments['active_positions']) != before
                or tuple(arguments['target_active_positions']) != after):
            raise ValueError('actual layer arguments differ from frozen sparse step')
        block = inner.layers[layer-1]
        self.context.adapter.check_deadline()
        repair = (self.plan['upstream_positions'] if layer < self.job['first_reuse_layer']
                  else self.plan['repair_positions_by_layer'][layer])
        historical = tuple(p for p in self.plan['upstream_positions'] if p not in set(repair))
        # The real engine's per-layer scatter has completed on the model stream
        # before this callback. Read the *working* rows before in-place RoPE.
        if historical:
            ticket = self.engine.tickets[self.job['upstream_segment_id']]
            if layer not in ticket.installed_layers:
                raise RuntimeError('historical Source rows were not installed by the engine')
            for absolute in historical:
                local = absolute-self.plan['upstream_positions'][0]
                for current, source in zip(inner.old_kvs[layer-1], self.source_layers[layer-1]):
                    if not torch.equal(current[absolute], source[local]):
                        raise ValueError('actual working historical K/V differs before RoPE')
        r1_active = self.target_r1 and layer >= self.job['first_reuse_layer']
        if r1_active:
            sid=self.job['target_id']
            commit=self.engine.session.commits.get(sid)
            ticket=self.engine.tickets.get(sid)
            if (commit is None or ticket is None or commit.source_id != self.job['target_source']['source_id']
                    or commit.repair_positions != self.plan['target_positions']
                    or commit.boundary != self.job['first_reuse_layer']
                    or layer not in ticket.installed_layers):
                raise ValueError('target r1 lacks actual all-row target ticket commit/install')
            for local,absolute in enumerate(self.plan['target_positions']):
                for current,source in zip(inner.old_kvs[layer-1],self.target_r1_layers[layer-1]):
                    if not torch.equal(current[absolute],source[local]):
                        raise ValueError('target r1 historical ticket was not actually installed')
        self._attention_outputs = []
        self._r1_writeback_calls = 0
        capture = CurrentLayerCapture(block.self_attn, layer_1based=layer,
            projection_positions=before, target_positions=self.plan['target_positions'],
            max_target_bytes=self.target_layer_bytes)
        self._current_capture = capture
        try:
            self._attention_handle = block.self_attn.register_forward_hook(self._attention_output)
            if r1_active:
                kernel=getattr(block.self_attn,'attn',None)
                if not callable(getattr(kernel,'register_forward_hook',None)):
                    raise ValueError('target r1 requires hookable actual attention backend')
                self._kernel_handle=kernel.register_forward_hook(self._target_r1_writeback)
            with capture:
                if capture._handle is not None:
                    capture._handle = _ObservedRemoval(capture._handle, self)
                result = run()
            packet = capture.packet_after_block()
            if packet is None:
                raise RuntimeError('actual sparse target projection missing: '+str(capture.failure))
            if self._attention_outputs != [len(after)]:
                raise ValueError('native self-attention did not execute the declared sparse query rows')
            if r1_active and self._r1_writeback_calls != 1:
                raise ValueError('target r1 lacks one actual current-KV backend writeback')
            if int(result.hidden_states.shape[0]) != len(after):
                raise ValueError('completed block retained a different query row count')
            debug = dict(result.runtime_debug)
            expected_status = 0 if layer < self.job['first_reuse_layer'] else (1 if layer == self.job['first_reuse_layer'] else 2)
            if (type(debug.get('status')) is not int or debug['status'] != expected_status
                    or debug.get('dense_full_repair') is not False):
                raise ValueError('native runtime status does not prove selective upstream execution')
            self.context.synchronize()
            tensors = tuple(t.detach().to(device='cpu', copy=True).contiguous()
                            for t in (packet.key, packet.value))
            if any(t.dtype != torch.bfloat16 or not bool(torch.isfinite(t).all()) for t in tensors):
                raise ValueError('nonfinite/non-BF16 sparse target capture')
            self._targets.append(tensors)
            self._trace.append(dict(layer_1based=layer, projected_positions=list(before),
                attention_query_positions=list(after), target_positions=list(self.plan['target_positions']),
                historical_kv_positions=list(historical), repair_positions=list(repair),
                target_full_projection=True, completed_block=True,
                actual_qkv_invocations=capture.invocation_count,
                actual_attention_output_rows=self._attention_outputs[0], native_runtime_status=debug,
                target_r1_commit_active=r1_active,
                target_r1_repair_positions=list(self.plan['target_positions']) if r1_active else [],
                target_r1_source_installed=r1_active,
                target_r1_current_kv_writeback_verified=r1_active and self._r1_writeback_calls==1))
            self._pending_step = None
            self._current_capture = None
            return result
        except BaseException as error:
            self.failure = type(error).__name__+': '+str(error)
            # CurrentLayerCapture retains no successful handle on exit; a remove
            # error is nevertheless quarantine-worthy, never cleanup success.
            if capture._handle is not None or not capture._closed:
                self.cleanup_failure = self.failure
            raise
        finally:
            cleanup_error=None
            for name in ('_attention_handle','_kernel_handle'):
                handle=getattr(self,name)
                if handle is None:
                    continue
                try:
                    handle.remove()
                except BaseException as error:
                    self.cleanup_failure = type(error).__name__+': '+str(error)
                    cleanup_error=cleanup_error or error
                else:
                    setattr(self,name,None)
            if cleanup_error is not None:
                raise cleanup_error

    def abort_execution(self, reason):
        self.failure = self.failure or str(reason)
        self._completed = False
        self._target_digest = None

    def finish(self):
        if (self.failure is not None or self.cleanup_failure is not None or self._pending_step is not None
                or len(self._trace) != self.layer_count
                or not any(row['historical_kv_positions'] for row in self._trace)):
            raise RuntimeError('incomplete actual sparse mixed execution')
        import torch
        digest = hashlib.sha256()
        for layer,pair in enumerate(self._targets,1):
            digest.update(str((layer,tuple(pair[0].shape),'BF16 pre-RoPE')).encode('ascii'))
            for tensor in pair:
                digest.update(memoryview(tensor.view(torch.uint8).numpy()).cast('B'))
        self._target_digest = digest.hexdigest()
        self._completed = True
        self._closed = True

    def release_device_references(self, *, caller_fenced):
        if caller_fenced is not True or self.cleanup_failure is not None:
            raise RuntimeError('sparse resources require successful completion fence and hook cleanup')
        # Failure may occur before finish(), but no hook can survive close.
        if self._attention_handle is not None or self._kernel_handle is not None:
            raise RuntimeError('sparse diagnostic still owns an attached attention hook')
        if self._current_capture is not None and self._current_capture._handle is not None:
            raise RuntimeError('sparse diagnostic still owns an attached projection hook')
        self._current_capture = None
        inner = self.context.adapter.inner
        inner.old_kvs = [[None,None] for _ in range(self.layer_count)]
        for block in inner.layers:
            block.self_attn.hack_kv = []
        self.engine._composite_old_kvs = []
        self.engine.tickets.clear()
        if self.engine.session is not None:
            self.engine.session.source_handles.clear()
            self.engine.session.provenance_capture = None
        self.source_layers = ()
        self.target_r1_layers = ()
        self._released = self._closed = True
        if not self._completed:
            self._targets.clear()

    @property
    def target_layers(self):
        if not self._completed or self.failure is not None:
            raise RuntimeError('target layers require completed sparse execution')
        return tuple(self._targets)

    def audit(self):
        return dict(kind='p0_actual_cacheblend_sparse_execution', completed=self._completed,
            failure=self.failure, cleanup_failure=self.cleanup_failure,
            device_references_released=self._released, executed_layers=list(self._trace),
            historical_kv_rows_replaced=sum(len(row['historical_kv_positions']) for row in self._trace),
            target_logical_digest=self._target_digest,
            geometry=[list(g) for g in self.layer_geometry], kv_geometry=list(self.geometry),
            projection_counts=[row['actual_qkv_invocations'] for row in self._trace],
            working_kv_device_bytes=self.working_bytes, transient_device_bytes_upper=self.transient_bytes,
            target_owned_host_bytes=len(self._targets)*self.target_layer_bytes,
            parent_owned_kv_bytes=0, prefix_shadow_bytes=0,
            target_r1_endpoint_exercised=self.target_r1 and self._completed,
            publication_allowed=False, native_runtime_qualified=False, P1_execution_allowed=False,
            paper_evidence=False)


def run_sparse_prefill(context, owner, plan, row, job, host_target_bytes, transient_device_bytes):
    """Shared bounded lifecycle calls this only after immutable Source prep."""
    from .cacheblend_v6_online_engine import CacheBlendV6OnlineEngine
    c, a = context, context.adapter
    if c.engine is not None:
        raise RuntimeError('sparse diagnostic cannot reuse a previous engine')
    # A dedicated callback validates the v2 snapshot, physical lease and HBM
    # reservation. Do not mutate the shared production loader's authorization.
    loader = PhysicalLayerwiseSourceLoader(a.loader.pool, authorize=owner.authorize,
        integrity_mode=a.loader.integrity_mode, device=a.loader.device)
    engine = CacheBlendV6OnlineEngine(inner_model=a.inner, model_spec=a.spec,
        source_loader=loader, prefetch_window=0, kv_layout_mode='legacy',
        contiguous_source_rows=False, component_timing=False)
    observer = SparseMixedExecutionObserver(c,engine,owner.source_layers,plan,job,
        host_target_bytes=host_target_bytes,transient_device_bytes=transient_device_bytes,
        target_r1_layers=getattr(owner,'target_r1_layers',()))
    owner.hooks = observer
    c.engine = engine
    a.inner.cache_fuse_metadata.update(check=False,collect=False,probekv_cfo_collector=None,
        probekv_defer_layer_timing=False,probekv_host_position_validation=False,
        probekv_resumable=False,reuse_active=False,dense_full_repair_endpoint=False,exact_prefix_tokens=0)
    try:
        session = engine.begin_prefill(model_signature=a.provenance['model_signature'],
            token_ids=tuple(c.request['token_ids']),absolute_positions=tuple(range(len(c.request['token_ids']))),
            exact_prefix_tokens=0,exact_prefix_layers=(),attention_metadata=c.attention,working_kv=a.kv)
        session.provenance_capture = observer
        ticket = engine.start_winner_prefetch(segment_id=job['upstream_segment_id'],
            source_id=row['source_id'],canonical_layers=owner.source_layers,
            segment_positions=plan['upstream_positions'],expected_artifact_digest=row['artifact_digest'],
            request_id=c.request['request_id'],replica_id='p0-diagnostic-source',
            resident_layers={layer:pair for layer,pair in enumerate(owner.source_layers,1)})
        ticket.wait_all(loader)
        if observer.target_r1:
            target_ticket=engine.start_winner_prefetch(segment_id=job['target_id'],
                source_id=owner.target_r1_row['source_id'],canonical_layers=owner.target_r1_layers,
                segment_positions=plan['target_positions'],
                expected_artifact_digest=owner.target_r1_row['artifact_digest'],
                request_id=c.request['request_id'],replica_id='p0-diagnostic-target-r1',
                resident_layers={layer:pair for layer,pair in enumerate(owner.target_r1_layers,1)})
            target_ticket.wait_all(loader)
        for layer in range(1,a.spec.num_layers+1):
            a.check_deadline()
            if layer == job['first_reuse_layer']:
                engine.commit_ready_segment(segment_id=job['upstream_segment_id'],boundary=layer,
                    segment_positions=plan['upstream_positions'],
                    repair_positions=plan['repair_positions_by_layer'][layer],scheduler_boundary=layer)
                if observer.target_r1:
                    engine.commit_ready_segment(segment_id=job['target_id'],boundary=layer,
                        segment_positions=plan['target_positions'],repair_positions=plan['target_positions'],
                        scheduler_boundary=layer)
            elif layer > job['first_reuse_layer']:
                previous = session.current_repair_positions_by_segment[job['upstream_segment_id']]
                if tuple(previous) != plan['repair_positions_by_layer'][layer]:
                    raise ValueError('fixed15 recipe unexpectedly changed repair support')
            engine.advance_to_layer(layer)
        observer.finish()
        hidden = engine.finish_prefill()
        if session.active_positions[-1] != len(c.request['token_ids'])-1:
            raise RuntimeError('sparse execution lost mandatory suffix')
        c.synchronize()
        session.resolve_completed_layer_timings()
        return hidden, observer.audit(), observer.target_layers
    except BaseException as error:
        observer.abort_execution(type(error).__name__+': '+str(error))
        raise
