"""Opt-in P0 bridge from issued v2 comparisons to actual winner preparation.

No selector, model call, economic bypass or legacy DENSE_EXACT publication is
introduced here. The existing engine still owns layer execution and final
admission. CPU fault tests do not qualify this bridge on a real model/GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from copy import deepcopy

from .source_store_v2 import TargetSourceStoreV2
from .source_manifest_v2 import ParentSourceMetadata
from .v8_schema10_staging import PhysicalLayerwiseSourceLoader
from .v8_schema6_hbm import HBMReservationKind


@dataclass
class _WinnerBinding:
    segment_id: str
    source_id: str
    metadata: dict
    current_positions: tuple
    lease: object
    layers: object
    reservation: object = None
    ticket: object = None
    lease_acquired: bool = False
    released: bool = False


class NativeV2ConsumptionSession:
    """One request, frozen winners, bounded SSD preparation, no parent owners.

    Keep the store lease and HBM reservation until a successful CUDA fence.
    Failure to fence quarantines both; cleanup must not pretend they are free.
    No GPU hot Replica is registered through the old exact-only Source pool.
    """
    def __init__(self, context, comparison_session):
        from .source_comparison_v2 import ComparisonSessionV2
        if type(comparison_session) is not ComparisonSessionV2:
            raise ValueError('issued v2 comparison session required')
        self.context, self.comparison = context, comparison_session
        self.store, self.snapshot = comparison_session.store, comparison_session.snapshot
        if type(self.store) is not TargetSourceStoreV2:
            raise ValueError('v2 target store required; legacy exact pool is not an adapter')
        c, a = context, context.adapter
        if (c.finished or c.closed or c.prepared or c.committed or c.frozen
                or c.selection_closed or c.cached_prefix_tokens or c.probe_fallback_reason
                or c.request.get('prefetch_window', 0) != 0
                or c.request.get('use_gpu_hot_cache') or c.request.get('retain_gpu_hot_cache')
                or c.request.get('capture_original_full_prefill')
                or c.request.get('publish_exact_prefix_shadow')
                or getattr(c, 'capture_reservation', None) is not None
                or getattr(c.native, 'prefix_shadow', None)):
            raise ValueError('v2 SSD bridge requires fresh preparation, Prefix/full-request capture off, no hot cache, window=0')
        if (self.snapshot.request_id != c.request['request_id']
                or self.store.config['model_signature'] != a.provenance['model_signature']
                or self.store.config['tokenizer_hash'] != a.provenance.get('tokenizer_hash')):
            raise ValueError('request/model/tokenizer differs from v2 snapshot')
        capture = getattr(c, 'source_capture_v2', None)
        if capture is not None and (capture.registry is not self.store.registry
                or capture.authorization_domain != self.store.config['authorization_domain']):
            raise ValueError('parent consumption and child capture must share verified registry/domain')
        with self.store.lock:
            self.store._snapshot(self.snapshot)
        c._begin()
        if c.engine.tickets or c.engine.session.commits or c.engine.prefetch_window != 0:
            raise ValueError('existing Source preparation/execution cannot be converted to v2')
        self.bindings, self.failures = {}, {}
        self.controlled_recipe_parents = {}
        self.closed = self.quarantined = False
        self.previous_loader = c.engine.source_loader
        loader = self.previous_loader
        # This constructor itself requires actual CUDA and pinned buffers.
        # CPU tests replace it explicitly and remain synthetic evidence.
        self.loader = PhysicalLayerwiseSourceLoader(loader.pool, authorize=self.authorize,
            integrity_mode=loader.integrity_mode, device=loader.device)
        self.loader.capture_hardware_trace = bool(getattr(loader, 'capture_hardware_trace', False))
        c.engine.source_loader = self.loader

    def _live(self):
        if self.closed or self.quarantined:
            raise RuntimeError('v2 consumption session closed or quarantined')
        if self.context.engine.source_loader is not self.loader:
            raise RuntimeError('v2 Source loader changed during active preparation')
        self.store._snapshot(self.snapshot)

    def _validate_binding(self, binding, *, require_ticket=False):
        self._live()
        c, manager = self.context, self.context.adapter.hbm
        row = self.store._visible_row(self.snapshot, binding.source_id)
        r = binding.reservation
        immutable = lambda value: {k:v for k,v in value.items()
                                   if k not in ('last_request_use_epoch','grace_until')}
        if (binding.released or not self.store._lease_counts.get(binding.source_id)
                or c.frozen.get(binding.segment_id) != binding.source_id
                or immutable(row) != immutable(binding.metadata) or r is None or r.released
                or manager.reservations.get(r.reservation_id) is not r
                or r.owner_request_id != c.request['request_id']
                or r.segment_id != binding.segment_id
                or r.bytes < row['target_kv_bytes']
                or r.kind not in (HBMReservationKind.WINNER_PREFETCH, HBMReservationKind.COMMITTED_EXECUTION)):
            raise ValueError('winner no longer has live bound Source/Replica/HBM ownership')
        if require_ticket:
            t = binding.ticket
            if (t is None or c.prepared.get(binding.segment_id) is not t
                    or t.source_id != binding.source_id or t.segment_id != binding.segment_id
                    or t.expected_artifact_digest != row['artifact_digest']
                    or tuple(t.segment_positions) != binding.current_positions
                    or t.expected_layer_count != row['layer_count']
                    or t.transfer_failed or t.preparation_cancelled):
                raise ValueError('preparation ticket differs from frozen v2 Artifact')
        return row

    def authorize(self, *, source_id, segment_id, bytes_required):
        with self.store.lock:
            binding = self.bindings.get(segment_id)
            if binding is None or binding.source_id != source_id:
                raise ValueError('full-KV transfer requires this session frozen winner')
            row = self._validate_binding(binding)
            if type(bytes_required) is not int or bytes_required != row['target_kv_bytes']:
                raise ValueError('requested transfer geometry differs from leased Artifact')

    def prepare_selected(self, receipt):
        return self._prepare_bound(receipt)

    def prepare_controlled_parent(self, receipt, *, job):
        """Explicit P0 prescribed-parent experiment, never selector admission.

        Keep real observations, unique eligible parent, ownership, geometry and
        stale checks. The frozen recipe, not a relaxed residual threshold,
        prescribes this diagnostic execution. Ordinary preparation is unchanged.
        """
        from .p0_lineage_control_v2 import validate_lineage_job, lineage_parents, lineage_targets
        from .v8_schema10_execution import digest_json
        validate_lineage_job(job, total_layers=self.context.adapter.spec.num_layers)
        parents, generations = lineage_parents(job)
        targets, _ = lineage_targets(job)
        capture = getattr(self.context, 'source_capture_v2', None)
        if (job.get('parent_preparation_mode') != 'preregistered_source_correctness_only'
                or job['request'] != self.context.request
                or job['comparison_profile'] != asdict(self.comparison.profile)
                or not self.comparison.profile.diagnostic_only
                or receipt.segment_id not in parents or capture is None
                or set(capture.targets) != set(targets)
                or tuple(receipt.eligible_ids) != (receipt.winner_source_id,)):
            raise ValueError('explicit matching P0 parent recipe and target capture required')
        declared = job['sources_by_segment'][receipt.segment_id][0]
        if 'source_id' in declared and declared['source_id'] != receipt.winner_source_id:
            raise ValueError('controlled parent differs from preregistered Source')
        with self.store.lock:
            self.comparison.verify(receipt, context=self.context)
            parent = self.store.parent_metadata(self.snapshot, receipt.winner_source_id)
            if parent.generation != generations[receipt.segment_id]:
                raise ValueError('controlled parent generation differs from frozen recipe')
        return self._prepare_bound(receipt, controlled_recipe=dict(
            recipe_digest=digest_json(job), source_id=receipt.winner_source_id,
            residual_admission_evaluated=False, production_execution_allowed=False))

    def _prepare_bound(self, receipt, *, controlled_recipe=None):
        c = self.context
        with self.store.lock:
            self._live()
            self.comparison.verify(receipt, context=c)
            sid, source_id = receipt.segment_id, receipt.winner_source_id
            if receipt.snapshot_id != self.snapshot.snapshot_id or not source_id:
                raise ValueError('comparison did not issue a winner in this snapshot')
            if not receipt.eligible_ids or set(receipt.compared_ids) != set(receipt.eligible_ids):
                raise ValueError('P0 winner preparation requires the full eligible comparison scope')
            if controlled_recipe is None and dict(receipt.scores).get(source_id,float('inf')) > self.comparison.profile.tau_reuse:
                raise ValueError('rank winner did not pass the bound residual reuse threshold')
            if sid in self.bindings or sid in c.frozen or sid in self.failures:
                raise ValueError('Source freeze is irreversible; no retry or second-place substitution')
            segment = c.segments[sid]
            if not c.execution_inventory[sid].comparison_eligible:
                raise ValueError('Prefix-covered/tail rows cannot become a Source target')
            positions = tuple(segment['positions'])
            tokens = tuple(c.request['token_ids'][p] for p in positions)
            row = self.store._visible_row(self.snapshot, source_id)
            attn = c.adapter.inner.layers[0].self_attn
            if (tuple(segment['token_ids']) != tokens or tuple(row['token_ids']) != tokens
                    or positions != tuple(range(positions[0], positions[0]+len(positions)))
                    or row['content_key'] != self.store.content_key(tokens)
                    or row['layer_count'] != c.adapter.spec.num_layers
                    or tuple(row['shape']) != (len(tokens),attn.num_kv_heads,attn.head_dim)):
                raise ValueError('winner token identity/model geometry differs from current Segment')
            lease = self.store.leased_target(self.snapshot, source_id)
            binding = _WinnerBinding(sid,source_id,deepcopy(row),positions,lease,None)
            self.bindings[sid] = binding
            if controlled_recipe is not None:
                self.controlled_recipe_parents[sid] = controlled_recipe
            # Record the irreversible selection attempt before file open. A
            # failed lease/read cannot silently authorize the runner-up.
            c.frozen[sid] = source_id
            try:
                layers = lease.__enter__()
                binding.layers, binding.lease_acquired = layers, True
                binding.reservation = c.adapter.hbm.reserve_batch(owner_request_id=c.request['request_id'],
                    rows=((sid,row['target_kv_bytes'],HBMReservationKind.WINNER_PREFETCH),))[0]
                c.replica_reservations[sid] = binding.reservation
                ticket = c.engine.start_winner_prefetch(segment_id=sid,source_id=source_id,
                    canonical_layers=layers,segment_positions=positions,
                    expected_artifact_digest=row['artifact_digest'],request_id=c.request['request_id'],
                    replica_id='v2-ssd:'+row['kv_file'],resident_layers=None)
                binding.ticket=ticket; c.prepared[sid]=ticket
                self._validate_binding(binding,require_ticket=True)
                parent = self.store.parent_metadata(self.snapshot,source_id)
                if getattr(c,'source_capture_v2',None) is not None:
                    self.bind_capture_parent(parent)
                self.store.mark_request_use(self.snapshot,source_id)
                # Binding a Source is a real use; reflect only its LRU update.
                binding.metadata=deepcopy(self.store._visible_row(self.snapshot,source_id))
                c.generation+=1
                return ticket
            except Exception as exc:
                self.failures[sid]=type(exc).__name__+':'+str(exc)
                if binding.lease_acquired:
                    try:
                        c.synchronize()
                    except Exception:
                        self.quarantined=True
                        raise RuntimeError('v2 preparation failure could not fence; resources quarantined') from exc
                self._drop_ticket(binding)
                self._release_binding(binding)
                c.generation+=1
                raise

    def bind_capture_parent(self, parent):
        if not isinstance(parent,ParentSourceMetadata):
            raise ValueError('typed verified parent metadata required')
        c=self.context
        capture=getattr(c,'source_capture_v2',None)
        if capture is None:
            raise RuntimeError('target capture was not reserved')
        with self.store.lock:
            matches=[b for b in self.bindings.values() if b.source_id==parent.source_id and not b.released]
            if not matches:
                raise ValueError('parent is not a leased v2 winner')
            for b in matches:
                self._validate_binding(b,require_ticket=True)
            expected=self.store.parent_metadata(self.snapshot,parent.source_id)
            if parent!=expected:
                raise ValueError('parent origin/G/digest differs from actual leased Artifact')
            capture.bind_parent(expected)

    def validate_selection_closure(self, frozen, prepared):
        self._live()
        if dict(frozen)!=self.context.frozen or set(prepared)!=set(self.context.prepared):
            raise ValueError('selection closure cannot replace or add v2 winners')
        if any(prepared[sid] is not self.context.prepared[sid] for sid in prepared):
            raise ValueError('selection closure changed a physical preparation ticket')

    def assert_can_commit(self,sid, *, controlled_recipe=False):
        with self.store.lock:
            if sid in self.controlled_recipe_parents and controlled_recipe is not True:
                raise ValueError('prescribed P0 parent cannot enter production commit')
            c = self.context
            boundary = c.current_completed_depth + 1
            if (not c.selection_closed or sid not in c.prepared or sid not in c.supports
                    or boundary not in c.supports[sid] or boundary > c.adapter.spec.num_layers):
                raise ValueError('selection closure and current-boundary repair support required before commit')
            binding=self.bindings[sid]
            self._validate_binding(binding,require_ticket=True)
            support = tuple(c.supports[sid][boundary])
            if (not binding.ticket.layer_ready(boundary) or support != tuple(sorted(set(support)))
                    or not set(support) <= set(binding.current_positions)):
                raise ValueError('current-boundary Source readiness and absolute repair support required')
            if binding.reservation.kind is not HBMReservationKind.WINNER_PREFETCH:
                raise ValueError('already committed or invalid preparation reservation')
            capture=getattr(self.context,'source_capture_v2',None)
            if capture is not None and capture.parents.get(binding.source_id)!=self.store.parent_metadata(self.snapshot,binding.source_id):
                raise ValueError('actual parent provenance must be bound before selective commit')

    def metadata_for_shape(self,sid,source_id):
        binding=self.bindings.get(sid)
        if binding is None or binding.source_id!=source_id:
            raise ValueError('cost shape must describe the frozen v2 winner')
        with self.store.lock:
            self._live()
            return deepcopy(binding.metadata)

    def _release_binding(self,binding):
        if binding.released:return
        if binding.reservation is not None:
            self.context.adapter.hbm.release(binding.reservation.reservation_id)
        if binding.lease_acquired:
            binding.lease.__exit__(None,None,None)
            binding.lease_acquired=False
        binding.released=True

    def _drop_ticket(self,binding):
        c=self.context; sid=binding.segment_id
        ticket=c.engine.tickets.pop(sid,None)
        if ticket is not None:ticket.cancel_pending()
        c.prepared.pop(sid,None)
        c.engine.session.source_handles.pop(sid,None)
        if hasattr(c.engine,'_source_row_indices'):c.engine._source_row_indices.pop(sid,None)
        binding.ticket=None

    def close(self, *, already_fenced=False):
        if self.closed:return
        if self.quarantined:
            raise RuntimeError('quarantined v2 resources require explicit recovery, not automatic release')
        if not already_fenced:
            try:self.context.synchronize()
            except Exception:
                self.quarantined=True
                raise
        for binding in self.bindings.values():
            self._drop_ticket(binding)
            self._release_binding(binding)
        self.context.engine.source_loader=self.previous_loader
        self.closed=True

    def audit(self):
        return dict(kind='native_v2_consumption',source_ids={sid:b.source_id for sid,b in self.bindings.items()},
            controlled_recipe_parents=deepcopy(self.controlled_recipe_parents),
            origins={sid:b.metadata['origin'] for sid,b in self.bindings.items()},
            generations={sid:b.metadata['generation'] for sid,b in self.bindings.items()},
            failed_segments=dict(self.failures),resources_quarantined=self.quarantined,
            legacy_exact_pool_modified=False,hot_replica_registered=False,
            transfer_path='SSD_STAGED_TO_GPU',prefetch_window=0,
            layerwise_overlap_qualified=False,native_runtime_qualified=False,
            final_commit_admission_bypassed=False,diagnostic_only=True)
