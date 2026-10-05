"""Issued, request-local evidence for P0 K-only Source comparison.

Uses the existing arithmetic unchanged. A diagnostic binding is NOT a frozen
quality Profile, runtime authorization or proof of real-model correctness.
No caller-supplied scores, QA labels, V, or full Artifact reads are accepted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import math
from pathlib import Path
import time
import uuid

from . import selection_comparison
from .source_provenance_v2 import PublicationScope
from .source_manifest_v2 import request_input_digest
from .source_store_v2 import TargetSourceStoreV2
from .v8_schema10_execution import digest_json
from .v8_schema10_storage import tensor_digest
from .source_metadata_selection import (MetadataSelectionPolicy, MetadataSelectionDecision,
    StateCandidate, select_source, validate_prefix_tokens)


def scorer_digest():
    return hashlib.sha256(Path(selection_comparison.__file__).read_bytes()).hexdigest()


def runtime_binding_digest():
    root = Path(__file__).parent
    names = ('source_comparison_v2.py', 'source_metadata_selection.py', 'source_store_v2.py', 'source_manifest_v2.py',
             'source_provenance_v2.py', 'selection_comparison.py', 'model_adapters.py',
             'cuda_comparison_v2.py', 'native_p0_operation_v2.py', 'native_p0_request_v2.py',
             'native_consumption_v2.py', 'native_publication_v2.py', 'v8_schema10_native_adapter.py',
             'p0_batch_v2.py', 'p0_evidence_v2.py', 'p0_decode_evidence_v2.py', 'p0_lineage_control_v2.py',
             'p0_exact_control_v2.py', 'p0_exact_pair_v2.py', 'p0_natural_input_v2.py', 'p1_input_consumer_v2.py',
             'p0_diagnostic_context_v2.py', 'p0_mixed_control_v2.py', 'p0_mixed_reference_v2.py',
             'p0_mixed_sparse_v2.py', 'p0_mixed_pair_v2.py', 'p0_stage_readiness_v2.py',
             'native_source_capture_v2.py', 'native_capture_hook_v2.py',
             'segment_capture_v2.py', 'resumable_prefill.py', 'v8_schema10_native_factory.py',
             'v8_schema10_measured_costs.py', 'v8_schema10_cost_provider.py',
             'v8_schema6_planner.py', 'v8_schema7_planner.py',
             'cacheblend_v6_online_engine.py', 'cacheblend_patch.py')
    files = {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
    entry = root.parents[1]/'scripts/server/run_decoupled_v2_p0.py'
    files['scripts/server/run_decoupled_v2_p0.py'] = hashlib.sha256(entry.read_bytes()).hexdigest()
    return digest_json(files)


def _sha(value):
    return (type(value) is str and len(value)==64 and
            all(c in '0123456789abcdef' for c in value))


@dataclass(frozen=True)
class ComparisonProfileBindingV2:
    model_signature: str
    provenance_policy: str
    completed_depth: int
    trim_ratio: float
    tau_reuse: float
    tau_add: float
    repair_policy_digest: str
    runtime_digest: str
    scoring_function_digest: str
    diagnostic_only: bool = True

    def __post_init__(self):
        if (type(self.model_signature) is not str or not self.model_signature or self.provenance_policy not in ('EXACT_ONLY','ALLOW_MIXED_G1')
                or type(self.completed_depth) is not int or self.completed_depth<1
                or self.diagnostic_only is not True):
            raise ValueError('explicit diagnostic model/policy/legal depth required; no quality qualification')
        for x in (self.trim_ratio,self.tau_reuse,self.tau_add):
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0:
                raise ValueError('finite nonnegative scalar binding required')
        if not 0<=self.trim_ratio<1 or self.tau_add!=self.tau_reuse:
            raise ValueError('initial v2 uses a separate trim ratio and one equal threshold pair')
        if not all(_sha(x) for x in (self.repair_policy_digest,self.runtime_digest,self.scoring_function_digest)):
            raise ValueError('explicit repair/runtime/scorer digests required')

    @property
    def binding_digest(self):
        return digest_json(asdict(self))


@dataclass(frozen=True)
class ComparisonReceiptV2:
    receipt_id: str
    snapshot_id: str
    request_id: str
    segment_id: str
    content_key: str
    token_ids: tuple
    absolute_positions: tuple
    completed_depth: int
    observation_layer: int
    context_generation: int
    profile_binding_digest: str
    request_input_digest: str
    current_k_digest: str | None
    stored_ids: tuple
    eligible_ids: tuple
    available_ids: tuple
    compared_ids: tuple
    source_signatures: tuple
    scores: tuple
    winner_source_id: str | None
    rejected_states: tuple
    host_ms: float
    diagnostic_only: bool = True
    # Populated only by an explicitly attached CUDA operation after its fence.
    comparison_device: str = 'cpu'
    measurement_digest: str | None = None
    metadata_selection: MetadataSelectionDecision | None = None

    @property
    def counts(self):
        return dict(stored=len(self.stored_ids),eligible=len(self.eligible_ids),
                    available=len(self.available_ids),compared=len(self.compared_ids))


class ComparisonSessionV2:
    """One issuer per store snapshot; immutable receipts retain no K tensors.

    A production-quality decision is not enabled by these P0 diagnostic
    bindings. CPU tests prove calculation/identity, not native GPU provenance.
    """
    def __init__(self, store, snapshot, profile, *, workspace_bytes, cuda_workspace=None,
                 metadata_policy=None):
        if type(store) is not TargetSourceStoreV2 or type(profile) is not ComparisonProfileBindingV2:
            raise ValueError('typed v2 store and diagnostic binding required')
        if type(workspace_bytes) is not int or workspace_bytes<=0:
            raise ValueError('bounded comparison workspace required')
        self.store,self.snapshot,self.profile=store,snapshot,profile
        self.metadata_policy = metadata_policy if metadata_policy is not None else MetadataSelectionPolicy()
        self.workspace_bytes=workspace_bytes
        if cuda_workspace is not None:
            from .cuda_comparison_v2 import CudaComparisonWorkspaceV2
            if type(cuda_workspace) is not CudaComparisonWorkspaceV2:
                raise ValueError('typed explicit CUDA comparison workspace required')
            if cuda_workspace.request_id!=snapshot.request_id:
                raise ValueError('CUDA workspace request differs from snapshot')
        self.cuda_workspace=cuda_workspace
        self._failed=False
        self._issued={}; self._used_for_publication=set(); self._closed=False
        self._observed_segments=set()
        self._check()

    def _check(self):
        if self._closed: raise RuntimeError('comparison session closed')
        if self._failed or (self.cuda_workspace is not None and self.cuda_workspace.lease.quarantined):
            raise RuntimeError('failed CUDA comparison session cannot issue or consume receipts')
        self.store._snapshot(self.snapshot)
        p=self.profile
        if (type(self.metadata_policy) is not MetadataSelectionPolicy or
                self.metadata_policy.enabled and self.metadata_policy.tau != p.tau_reuse):
            raise ValueError('metadata tau must match the bound state compatibility threshold')
        if (p.model_signature!=self.store.config['model_signature']
                or p.provenance_policy!=self.store.config['policy']
                or p.scoring_function_digest!=scorer_digest() or p.runtime_digest!=runtime_binding_digest()):
            raise ValueError('stale or wrong model/policy/runtime/scoring binding')

    def _context(self, context, segment_id, *, require_open):
        self._check()
        if (context.request['request_id']!=self.snapshot.request_id
                or context.adapter.provenance['model_signature']!=self.profile.model_signature
                or require_open and context.finished
                or context.current_completed_depth!=self.profile.completed_depth
                or type(context.generation) is not int or context.generation<0):
            raise ValueError('current request/model/completed depth differs from comparison binding')
        spec=getattr(context.adapter,'spec',None)
        if spec is not None and self.profile.completed_depth>=spec.num_layers:
            raise ValueError('observation layer must exist after completed depth')
        positions=tuple(context.segments[segment_id]['positions'])
        if (len(positions)<2 or any(type(p) is not int or p<0 for p in positions)
                or positions!=tuple(range(positions[0],positions[0]+len(positions)))
                or positions[-1]>=len(context.request['token_ids'])):
            raise ValueError('ordered contiguous absolute target positions required')
        tokens=tuple(context.request['token_ids'][p] for p in positions)
        owner=context.execution_inventory[segment_id]
        if not owner.comparison_eligible:
            raise ValueError('Prefix/partial-tail/suffix cannot enter Source comparison')
        if hasattr(owner,'remaining_positions') and tuple(owner.remaining_positions)!=positions:
            raise ValueError('comparison rows differ from actual execution ownership')
        return tokens,positions

    @staticmethod
    def _signature(row):
        return (row['source_id'],row['artifact_digest'],row['selection_file_digest'],
                row['proof_digest'],row['publication_epoch'],row['origin'],row['generation'])

    def compare_native(self, context, segment_id, candidate_ids=None, *,
                       current_prefix=None, historical_prefixes=None):
        metadata = dict(current_prefix=current_prefix, historical_prefixes=historical_prefixes)
        if self.cuda_workspace is None:
            return self._compare_native_impl(context,segment_id,candidate_ids,**metadata)
        started=time.perf_counter_ns()
        with self.store.lock:
            tokens,_=self._context(context,segment_id,require_open=True)
            self.cuda_workspace._check(context)
            # A real miss needs neither projection nor CUDA events/reservation.
            if not self.store.lookup(self.snapshot,tokens):
                return self._compare_native_impl(context,segment_id,candidate_ids,**metadata)
            try:
                with self.cuda_workspace.operation(context,segment_id):
                    receipt=self._compare_native_impl(context,segment_id,candidate_ids,**metadata)
                receipt=replace(receipt,host_ms=(time.perf_counter_ns()-started)/1e6,
                    comparison_device=str(self.cuda_workspace.device),
                    measurement_digest=digest_json(self.cuda_workspace.events[-1]))
                self._issued[receipt.receipt_id]=receipt
                return receipt
            except BaseException:
                # Includes a post-arithmetic CUDA fence failure. No apparently
                # valid receipt may survive even if the scalar was computed.
                self._failed=True
                self._issued.clear()
                raise

    def _compare_native_impl(self, context, segment_id, candidate_ids=None, *,
                             current_prefix=None, historical_prefixes=None):
        """Read current K from the context, never accept scores as input.

        Full source enumeration comes from the frozen store snapshot. Partial
        explicit comparison is a diagnostic control and cannot authorize growth.
        No V or full Artifact is read even when a SelectionState is unavailable.
        """
        import torch
        started=time.perf_counter_ns()
        with self.store.lock:
            tokens,positions=self._context(context,segment_id,require_open=True)
            if segment_id in self._observed_segments:
                raise RuntimeError('one observation per target/depth; do not overwrite prior receipts')
            rows=self.store.lookup(self.snapshot,tokens)
            stored=tuple(r['source_id'] for r in rows)
            generation=context.generation
            input_digest=request_input_digest(context.request['token_ids'],
                tuple(range(len(context.request['token_ids']))))
            eligible=[]; available=[]; failures=[]; states={}
            selected=stored if candidate_ids is None else tuple(candidate_ids)
            if len(set(selected))!=len(selected) or not set(selected)<=set(stored):
                raise ValueError('candidate subset must be distinct actually visible Source IDs')
            current_digest=None; scores=[]; winner=None
            # No current-state projection is necessary to prove a real miss.
            if stored:
                current=context.observe_current_k(segment_id,self.profile.completed_depth)
                if (not torch.is_tensor(current) or current.dtype!=torch.bfloat16 or current.ndim!=3
                        or current.shape[0]!=len(tokens) or min(current.shape)<1):
                    raise ValueError('finite BF16 current K with actual target geometry required')
                if self.cuda_workspace is not None:
                    self.cuda_workspace.validate_current(current)
                elif current.device.type!='cpu':
                    raise RuntimeError('P0 receipt comparator is CPU-only; GPU workspace qualification pending')
                if not bool(torch.isfinite(current).all()):
                    raise ValueError('finite BF16 current K with actual target geometry required')
                per_source=current.numel()*32+current.shape[0]*32
                if 2*per_source>self.workspace_bytes:
                    raise MemoryError('bounded current/source comparison workspace unavailable')
                current_digest=(self.cuda_workspace.current_digest(current) if self.cuda_workspace
                                else tensor_digest((current,)))
                for row in rows:
                    sid=row['source_id']
                    if (row['generation'] not in (0,1)
                            or self.profile.provenance_policy=='EXACT_ONLY' and row['generation']!=0):
                        failures.append((sid,'provenance_filtered'));continue
                    eligible.append(sid)
                    try:
                        # The current file format loads all saved checkpoints
                        # before returning one. Charge that transient payload,
                        # not only the single returned layer's tensor bytes.
                        retained=sum(t.numel()*t.element_size() for t in states.values())
                        if retained+row['selection_state_bytes']+2*per_source>self.workspace_bytes:
                            raise MemoryError('SelectionState file payload exceeds bounded workspace')
                        key=self.store.read_selection(self.snapshot,sid,self.profile.completed_depth)
                        if (key.dtype!=torch.bfloat16 or key.shape!=current.shape
                                or key.device.type!='cpu' or not bool(torch.isfinite(key).all())):
                            raise ValueError('SelectionState shape/dtype/finite mismatch')
                        available.append(sid)
                        if sid in selected:
                            # Keep at most K<=4 target K states; charge persistent
                            # auxiliary bytes too, not just arithmetic scratch.
                            if sum(t.numel()*t.element_size() for t in states.values())+key.numel()*key.element_size()+2*per_source>self.workspace_bytes:
                                raise MemoryError('SelectionState plus arithmetic workspace exhausted')
                            states[sid]=key
                    except (KeyError,ValueError,RuntimeError,OSError) as exc:
                        failures.append((sid,type(exc).__name__+':'+str(exc)))
                normalized=selection_comparison.prepare_current_k(current)
                for sid in selected:
                    if sid not in states: continue
                    source=(self.cuda_workspace.source_to_device(states[sid]) if self.cuda_workspace else states[sid])
                    result=selection_comparison.residual_scores(source.unsqueeze(0),
                        *normalized,self.profile.trim_ratio)
                    value=float(result[0].item())
                    if not math.isfinite(value) or value<0:
                        raise ValueError('legacy arithmetic returned nonfinite residual')
                    scores.append((sid,value))
                    del source,result  # Do not retain previous candidate GPU allocation.
                by_id={r['source_id']:r for r in rows}
                if scores:
                    winner=min(scores,key=lambda p:(p[1],by_id[p[0]]['generation'],
                        by_id[p[0]]['publication_epoch'],p[0]))[0]
            metadata_decision = None
            if self.metadata_policy.enabled:
                by_id = {r['source_id']: r for r in rows}
                ordered = sorted(scores, key=lambda p: (p[1], by_id[p[0]]['generation'],
                    by_id[p[0]]['publication_epoch'], p[0]))
                compatible = [p for p in ordered if p[1] <= self.metadata_policy.tau]
                band = ([p for p in compatible if p[1] <= compatible[0][1]+self.metadata_policy.delta]
                        if compatible else [])
                if len(band) > 1:
                    # Validate lightweight metadata against already-shared input
                    # manifests. No parent KV, attention or inferred role reads.
                    from .segment_capture_v2 import ManifestReference
                    tokenizer = self.store.config['tokenizer_hash']
                    validate_prefix_tokens(current_prefix, context.request['token_ids'],
                        target_start=positions[0], tokenizer_signature=tokenizer)
                    for sid, _ in band:
                        prefix = (historical_prefixes or {}).get(sid)
                        if prefix is not None:
                            row = by_id[sid]
                            manifest, target = self.store._manifest_record(
                                ManifestReference(**row['manifest_reference']),
                                row['token_ids'], row['birth_target_positions'])
                            validate_prefix_tokens(prefix, manifest['token_ids'],
                                target_start=target['start'], tokenizer_signature=tokenizer)
                metadata_decision = select_source(
                    tuple(StateCandidate(sid, score) for sid, score in ordered),
                    policy=self.metadata_policy, current_prefix=current_prefix,
                    historical_prefixes=historical_prefixes)
                # Keep the raw best for the existing diagnostic mismatch path;
                # native preparation will still reject its incompatible score.
                if metadata_decision.selected_source_id is not None:
                    winner = metadata_decision.selected_source_id
            observed_tokens,observed_positions=self._context(context,segment_id,require_open=True)
            if (context.generation!=generation or observed_tokens!=tokens
                    or observed_positions!=positions or input_digest!=request_input_digest(
                        context.request['token_ids'],tuple(range(len(context.request['token_ids']))))):
                raise ValueError('request execution or input identity changed during comparison')
            receipt=ComparisonReceiptV2(uuid.uuid4().hex,self.snapshot.snapshot_id,self.snapshot.request_id,
                segment_id,self.store.content_key(tokens),tokens,positions,self.profile.completed_depth,
                self.profile.completed_depth+1,generation,self.profile.binding_digest,input_digest,current_digest,
                stored,tuple(eligible),tuple(available),tuple(s for s,_ in scores),
                tuple(self._signature(r) for r in rows),tuple(scores),winner,tuple(failures),
                (time.perf_counter_ns()-started)/1e6, metadata_selection=metadata_decision)
            self._issued[receipt.receipt_id]=receipt
            self._observed_segments.add(segment_id)
            return receipt

    def verify(self, receipt, context=None):
        with self.store.lock:
            self._check()
            if (type(receipt) is not ComparisonReceiptV2 or self._issued.get(receipt.receipt_id)!=receipt
                    or receipt.snapshot_id!=self.snapshot.snapshot_id
                    or receipt.profile_binding_digest!=self.profile.binding_digest):
                raise ValueError('unissued, altered or cross-session comparison receipt')
            if ((receipt.metadata_selection is None) != (not self.metadata_policy.enabled)
                    or receipt.metadata_selection is not None and
                    receipt.metadata_selection.policy != self.metadata_policy):
                raise ValueError('metadata selection policy changed after observation')
            rows=self.store.lookup(self.snapshot,receipt.token_ids)
            current_ids=tuple(r['source_id'] for r in rows)
            all_current=tuple(sorted(s for s,r in self.store._catalog['rows'].items()
                                     if r['content_key']==receipt.content_key))
            if (current_ids!=receipt.stored_ids or all_current!=tuple(sorted(receipt.stored_ids))
                    or tuple(self._signature(r) for r in rows)!=receipt.source_signatures):
                raise ValueError('Source pool changed after observation')
            # Detect changed/missing selection files without rereading full KV.
            from .v8_schema10_storage import file_digest
            for row in rows:
                if row['source_id'] in receipt.available_ids:
                    path=self.store._path(row['selection_file'])
                    if not path.is_file() or file_digest(path)!=row['selection_file_digest']:
                        raise ValueError('SelectionState changed after comparison')
            if context is not None:
                tokens,positions=self._context(context,receipt.segment_id,require_open=True)
                if (tokens!=receipt.token_ids or positions!=receipt.absolute_positions
                        or context.generation!=receipt.context_generation
                        or request_input_digest(context.request['token_ids'],tuple(range(len(context.request['token_ids']))))!=receipt.request_input_digest):
                    raise ValueError('stale observation for winner freeze')
            return receipt

    def publication_scope(self, receipt):
        self.verify(receipt)
        if receipt.receipt_id in self._used_for_publication:
            raise ValueError('comparison publication receipt already consumed')
        if not receipt.stored_ids:
            return PublicationScope('content_miss',0,0,0,0)
        counts=receipt.counts
        best=min((v for _,v in receipt.scores),default=None)
        scope=PublicationScope('complete_scope_mismatch',**counts,best_score=best,threshold=self.profile.tau_add)
        if scope.rejection(): raise ValueError('no eligible growth trigger: '+scope.rejection())
        return scope

    def mark_publication_used(self, receipt):
        # Called only after store commit has revalidated the issued receipt.
        # A closed issuer cannot be reused; do not hide a rejected transaction
        # behind a second exception when closing invalidated the planned input.
        if self._closed: return
        if self._issued.get(receipt.receipt_id)!=receipt:
            raise ValueError('unknown publication receipt')
        self._used_for_publication.add(receipt.receipt_id)

    def close(self):
        self._issued.clear();self._observed_segments.clear();self._used_for_publication.clear()
        self._closed=True
