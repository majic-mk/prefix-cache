"""Bounded P0 native comparison/preparation operation, not a GPU launcher.

The caller must first obtain an authorized, environment-verified native context
and freeze the input/Source action. This function creates no instance, fetches
no data, builds no Source, and never commits reuse without the existing final
admission. Full request execution/qualification remains a separate stage.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

from .source_comparison_v2 import ComparisonSessionV2, ComparisonProfileBindingV2
from .source_manifest_v2 import request_input_digest
from .source_store_v2 import TargetSourceStoreV2
from .v8_schema10_execution import digest_json


@dataclass(frozen=True)
class P0ComparisonActionV2:
    request_id: str
    segment_id: str
    snapshot_id: str
    request_input_digest: str
    completed_depth: int
    source_ids: tuple
    profile_binding_digest: str
    host_workspace_bytes: int
    cuda_workspace_bytes: int
    mode: str = 'comparison_and_preparation_only'

    def __post_init__(self):
        if any(type(x) is not str or not x for x in (self.request_id,self.segment_id,self.snapshot_id)):
            raise ValueError('explicit action identities required')
        for sha in (self.request_input_digest,self.profile_binding_digest):
            if type(sha) is not str or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
                raise ValueError('bound input and diagnostic policy digests required')
        if (type(self.completed_depth) is not int or self.completed_depth<1
                or type(self.source_ids) is not tuple or not 0<=len(self.source_ids)<=4
                or any(type(s) is not str or not s for s in self.source_ids)
                or len(set(self.source_ids))!=len(self.source_ids)
                or any(type(n) is not int or n<=0 for n in (self.host_workspace_bytes,self.cuda_workspace_bytes))
                or self.mode!='comparison_and_preparation_only'):
            raise ValueError('bounded 0..4 Source comparison/preparation action required')

    @property
    def digest(self):return digest_json(asdict(self))


def validate_p0_comparison_action(context, store, snapshot, profile, action):
    """Read-only checks happen before advancing layers or touching CUDA."""
    from .v8_schema10_native_adapter import NativeRequestContext
    if (type(action) is not P0ComparisonActionV2 or type(store) is not TargetSourceStoreV2
            or type(profile) is not ComparisonProfileBindingV2 or not isinstance(context,NativeRequestContext)):
        raise ValueError('typed native operation inputs required')
    c=context
    if (c.finished or c.closed or c.prepared or c.frozen or c.committed or c.selection_closed
            or c.cached_prefix_tokens or c.probe_fallback_reason
            or c.request.get('capture_original_full_prefill') or c.request.get('publish_exact_prefix_shadow')
            or c.request.get('use_gpu_hot_cache') or c.request.get('retain_gpu_hot_cache')
            or c.request.get('prefetch_window',0)!=0
            or getattr(c,'comparison_workspace_v2',None) is not None
            or getattr(c,'source_consumption_v2',None) is not None):
        raise ValueError('P0 requires untouched preparation, Prefix/full capture off, SSD window=0')
    if (action.request_id!=c.request['request_id'] or action.snapshot_id!=snapshot.snapshot_id
            or snapshot.request_id!=action.request_id
            or action.request_input_digest!=request_input_digest(c.request['token_ids'],tuple(range(len(c.request['token_ids']))))
            or action.profile_binding_digest!=profile.binding_digest
            or action.completed_depth!=profile.completed_depth
            or action.completed_depth not in c.adapter.depths
            or action.completed_depth>=c.adapter.spec.num_layers
            or c.current_completed_depth>action.completed_depth
            or profile.model_signature!=c.adapter.provenance['model_signature']
            or store.config['model_signature']!=profile.model_signature
            or store.config['tokenizer_hash']!=c.adapter.provenance.get('tokenizer_hash')
            or store.config['policy']!=profile.provenance_policy):
        raise ValueError('action request/snapshot/policy/model/tokenizer/depth mismatch')
    owner=c.execution_inventory[action.segment_id]
    positions=tuple(c.segments[action.segment_id]['positions'])
    if (not owner.comparison_eligible or not positions
            or positions!=tuple(range(positions[0],positions[0]+len(positions)))
            or positions[0]<0 or positions[-1]>=len(c.request['token_ids'])
            or tuple(owner.remaining_positions)!=positions):
        raise ValueError('actual full target ownership required')
    tokens=tuple(c.request['token_ids'][p] for p in positions)
    rows=store.lookup(snapshot,tokens)
    if tuple(r['source_id'] for r in rows)!=action.source_ids:
        raise ValueError('frozen action must enumerate actual visible candidates in birth order')
    # Validate profile/runtime digests before any model step. No receipt or
    # materialization is produced by this short-lived issuer.
    check=ComparisonSessionV2(store,snapshot,profile,workspace_bytes=action.host_workspace_bytes)
    check.close()
    return rows


def execute_p0_comparison_preparation(context, store, snapshot, profile, action):
    """Actual native calls only; returned handle is NOT reuse admission.

    The caller owns context/snapshot lifetime and calls context.close() before
    store.end_request(). On exceptions, native cleanup runs; a failed fence
    preserves/quarantines reservations and must stop the surrounding runner.
    """
    validate_p0_comparison_action(context,store,snapshot,profile,action)
    issuer=None
    try:
        context.adapter.check_deadline()
        context.advance_to_depth(action.completed_depth)
        workspace=context.configure_cuda_comparison_v2(capacity_bytes=action.cuda_workspace_bytes)
        issuer=ComparisonSessionV2(store,snapshot,profile,workspace_bytes=action.host_workspace_bytes,
                                   cuda_workspace=workspace)
        context.configure_source_consumption_v2(issuer)
        receipt=issuer.compare_native(context,action.segment_id)
        complete=(receipt.eligible_ids==receipt.available_ids and set(receipt.eligible_ids)==set(receipt.compared_ids))
        compatible=receipt.winner_source_id is not None and dict(receipt.scores)[receipt.winner_source_id]<=profile.tau_reuse
        ticket=None
        if complete and compatible:
            context.adapter.check_deadline()
            ticket=context.prepare_selected_v2(receipt)
        audit=dict(action_digest=action.digest,comparison_receipt=asdict(receipt),
            comparison_workspace=workspace.audit(),source_consumption=context.source_consumption_v2.audit(),
            outcome='PREPARED_NOT_COMMITTED' if ticket is not None else 'DENSE_REQUIRED',
            reason='complete_compatible_winner' if ticket is not None else
                   'incomplete_comparison' if not complete else 'no_compatible_winner',
            reuse_committed=False,request_finished=False,extra_forward_count=0,
            native_runtime_qualified=False,paper_evidence=False)
        return dict(issuer=issuer,receipt=receipt,ticket=ticket,audit=audit)
    except BaseException:
        try:context.close()
        finally:
            if issuer is not None:issuer.close()
        raise
