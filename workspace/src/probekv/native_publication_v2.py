"""Explicit post-answer bridge to the isolated SSD-only P0 Source store.

This module never runs a model, prepares native parent KV, creates a Prefix
shadow, or enables the production dispatcher. Missing capture/scope is a skip,
not permission to build a Source or infer a growth trigger.
"""
from __future__ import annotations

from collections.abc import Mapping
import time

from .native_source_capture_v2 import NativeTargetCaptureV2
from .segment_capture_v2 import ManifestReference
from .source_provenance_v2 import PublicationScope, TargetProof
from .source_store_v2 import TargetSourceStoreV2


def publish_target_candidates_v2(context, store, snapshot, scope_by_target, *, comparison_session=None):
    """Publish once after native finish; leave the answer and candidates intact.

    The caller owns the existing birth-request snapshot and must end it after
    publication. Results remain invisible to that snapshot. Invalid operation
    identity is an API error; individual candidate/store failures are reported
    without retry, replay, fallback, or changes to the completed model output.
    """
    started = time.perf_counter_ns()
    if getattr(context, 'finished', False) is not True:
        raise RuntimeError('publication requires completed birth request')
    if getattr(context, '_target_publication_attempted_v2', False):
        raise RuntimeError('target publication is not repeatable')
    capture = getattr(context, 'source_capture_v2', None)
    if not isinstance(capture, NativeTargetCaptureV2) or not capture._finalized:
        raise ValueError('finalized actual native capture required; no legacy build fallback')
    if not isinstance(store, TargetSourceStoreV2):
        raise ValueError('isolated TargetSourceStoreV2 required')
    if not isinstance(scope_by_target, Mapping):
        raise ValueError('explicit per-target publication scope mapping required')
    if comparison_session is not None:
        from .source_comparison_v2 import ComparisonSessionV2
        if (type(comparison_session) is not ComparisonSessionV2
                or comparison_session.store is not store or comparison_session.snapshot!=snapshot):
            raise ValueError('comparison issuer must own this actual store/snapshot')
    tokens = tuple(context.request['token_ids'])
    request_id = context.request['request_id']
    model = context.adapter.provenance['model_signature']
    if (store.registry is not capture.registry or not capture.registry.durable
            or store.config['model_signature'] != model or capture.model_signature != model
            or store.config['authorization_domain'] != capture.authorization_domain
            or capture.request_id != request_id or capture.ledger.request_id != request_id
            or capture.ledger.model_signature != model or capture.tokens != tokens
            or capture.positions != tuple(range(len(tokens)))
            or capture.ledger.num_layers != context.adapter.spec.num_layers):
        raise ValueError('store/capture registry, domain, model or birth identity mismatch')
    # Validate the actual issued snapshot before marking the operation used.
    # No implicit begin/end/rebase can make a newly published child self-visible.
    with store.lock:
        store._snapshot(snapshot)
        if snapshot.request_id != request_id:
            raise ValueError('snapshot must belong to the completed birth request')
    candidates = context.export_target_candidates_v2()
    context._target_publication_attempted_v2 = True
    outcomes = {}
    recovery_required = False
    for sid in capture.targets:
        if recovery_required:
            outcomes[sid] = dict(status='SKIPPED', publication_performed=False,
                reason='prior_commit_outcome_uncertain', recovery_required=True)
            continue
        candidate = candidates.get(sid)
        if candidate is None:
            outcomes[sid] = dict(status='SKIPPED', publication_performed=False,
                reason=capture.rejections.get(sid, 'no_completed_target_candidate'))
            continue
        if sid not in scope_by_target:
            outcomes[sid] = dict(status='SKIPPED', publication_performed=False,
                reason='missing_publication_scope')
            continue
        phase = 'validate'
        try:
            scope = scope_by_target[sid]
            if comparison_session is None and type(scope) is not PublicationScope:
                raise ValueError('typed diagnostic publication scope required')
            positions = capture.targets[sid]
            proof = candidate.get('proof')
            reference = candidate.get('manifest_reference')
            if (capture._failed or capture._submitted_layers != capture.ledger.num_layers
                    or len(capture.ledger.layers) != capture.ledger.num_layers
                    or tuple(context.segments[sid]['positions']) != positions
                    or not isinstance(proof, TargetProof)
                    or proof != capture.ledger.target_proof(positions)
                    or not isinstance(reference, ManifestReference)
                    or reference.target_occurrence != sid):
                raise ValueError('candidate differs from completed actual native target execution')
            target_tokens = tuple(tokens[p] for p in positions)
            phase = 'plan'
            if comparison_session is None:
                plan = store.plan_publication(snapshot, candidate,
                    token_ids=target_tokens, scope=scope)
            else:
                from .source_comparison_v2 import ComparisonReceiptV2
                if (type(scope) is not ComparisonReceiptV2 or scope.segment_id!=sid
                        or scope.absolute_positions!=positions):
                    raise ValueError('issued comparison receipt must match actual target occurrence')
                plan = store.plan_from_comparison(snapshot, candidate, token_ids=target_tokens,
                    comparison=comparison_session, receipt=scope)
            phase = 'commit'
            outcomes[sid] = store.commit_publication(snapshot, plan, candidate,
                token_ids=target_tokens)
            recovery_required = bool(outcomes[sid].get('recovery_required'))
        except Exception as exc:
            # Publication is optional post-answer host work. In particular an
            # I/O fault must not trigger a second forward or discard the answer.
            # BaseException still propagates for cancellation/process shutdown.
            # An unexpected commit exception might follow the atomic replace.
            # Do not misreport that ambiguous state as a definite non-write.
            outcomes[sid] = dict(status='COMMIT_UNCERTAIN' if phase == 'commit' else 'REJECTED',
                publication_performed=None if phase == 'commit' else False,
                recovery_required=phase == 'commit',
                reason=type(exc).__name__ + ':' + str(exc))
            recovery_required = phase == 'commit'
    audit = dict(event='native_target_publication_v2', request_id=request_id,
        snapshot_id=snapshot.snapshot_id, snapshot_epoch=snapshot.epoch,
        targets=outcomes, post_request_host_ms=(time.perf_counter_ns()-started)/1e6,
        recovery_required=recovery_required,
        extra_forward_count=0, prefix_shadow_created=False,
        publication_evidence_mode='ISSUED_K_COMPARISON' if comparison_session else 'RULE_ONLY_CPU_DIAGNOSTIC',
        physical_backing='SSD_ONLY_P0', diagnostic_only=True,
        production_dispatch_integrated=False, native_runtime_qualified=False,
        native_parent_consumption_qualified=False)
    context.source_publication_v2 = audit
    capture.events.append(audit)
    return audit


def publish_compared_targets_v2(context, comparison_session, receipts_by_target):
    """Native post-answer opt-in: caller-supplied scores/scopes are forbidden.

    These receipts establish local computation and identity, not GPU/quality
    qualification. The rule-only bridge above remains for old CPU fault tests.
    """
    from .source_comparison_v2 import ComparisonSessionV2
    if type(comparison_session) is not ComparisonSessionV2:
        raise ValueError('actual comparison issuer required')
    return publish_target_candidates_v2(context, comparison_session.store,
        comparison_session.snapshot, receipts_by_target, comparison_session=comparison_session)
