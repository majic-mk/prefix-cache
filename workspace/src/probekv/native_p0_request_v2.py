"""Bounded P0 request driver; no server, authority or qualification bypass.

An outer preflight must supply an authorized native request context and frozen
actions. This driver owns that context's close and the supplied store snapshot;
the native block context manager remains the outer caller's responsibility.
It never opens a model, builds extra Sources or runs another forward for QA.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import time

from .native_p0_operation_v2 import P0ComparisonActionV2, validate_p0_comparison_action
from .native_publication_v2 import publish_compared_targets_v2
from .source_comparison_v2 import ComparisonSessionV2
from .v8_schema10_cost_provider import EXECUTION_SHAPE_KEY, UnsupportedTimelineCost
from .v8_schema10_measured_costs import MeasuredRequestCostProvider
from .v8_schema7_planner import FinalCommitPlanner


class P0RequestFailure(RuntimeError):
    """Retains completed answer/partial events and cleanup failures for logging."""
    def __init__(self, audit, cause):
        super().__init__('P0 request failed: ' + str(cause))
        self.audit, self.cause = audit, cause


def _validate(context, store, snapshot, profile, actions, host_capture_bytes):
    if (type(actions) is not tuple or not actions
            or any(type(a) is not P0ComparisonActionV2 for a in actions)
            or len({a.segment_id for a in actions}) != len(actions)
            or type(host_capture_bytes) is not int or host_capture_bytes <= 0):
        raise ValueError('distinct frozen actions and explicit capture budget required')
    if (context.current_completed_depth != 0
            or getattr(context, 'source_capture_v2', None) is not None
            or context.repair_ratio != .15
            or context.adapter.native_repair_metric != 'normalized_kv_deviation'
            or type(context.arrival_ns) is not int
            or not 0 < context.arrival_ns <= time.perf_counter_ns()):
        raise ValueError('fresh d0 context, fixed15 normalized K/V and real arrival required')
    for action in actions:
        validate_p0_comparison_action(context, store, snapshot, profile, action)
    if len({(a.completed_depth, a.host_workspace_bytes, a.cuda_workspace_bytes) for a in actions}) != 1:
        raise ValueError('one frozen depth and shared workspace budget per P0 request')
    costs = context.adapter.costs
    if costs is not None and (type(costs) is not MeasuredRequestCostProvider
                             or costs.key_contract != EXECUTION_SHAPE_KEY):
        raise ValueError('v2 accepts only measured execution-shape costs, or explicit missing costs')


def _final_admission(context, ready, union_digest, events):
    """Reuse the measured joint planner; null/missing support stays dense."""
    costs = context.adapter.costs
    if not ready:
        return dict(status='DENSE', reason='no_prepared_compatible_source', accepted=[])
    if costs is None:
        return dict(status='UNSUPPORTED', reason='missing_measured_cost_provider', accepted=[])
    dense = costs.dense_reference(context)
    if dense is None:
        return dict(status='UNSUPPORTED', reason='missing_matched_dense_reference', accepted=[])
    snapshot = context.planner_snapshot(context.adapter.hbm.epoch)
    started = time.perf_counter_ns()
    try:
        estimator = costs.joint_estimator(context)
        result = FinalCommitPlanner(estimator).plan_ready_subset(
            inventory_segment_ids=tuple(context.execution_inventory),
            eligible_ready_segment_ids=tuple(ready), committed_segment_ids=tuple(context.committed),
            actual_boundary_by_segment=ready,
            actual_sunk_ms=(started-context.arrival_ns)/1e6,
            dense_reference_total_ms=dense, snapshot=snapshot,
            current_snapshot=context.planner_snapshot(context.adapter.hbm.epoch),
            union_mask_digest=union_digest)
        snapshot.assert_current(context.planner_snapshot(context.adapter.hbm.epoch))
        elapsed = (time.perf_counter_ns()-started)/1e6
        result = replace(result, request_total_ms=result.request_total_ms+elapsed,
            cost_audit={**result.cost_audit, 'driver_planner_elapsed_ms': elapsed})
        if result.accepted_ready_segment_ids and result.request_total_ms > .8*dense:
            return dict(status='DENSE', reason='planner_elapsed_exceeds_gamma', accepted=[],
                        proposed_decision=asdict(result))
        if result.accepted_ready_segment_ids:
            context.commit_reuse(result)
        return dict(status='DECIDED', accepted=list(result.accepted_ready_segment_ids), decision=asdict(result))
    except UnsupportedTimelineCost as exc:
        return dict(status='UNSUPPORTED', reason=str(exc), accepted=[])
    except RuntimeError as exc:
        if str(exc) != 'stale Planner snapshot cannot be applied':
            raise
        # Bounded P0: no unregistered retry, second winner, or threshold search.
        return dict(status='DENSE', reason='stale_planner_snapshot', accepted=[])
    finally:
        events.append(dict(kind='final_admission_interval', start_ns=started,
                           end_ns=time.perf_counter_ns()))


def execute_p0_request(context, store, snapshot, profile, actions, *, host_capture_bytes):
    """Capture -> compare -> prepare -> final admission -> answer -> publish.

    Same-depth actions can include a reused upstream Segment and a fully
    recomputed downstream target; their actual execution ledger determines G.
    Actions enumerate only previously visible 0..4 Sources. Non-action rows
    remain dense in the complete request inventory, never silently omitted.
    Read-only validation failures leave ownership with caller. Once started,
    cleanup and snapshot release are this driver's responsibility.
    """
    _validate(context, store, snapshot, profile, actions, host_capture_bytes)
    audit = dict(kind='native_p0_request_v2', status='RUNNING',
        request_id=context.request['request_id'], snapshot_id=snapshot.snapshot_id,
        action_digests=[a.digest for a in actions], profile_binding_digest=profile.binding_digest,
        cost_measurement_digest=getattr(context.adapter.costs, 'sha', None),
        events=[], comparison_receipts={}, preparation={}, answer=None, publication=None,
        final_admission=None, cleanup=None, extra_forward_count=0,
        native_runtime_qualified=False, gpu_execution_authorized_by_driver=False,
        paper_evidence=False, P1_execution_allowed=False)
    issuer = workspace = None
    error = None
    try:
        context.adapter.check_deadline()
        audit['capture_configuration'] = context.configure_target_capture_v2(
            target_ids=tuple(a.segment_id for a in actions), registry=store.registry,
            authorization_domain=store.config['authorization_domain'], host_budget_bytes=host_capture_bytes)
        context.advance_to_depth(profile.completed_depth)
        workspace = context.configure_cuda_comparison_v2(capacity_bytes=actions[0].cuda_workspace_bytes)
        issuer = ComparisonSessionV2(store, snapshot, profile,
            workspace_bytes=actions[0].host_workspace_bytes, cuda_workspace=workspace)
        context.configure_source_consumption_v2(issuer)
        receipts = {}
        for action in actions:
            context.adapter.check_deadline()
            receipt = issuer.compare_native(context, action.segment_id)
            receipts[action.segment_id] = receipt
            audit['comparison_receipts'][action.segment_id] = asdict(receipt)
            complete = (receipt.eligible_ids == receipt.available_ids
                        and set(receipt.eligible_ids) == set(receipt.compared_ids))
            compatible = (receipt.winner_source_id is not None
                          and dict(receipt.scores)[receipt.winner_source_id] <= profile.tau_reuse)
            if complete and compatible:
                context.prepare_selected_v2(receipt)
                audit['preparation'][action.segment_id] = 'WINNER_PREPARED'
            else:
                audit['preparation'][action.segment_id] = ('INCOMPLETE_COMPARISON' if not complete
                                                          else 'NO_COMPATIBLE_SOURCE')
        context.finish_selection(context.frozen, context.prepared)
        ready, union_digest = context.ready_for_final_commit(context.prepared)
        audit['final_admission'] = _final_admission(context, ready, union_digest, audit['events'])
        audit['cancelled_preparation'] = context.cancel_uncommitted_preparation()
        audit['selected_sources'] = dict(context.frozen)
        audit['committed_boundaries'] = dict(context.committed)
        first = []
        audit['answer'] = context.finish(lambda: first.append(time.perf_counter_ns()))
        answer_done = time.perf_counter_ns()
        if not context.finished or len(first) != 1 or not context.arrival_ns <= first[0] <= answer_done:
            raise RuntimeError('native finish must produce exactly one actual first-token endpoint')
        audit['timing'] = dict(arrival_ns=context.arrival_ns, first_token_ns=first[0],
            answer_complete_ns=answer_done, ttft_ms=(first[0]-context.arrival_ns)/1e6,
            answer_service_ms=(answer_done-context.arrival_ns)/1e6,
            semantics='host wall-clock; CUDA envelopes are not additive')
        capture = context.source_capture_v2
        if capture is not None and capture._finalized:
            audit['publication'] = publish_compared_targets_v2(context, issuer, receipts)
        else:
            audit['publication'] = dict(status='SKIPPED', reason='capture_not_finalized', extra_forward_count=0)
        audit['status'] = ('RECOVERY_REQUIRED' if audit['publication'].get('recovery_required') else 'COMPLETED')
    except BaseException as exc:
        error = exc
        audit.update(status='FAILED', failure=dict(type=type(exc).__name__, detail=str(exc)))
    finally:
        if workspace is not None:
            audit['comparison_workspace'] = workspace.audit()
        if getattr(context, 'source_consumption_v2', None) is not None:
            audit['consumption'] = context.source_consumption_v2.audit()
        cleanup = []
        try:
            context.close()
        except BaseException as exc:
            cleanup.append(dict(phase='context_close', type=type(exc).__name__, detail=str(exc)))
            error = error or exc
        if issuer is not None:
            issuer.close()
        # A failed CUDA fence must retain Source/snapshot ownership too.
        if not cleanup:
            try:
                store.end_request(snapshot)
            except BaseException as exc:
                cleanup.append(dict(phase='snapshot_end', type=type(exc).__name__, detail=str(exc)))
                error = error or exc
        audit['cleanup'] = dict(passed=not cleanup, failures=cleanup, snapshot_retained=bool(cleanup))
        audit['service_end_ns'] = time.perf_counter_ns()
        audit['total_service_ms'] = (audit['service_end_ns']-context.arrival_ns)/1e6
    if error is not None:
        audit['status'] = 'FAILED'
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            error.p0_audit = audit
            raise error
        raise P0RequestFailure(audit, error) from error
    return audit
