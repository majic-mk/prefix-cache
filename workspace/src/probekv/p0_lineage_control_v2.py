"""Bounded, isolated Source-lineage exercise; never production economics.

Unlike teacher/reference comparisons, this performs one ordinary request and
captures its full downstream target in that very execution. The preregistered
upstream must still win an authentic K comparison and hold normal physical
ownership. Only this explicit P0 operation may exercise a recipe commit without
a measured economic decision. No diagnostic reference output enters the pool.
"""
from dataclasses import asdict
import math
import time

from .native_p0_request_v2 import _validate, P0RequestFailure
from .native_publication_v2 import publish_compared_targets_v2
from .source_comparison_v2 import ComparisonSessionV2
from .v8_schema6_hbm import HBMReservationKind


def lineage_targets(job):
    """Explicit bounded multi-target recipe; preserve old single-target format."""
    if 'target_ids' in job:
        if 'target_id' in job or 'expected_target_generation' in job:
            raise ValueError('do not mix single and multi-target recipe formats')
        targets = job['target_ids']
        generations = job.get('expected_target_generations', {})
        if (type(targets) is not list or not 1 <= len(targets) <= 10
                or any(type(t) is not str for t in targets)
                or len(set(targets)) != len(targets)
                or set(generations) != set(targets)):
            raise ValueError('one to ten explicit targets and complete generation table required')
        return tuple(targets), generations
    return (job.get('target_id'),), {job.get('target_id'): job.get('expected_target_generation')}


def lineage_parents(job):
    if 'reuse_segment_ids' in job:
        if 'upstream_segment_id' in job or 'expected_parent_generation' in job:
            raise ValueError('do not mix single and multiple parent formats')
        parents = job['reuse_segment_ids']
        generations = job.get('expected_parent_generations', {})
        if (type(parents) is not list or not 1 <= len(parents) <= 9
                or any(type(s) is not str for s in parents)
                or len(set(parents)) != len(parents) or set(generations) != set(parents)):
            raise ValueError('one to nine explicit reused segments and generation table required')
        return tuple(parents), generations
    parent = job.get('upstream_segment_id')
    return (parent,), {parent: job.get('expected_parent_generation')}


def validate_lineage_job(job, *, total_layers=None):
    q = job['request']
    if job.get('parent_preparation_mode', 'residual_admitted') not in (
            'residual_admitted', 'preregistered_source_correctness_only'):
        raise ValueError('unknown controlled parent preparation mode')
    if (job.get('operation') != 'controlled_lineage_birth'
            or job.get('input_origin') != 'controlled_provenance_diagnostic'
            or job.get('economic_admission_evaluated') is not False
            or job.get('production_reuse_commit_observed') is not False
            or 'teacher_token_ids' in q or q.get('capture_logits')
            or q.get('max_new_tokens') != 1 or q.get('native_dense_continuation')
            or q.get('capture_original_full_prefill') or q.get('publish_exact_prefix_shadow')
            or q.get('use_gpu_hot_cache') or q.get('retain_gpu_hot_cache')
            or q.get('prefetch_window', 0) != 0
            or q.get('correctness_repair_ratio', .15) != .15):
        raise ValueError('explicit isolated ordinary lineage request required; no teacher/reference publication')
    segments = {s['segment_id']: s for s in q['segments']}
    parents, parent_generations = lineage_parents(job)
    targets, generations = lineage_targets(job)
    if (len(segments) != len(q['segments']) or set(parents).intersection(targets)
            or any(t not in segments for t in (*parents, *targets))
            or set(job.get('sources_by_segment', {})) != {*parents, *targets}
            or any(len(job['sources_by_segment'][p]) != 1 for p in parents)):
        raise ValueError('preregistered reused segments and separate full targets required')
    occupied = set()
    for sid in (*parents, *targets):
        s = segments[sid]; rows = s['positions']
        if (not rows or any(type(p) is not int or not 0 <= p < len(q['token_ids']) for p in rows)
                or rows != list(range(rows[0], rows[-1]+1))
                or s['token_ids'] != [q['token_ids'][p] for p in rows]
                or occupied.intersection(rows)):
            raise ValueError('unchanged contiguous target token/position identity required')
        occupied.update(rows)
    if (max(occupied) >= len(q['token_ids'])-1
            or any(math.ceil(.15*len(segments[p]['positions'])) >= len(segments[p]['positions']) for p in parents)):
        raise ValueError('real sparse reused segment, full targets and mandatory suffix required')
    if (any(type(g) is not int or g not in (0, 1) for g in parent_generations.values())
            or job['comparison_profile']['provenance_policy'] != 'ALLOW_MIXED_G1'):
        raise ValueError('explicit G0->G1 or G1->G2 isolated propagation required')
    for target in targets:
        preceding = [parent_generations[p]+1 for p in parents
                     if segments[p]['positions'][-1] < segments[target]['positions'][0]]
        expected = max(preceding, default=0)
        if (type(generations[target]) is not int or generations[target] != expected
                or (not preceding and 'target_ids' not in job)):
            raise ValueError('target generation must match causal position, without per-target increment')
    depth = job['comparison_profile']['completed_depth']
    if type(depth) is not int or depth < 1 or (total_layers is not None and depth >= total_layers):
        raise ValueError('legal d+1 selective boundary required')


def _commit_recipe(context, sid):
    """No forged Planner decision: named controlled operation only."""
    c = context
    parents = (sid,) if isinstance(sid, str) else tuple(sid)
    if (not parents or len(set(parents)) != len(parents) or c.finished or c.closed
            or c.committed or set(c.prepared) != set(parents)):
        raise ValueError('lineage recipe requires exactly its live prepared parents')
    boundary = c.current_completed_depth+1
    # Check all ownership/support first. Any execution exception aborts the
    # request and uses normal cleanup; never continue after a partial commit.
    rows = []
    for parent in parents:
        if parent in getattr(c.source_consumption_v2, 'controlled_recipe_parents', {}):
            c.source_consumption_v2.assert_can_commit(parent, controlled_recipe=True)
        else:
            c.source_consumption_v2.assert_can_commit(parent)
        support = tuple(c.supports[parent][boundary])
        positions = tuple(c.segments[parent]['positions'])
        if len(support) != math.ceil(.15*len(positions)):
            raise ValueError('lineage operation must retain fixed15 repair')
        rows.append((parent,positions,support))
    for parent,positions,support in sorted(rows, key=lambda row:row[1][0]):
        c.engine.commit_ready_segment(segment_id=parent, boundary=boundary,
            segment_positions=positions, repair_positions=support, scheduler_boundary=boundary)
        c.adapter.hbm.promote(c.replica_reservations[parent].reservation_id,
            expected=HBMReservationKind.WINNER_PREFETCH, target=HBMReservationKind.COMMITTED_EXECUTION)
        c.committed[parent] = boundary
        c.generation += 1


def _observe_and_prepare_parents(context, store, snapshot, issuer, parents,
                                 expected, parent_generations, audit, job=None):
    """Prepare changes context generation; never freeze later stale receipts.

    Each parent observes the current state immediately before its own freeze.
    No layer is advanced here and no previously issued receipt is relabeled.
    """
    receipts = {}
    for parent in parents:
        receipt = issuer.compare_native(context, parent)
        receipts[parent] = receipt
        audit['comparison_receipts'][parent] = asdict(receipt)
        ids = expected[parent]
        if (tuple(receipt.eligible_ids) != ids or receipt.winner_source_id != ids[0]
                or tuple(receipt.available_ids) != ids or tuple(receipt.compared_ids) != ids):
            raise ValueError('lineage recipe parent not the unique eligible measured winner')
        if store.parent_metadata(snapshot, ids[0]).generation != parent_generations[parent]:
            raise ValueError('lineage parent generation differs from frozen action')
        if job is not None and job.get('parent_preparation_mode') == 'preregistered_source_correctness_only':
            context.source_consumption_v2.prepare_controlled_parent(receipt, job=job)
        else:
            context.prepare_selected_v2(receipt)
    return receipts


def execute_lineage_birth(context, store, snapshot, profile, actions, job, *, host_capture_bytes):
    validate_lineage_job(job, total_layers=context.adapter.spec.num_layers)
    _validate(context, store, snapshot, profile, actions, host_capture_bytes)
    from .p0_diagnostic_context_v2 import diagnostic_context_v2
    if (context.request != job['request'] or diagnostic_context_v2(context) is not None
            or store.config['policy'] != 'ALLOW_MIXED_G1'):
        raise ValueError('ordinary matching request and isolated mixed-capable pool required')
    parents, parent_generations = lineage_parents(job)
    targets, generations = lineage_targets(job)
    expected = {p:next(a for a in actions if a.segment_id == p).source_ids for p in parents}
    audit = dict(kind='controlled_lineage_birth_v2', status='RUNNING',
        request_id=context.request['request_id'], comparison_receipts={}, answer=None,
        publication=None, cleanup=None, extra_forward_count=0,
        economic_admission_evaluated=False, production_reuse_commit_observed=False,
        controlled_recipe_commit_observed=False, native_runtime_qualified=False,
        P1_execution_allowed=False, paper_evidence=False)
    issuer = workspace = None
    error = None
    try:
        context.configure_target_capture_v2(target_ids=targets, registry=store.registry,
            authorization_domain=store.config['authorization_domain'], host_budget_bytes=host_capture_bytes)
        context.advance_to_depth(profile.completed_depth)
        workspace = context.configure_cuda_comparison_v2(capacity_bytes=actions[0].cuda_workspace_bytes)
        issuer = ComparisonSessionV2(store, snapshot, profile,
            workspace_bytes=actions[0].host_workspace_bytes, cuda_workspace=workspace)
        context.configure_source_consumption_v2(issuer)
        receipts = _observe_and_prepare_parents(context, store, snapshot, issuer,
            parents, expected, parent_generations, audit, job=job)
        for target in targets:
            receipts[target] = issuer.compare_native(context, target)
            audit['comparison_receipts'][target] = asdict(receipts[target])
        context.finish_selection(context.frozen, context.prepared)
        context.ready_for_final_commit(context.prepared)
        _commit_recipe(context, parents)
        audit['controlled_recipe_commit_observed'] = True
        audit['selected_sources'] = {p:expected[p][0] for p in parents}
        if len(parents) == 1:
            audit['selected_source'] = expected[parents[0]][0]
        audit['repair_support'] = context.supports
        endpoints = []
        audit['answer'] = context.finish(lambda: endpoints.append(time.perf_counter_ns()))
        if not context.finished or len(endpoints) != 1:
            raise RuntimeError('ordinary lineage request did not finish exactly once')
        capture = context.source_capture_v2
        proofs = {t: capture.ledger.target_proof(capture.targets[t]) for t in targets}
        audit['target_proofs'] = {t: asdict(p) for t,p in proofs.items()}
        if len(targets) == 1:
            audit['target_proof'] = asdict(proofs[targets[0]])
        audit['capture'] = capture.audit()
        for target, proof in proofs.items():
            origin = 'EXACT_CONTEXT' if generations[target] == 0 else 'MIXED_CONTEXT_FULL_SEGMENT'
            if (proof.origin.value != origin or proof.generation != generations[target]
                    or capture._submitted_layers != context.adapter.spec.num_layers):
                raise ValueError('actual full-target propagation differs from frozen expectation')
        audit['publication'] = publish_compared_targets_v2(context, issuer, {t: receipts[t] for t in targets})
        audit['visibility_by_target'] = {}
        audit['rejection_by_target'] = {}
        for target in targets:
            published = audit['publication']['targets'][target]
            if generations[target] <= 1:
                if published.get('publication_performed') is not True:
                    raise ValueError('full target was not atomically published: '+str(published))
                seen = any(r['source_id'] == published['source_id'] for r in store.lookup(snapshot, receipts[target].token_ids))
                audit['visibility_by_target'][target] = seen
                if seen:
                    raise ValueError('birth request saw its newly published Source')
            else:
                if target in context.target_candidates_v2 or published.get('publication_performed') is not False:
                    raise ValueError('G2 entered publishable candidates or the Source pool')
                reason = capture.rejections.get(target)
                audit['rejection_by_target'][target] = reason
                if not reason:
                    raise ValueError('missing explicit G2 rejection evidence')
        if len(targets) == 1:
            audit['birth_snapshot_sees_child'] = audit['visibility_by_target'].get(targets[0])
            audit['g2_rejection_reason'] = audit['rejection_by_target'].get(targets[0])
        audit['status'] = 'COMPLETED'
    except BaseException as exc:
        error = exc
        audit.update(status='FAILED', failure=dict(type=type(exc).__name__, detail=str(exc)))
    finally:
        if workspace is not None:
            audit['comparison_workspace'] = workspace.audit()
        if getattr(context, 'source_consumption_v2', None) is not None:
            audit['consumption'] = context.source_consumption_v2.audit()
            audit['consumption']['economic_admission_evaluated'] = False
            audit['consumption']['controlled_recipe_not_production_admission'] = True
        failures = []
        try:
            context.close()
        except BaseException as exc:
            failures.append(dict(phase='context_close', detail=str(exc))); error = error or exc
        if issuer is not None:
            issuer.close()
        if not failures:
            try:
                store.end_request(snapshot)
            except BaseException as exc:
                failures.append(dict(phase='snapshot_end', detail=str(exc))); error = error or exc
        audit['cleanup'] = dict(passed=not failures, failures=failures, snapshot_retained=bool(failures))
    if error is not None:
        audit['status'] = 'FAILED'
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            error.p0_audit = audit
            raise error
        raise P0RequestFailure(audit, error) from error
    return audit
