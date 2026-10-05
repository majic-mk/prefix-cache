"""Read-only QA/headroom and exact shallow-signature diagnostics.

The input is one JSON-serializable, frozen content-group matrix. ``contract``
binds model/code/patch/execution/repair/boundary/timing; ``sources`` describe
immutable artifacts; ``requests`` carry causal visibility, dense/Source QA and
sealed K observations. Every outcome and observation has an evidence_ref with
``path`` and ``sha256``. These refs are preserved, NOT opened or authenticated.

Headroom/safe-coverage definitions match audit_source_dominance.summarize_group;
quality tolerance matches measured_quality_cost_oracle's explicit F1 drop.
This is not that timing oracle: durations remain optional, and neither residual
scores nor deep residual winners substitute for QA. No model/GPU/filesystem
operation, threshold fitting, policy mutation or qualification happens here.
The historical catalog cap of four does NOT include independent S0: S0 remains
an obligatory strong control in the existing P1 matrix, outside this API.
Controlled E/M1 pairs are not natural historical headroom cohorts.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math


_CONTRACT_FIELDS = {
    'model_signature', 'code_commit', 'runtime_worktree_sha256', 'patch_sha256', 'execution_policy_sha256',
    'repair_policy_sha256', 'first_reuse_layer', 'timing_scope',
    'content_group_id', 'content_key', 'comparison_depths', 'max_answer_f1_drop',
}
_TIMING_SCOPES = {'request_ttft', 'executor_prefill_start_not_request_arrival'}
_AUTHORITY_FLAGS = {'paper_evidence', 'native_runtime_qualified', 'gpu_runtime_qualified', 'gpu_execution_allowed',
                    'P1_execution_allowed', 'P1_E_execution_allowed', 'P1_M_execution_allowed'}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def _text(value, name):
    if type(value) is not str or not value.strip():
        raise ValueError(name + ' must be a nonempty string')
    return value


def _sha(value, name, length=64):
    if type(value) is not str or len(value) != length or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(name + ' must be a lowercase digest of length ' + str(length))
    return value


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return value


def _number(value, name, *, unit_interval=False, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(name + ' must be finite and nonnegative')
    if unit_interval and value > 1 or positive and value <= 0:
        raise ValueError(name + ' is outside its valid range')
    return float(value)


def _reference(value):
    if type(value) is not dict or set(value) != {'path', 'sha256'}:
        raise ValueError('evidence_ref requires path and sha256; referenced bytes are not verified')
    _text(value['path'], 'evidence path')
    _sha(value['sha256'], 'evidence digest')


def _reject_authority(value):
    if type(value) is dict:
        for key, child in value.items():
            if key in _AUTHORITY_FLAGS and child is not False:
                raise ValueError('analysis input cannot grant qualification or execution authority')
            _reject_authority(child)
    elif type(value) is list:
        for child in value:
            _reject_authority(child)


def _outcome(value, contract_sha, *, request_id, input_sha256, artifact_sha=None, repair_mask_sha=None):
    if type(value) is not dict:
        raise ValueError('every planned QA cell must exist; missing QA must be explicit null')
    if value.get('contract_sha256') != contract_sha:
        raise ValueError('QA execution/repair/boundary/timing identity differs')
    if value.get('request_id') != request_id or value.get('input_sha256') != input_sha256:
        raise ValueError('QA request/input identity differs')
    if 'repair_mask_sha256' not in value or value['repair_mask_sha256'] != repair_mask_sha:
        raise ValueError('QA actual repair mask differs from frozen request/Source action')
    _reference(value.get('evidence_ref'))
    if artifact_sha is not None and value.get('artifact_sha256') != artifact_sha:
        raise ValueError('QA Source artifact identity differs')
    if 'f1' not in value:
        raise ValueError('missing QA must be explicit null')
    f1 = value['f1']
    if f1 is not None:
        f1 = _number(f1, 'answer F1', unit_interval=True)
    duration = value.get('duration_ms')
    if duration is not None:
        duration = _number(duration, 'duration_ms', positive=True)
    return dict(f1=f1, duration_ms=duration, repair_mask_sha256=repair_mask_sha,
                evidence_ref=copy.deepcopy(value['evidence_ref']))


def _best_ids(scores):
    best = max(scores.values())
    # Same numeric tie slack as the existing post-outcome dominance audit.
    return sorted(s for s, value in scores.items() if value >= best - 1e-12)


def analyze_source_selection_identifiability_v2(payload):
    """Analyze declared measurements; never establish authenticity or a PASS.

    ``observation_sha256`` seals the observation dict without that field using
    canonical JSON (sorted keys, compact separators, UTF-8, ensure_ascii=False).
    Its current/source K digests are tensor-content identities, not receipt IDs.
    A different pool per request is allowed for row diagnostics, but cannot
    produce a global post-hoc best-fixed comparison. Source publication must be
    strictly before request_epoch and birth_request_id must differ.
    """
    if type(payload) is not dict or payload.get('kind') != 'source_selection_identifiability_input_v2':
        raise ValueError('explicit diagnostic input kind required')
    _reject_authority(payload)
    try:
        input_sha = _digest(payload)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('finite JSON-serializable input required') from exc
    if payload.get('evidence_origin') not in {'mock_cpu_test', 'native_measurements_unverified',
                                             'controlled_provenance_diagnostic'}:
        raise ValueError('explicit mock or declared-measurement origin required')
    cohort_kind = payload.get('cohort_kind')
    if cohort_kind not in {'historical_source_cohort', 'controlled_origin_pair'}:
        raise ValueError('explicit historical cohort or controlled origin pair required')
    if payload['evidence_origin'] == 'controlled_provenance_diagnostic' and cohort_kind != 'controlled_origin_pair':
        raise ValueError('controlled provenance cannot be relabeled natural historical headroom')
    c = payload.get('contract')
    if type(c) is not dict or set(c) != _CONTRACT_FIELDS:
        raise ValueError('complete frozen comparison contract required')
    for name in ('model_signature', 'content_group_id', 'content_key'):
        _text(c[name], name)
    _sha(c['code_commit'], 'code_commit', 40)
    for name in ('runtime_worktree_sha256', 'patch_sha256', 'execution_policy_sha256', 'repair_policy_sha256'):
        _sha(c[name], name)
    _integer(c['first_reuse_layer'], 'first_reuse_layer', 1)
    if c['timing_scope'] not in _TIMING_SCOPES:
        raise ValueError('explicit known first-token timing scope required')
    tolerance = _number(c['max_answer_f1_drop'], 'max_answer_f1_drop', unit_interval=True)
    depths = c['comparison_depths']
    if type(depths) is not list or not depths or any(type(d) is not int or d < 1 for d in depths):
        raise ValueError('explicit legal completed depths required')
    if depths != sorted(set(depths)):
        raise ValueError('comparison depths must be ordered and unique')
    contract_sha = _digest(c)
    if payload.get('contract_sha256') != contract_sha:
        raise ValueError('frozen contract digest mismatch')
    sources = payload.get('sources')
    if type(sources) is not list or not 1 <= len(sources) <= 4:
        raise ValueError('v2 fixed cohort contains one to four Sources')
    by_id = {}
    for source in sources:
        if type(source) is not dict:
            raise ValueError('Source descriptor required')
        sid = _text(source.get('source_id'), 'source_id')
        if sid in by_id or source.get('content_key') != c['content_key']:
            raise ValueError('duplicate Source or different exact content')
        _sha(source.get('artifact_sha256'), 'artifact_sha256')
        _integer(source.get('generation'), 'generation')
        if source['generation'] not in (0, 1):
            raise ValueError('G2/unknown is outside this v2 stored-pool diagnostic')
        _integer(source.get('publication_epoch'), 'publication_epoch')
        _text(source.get('birth_request_id'), 'birth_request_id')
        _reference(source.get('evidence_ref'))
        states = source.get('selection_k_sha256_by_depth')
        if type(states) is not dict or set(states) != {str(d) for d in depths}:
            raise ValueError('Source must identify all frozen shallow K states')
        for value in states.values():
            _sha(value, 'source K digest')
        by_id[sid] = source
    requests = payload.get('requests')
    if type(requests) is not list or not requests:
        raise ValueError('nonempty request matrix required')
    rows, request_ids, last_epoch = [], set(), -1
    signature_classes = {}
    candidate_collisions = []
    for request in requests:
        if type(request) is not dict:
            raise ValueError('request descriptor required')
        rid = _text(request.get('request_id'), 'request_id')
        epoch = _integer(request.get('request_epoch'), 'request_epoch')
        if rid in request_ids or epoch <= last_epoch:
            raise ValueError('requests must have unique identities and strictly increasing epochs')
        request_ids.add(rid)
        last_epoch = epoch
        input_digest = _sha(request.get('input_sha256'), 'request input digest')
        visible = request.get('visible_source_ids')
        if (type(visible) is not list or any(type(s) is not str for s in visible)
                or len(set(visible)) != len(visible) or not set(visible) <= set(by_id)):
            raise ValueError('visible source cohort is invalid')
        visible = sorted(visible)
        for sid in visible:
            if by_id[sid]['publication_epoch'] >= epoch or by_id[sid]['birth_request_id'] == rid:
                raise ValueError('future/current-request Source cannot be visible')
        masks = request.get('expected_repair_mask_sha256_by_source')
        if type(masks) is not dict or set(masks) != set(visible):
            raise ValueError('frozen per-Source repair mask inventory must match visible cohort')
        for mask in masks.values():
            _sha(mask, 'frozen actual repair mask digest')
        dense = _outcome(request.get('dense'), contract_sha, request_id=rid, input_sha256=input_digest)
        cells = request.get('source_outcomes')
        if type(cells) is not dict or set(cells) != set(visible):
            raise ValueError('QA cell inventory must equal actually visible candidate cohort')
        outcomes = {s: _outcome(cells[s], contract_sha, request_id=rid, input_sha256=input_digest,
                                artifact_sha=by_id[s]['artifact_sha256'], repair_mask_sha=masks[s])
                    for s in visible}
        complete = bool(visible) and dense['f1'] is not None and all(x['f1'] is not None for x in outcomes.values())
        f1 = {s: x['f1'] for s, x in outcomes.items()}
        safe = ([s for s in visible if f1[s] >= dense['f1']-tolerance-1e-12] if complete else None)
        best = _best_ids(f1) if complete else None
        observations = request.get('observations')
        if type(observations) is not list or len(observations) != len(depths):
            raise ValueError('one observation for every frozen depth required')
        seen_depths, selected_rows = set(), {}
        for observation in observations:
            if type(observation) is not dict:
                raise ValueError('observation descriptor required')
            digest = _sha(observation.get('observation_sha256'), 'observation digest')
            if digest != _digest({k: v for k, v in observation.items() if k != 'observation_sha256'}):
                raise ValueError('observation content digest mismatch')
            d = observation.get('completed_depth')
            if type(d) is not int or d not in depths or d in seen_depths:
                raise ValueError('observation depth differs or repeats')
            seen_depths.add(d)
            if (observation.get('contract_sha256') != contract_sha or observation.get('request_id') != rid
                    or observation.get('input_sha256') != input_digest):
                raise ValueError('observation request or frozen policy identity differs')
            _reference(observation.get('evidence_ref'))
            current_k = _sha(observation.get('current_k_sha256'), 'current K digest')
            source_k = observation.get('source_k_sha256_by_id')
            expected_k = {s: by_id[s]['selection_k_sha256_by_depth'][str(d)] for s in visible}
            if source_k != expected_k:
                raise ValueError('observation Source K identities or candidate coverage differ')
            scores = observation.get('scores')
            if type(scores) is not dict or set(scores) != set(visible):
                raise ValueError('scores must cover exactly the actually visible candidates')
            for score in scores.values():
                _number(score, 'residual score')
            selected = observation.get('selected_source_id')
            if selected is not None and (selected not in visible or scores[selected] != min(scores.values())):
                raise ValueError('selected Source must be a measured residual minimum or explicit abstention')
            selected_rows[str(d)] = dict(selected_source_id=selected,
                selected_f1=f1.get(selected),
                qa_regret=(max(f1.values())-f1[selected] if complete and selected is not None else None),
                qa_oracle_tie_hit=(selected in best if complete and selected is not None else None),
                selected_quality_safe=(selected in safe if complete and selected is not None else None),
                status='EVALUATED' if complete and selected is not None else 'NOT_EVALUATED',
                observation_sha256=digest, evidence_ref=copy.deepcopy(observation['evidence_ref']))
            signature = _digest(dict(contract=contract_sha, depth=d, current_k=current_k,
                                     candidate_ids=visible, source_k=source_k))
            signature_classes.setdefault((d, signature), []).append(dict(
                request_id=rid, f1_by_source=f1, qa_complete=complete, qa_oracle_ids=best,
                scores=copy.deepcopy(scores), selected_source_id=selected))
            for index, left in enumerate(visible):
                for right in visible[index+1:]:
                    if source_k[left] != source_k[right]:
                        continue
                    different_artifact = by_id[left]['artifact_sha256'] != by_id[right]['artifact_sha256']
                    qa_difference = (abs(f1[left]-f1[right]) if f1[left] is not None and f1[right] is not None else None)
                    if different_artifact or qa_difference is not None and qa_difference > 1e-12:
                        candidate_collisions.append(dict(request_id=rid, completed_depth=d,
                            source_ids=[left, right], selection_k_sha256=source_k[left],
                            artifact_differs=different_artifact, absolute_f1_difference=qa_difference,
                            scores_disagree=scores[left] != scores[right],
                            interpretation=('SCORE_OR_OBSERVATION_BINDING_INCONSISTENCY'
                                if scores[left] != scores[right] else 'EXACT_K_COLLISION_DIAGNOSTIC'),
                            shallow_information_insufficiency_evaluable=scores[left] == scores[right],
                            numerical_error_bound_implied=False))
        rows.append(dict(request_id=rid, request_epoch=epoch, visible_source_ids=visible,
            qa_status='EVALUATED' if complete else 'NOT_EVALUATED', dense=dense,
            source_outcomes=outcomes, oracle_max_f1=(max(f1.values()) if complete else None),
            oracle_best_source_ids=best, quality_safe_source_ids=safe, selection_by_depth=selected_rows))

    fixed = bool(rows[0]['visible_source_ids']) and all(r['visible_source_ids'] == rows[0]['visible_source_ids'] for r in rows)
    full_qa = all(r['qa_status'] == 'EVALUATED' for r in rows)
    summary = dict(status=('NOT_APPLICABLE_CONTROLLED_ORIGIN_PAIR' if cohort_kind == 'controlled_origin_pair'
                           else 'UNSUPPORTED_FIXED_COHORT' if not fixed else 'NOT_EVALUATED'),
        request_count=len(rows), qa_complete_request_count=sum(r['qa_status']=='EVALUATED' for r in rows),
        oracle_mean_f1=None, posthoc_best_fixed_mean_f1=None, headroom_f1=None,
        oracle_safe_coverage=None, best_fixed_safe_coverage=None, safe_coverage_headroom=None,
        fixed_mean_f1=None, fixed_safe_coverage=None, selection_by_depth=None)
    if fixed and full_qa and cohort_kind == 'historical_source_cohort':
        cohort = rows[0]['visible_source_ids']
        means = {s: math.fsum(r['source_outcomes'][s]['f1'] for r in rows)/len(rows) for s in cohort}
        coverage = {s: sum(s in r['quality_safe_source_ids'] for r in rows)/len(rows) for s in cohort}
        oracle_mean = math.fsum(r['oracle_max_f1'] for r in rows)/len(rows)
        oracle_safe = sum(bool(r['quality_safe_source_ids']) for r in rows)/len(rows)
        selections = {}
        for d in depths:
            values = [r['selection_by_depth'][str(d)] for r in rows]
            selected = [r for r in values if r['selected_source_id'] is not None]
            selections[str(d)] = dict(request_count=len(rows), selected_count=len(selected),
                mean_qa_regret=(math.fsum(r['qa_regret'] for r in selected)/len(selected) if selected else None),
                regret_denominator='selected requests only; abstentions explicitly counted',
                abstention_count=len(rows)-len(selected),
                oracle_tie_hit_count=sum(r['qa_oracle_tie_hit'] for r in selected),
                selected_quality_safe_coverage=sum(r['selected_quality_safe'] for r in selected)/len(rows))
        summary.update(status='EVALUATED', oracle_mean_f1=oracle_mean,
            posthoc_best_fixed_mean_f1=max(means.values()), headroom_f1=oracle_mean-max(means.values()),
            oracle_safe_coverage=oracle_safe, best_fixed_safe_coverage=max(coverage.values()),
            safe_coverage_headroom=oracle_safe-max(coverage.values()), fixed_mean_f1=means,
            fixed_safe_coverage=coverage, selection_by_depth=selections)
    cross_request = []
    for (depth, signature), group in sorted(signature_classes.items()):
        if len(group) < 2:
            continue
        measured = all(r['qa_complete'] for r in group)
        common = (sorted(set.intersection(*(set(r['qa_oracle_ids']) for r in group))) if measured else None)
        scores_consistent = all(r['scores'] == group[0]['scores'] for r in group)
        choices_consistent = all(r['selected_source_id'] == group[0]['selected_source_id'] for r in group)
        consistent = scores_consistent and choices_consistent
        cross_request.append(dict(completed_depth=depth, shallow_signature_sha256=signature,
            request_ids=[r['request_id'] for r in group], common_qa_oracle_source_ids=common,
            no_common_qa_optimal_source=(not common if measured and consistent else None),
            qa_vectors_differ=(any(r['f1_by_source'] != group[0]['f1_by_source'] for r in group[1:]) if measured else None),
            identical_signature_scores_consistent=scores_consistent,
            identical_signature_choices_consistent=choices_consistent,
            shallow_information_insufficiency_evaluable=consistent and measured,
            status=('INCONSISTENT_SCORE_OR_SELECTION_BINDING' if not consistent
                    else 'EVALUATED' if measured else 'NOT_EVALUATED')))
    report = dict(kind='source_selection_identifiability_audit_v2', input_sha256=input_sha,
        contract=copy.deepcopy(c), contract_sha256=contract_sha, evidence_origin=payload['evidence_origin'],
        cohort_kind=cohort_kind, historical_headroom_applicable=cohort_kind == 'historical_source_cohort',
        sources=copy.deepcopy(sources),
        summary=summary, requests=rows, candidate_shallow_collisions=candidate_collisions,
        cross_request_shallow_equivalence=cross_request,
        oracle_definition='actual declared answer-F1 maximum; never minimum deep residual',
        posthoc_best_fixed_is_diagnostic_upper_bound=True,
        references_verified=False, inline_observation_digests_verified=True,
        evidence_authenticity='external evidence refs and tensor digests are declarations, not independently verified',
        timing_oracle_evaluated=False, threshold_fitted=False, no_gpu_or_model_execution=True,
        independent_S0_control='outside this bounded API; must remain in the P1 matrix',
        natural_multisource_go_decision=None, native_runtime_qualified=False,
        source_publication_performed=False, paper_evidence=False, gpu_runtime_qualified=False,
        gpu_execution_allowed=False, P1_execution_allowed=False)
    report['report_sha256'] = _digest(report)
    return report
