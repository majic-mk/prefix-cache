"""Build bounded controlled-P0 jobs from real assets, never grant GPU authority.

No weights/GPU load, Source construction, pool mutation, or file writes. Inputs
are literal controlled requests and an already opened isolated Source store.
The native entry must still re-observe GPU/imports and the authority window.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re

from .p0_batch_v2 import validate_p0_batch
from .p0_stage_readiness_v2 import validate_p0_model_geometry
from .source_comparison_v2 import runtime_binding_digest
from .source_manifest_v2 import RequestManifestRegistry
from .source_store_v2 import TargetSourceStoreV2
from .v8_schema10_execution import digest_json
from .v8_schema10_native_factory import validate_native_attachment, verified_model_asset_path
from .v8_schema10_native_adapter import validate_native_sampling_request
from .v8_schema10_storage import file_digest


_EXACT_POLICY = {'relative_l2_limit', 'minimum_positions', 'require_predicted_token_ids_equal'}
_MIXED_POLICY = _EXACT_POLICY | {'target_relative_l2_limit', 'target_absolute_max_limit',
                               'target_read_bytes'}
_CONTROL = {'control_id', 'request', 'upper_seconds_by_arm'}
_EXACT = _CONTROL | {'target_ids'}
_BIRTH = {'control_id', 'request', 'upper_seconds', 'target_ids', 'comparison_profile'}
_MIXED = _CONTROL | {'upstream_segment_id', 'target_id', 'source', 'first_reuse_layer',
                     'repair_positions_by_layer'}


def _safe(value):
    return type(value) is str and re.fullmatch('[A-Za-z0-9_-]{1,48}', value)


def _literal_json(value):
    # Reject non-JSON values/NaN before copying; JSON equality must not silently
    # change tuple masks, integer layer keys, or user-provided request objects.
    copied = json.loads(json.dumps(value, allow_nan=False))
    if type(value) is not dict or copied != value:
        raise ValueError('literal JSON object required; no implicit recipe coercion')
    return deepcopy(value)


def _pair(policy, control_id, reference, candidate):
    return dict(policy, pair_id=control_id, reference_action_id=reference,
                candidate_action_id=candidate)


def _request(request):
    if type(request) is not dict:
        raise ValueError('literal actual controlled request required')
    validate_native_sampling_request(request)
    tokens, segments = request.get('token_ids'), request.get('segments')
    if type(tokens) is not list or not tokens or type(segments) is not list or not segments:
        raise ValueError('nonempty original tokens and explicit Segment inventory required')
    identities, occupied = set(), set()
    for segment in segments:
        sid, positions = segment.get('segment_id'), segment.get('positions')
        if (type(sid) is not str or not sid or sid in identities
                or type(positions) is not list or not positions
                or any(type(p) is not int or not 0 <= p < len(tokens) for p in positions)
                or positions != list(range(positions[0], positions[-1]+1))
                or occupied.intersection(positions)
                or segment.get('token_ids') != [tokens[p] for p in positions]):
            raise ValueError('Segment tokens/absolute positions must match unique nonoverlapping original slices')
        identities.add(sid); occupied.update(positions)


def _job(control, arm, operation, **fields):
    action_id = control['control_id'] + '_' + arm
    request = deepcopy(control['request'])
    # The input recipe is preserved. Only per-arm request identity is derived;
    # all tokens/teacher conditioning/geometry are compared by existing rules.
    if not isinstance(request.get('request_id'), str) or not request['request_id']:
        raise ValueError('original controlled request identity required')
    request['request_id'] = request['request_id'] + ':' + action_id
    return dict(action_id=action_id, operation=operation, request=request,
        request_sha256=digest_json(request), input_origin='controlled_provenance_diagnostic',
        upper_seconds=control['upper_seconds_by_arm'][arm], **deepcopy(fields))


def _source(store, ref, request, segment_id, config, layers, *, verification):
    if (type(ref) is not dict or set(ref) != {'source_id', 'artifact_digest'}
            or type(ref['source_id']) is not str or not ref['source_id']):
        raise ValueError('literal Source ID and artifact digest required, no birth-action references')
    row = store._catalog['rows'].get(ref['source_id'])
    if row is None:
        raise ValueError('Source is not already published in the bound isolated pool')
    segments = {s['segment_id']: s for s in request['segments']}
    segment = segments[segment_id]
    positions = segment['positions']
    tokens = [request['token_ids'][p] for p in positions]
    heads = config.get('num_key_value_heads')
    qheads, hidden = config.get('num_attention_heads'), config.get('hidden_size')
    dim = config.get('head_dim')
    if dim is None and type(qheads) is int and qheads > 0 and type(hidden) is int and hidden % qheads == 0:
        dim = hidden // qheads
    if (type(heads) is not int or heads <= 0 or type(qheads) is not int or qheads % heads
            or type(dim) is not int or dim <= 0):
        raise ValueError('explicit actual GQA geometry required for Source binding')
    if (row['artifact_digest'] != ref['artifact_digest'] or row['origin'] != 'EXACT_CONTEXT'
            or type(row['generation']) is not int or row['generation'] != 0
            or row['token_ids'] != tokens or row['layer_count'] != layers
            or row['shape'] != [len(tokens), heads, dim]
            or row['model_signature'] != store.config['model_signature']
            or row['birth_request_id'] == request['request_id']
            or not 0 < row['publication_epoch'] <= store._catalog['epoch']):
        raise ValueError('Source G0/token/Artifact/model/geometry/chronology identity differs')
    if (row['target_kv_bytes'] + row['selection_state_bytes']) * 3 > store.config['staging_bytes']:
        raise MemoryError('full Source verification exceeds explicit host staging budget')
    if row['source_id'] not in verification:
        store._verify_row(row, full=True)
        verification[row['source_id']] = dict(source_id=row['source_id'],
            artifact_digest=row['artifact_digest'], origin=row['origin'], generation=row['generation'],
            publication_epoch=row['publication_epoch'], layer_count=row['layer_count'],
            kv_file_sha256=row['kv_file_digest'], selection_file_sha256=row['selection_file_digest'])


def build_controlled_p0_manifest(*, native_manifest_path, native_manifest_sha256, instance_id,
        store, authority, limits, registry_budget, numerical_policy,
        exact_controls, mixed_controls, now_unix, birth_controls=None):
    """Return ``{manifest, preflight}``; missing prerequisites raise, not default.

    A mixed control emits T21 full-query/reference versus independent sparse
    arms. An explicit ``target_r1_source`` adds a separate target-r1 pair. Every
    arm has its own declared upper duration; no Source is constructed here.
    ``numerical_policy`` has policy_id/rationale and exact/mixed objects (null
    for an unused branch). Numeric thresholds are never chosen by this helper.
    Optional birth_controls are separate ordinary source_request actions against
    an empty isolated pool, never exports from the extra reference forwards.
    """
    if type(store) is not TargetSourceStoreV2:
        raise ValueError('actual opened isolated v2 Source store required')
    if type(instance_id) is not str or not instance_id.strip():
        raise ValueError('explicit current authorized instance identity required')
    path = Path(native_manifest_path)
    if (not path.is_file() or path.stat().st_size > 32 * 1024 * 1024
            or file_digest(path) != native_manifest_sha256):
        raise ValueError('actual bounded native manifest file/hash required')
    native = RequestManifestRegistry._parse(path.read_bytes())
    runtime = validate_native_attachment(native, allow_unmeasured=True)
    source = runtime['source_provenance']
    if (source['model_signature'] != store.config['model_signature']
            or source['tokenizer_hash'] != store.config['tokenizer_hash']):
        raise ValueError('isolated pool differs from native model/tokenizer')
    births = [] if birth_controls is None else birth_controls
    if (type(exact_controls) is not list or type(mixed_controls) is not list or type(births) is not list
            or not exact_controls and not mixed_controls):
        raise ValueError('explicit nonempty finite controlled-P0 recipe lists required')
    # Literal tokens/positions remain immutable; derive the required native
    # occurrence key using this actual v2 namespace, not legacy caller labels.
    # The original request digest below is retained before metadata derivation.
    exact_controls = deepcopy(exact_controls)
    mixed_controls = deepcopy(mixed_controls)
    births = deepcopy(births)
    policy = _literal_json(numerical_policy)
    if (set(policy) != {'policy_id', 'rationale', 'exact', 'mixed'}
            or not _safe(policy['policy_id']) or type(policy['rationale']) is not str
            or not policy['rationale'].strip()):
        raise ValueError('preregistered named numerical policy with rationale required')
    for name, controls, fields in (('exact', exact_controls, _EXACT_POLICY),
                                    ('mixed', mixed_controls, _MIXED_POLICY)):
        if controls:
            if type(policy[name]) is not dict or set(policy[name]) != fields:
                raise ValueError('explicit exact/mixed numerical policy fields required')
        elif policy[name] is not None:
            raise ValueError('unused numerical policy branch must be null')
    jobs, exact_pairs, mixed_pairs, original_inputs, seen = [], [], [], [], set()
    for control in exact_controls + mixed_controls + births:
        if not _safe(control.get('control_id')) or control['control_id'] in seen:
            raise ValueError('unique safe controlled case IDs required')
        seen.add(control['control_id'])
        _request(control['request'])
        original_inputs.append(dict(control_id=control['control_id'],
            input_request_sha256=digest_json(control['request'])))
        for segment in control['request']['segments']:
            expected = store.content_key(segment['token_ids'])
            if 'content_key' in segment and segment['content_key'] != expected:
                raise ValueError('supplied Segment content key differs from actual v2 namespace')
            segment['content_key'] = expected
    for raw in exact_controls:
        control = _literal_json(raw)
        if set(control) != _EXACT or set(control['upper_seconds_by_arm']) != {'off', 'on'}:
            raise ValueError('T20 needs only explicit off/on durations and capture targets')
        arms = [_job(control, arm, 'exact_capture_control',
                     capture_enabled=capture, target_ids=control['target_ids'])
                for arm, capture in (('off', False), ('on', True))]
        jobs.extend(arms)
        exact_pairs.append(_pair(policy['exact'], control['control_id'],
                                  arms[0]['action_id'], arms[1]['action_id']))
    for raw in mixed_controls:
        control = _literal_json(raw)
        r1 = 'target_r1_source' in control
        endpoint_fields = {'upstream_repair_endpoint'} if 'upstream_repair_endpoint' in control else set()
        names = {'reference', 'sparse'} | ({'r1_reference', 'r1_sparse'} if r1 else set())
        if (set(control) != _MIXED | ({'target_r1_source'} if r1 else set()) | endpoint_fields
                or set(control['upper_seconds_by_arm']) != names):
            raise ValueError('T21 needs explicit masks/Source and duration for every full/r1 arm')
        common = {k:control[k] for k in ('upstream_segment_id', 'target_id', 'source',
                   'first_reuse_layer', 'repair_positions_by_layer')}
        if endpoint_fields:
            common['upstream_repair_endpoint'] = control['upstream_repair_endpoint']
        for target_r1 in ([False, True] if r1 else [False]):
            prefix = 'r1_' if target_r1 else ''
            fields = dict(common, target_execution='R1_ALL_LAYERS' if target_r1 else 'FULL_ALL_LAYERS')
            if target_r1:
                fields['target_source'] = control['target_r1_source']
            arms = [_job(control, prefix+arm, operation, **fields) for arm, operation in
                    (('reference', 'explicit_mixed_reference'), ('sparse', 'mixed_sparse_control'))]
            jobs.extend(arms)
            mixed_pairs.append(_pair(policy['mixed'], control['control_id']+('_r1' if target_r1 else ''),
                                      arms[0]['action_id'], arms[1]['action_id']))
    if births and (mixed_controls or not exact_controls):
        raise ValueError('Source births belong to first exact batch; mixed batch binds resulting IDs separately')
    for raw in births:
        control = _literal_json(raw)
        if set(control) != _BIRTH:
            raise ValueError('birth needs explicit normal request, targets, comparison profile and duration')
        targets = control['target_ids']
        if (type(targets) is not list or not targets or any(type(s) is not str or not s for s in targets)
                or len(targets) != len(set(targets))):
            raise ValueError('birth targets must be unique explicit segment IDs')
        if control['comparison_profile'].get('provenance_policy') != store.config['policy']:
            raise ValueError('birth comparison policy differs from isolated pool')
        request = deepcopy(control['request'])
        if (not isinstance(request.get('request_id'), str) or not request['request_id']
                or request.get('capture_logits') or 'teacher_token_ids' in request):
            raise ValueError('ordinary birth request cannot be a teacher/logit numerical reference')
        action_id = control['control_id'] + '_birth'
        request['request_id'] += ':' + action_id
        jobs.append(dict(action_id=action_id, operation='source_request', request=request,
            request_sha256=digest_json(request), input_origin='controlled_provenance_diagnostic',
            upper_seconds=control['upper_seconds'], sources_by_segment={s:[] for s in targets},
            comparison_profile=deepcopy(control['comparison_profile'])))
    with store.lock:
        store._guard()
        if registry_budget != dict(max_bytes=store.registry._max_bytes,
                                   max_manifest_bytes=store.registry._max_manifest_bytes):
            raise ValueError('batch registry limits differ from the actual opened registry')
        if store._snapshots or store._plans or any(store._lease_counts.values()):
            raise ValueError('manifest freeze requires a quiescent isolated pool')
        if births and store._catalog['rows']:
            raise ValueError('controlled birth bootstrap requires actual empty isolated pool')
        catalog_sha = file_digest(store.root/'catalog.json')
        registry_sha = file_digest(store.registry._root/'registry.json')
        binding = dict(code_commit=source['code_commit'], runtime_digest=runtime_binding_digest(),
            patch_sha256=native['binding']['patch_sha256'], model_signature=source['model_signature'],
            tokenizer_hash=source['tokenizer_hash'], instance_id=instance_id,
            gpu_uuid=runtime['cost_provenance']['gpu'], initial_pool_sha256=catalog_sha,
            initial_registry_sha256=registry_sha,
            input_manifest_sha256=digest_json([dict(action_id=j['action_id'],
                request_sha256=j['request_sha256'], input_origin=j['input_origin']) for j in jobs]))
        manifest = dict(kind='bounded_native_p0_batch_v2', phase='P0',
            automatic_rental_allowed=False, locked_test_accessed=False,
            native_manifest_sha256=native_manifest_sha256, binding=binding,
            authority=_literal_json(authority), limits=_literal_json(limits),
            registry_budget=_literal_json(registry_budget), jobs=jobs,
            exact_capture_pairs=exact_pairs, mixed_reference_pairs=mixed_pairs,
            controlled_recipe=dict(kind='controlled_p0_launch_recipe_v2',
                numerical_policy=policy, numerical_policy_sha256=digest_json(policy),
                original_inputs=original_inputs, natural_workload_evidence=False,
                derived_segment_identity_policy='v2_store_tokens_model_tokenizer_domain',
                reference_forward_publication_allowed=False,
                controlled_birth_actions=[j['action_id'] for j in jobs if j['operation']=='source_request'],
                controlled_birth_destination='isolated_P0_diagnostic' if births else None))
        manifest['manifest_sha256'] = digest_json(manifest)
        validation = validate_p0_batch(manifest, actual_binding=binding, now_unix=now_unix)
        geometry = validate_p0_model_geometry(manifest, runtime)
        config = json.loads(verified_model_asset_path(runtime['model_path'], 'config.json').read_text(encoding='utf-8'))
        verification = {}
        for control in mixed_controls:
            _source(store, control['source'], control['request'], control['upstream_segment_id'],
                    config, geometry['num_layers'], verification=verification)
            if 'target_r1_source' in control:
                _source(store, control['target_r1_source'], control['request'], control['target_id'],
                        config, geometry['num_layers'], verification=verification)
        store._guard()
        if (file_digest(store.root/'catalog.json') != catalog_sha
                or file_digest(store.registry._root/'registry.json') != registry_sha):
            raise ValueError('isolated pool/registry changed during recipe verification')
    return dict(manifest=manifest, preflight=dict(validation,
        status='CONTROLLED_INPUT_RECIPE_VALIDATED_RUNTIME_RECHECK_REQUIRED',
        model_geometry=geometry, source_files_verified=list(verification.values()),
        model_loaded=False, gpu_observed=False, gpu_execution_allowed=False,
        P0_qualified=False, P1_execution_allowed=False, paper_evidence=False,
        automatic_rental_allowed=False, locked_test_accessed=False))
