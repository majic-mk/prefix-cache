"""Compile one frozen natural input into existing P0 operations, CPU only.

This is a diagnostic bridge, NOT a P1 dispatcher. Each recipe requires its own
empty isolated pool. Reference captures cannot supply a Source. A separately
accounted ordinary request may publish its complete target after finishing.
"""
from copy import deepcopy
import math

from .p1_input_consumer_v2 import resolve_input_graph
from .p0_launch_recipe_v2 import _request
from .p0_exact_control_v2 import validate_exact_control_job
from .source_comparison_v2 import ComparisonProfileBindingV2, runtime_binding_digest, scorer_digest
from .v8_schema10_execution import digest_json


def compile_natural_p0_controls(documents, *, identity, build_id, teacher_token_ids,
                                comparison_profile, upper_seconds_by_arm):
    """Re-resolve frozen inputs and return three existing, non-executable arms.

    Caller must freeze teacher conditioning BEFORE results, pass the approved
    numerical policy to the ordinary launch builder, and obtain current server
    authority. Original prompt tokens/positions are never changed. Only request
    ID, decoding/capture controls differ, explicitly audited below.
    """
    graph = resolve_input_graph(documents, **identity)
    matches = [b for b in graph['source_builds'] if b['build_id'] == build_id]
    if len(matches) != 1:
        raise ValueError('one existing frozen Source build ID required')
    build = matches[0]
    if build['role'] not in ('historical_exact', 'independent_S0', 'exact_E', 'G0_parent'):
        raise ValueError('mixed Source requires separately bound actual parent and mixed P0 recipe')
    target = 'U' if build['role'] == 'G0_parent' else 'C'
    if (type(teacher_token_ids) is not list or len(teacher_token_ids) != 31
            or any(type(t) is not int or t < 0 for t in teacher_token_ids)):
        raise ValueError('explicit 31 teacher tokens for 32 aligned positions required')
    if (type(upper_seconds_by_arm) is not dict
            or set(upper_seconds_by_arm) != {'off', 'on', 'birth'}
            or any(type(t) not in (int, float) or not math.isfinite(t)
                   or not 0 < t <= build['maximum_seconds'] for t in upper_seconds_by_arm.values())):
        raise ValueError('explicit positive arm bounds within frozen per-action limit required')
    profile = ComparisonProfileBindingV2(**comparison_profile)
    if (profile.model_signature != identity['model_signature']
            or profile.runtime_digest != runtime_binding_digest()
            or profile.scoring_function_digest != scorer_digest()):
        raise ValueError('current code/scorer/model diagnostic binding required')
    q = deepcopy(build['request'])
    _request(q)
    forbidden = ('teacher_token_ids', 'capture_logits', 'capture_original_full_prefill',
                 'publish_exact_prefix_shadow', 'use_gpu_hot_cache', 'retain_gpu_hot_cache',
                 'native_dense_continuation')
    if (any(q.get(k) for k in forbidden)
            or type(q.get('prefetch_window', 0)) is not int or q.get('prefetch_window', 0) not in (0, 1)
            or q.get('correctness_repair_ratio', .15) != .15):
        raise ValueError('frozen input contains incompatible execution overrides')
    # Native identifiers are derived by the existing builder, never fabricated
    # from metadata content keys of a different namespace.
    if not any(s['segment_id'] == target for s in q['segments']):
        raise ValueError('original requested target is missing')
    original_keys = {s['segment_id']:s.get('content_key') for s in q['segments']}
    original_window = q.get('prefetch_window', 0)
    # Input plans predate the isolated v2 authorization namespace. Preserve
    # those keys in audit, but derive actual store keys only at server binding.
    for segment in q['segments']:
        segment.pop('content_key', None)
    q['prefetch_window'] = 0  # P0 full-dense capture, not a P1 performance arm.
    control_id = 'natural_' + digest_json([graph['graph_sha256'], build_id])[:20]
    diagnostic = deepcopy(q)
    # Teacher-forcing diagnostics are not free-generation QA. Preserve the
    # original QA contract on the ordinary birth, but do not combine its stop
    # rule or quality claims with teacher-conditioned logits.
    diagnostic_qa_fields = {k:diagnostic.pop(k) for k in (
        'answer_boundary_contract', 'answers', 'matched_dense_answer_evidence',
        'quality_contract') if k in diagnostic}
    diagnostic.update(request_id=q['request_id'] + ':P0capture', max_new_tokens=32,
                      teacher_token_ids=list(teacher_token_ids), capture_logits=True)
    birth = deepcopy(q)
    birth.update(request_id=q['request_id'] + ':P0birth', max_new_tokens=1)
    _request(diagnostic)
    _request(birth)
    validate_exact_control_job(dict(operation='exact_capture_control', request=diagnostic,
                                    capture_enabled=True, target_ids=[target]))
    result = dict(kind='natural_input_controlled_P0_v2',
        status='CONTROLS_COMPILED_SERVER_BINDING_REQUIRED',
        input_graph_sha256=graph['graph_sha256'], original_build_id=build_id,
        original_request_sha256=build['request_sha256'], partition_digest=graph['partition_digest'],
        original_prompt_token_sha256=digest_json(q['token_ids']),
        teacher_token_sha256=digest_json(teacher_token_ids),
        execution_overrides=dict(original_prefetch_window=original_window, prefetch_window=0,
            diagnostic_removed_qa_fields=diagnostic_qa_fields,
            original_content_keys=original_keys, content_key_policy='derive_from_actual_v2_store_at_binding',
            diagnostic_max_new_tokens=32, ordinary_birth_max_new_tokens=1),
        exact_controls=[dict(control_id=control_id, request=diagnostic, target_ids=[target],
                             upper_seconds_by_arm={k:upper_seconds_by_arm[k] for k in ('off','on')})],
        birth_controls=[dict(control_id=control_id+'_src', request=birth, target_ids=[target],
                             upper_seconds=upper_seconds_by_arm['birth'],
                             comparison_profile=deepcopy(comparison_profile))], mixed_controls=[],
        maximum_actions=3, maximum_action_seconds=sum(upper_seconds_by_arm.values()),
        required_context_tokens=max(len(q['token_ids'])+32, len(q['token_ids'])+1),
        isolated_empty_pool_required=True, diagnostic_reference_publication_allowed=False,
        ordinary_birth_publication_scope='ISOLATED_P0_ONLY',
        request_prompt_modified=False, metadata_reranking_enabled=False,
        P1_execution_allowed=False, gpu_execution_allowed=False, automatic_rental_allowed=False,
        paper_evidence=False, locked_test_accessed=False,
        remaining=['NATIVE_ASSETS_AND_PATCH_RECHECK', 'CURRENT_INSTANCE_AUTHORITY',
                   'FRESH_EMPTY_POOL', 'APPROVED_NUMERICAL_POLICY', 'GPU_EXECUTION_AND_RAW_VERIFICATION'])
    result['controls_sha256'] = digest_json(result)
    return result


def bind_natural_p0_manifest(controls, **launch_arguments):
    """Use the existing launch builder, including all real asset/Source checks.

    No authority defaults, no model load, no new execution bypass. Caller gives
    native_manifest_path/SHA, store, authority, limits, registry/numerical policy,
    instance ID and current time exactly as required by the existing builder.
    """
    from .p0_launch_recipe_v2 import build_controlled_p0_manifest
    body = {k:v for k,v in controls.items() if k != 'controls_sha256'}
    if (controls.get('kind') != 'natural_input_controlled_P0_v2'
            or digest_json(body) != controls.get('controls_sha256')
            or controls.get('gpu_execution_allowed') is not False
            or controls.get('P1_execution_allowed') is not False):
        raise ValueError('unaltered non-executable natural P0 controls required')
    policy = launch_arguments['numerical_policy']['exact']
    if policy != dict(relative_l2_limit=1e-4, minimum_positions=32,
                      require_predicted_token_ids_equal=True):
        raise ValueError('approved strict 32-position exact policy required')
    if launch_arguments['limits']['maximum_actions'] != 3:
        raise ValueError('this natural P0 recipe is exactly three actions')
    result = build_controlled_p0_manifest(
        exact_controls=controls['exact_controls'], birth_controls=controls['birth_controls'],
        mixed_controls=controls['mixed_controls'], **launch_arguments)
    manifest = result['manifest']
    manifest['natural_input_binding'] = {k:controls[k] for k in (
        'controls_sha256', 'input_graph_sha256', 'original_build_id',
        'original_request_sha256', 'partition_digest', 'original_prompt_token_sha256',
        'teacher_token_sha256', 'ordinary_birth_publication_scope')}
    manifest['manifest_sha256'] = digest_json({k:v for k,v in manifest.items() if k != 'manifest_sha256'})
    return result


def verify_natural_p0_birth(store, job, *, source_id, target_id, expected_num_layers):
    """Read-only disk/recipe verification, NOT CUDA numerical qualification.

    Call after raw batch evidence verification, with the birth job from the
    frozen manifest. In particular, target token equality alone cannot bind
    the Source to its historical prefix. No comparison/use/LRU event is emitted.
    """
    from .source_store_v2 import TargetSourceStoreV2
    from .source_manifest_v2 import ManifestReference, request_input_digest
    if type(store) is not TargetSourceStoreV2:
        raise ValueError('actual isolated target store required')
    q = job['request']
    if (job.get('operation') != 'source_request' or job.get('request_sha256') != digest_json(q)
            or q.get('capture_logits') or 'teacher_token_ids' in q
            or q.get('max_new_tokens') != 1
            or job.get('sources_by_segment', {}).get(target_id) != []
            or type(expected_num_layers) is not int or expected_num_layers < 2):
        raise ValueError('frozen ordinary cold-miss birth required, not a numerical reference')
    _request(q)
    targets = [s for s in q['segments'] if s['segment_id'] == target_id]
    if len(targets) != 1:
        raise ValueError('unique frozen target required')
    segment = targets[0]
    with store.lock:
        store._guard()
        if store._snapshots or store._plans or any(store._lease_counts.values()):
            raise ValueError('post-request verification requires quiescent store')
        row = store._catalog['rows'].get(source_id)
        if row is None:
            raise ValueError('Source has not been atomically published')
        if (row['birth_request_id'] != q['request_id']
                or row['token_ids'] != segment['token_ids']
                or row['birth_target_positions'] != segment['positions']
                or row['layer_count'] != expected_num_layers
                or row['origin'] != 'EXACT_CONTEXT' or type(row['generation']) is not int or row['generation'] != 0
                or row['parent_owned_kv_bytes'] != 0 or row['prefix_shadow_bytes'] != 0
                or not 0 < row['publication_epoch'] <= store._catalog['epoch']):
            raise ValueError('published Source birth/target/geometry/origin differs from frozen job')
        if 3 * (row['target_kv_bytes'] + row['selection_state_bytes']) > store.config['staging_bytes']:
            raise MemoryError('explicit host verification budget insufficient')
        reference = ManifestReference(**row['manifest_reference'])
        manifest, record = store._manifest_record(reference, row['token_ids'], row['birth_target_positions'])
        positions = list(range(len(q['token_ids'])))
        if (manifest['token_ids'] != q['token_ids'] or manifest['absolute_positions'] != positions
                or manifest['input_digest'] != request_input_digest(q['token_ids'], positions)
                or manifest['num_layers'] != expected_num_layers
                or manifest['parent_sources'] or manifest['unverified_noncausal_parent_ids']):
            raise ValueError('complete historical context or parent recipe differs from frozen birth')
        layers = manifest['executed_layers']
        if (len(layers) != expected_num_layers
                or [x['layer_1based'] for x in layers] != list(range(1, expected_num_layers+1))
                or any(x['imported_kv'] or any(x[field] != positions for field in
                    ('qkv_rows','current_kv_rows','attention_rows','output_mlp_rows')) for x in layers)):
            raise ValueError('ordinary exact birth must execute all input rows in every layer')
        store._verify_row(row, full=True)
        store._guard()
        result = dict(kind='natural_P0_birth_disk_verification_v2',
            source_id=source_id, artifact_digest=row['artifact_digest'],
            birth_request_id=row['birth_request_id'], frozen_job_sha256=digest_json(job),
            full_input_digest=manifest['input_digest'], target_positions_digest=digest_json(segment['positions']),
            manifest_id=reference.manifest_id, proof_digest=record['proof_digest'],
            kv_file_sha256=row['kv_file_digest'], selection_file_sha256=row['selection_file_digest'],
            target_token_count=len(segment['positions']), layer_count=expected_num_layers,
            generation=0, parent_owned_kv_bytes=0, prefix_shadow_bytes=0,
            evidence_scope='disk_and_execution_recipe_only_not_GPU_numerics',
            actual_GPU_execution_verified=False, P1_execution_allowed=False, paper_evidence=False)
        result['verification_digest'] = digest_json(result)
        return result
