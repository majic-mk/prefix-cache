"""CPU byte binding of the existing formal natural token-ID trace contract.

This adds no trace generator, GPU authority, table issuer, resource owner or
request executor. Legacy qualification callers must retain their original
validate_workload/check_phase calls. Effect input metadata is distinct from
actual finite-cell, device, lifecycle and original-guard qualification.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time

PROTOCOL_SHA = '7fbc544597c9abe334122d354e7494b590e810ab0682f1d2d155408cc4f618eb'
AUTHOR_TRACE_SHA = '83713db1b011a954616896ec0ccfa05dc766805e80ed8defdef0da65100ff99e'
AUTHOR_COMMON_SHA = 'a331d1595b815c3c8fc63d3e2f236264c2f181ece7a9de51f50beb4703338df5'
STRONG_SHA = 'd2d1d8e35c582df3a6f4857f91b21c224d524ed2cb0ca9a35948f19195396478'
PARTITIONS = ('calibration', 'development', 'evaluation')
MAX_SELECTED_REQUESTS = 32
MAX_TOKEN_ID = 152064
INITIAL = 'fresh_equal_namespace_preserved_through_whole_partition'
BINDING_FIELDS = {'schema', 'manifest_ref', 'dataset_ref', 'author_trace_ref', 'author_common_ref',
                  'protocol_source_ref', 'declaration_ref', 'tokenizer_receipt_ref', 'model_manifest_ref',
                  'namespace_contract_ref', 'independent_deadline_ref', 'strong_pair_validator_ref'}
MANIFEST_FIELDS = {'schema', 'origin', 'dataset_sha256', 'ordered_prompt_digest', 'author_trace_sha256',
                   'author_common_sha256', 'tokenizer_receipt_digest', 'declaration_digest',
                   'model_manifest_sha256', 'records', 'partition_counts', 'max_concurrency', 'arrival_rate',
                   'schedule_seed', 'schedule_origin', 'recorded_natural_arrival_claim', 'no_prefix_injection',
                   'no_per_request_cache_reset', 'no_request_drops_or_reorder', 'initial_cache_state',
                   'cost_domain_covered', 'gpu_effect_qualified', 'gpu_operations', 'workload_sha256'}
ROW_FIELDS = {'request_id', 'prompt_sha256', 'prompt_token_ids', 'scheduled_ns', 'max_tokens',
              'min_tokens', 'seed', 'split', 'prefix_family'}


def require(value, reason):
    if not value:
        raise ValueError('FORMAL_TRACE_BINDING_REJECTED: ' + reason)


def integer(value, name, lower=0, upper=None):
    require(type(value) is int and value >= lower and (upper is None or value <= upper), name)
    return value


def sha(value, name):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), name)
    return value


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require_independent_service_SLO(value):
    require(type(value) is dict and set(value) == {'schema', 'origin', 'TTFT_ns', 'request_ITL_P95_ns'} and
            value.get('schema') == 'independent_service_SLO_v1' and
            value.get('origin') == 'independent_requirement_before_development',
            'independent service SLO missing or posthoc; formal inputs remain UNBOUND')
    integer(value.get('TTFT_ns'), 'independent TTFT service target', 1)
    integer(value.get('request_ITL_P95_ns'), 'independent request ITL P95 service target', 1)
    return value


def exact(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(exact(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return left == right


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: require(False, 'nonfinite JSON'))


def safe(root, relative):
    root = Path(root).resolve(strict=True)
    require(type(relative) is str and relative and not relative.startswith('/') and
            ':' not in relative and '\\' not in relative and '\0' not in relative and
            all(part not in ('', '.', '..') for part in relative.split('/')), 'project-relative POSIX path')
    path = root
    for part in relative.split('/'):
        path /= part
        require(not path.is_symlink(), 'symlink evidence/namespace')
    require(path.resolve().is_relative_to(root), 'project containment')
    return path


def normalize_ref(root, row):
    require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'exact frozen file ref')
    integer(row['bytes'], 'source bytes')
    sha(row['sha256'], 'source SHA')
    text = row['path']
    require(type(text) is str and text, 'source path')
    path = Path(text)
    if path.is_absolute():
        try:
            text = path.relative_to(Path(root).resolve(strict=True)).as_posix()
        except ValueError:
            require(False, 'absolute source outside actual project')
    safe(root, text)
    return dict(row, path=text)


def file_ref(root, relative):
    path = safe(root, relative)
    require(path.is_file(), 'actual regular source')
    size = path.stat().st_size
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            hasher.update(block)
    require(path.stat().st_size == size, 'source changed while hashing')
    return dict(path=relative, bytes=size, sha256=hasher.hexdigest())


def closed(root, row, refs):
    row = normalize_ref(root, row)
    require(refs.get(row['path']) is not None and exact(refs[row['path']], row), 'source leaf outside frozen closure')
    require(exact(file_ref(root, row['path']), row), 'actual source bytes changed: ' + row['path'])
    return row, safe(root, row['path'])


def json_leaf(root, row, refs):
    row, path = closed(root, row, refs)
    require(row['bytes'] <= 32 * 1024**2, 'bounded JSON source')
    return row, parse(path.read_bytes())


def load_pure(root, row, refs, expected_sha):
    row, path = closed(root, row, refs)
    require(row['sha256'] == expected_sha and path.suffix == '.py' and row['bytes'] <= 1024**2,
            'exact existing pure CPU protocol/helper source')
    raw = path.read_bytes()
    tree = ast.parse(raw)
    allowed_imports = {'__future__', 'argparse', 'ast', 'copy', 'hashlib', 'json', 'math', 'random',
                       're', 'dataclasses', 'pathlib', 'typing'}
    for node in tree.body:
        if isinstance(node, ast.Import):
            require(all(alias.name in allowed_imports for alias in node.names), 'pure helper import boundary')
        elif isinstance(node, ast.ImportFrom):
            require(node.module in allowed_imports, 'pure helper import boundary')
    name = '_actual_formal_CPU_helper_' + str(time.monotonic_ns())
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        require(exact(file_ref(root, row['path']), row), 'pure helper drift during loading')
        return module
    finally:
        sys.modules.pop(name, None)


def validate_manifest_shape(document):
    """Exact unchanged freeze_trace schema. Shape validation never qualifies data."""
    require(type(document) is dict and set(document) == MANIFEST_FIELDS and
            document['schema'] == 'natural_trace_workload_v1', 'original formal freeze_trace schema; not qualification relabeling')
    require(document['origin'] == 'frozen_actual_existing_trace_no_new_gpu_outcomes' and
            document['author_trace_sha256'] == AUTHOR_TRACE_SHA and
            document['author_common_sha256'] == AUTHOR_COMMON_SHA, 'original author parser/schedule provenance')
    for key in ('dataset_sha256', 'ordered_prompt_digest', 'tokenizer_receipt_digest', 'declaration_digest',
                'model_manifest_sha256', 'workload_sha256'):
        sha(document[key], key)
    require(digest({key: value for key, value in document.items() if key != 'workload_sha256'}) ==
            document['workload_sha256'], 'complete original formal manifest digest')
    for key in ('no_prefix_injection', 'no_per_request_cache_reset', 'no_request_drops_or_reorder'):
        require(document[key] is True, 'natural replay does not manufacture/reorder/drop requests')
    require(document['initial_cache_state'] == INITIAL and document['recorded_natural_arrival_claim'] is False,
            'fresh whole-partition cache and author-generated schedule, not recorded natural arrivals')
    require(document['cost_domain_covered'] is document['gpu_effect_qualified'] is False and
            type(document['gpu_operations']) is int and document['gpu_operations'] == 0,
            'frozen data is never GPU qualification')
    counts = document['partition_counts']
    require(type(counts) is dict and set(counts) == set(PARTITIONS), 'all three formal partitions')
    for name in PARTITIONS:
        integer(counts[name], 'complete bounded partition count', 1, MAX_SELECTED_REQUESTS)
    records = document['records']
    require(type(records) is list and len(records) == sum(counts.values()), 'no hidden selection or dropped author inputs')
    integer(document['max_concurrency'], 'existing bounded concurrency', 1, 8)
    integer(document['schedule_seed'], 'schedule seed')
    rate = document['arrival_rate']
    require(rate is None or type(rate) in (int, float) and math.isfinite(rate) and rate > 0, 'prospective author arrival rate')
    require(document['schedule_origin'] == 'unchanged_author_build_global_specs', 'original scheduler schedule')
    families, prompts, previous_ns = {}, {}, -1
    cuts = counts['calibration'], counts['calibration'] + counts['development']
    for index, record in enumerate(records):
        require(type(record) is dict and set(record) == ROW_FIELDS, 'exact original formal request row')
        require(type(record['request_id']) is int and record['request_id'] == index, 'all original IDs/order')
        expected = 'calibration' if index < cuts[0] else 'development' if index < cuts[1] else 'evaluation'
        require(record['split'] == expected, 'prospective contiguous split, never qualification labels')
        family = record['prefix_family']
        require(type(family) is str and 0 < len(family) <= 512, 'verified frozen prefix family identity')
        prompt = sha(record['prompt_sha256'], 'natural text prompt SHA')
        require(families.setdefault(family, expected) == expected and prompts.setdefault(prompt, expected) == expected,
                'family/session/document or identical prompt leaks across partitions')
        tokens = record['prompt_token_ids']
        require(type(tokens) is list and 1 <= len(tokens) <= 4096 and
                all(type(token) is int and 0 <= token < MAX_TOKEN_ID for token in tokens), 'actual bounded model token-ID lane')
        integer(record['scheduled_ns'], 'unchanged author planned arrival')
        require(record['scheduled_ns'] >= previous_ns, 'original arrival order')
        previous_ns = record['scheduled_ns']
        integer(record['max_tokens'], 'sustained complete outputs', 128)
        require(type(record['min_tokens']) is int and record['min_tokens'] == record['max_tokens'], 'unchanged fixed diagnostic output length')
        require(type(record['seed']) is int and record['seed'] == document['schedule_seed'], 'original frozen request seed')
    return dict(schema='formal_manifest_shape_CPU_only_v1', cpu_metadata_only=True,
                gpu_eligible=False, partition_counts=dict(counts), complete_records=len(records))


def common_domain_sha(pair):
    normalized = deepcopy(pair['U'])
    extra = normalized['engine']['kv_transfer_config']['kv_connector_extra_config']
    extra.pop('shared_storage_path')
    extra.pop('prefix_io_p4_policy')
    extra['prefix_io_parent_admission'].pop('run_id')
    extra.pop('prefix_io_observation_run_id', None)
    return digest(normalized)


def validate_namespace(document, *, root, manifest, pair, partition):
    fields = {'schema', 'workload_sha256', 'model_manifest_sha256', 'tokenizer_receipt_digest',
              'common_runtime_domain_sha256', 'initial_cache_state', 'no_per_request_reset', 'partition_namespaces'}
    require(type(document) is dict and set(document) == fields and
            document['schema'] == 'formal_trace_namespace_contract_v1', 'closed formal namespace contract')
    for key in ('workload_sha256', 'model_manifest_sha256', 'tokenizer_receipt_digest'):
        require(document[key] == manifest[key], 'namespace bound to these exact formal input identities')
    require(document['common_runtime_domain_sha256'] == common_domain_sha(pair) and
            document['initial_cache_state'] == INITIAL and document['no_per_request_reset'] is True,
            'same actual strong U/I domain and whole-partition lifetime')
    namespaces = document['partition_namespaces']
    require(type(namespaces) is dict and set(namespaces) == set(PARTITIONS), 'all three independent namespace partitions')
    seen = set()
    for name in PARTITIONS:
        arms = namespaces[name]
        require(type(arms) is dict and set(arms) == {'U', 'I'}, 'distinct namespaces for both existing arms')
        for arm, absolute in arms.items():
            require(type(absolute) is str and Path(absolute).is_absolute(), 'actual absolute namespace')
            try:
                relative = Path(absolute).relative_to(Path(root).resolve(strict=True)).as_posix()
            except ValueError:
                require(False, 'namespace outside project')
            require(relative.startswith('experiments/prefix_io_v1/runs/'), 'original bounded private runs only')
            path = safe(root, relative)
            require(absolute not in seen, 'partition/arm namespace alias')
            seen.add(absolute)
            if name == partition:
                actual = pair[arm]['engine']['kv_transfer_config']['kv_connector_extra_config']['shared_storage_path']
                require(absolute == actual and not path.exists(), 'fresh selected whole-partition strong U/I namespaces')
    return namespaces[partition]


def validate_formal_workload(document, *, root, workload_ref, binding_ref, refs, pair, partition):
    """Return the unchanged selected token-ID rows after full original replay.

    The declaration, actual CPU tokenizer receipt/results/source files, original
    parser and model manifest are separate frozen byte leaves. Digests or
    self-reported family names alone cannot substitute this replay.
    """
    require(partition in PARTITIONS, 'explicit formal partition')
    validate_manifest_shape(document)
    manifest_row, actual_manifest = json_leaf(root, workload_ref, refs)
    require(exact(document, actual_manifest), 'consumer has actual frozen manifest bytes')
    _, binding = json_leaf(root, binding_ref, refs)
    require(type(binding) is dict and set(binding) == BINDING_FIELDS and
            binding['schema'] == 'formal_natural_trace_binding_v1', 'actual closed formal input descriptor')
    leaves = {}
    for key in BINDING_FIELDS - {'schema'}:
        leaves[key] = closed(root, binding[key], refs)
    require(exact(leaves['manifest_ref'][0], manifest_row), 'binding names this full original manifest')
    require(leaves['protocol_source_ref'][0]['sha256'] == PROTOCOL_SHA and
            leaves['author_trace_ref'][0]['sha256'] == AUTHOR_TRACE_SHA and
            leaves['author_common_ref'][0]['sha256'] == AUTHOR_COMMON_SHA and
            leaves['strong_pair_validator_ref'][0]['sha256'] == STRONG_SHA, 'exact existing protocol/parser/strong pair helpers')
    strong = load_pure(root, binding['strong_pair_validator_ref'], refs, STRONG_SHA)
    strong.validate_runtime_pair(pair)
    for arm in ('U', 'I'):
        engine = pair[arm]['engine']
        require(engine.get('skip_tokenizer_init') is True and engine.get('speculative_config') is None,
                'same calibrated token-ID lane; no tokenizer or speculative-domain switch')
        require(engine['max_num_seqs'] >= document['max_concurrency'], 'same original engine supports selected concurrency')
    require(leaves['dataset_ref'][0]['sha256'] == document['dataset_sha256'] and
            leaves['model_manifest_ref'][0]['sha256'] == document['model_manifest_sha256'], 'actual natural dataset/model source bytes')
    require(leaves['dataset_ref'][0]['bytes'] <= 32 * 1024**2, 'bounded complete natural input; never silently truncate')
    _, declaration = json_leaf(root, binding['declaration_ref'], refs)
    _, receipt = json_leaf(root, binding['tokenizer_receipt_ref'], refs)
    require(digest(declaration) == document['declaration_digest'] and
            digest(receipt) == document['tokenizer_receipt_digest'], 'complete declaration/tokenizer byte closure')
    require(type(receipt.get('tokenizer_source_refs')) is list and receipt['tokenizer_source_refs'], 'actual CPU tokenizer sources')
    for row in receipt['tokenizer_source_refs']:
        closed(root, row, refs)
    _, tokenizer_result = json_leaf(root, receipt.get('tokenization_result_ref'), refs)
    require(tokenizer_result.get('schema') == 'actual_cpu_tokenizer_result_v1' and
            type(tokenizer_result.get('exit_code')) is int and tokenizer_result['exit_code'] == 0 and
            tokenizer_result.get('synthetic_fixture') is False, 'actual successful CPU tokenizer receipt contract')
    protocol = load_pure(root, binding['protocol_source_ref'], refs, PROTOCOL_SHA)
    replay = protocol.freeze_trace(leaves['dataset_ref'][1], leaves['author_trace_ref'][1],
                                   leaves['author_common_ref'][1], declaration, receipt)
    require(exact(replay, document), 'full unchanged freeze_trace replay, not qualification relabeling')
    _, deadline = json_leaf(root, binding['independent_deadline_ref'], refs)
    require(deadline.get('schema') == 'independent_deadline_declaration_v1' and
            deadline.get('origin') == 'independent_requirement_before_development', 'independent prospective deadline source')
    integer(deadline.get('full_control_window_deadline_ns'), 'independent full-control deadline', 1)
    integer(deadline.get('declared_monotonic_ns'), 'independent declaration time', 1)
    service = require_independent_service_SLO(deadline.get('service_SLO'))
    require(exact(protocol.validate_service_SLO(service), service), 'same original independent service SLO contract')
    protocol.clock_scope(deadline.get('clock_scope'))
    _, authority = json_leaf(root, deadline.get('authority_ref'), refs)
    require(authority.get('schema') == 'independent_deadline_authority_v1' and
            authority.get('origin') == 'independent_requirement_before_development' and
            type(authority.get('full_control_window_deadline_ns')) is int and
            authority['full_control_window_deadline_ns'] == deadline['full_control_window_deadline_ns'] and
            exact(authority.get('service_SLO'), service), 'actual independent authority binds deadline and SLO')
    _, namespace = json_leaf(root, binding['namespace_contract_ref'], refs)
    selected_namespaces = validate_namespace(namespace, root=root, manifest=document, pair=pair, partition=partition)
    selected = [deepcopy(row) for row in document['records'] if row['split'] == partition]
    engine = pair['U']['engine']
    for row in selected:
        require(len(row['prompt_token_ids']) + row['max_tokens'] <= engine['max_model_len'], 'actual native model context capacity')
        require(row['min_tokens'] == pair['U']['sampling']['min_tokens'] == pair['I']['sampling']['min_tokens'] and
                row['max_tokens'] == pair['U']['sampling']['max_tokens'] == pair['I']['sampling']['max_tokens'], 'unchanged shared full output work')
    max_steps = sum(math.ceil(len(row['prompt_token_ids']) / engine['max_num_batched_tokens']) + row['max_tokens'] for row in selected)
    require(max_steps <= 4096, 'whole selected partition fits existing bounded collector; no truncation')
    return dict(schema='formal_natural_token_id_partition_CPU_binding_v1',
                origin='actual_byte_closed_original_formal_manifest_replay_CPU_only', partition=partition,
                records=selected, max_concurrency=document['max_concurrency'],
                workload_ref=manifest_row, binding_ref=normalize_ref(root, binding_ref),
                original_manifest_workload_sha256=document['workload_sha256'],
                model_manifest_sha256=document['model_manifest_sha256'],
                tokenizer_receipt_digest=document['tokenizer_receipt_digest'],
                partition_families=sorted({row['prefix_family'] for row in selected}),
                complete_selected_record_count=len(selected), source_request_ids=[row['request_id'] for row in selected],
                selected_namespaces=selected_namespaces, independent_deadline_ref=leaves['independent_deadline_ref'][0],
                service_SLO=service, common_runtime_domain_sha256=common_domain_sha(pair),
                skip_tokenizer_init=True, no_request_drop_or_relabel=True, cpu_metadata_only=True,
                gpu_eligible=False, formal_goodput_allowed=False, actual_GPU_operations=0)


def validate_formal_phase(config, *, root, refs, pair, formal_workload, finite_activation=None,
                          development_replay=None):
    """Both effect arms consume the same independently frozen real dev gates.

    development_replay is the existing reserve_join read-only replay output,
    anchored by the descriptor's actual raw proof refs. This module never issues
    a qualified table or infers one from CPU fixtures/JSON flags.
    """
    require(type(config) is dict, 'formal phase configuration')
    role = config.get('phase'), config.get('mode'), config.get('arm')
    allowed = {('calibration', 'off', 'U'): 'calibration',
               ('development', 'off', 'U'): 'development',
               ('development', 'shadow', 'I'): 'development',
               ('effect', 'off', 'U'): 'evaluation', ('effect', 'on', 'I'): 'evaluation'}
    require(role in allowed, 'formal Uoff/Ion phase; qualification must use original validator')
    require(type(formal_workload) is dict and formal_workload.get('schema') == 'formal_natural_token_id_partition_CPU_binding_v1'
            and formal_workload.get('partition') == allowed[role] and formal_workload.get('gpu_eligible') is False,
            'real selected formal partition, no qualification promotion')
    actual = validate_formal_workload(
        json_leaf(root, config.get('workload_ref'), refs)[1], root=root,
        workload_ref=config['workload_ref'], binding_ref=config.get('formal_trace_binding_ref'),
        refs=refs, pair=pair, partition=allowed[role])
    require(exact(actual, formal_workload), 'no forged partition/family/namespace metadata')
    require(config.get('run_id') == pair[config['arm']]['engine']['kv_transfer_config']['kv_connector_extra_config']
            ['prefix_io_parent_admission']['run_id'], 'actual selected original parent/journal identity')
    if role[0] == 'effect':
        require(type(finite_activation) is dict and finite_activation.get('phase') == 'effect',
                'both effect Uoff and Ion require original finite activation CPU preflight')
        descriptor = finite_activation.get('descriptor')
        _, actual_descriptor = json_leaf(root, config.get('activation_ref'), refs)
        require(exact(descriptor, actual_descriptor), 'actual original activation descriptor bytes')
        required = ('independent_deadline_ref', 'development_reserve_ref', 'development_budget_ref',
                    'reserve_verification_ref', 'development_plan_ref', 'development_guard_ref',
                    'development_observation_ref', 'development_native_result_ref', 'calibration_source_lock_ref')
        for key in required:
            require(type(descriptor.get(key)) is dict, 'actual development prerequisite missing: ' + key)
            closed(root, descriptor[key], refs)
        require(exact(normalize_ref(root, descriptor['independent_deadline_ref']), actual['independent_deadline_ref']),
                'same prospectively frozen independent SLO/deadline for both arms')
        _, budget = json_leaf(root, descriptor['development_budget_ref'], refs)
        require(budget.get('schema') == 'frozen_independent_development_budget_v1' and
                budget.get('derived_from_A_max') is False and budget.get('evaluation_used_to_fit') is False and
                exact(budget.get('service_SLO'), actual['service_SLO']) and
                exact(finite_activation.get('independent_budget'), budget), 'same actual frozen independent development budget')
        integer(budget.get('non_gpu_reserve_ns'), 'actual measured scheduler/sampling/output/controller reserve', 1)
        require(type(budget.get('internal_step_budget_ns')) is int and budget['internal_step_budget_ns'] > 0 and
                budget['internal_step_budget_ns'] == budget['full_control_window_deadline_ns'] - budget['non_gpu_reserve_ns'],
                'same retained full-control reserve and positive internal budget')
        _, proof = json_leaf(root, descriptor['reserve_verification_ref'], refs)
        require(proof.get('schema') == 'actual_development_native_reserve_verification_v1' and
                proof.get('status') == 'PASS_ACTUAL_DEVELOPMENT_RESERVE_ONLY' and
                proof.get('actual_native_gpu_run') is True and proof.get('synthetic_fixture') is False and
                proof.get('independent_deadline_prospective') is True and
                proof.get('actual_guard_native_SDK_CUDA_outputs_and_all4host_categories_verified') is True and
                exact(proof.get('frozen_budget'), budget), 'actual native development qualification proof, not CPU algebra')
        require(type(development_replay) is dict and
                development_replay.get('schema') == 'existing_actual_native_development_reserve_verified_v1' and
                development_replay.get('status') == 'PASS_READ_ONLY_NATIVE_RESERVE_REPLAY' and
                development_replay.get('actual_native_gpu_run') is True and
                development_replay.get('synthetic_fixture') is False and
                development_replay.get('native_qualification_verifier_required_separately') is False and
                exact(development_replay.get('frozen_budget'), budget), 'original raw read-only native-reserve replay required for both effect arms')
        require(exact(normalize_ref(root, development_replay.get('proof_ref')),
                      normalize_ref(root, descriptor['reserve_verification_ref'])), 'same original actual development proof anchor')
        plan = finite_activation.get('calibration_plan')
        require(type(plan) is dict and plan.get('common_runtime_domain_sha256') == actual['common_runtime_domain_sha256'] and
                plan.get('gpu_uuid') == config.get('gpu_uuid'), 'same measured token-ID domain/device; no planner/tokenizer switch')
        _, cal_lock = json_leaf(root, descriptor['calibration_source_lock_ref'], refs)
        require(type(cal_lock.get('files')) is list and cal_lock['files'] and all(
            exact(refs.get(normalize_ref(root, row)['path']), normalize_ref(root, row)) for row in cal_lock['files']),
            'every immutable calibration/V9 source row inherited unchanged; additions do not replace old rows')
    return dict(schema='formal_trace_phase_CPU_preflight_v1', role=list(role), partition=actual['partition'],
                input_bindings_validated=True, same_strong_U_I_domain=True,
                independent_service_SLO=actual['service_SLO'], cpu_metadata_only=True,
                gpu_eligible=False, formal_goodput_allowed=False, actual_GPU_operations=0,
                required_runtime_gates=['original_guard_and_live_device', 'original_private_issuer_finite_cost_coverage',
                                        'original_native_reserve_read_only_replay', 'native_shutdown_and_session_drain',
                                        'all_evaluation_requests_and_IO_accounted'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--source-lock', required=True, type=Path)
    parser.add_argument('--manifest-ref', required=True, type=Path, help='JSON exact {path,bytes,sha256}, not a bare path')
    parser.add_argument('--binding-ref', required=True, type=Path, help='JSON exact {path,bytes,sha256}')
    parser.add_argument('--pair-ref', required=True, type=Path, help='JSON exact source-closed strong U/I configuration ref')
    parser.add_argument('--partition', required=True, choices=PARTITIONS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    lock = parse(args.source_lock.read_bytes())
    require(type(lock) is dict and type(lock.get('files')) is list, 'actual existing source lock')
    refs = {}
    for row in lock['files']:
        row = normalize_ref(args.project_root, row)
        require(row['path'] not in refs, 'duplicate source lock row')
        refs[row['path']] = row
    workload_ref = parse(args.manifest_ref.read_bytes())
    binding_ref = parse(args.binding_ref.read_bytes())
    pair_ref = parse(args.pair_ref.read_bytes())
    _, document = json_leaf(args.project_root, workload_ref, refs)
    _, pair_doc = json_leaf(args.project_root, pair_ref, refs)
    require(pair_doc.get('schema') == 'strong_native_u_i_cpu_configuration_v1', 'existing strong pair schema')
    result = validate_formal_workload(document, root=args.project_root, workload_ref=workload_ref,
                                      binding_ref=binding_ref, refs=refs, pair=pair_doc['configurations'], partition=args.partition)
    summary = {key: value for key, value in result.items() if key != 'records'}
    if args.output:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
