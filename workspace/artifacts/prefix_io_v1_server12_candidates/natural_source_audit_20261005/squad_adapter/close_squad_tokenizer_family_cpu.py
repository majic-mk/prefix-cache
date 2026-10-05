"""Close actual CPU SQuAD tokenizer/family evidence into original receipt schemas.

All tokenizer work is the frozen raw producer's real local backend. This tool
does not download, create a model/cache executor, declare a deadline or use GPU.
Fresh raw/result/receipt files enter a later lock, avoiding a V13 hash cycle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from types import ModuleType


ADAPTER_SHA = 'b309bd1d8b08c52e9541b9f9c7738441546afbee3bc6bc50b790cc7ab7900649'
RAW_PRODUCER_SHA = '2d252d0647f6a1617a4222fd86009da27344d29391c109fa3ccecf7cd4f68c63'
PROTOCOL_SHA = '7fbc544597c9abe334122d354e7494b590e810ab0682f1d2d155408cc4f618eb'
TRACE_SHA = '83713db1b011a954616896ec0ccfa05dc766805e80ed8defdef0da65100ff99e'
COMMON_SHA = 'a331d1595b815c3c8fc63d3e2f236264c2f181ece7a9de51f50beb4703338df5'
SOURCE_SHA = '95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9'
SOURCE_GIT_BLOB_SHA = 'e9a3f913ad1468ebe105b891334ca7b0bc0e2510'
V12_PATH = 'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/PRERENT_SOURCE_LOCK_V12.json'
V12_SHA = '23ab34887c7f4340a37f2814936c639c46890ca31e4b3ba1b07359d41fc13f77'
FREEZER_PATH = 'artifacts/prefix_io_v1/server12-natural-source-audit-20261005/source_freeze/append_public_source_lock_v13.py'
FREEZER_SHA = '63db22e5cb91a117e6f65c3558309b0c5854861de963cff70866f8df1b4cefc7'
PROOF = 'all_sessions_documents_and_public_prefix_families_closed_before_partitioning'
REF_FIELDS = ('source_ref', 'selection_contract_ref', 'trace_ref', 'mapping_ref', 'adapter_report_ref',
              'adapter_source_ref', 'raw_producer_source_ref', 'protocol_source_ref',
              'author_trace_ref', 'author_common_ref', 'raw_tokenizer_request_ref')
REQUEST_FIELDS = {'schema', 'raw_result_relative_path', 'block_size', *REF_FIELDS}
MAX_BYTES = 32 * 1024**2


def require(value, reason):
    if not value:
        raise ValueError('CPU_SQUAD_FAMILY_CLOSE_REJECTED: ' + reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def parse(raw):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: require(False, 'nonfinite JSON'))


def safe(root, relative):
    root = Path(root).resolve(strict=True)
    require(type(relative) is str and relative and not relative.startswith('/') and ':' not in relative and
            '\\' not in relative and '\0' not in relative and
            all(part not in ('', '.', '..') for part in relative.split('/')), 'project-relative POSIX path')
    path = root
    for part in relative.split('/'):
        path /= part
        require(not path.is_symlink(), 'symlink source/output')
    require(path.resolve().is_relative_to(root), 'actual project containment')
    return path


def file_ref(root, relative):
    path = safe(root, relative)
    require(path.is_file(), 'actual regular file')
    hasher = hashlib.sha256()
    size = path.stat().st_size
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            hasher.update(chunk)
    require(path.stat().st_size == size, 'file changed during hashing')
    return dict(path=relative, bytes=size, sha256=hasher.hexdigest())


def check_ref(root, row):
    require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'} and
            type(row['bytes']) is int and row['bytes'] >= 0 and type(row['sha256']) is str and
            re.fullmatch('[0-9a-f]{64}', row['sha256']), 'strict actual file ref')
    require(file_ref(root, row['path']) == row, 'actual source byte SHA/size mismatch')
    return safe(root, row['path'])


def read_json(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_BYTES, 'bounded actual JSON')
    return parse(path.read_bytes())


def closed(root, row, refs, pin=None):
    require(type(row) is dict and refs.get(row.get('path')) == row, 'actual leaf inside current V13 closure')
    require(pin is None or row['sha256'] == pin, 'exact frozen source version')
    return check_ref(root, row)


def load_source_only(root, row, refs, pin):
    path = closed(root, row, refs, pin)
    raw = path.read_bytes()
    require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == pin, 'exact source execution bytes')
    module = ModuleType('_actual_CPU_squad_family_source_' + pin[:12])
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


def source_rows(root, rows):
    require(type(rows) is list and 1 <= len(rows) <= 8192, 'bounded source row list')
    result = {}
    for row in rows:
        require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'} and
                type(row['bytes']) is int and row['bytes'] >= 0 and type(row['sha256']) is str and
                re.fullmatch('[0-9a-f]{64}', row['sha256']), 'strict source row shape/bytes/SHA')
        name = row['path']
        require(type(name) is str and name and not name.startswith('/') and ':' not in name and
                '\\' not in name and '\0' not in name and
                all(part not in ('', '.', '..') for part in name.split('/')), 'strict contained row path syntax')
        require(row['path'] not in result, 'unique source row path')
        result[row['path']] = row
    return result


def verify_v13_closure(root, source_lock_ref):
    """Recheck actual V12 metadata ancestry, not a second 16GB byte pass.

    The root's original append freezer performs that full pass. Here all 4953
    inherited rows stay exact; the finite actually-used leaves are rehashed.
    """
    require(safe(root, source_lock_ref['path']).name == 'PRERENT_SOURCE_LOCK_V13.json', 'actual prescribed V13 lock')
    lock = read_json(check_ref(root, source_lock_ref))
    fields = {'schema', 'files', 'ancestry_ref', 'addition_manifest_ref', 'append_freezer_ref',
              'GPU_launch_allowed', 'production_qualified', 'strategy_effect_verified',
              'model_and_SDK_loaded', 'observation_scope'}
    require(type(lock) is dict and set(lock) == fields and lock['schema'] == 'strong_trace_source_lock_v1' and
            all(lock[key] is False for key in ('GPU_launch_allowed', 'production_qualified',
                'strategy_effect_verified', 'model_and_SDK_loaded')) and
            lock['observation_scope'] == 'this_CPU_public_source_append_freeze_only', 'strict CPU V13 source-lock schema')
    refs = source_rows(root, lock['files'])
    ancestor_ref = lock['ancestry_ref']
    require(type(ancestor_ref) is dict and ancestor_ref.get('path') == V12_PATH and
            ancestor_ref.get('sha256') == V12_SHA, 'exact actual V12 ancestor identity')
    ancestor = read_json(closed(root, ancestor_ref, refs, V12_SHA))
    require(type(ancestor) is dict and ancestor.get('schema') == 'strong_trace_source_lock_v1' and
            type(ancestor.get('files')) is list and len(ancestor['files']) == 4953, 'all 4953 actual V12 inherited source rows')
    inherited = source_rows(root, ancestor['files'])
    for name, row in inherited.items():
        require(refs.get(name) == row, 'strict complete V12 row inheritance')
    freezer_ref = lock['append_freezer_ref']
    require(type(freezer_ref) is dict and freezer_ref.get('path') == FREEZER_PATH, 'actual original append freezer path')
    closed(root, freezer_ref, refs, FREEZER_SHA)
    manifest_ref = lock['addition_manifest_ref']
    manifest_path = closed(root, manifest_ref, refs)
    require(manifest_ref['path'].startswith('artifacts/prefix_io_v1/server12-natural-source-audit-20261005/') and
            manifest_ref['bytes'] <= 1024**2, 'actual bounded audit addition manifest')
    manifest = read_json(manifest_path)
    additions = manifest.get('files') if type(manifest) is dict else None
    require(type(manifest) is dict and manifest.get('schema') == 'bounded_public_source_append_manifest_v1' and
            type(additions) is list and 1 <= len(additions) <= 1024 and
            all(type(name) is str for name in additions) and len(set(additions)) == len(additions),
            'exact explicit addition paths')
    for name in additions:
        safe(root, name)
        require(name in refs and not name.endswith(('.partial', '.part', '.tmp')), 'complete declared addition leaf')
    expected_paths = set(inherited) | {ancestor_ref['path'], freezer_ref['path'], manifest_ref['path']} | set(additions)
    require(set(refs) == expected_paths and source_lock_ref['path'] not in refs,
            'all and only inherited/declared source leaves; no lock self-cycle')
    return refs


def derive_family_records(rows, mapping, context_ids):
    """Pure check of source groups and actual complete first block IDs.

    Tests may call this with clearly synthetic rows; this never writes receipts.
    Production passes only independently source-replayed and re-encoded inputs.
    """
    require(type(rows) is list and rows and len(rows) == len(mapping.get('records', [])), 'whole raw/mapping denominator')
    paragraphs = {}
    for paragraph in mapping['selected_paragraphs']:
        key = (paragraph['article_index'], paragraph['paragraph_index'])
        require(key not in paragraphs, 'unique actual source paragraph')
        identity = paragraph['source_identity']
        require(identity['source_sha256'] == mapping['source_sha256'] and
                identity['article_index'] == key[0] and identity['paragraph_index'] == key[1] and
                identity['partition'] == paragraph['partition'] and
                identity['context_sha256'] == paragraph['context_sha256'] and
                paragraph['source_prefix_family'] == 'squad-source-context:' + digest(identity),
                'actual source family identity, not a supplied label')
        paragraphs[key] = paragraph
    article_parts, context_parts, blocks, family_blocks, output = {}, {}, {}, {}, []
    for index, (row, source) in enumerate(zip(rows, mapping['records'])):
        require(all(type(source.get(key)) is int and source[key] >= 0 for key in
                    ('request_id', 'raw_request_pos', 'article_index', 'paragraph_index', 'qa_index')),
                'actual integer source positions')
        require(type(row) is dict and set(row) == {'request_id', 'prompt_sha256', 'prompt_token_ids'} and
                type(row['request_id']) is int and row['request_id'] == index == source['request_id'] == source['raw_request_pos'] and
                row['prompt_sha256'] == source['prompt_sha256'], 'complete accepted raw row order/identity')
        ids = row['prompt_token_ids']
        require(type(ids) is list and 16 <= len(ids) <= 4096 and
                all(type(token) is int and 0 <= token < 152064 for token in ids), 'complete actual first 16-token block')
        key = (source['article_index'], source['paragraph_index'])
        require(key in paragraphs, 'raw row has actual source paragraph')
        paragraph = paragraphs[key]
        partition, family = paragraph['partition'], paragraph['source_prefix_family']
        require(source['partition'] == partition and source['source_prefix_family'] == family and
                source['context_sha256'] == paragraph['context_sha256'], 'row actual source group/partition')
        require(article_parts.setdefault(key[0], partition) == partition, 'article ancestry leaks across partitions')
        require(context_parts.setdefault(paragraph['context_sha256'], partition) == partition, 'same context leaks across partitions')
        context = context_ids.get(key)
        require(type(context) is list and len(context) >= 16 and
                all(type(token) is int and 0 <= token < 152064 for token in context), 'actual independently encoded full context block')
        block = tuple(ids[:16])
        require(block == tuple(context[:16]), 'prompt first block not covered by actual original context')
        require(blocks.setdefault(block, partition) == partition, 'same complete first 16-token block leaks across partitions')
        require(family_blocks.setdefault(family, block) == block, 'one source family lacks same actual first block')
        output.append(dict(row, prefix_family=family))
    proof = dict(schema='CPU_source_group_and_first_block_check_v1', block_size=16,
                 entire_selected_trace_checked=True, article_ancestry_cross_partition_disjoint=True,
                 same_context_cross_partition_disjoint=True, first_complete_blocks_cross_partition_disjoint=True,
                 first_complete_block_covered_by_actual_context=True, family_count=len(family_blocks),
                 family_first_blocks=[dict(prefix_family=family, prompt_token_ids=list(block))
                                      for family, block in sorted(family_blocks.items())],
                 synthetic_fixture=mapping.get('synthetic_fixture'), actual_GPU_operations=0)
    return output, proof


def validate_raw_header(document):
    require(type(document) is dict and document.get('schema') == 'actual_cpu_raw_tokenization_v1' and
            document.get('status') == 'ACTUAL_CPU_RAW_IDS_ONLY' and document.get('synthetic_fixture') is False and
            type(document.get('exit_code')) is int and document['exit_code'] == 0 and
            document.get('tokenizer_imported') is True and type(document.get('actual_GPU_operations')) is int and
            document['actual_GPU_operations'] == 0, 'actual successful raw CPU result; no fixture/mock/GPU promotion')
    require(all(document.get(key) is False for key in
                ('family_proof_available', 'formal_receipt_ready', 'gpu_eligible', 'formal_effect_qualified')),
            'raw artifact remains raw; cannot inherit a supplied family/effect claim')
    backend = document.get('tokenizer_backend')
    require(type(backend) is dict and backend.get('schema') == 'actual_local_CPU_tokenizer_backend_v1' and
            backend.get('version') == '0.22.2' and backend.get('backend_mocked') is False and
            backend.get('from_file_only') is True and backend.get('add_special_tokens') is False and
            backend.get('chat_template_applied') is False, 'actual original nonmocked local backend header')


def close_family(root, *, request_ref, source_lock_ref, result_output_relative, receipt_output_relative):
    refs = verify_v13_closure(root, source_lock_ref)
    self_relative = Path(__file__).resolve().relative_to(Path(root).resolve()).as_posix()
    self_ref = refs.get(self_relative)
    closed(root, self_ref, refs)
    request = read_json(closed(root, request_ref, refs))
    require(type(request) is dict and set(request) == REQUEST_FIELDS and
            request['schema'] == 'CPU_squad_tokenizer_family_close_request_v1' and
            type(request['block_size']) is int and request['block_size'] == 16, 'exact frozen family-close request')
    paths = {key: closed(root, request[key], refs) for key in REF_FIELDS}
    adapter = load_source_only(root, request['adapter_source_ref'], refs, ADAPTER_SHA)
    raw_api = load_source_only(root, request['raw_producer_source_ref'], refs, RAW_PRODUCER_SHA)
    protocol = load_source_only(root, request['protocol_source_ref'], refs, PROTOCOL_SHA)
    raw_api.cpu_modules_only()
    closed(root, request['author_trace_ref'], refs, TRACE_SHA)
    closed(root, request['author_common_ref'], refs, COMMON_SHA)
    source = paths['source_ref'].read_bytes()
    closed(root, request['source_ref'], refs, SOURCE_SHA)
    require(hashlib.sha1(b'blob ' + str(len(source)).encode('ascii') + b'\0' + source).hexdigest() == SOURCE_GIT_BLOB_SHA,
            'actual official frozen Git blob payload')
    contract, trace, mapping, adapter_report = [read_json(paths[key]) for key in (
        'selection_contract_ref', 'trace_ref', 'mapping_ref', 'adapter_report_ref')]
    source_replay = adapter.replay_adapter(source, contract, trace, mapping, adapter_report)
    require(contract['synthetic_fixture'] is False and mapping['synthetic_fixture'] is False and
            adapter_report['synthetic_fixture'] is False, 'actual source only; fixtures never issue receipts')
    inspection = protocol.inspect_trace(paths['trace_ref'], paths['author_trace_ref'], paths['author_common_ref'],
                                        request['trace_ref']['sha256'])
    prompts = protocol.author_cpu_namespace(paths['author_trace_ref'], paths['author_common_ref'])[
        'load_trace_prompts'](str(paths['trace_ref']), None)
    raw_ref = file_ref(root, request['raw_result_relative_path'])
    require(raw_ref['path'] not in refs, 'fresh raw output cannot predate/hash-cycle its own V13 lock')
    actual_raw = read_json(check_ref(root, raw_ref))
    raw_request = read_json(paths['raw_tokenizer_request_ref'])
    validate_raw_header(actual_raw)
    require(canonical(actual_raw.get('source_lock_ref')) == canonical(source_lock_ref) and
            canonical(actual_raw.get('request_ref')) == canonical(request['raw_tokenizer_request_ref']) and
            canonical(actual_raw.get('dataset_ref')) == canonical(request['trace_ref']) and
            canonical(actual_raw.get('inspection')) == canonical(inspection), 'actual raw input/source/order byte closure')
    require(type(raw_request) is dict and set(raw_request) == raw_api.REQUEST_FIELDS and
            raw_request['schema'] == raw_api.REQUEST_SCHEMA and raw_request['synthetic_fixture'] is False and
            canonical(raw_request['dataset_ref']) == canonical(request['trace_ref']), 'same frozen actual raw request')
    require(canonical(actual_raw.get('model_manifest_ref')) == canonical(raw_request['model_manifest_ref']),
            'actual raw model manifest byte identity')
    for key, family_key in (('protocol_ref', 'protocol_source_ref'), ('author_trace_ref', 'author_trace_ref'),
                            ('author_common_ref', 'author_common_ref')):
        require(canonical(raw_request[key]) == canonical(request[family_key]), 'same original author/parser raw domain')
    for key, pin in (('model_manifest_ref', raw_api.MODEL_PLAN_SHA), ('tokenizer_json_ref', raw_api.TOKENIZER_JSON_SHA),
                     ('tokenizer_config_ref', raw_api.TOKENIZER_CONFIG_SHA)):
        closed(root, raw_request[key], refs, pin)
    raw_api.validate_assets(read_json(safe(root, raw_request['tokenizer_json_ref']['path'])),
                            read_json(safe(root, raw_request['tokenizer_config_ref']['path'])))
    backend_sources = raw_api.backend_sources(root, raw_request, refs)
    pins_order = ('protocol_ref', 'author_trace_ref', 'author_common_ref', 'model_manifest_ref',
                  'tokenizer_json_ref', 'tokenizer_config_ref')
    expected_sources = [request['raw_producer_source_ref'], request['raw_tokenizer_request_ref'],
                        request['trace_ref'], *[raw_request[key] for key in pins_order], *backend_sources]
    require(canonical(actual_raw.get('sources')) == canonical(expected_sources), 'entire original raw source set closed')
    rows = actual_raw.get('records')
    require(type(rows) is list and actual_raw.get('raw_records_sha256') == digest(rows) and
            actual_raw.get('accepted_prompt_count') == len(prompts), 'entire raw record digest and denominator')
    raw_api.validate_author_rows(prompts, inspection)
    context_ids = {}
    with raw_api.actual_local_backend(root, raw_request, refs, backend_sources) as (encode, backend_audit):
        require(backend_audit.get('backend_mocked') is False and
                canonical(backend_audit) == canonical(actual_raw.get('tokenizer_backend')), 'same actual nonmocked backend identity')
        replayed_rows = raw_api.tokenize_author_rows(prompts, inspection, encode)
        require(canonical(replayed_rows) == canonical(rows), 'actual CPU re-encoding differs from frozen raw IDs')
        for paragraph in mapping['selected_paragraphs']:
            # Only the source's leading boundary is stripped in an accepted
            # context+question prompt; do not trim its interior/trailing context.
            context_ids[(paragraph['article_index'], paragraph['paragraph_index'])] = encode(paragraph['context'].lstrip())
    family_records, family_blocks = derive_family_records(rows, mapping, context_ids)
    used = [self_ref, request_ref, *[request[key] for key in REF_FIELDS], *expected_sources]
    unique = {row['path']: row for row in used}
    for row in unique.values():
        closed(root, row, refs)
    require(canonical(verify_v13_closure(root, source_lock_ref)) == canonical(refs), 'same complete V13 metadata after encoding')
    check_ref(root, raw_ref)
    check_ref(root, source_lock_ref)
    raw_api.cpu_modules_only()
    result_path, receipt_path = safe(root, result_output_relative), safe(root, receipt_output_relative)
    require(result_path != receipt_path, 'distinct result and receipt outputs')
    for text, path in ((result_output_relative, result_path), (receipt_output_relative, receipt_path)):
        require(text.startswith('artifacts/') and path.parent.is_dir() and not path.exists(), 'fresh output in existing artifact parent')
    def absolute(row):
        return dict(row, path=str(safe(root, row['path']).resolve()))
    evidence = dict(schema='actual_CPU_squad_source_and_token_family_evidence_v1',
                    scope='entire_derived_selected_trace_only', source_replay=source_replay,
                    source_ref=absolute(request['source_ref']), mapping_ref=absolute(request['mapping_ref']),
                    official_git_blob_sha1=SOURCE_GIT_BLOB_SHA,
                    adapter_report_ref=absolute(request['adapter_report_ref']), raw_tokenization_ref=absolute(raw_ref),
                    source_lock_ref=absolute(source_lock_ref), family_request_ref=absolute(request_ref),
                    actual_backend=backend_audit, all_actual_prompt_ids_reencoded_equal=True,
                    original_author_boundary_semantics_preserved=True, first_block_check=family_blocks,
                    actual_GPU_operations=0, GPU_launch_allowed=False)
    token_result = dict(schema='actual_cpu_tokenizer_result_v1', exit_code=0, synthetic_fixture=False,
                        dataset_sha256=inspection['dataset_sha256'], ordered_prompt_digest=inspection['ordered_prompt_digest'],
                        model_manifest_sha256=raw_request['model_manifest_ref']['sha256'],
                        records=family_records, tokenized_records_sha256=digest(family_records),
                        family_closure_evidence=evidence, formal_workload_binding_ready=False,
                        gpu_eligible=False, formal_effect_qualified=False, actual_GPU_operations=0)
    with result_path.open('xb') as stream:
        stream.write(json.dumps(token_result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode('utf-8') + b'\n')
    result_ref = file_ref(root, result_output_relative)
    receipt = dict(schema='cpu_actual_tokenizer_family_receipt_v1', synthetic_fixture=False,
                   dataset_sha256=inspection['dataset_sha256'], ordered_prompt_digest=inspection['ordered_prompt_digest'],
                   model_manifest_sha256=raw_request['model_manifest_ref']['sha256'], records=family_records,
                   family_proof=PROOF, family_closure_evidence=evidence,
                   tokenizer_source_refs=[absolute(unique[name]) for name in sorted(unique)] + [absolute(raw_ref), absolute(source_lock_ref)],
                   tokenization_result_ref=absolute(result_ref), formal_workload_binding_ready=False,
                   gpu_eligible=False, formal_effect_qualified=False, actual_GPU_operations=0)
    # Recheck all actual inputs after the first write, before the receipt.
    for row in unique.values():
        closed(root, row, refs)
    require(canonical(verify_v13_closure(root, source_lock_ref)) == canonical(refs), 'same complete V13 metadata before receipt')
    check_ref(root, raw_ref)
    check_ref(root, source_lock_ref)
    check_ref(root, result_ref)
    with receipt_path.open('xb') as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode('utf-8') + b'\n')
    return dict(schema='actual_CPU_squad_family_receipt_close_report_v1', status='CPU_FAMILY_RECEIPT_CLOSED_ONLY',
                tokenization_result_ref=absolute(result_ref), tokenizer_family_receipt_ref=absolute(file_ref(root, receipt_output_relative)),
                selected_request_count=len(family_records), prefix_family_count=family_blocks['family_count'],
                block_size=16, actual_GPU_operations=0, gpu_eligible=False, formal_effect_qualified=False,
                requires_later_source_lock_for_formal_consumption=True, independent_deadline_or_SLO_created=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    parser.add_argument('--source-lock')
    parser.add_argument('--request')
    parser.add_argument('--result-output-relative')
    parser.add_argument('--receipt-output-relative')
    args = parser.parse_args(argv)
    try:
        require(args.source_lock and args.request and args.result_output_relative and args.receipt_output_relative,
                'actual frozen V13/request/raw inputs and fresh outputs required; UNBOUND')
        result = close_family(args.project_root, request_ref=file_ref(args.project_root, args.request),
                              source_lock_ref=file_ref(args.project_root, args.source_lock),
                              result_output_relative=args.result_output_relative,
                              receipt_output_relative=args.receipt_output_relative)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, TypeError, KeyError, ImportError, AttributeError) as error:
        print(json.dumps(dict(schema='CPU_squad_family_close_rejection_v1', status='REJECTED_CPU_FAMILY_CLOSE',
                              error=str(error), actual_GPU_operations=0, gpu_eligible=False,
                              formal_effect_qualified=False), ensure_ascii=False, sort_keys=True))
        return 78


if __name__ == '__main__':
    raise SystemExit(main())
