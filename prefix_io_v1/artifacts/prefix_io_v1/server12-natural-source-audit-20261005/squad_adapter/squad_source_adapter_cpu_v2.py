"""CPU adaptation/replay of one prospectively fixed public SQuAD selection.

No download, GPU, tokenizer, model, family receipt or formal manifest is created.
Source acquisition authenticity and the contract's pre-download timestamp must
be closed by the caller's actual source/guard records, not asserted by this API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


COMMIT = 'eee5fdbf62f8613a7812b03419e6b29617b74fd1'
URL = 'https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/' + COMMIT + '/dataset/dev-v1.1.json'
PARTITIONS = ['calibration', 'development', 'evaluation']
MAX_SOURCE_BYTES = 16 * 1024**2
FIXED_CONTRACT = {
    'schema': 'prospective_official_squad_fixed_articles_v1',
    'origin': 'prospective_before_source_download_and_any_new_gpu_outcome',
    'source_repository': 'rajpurkar/SQuAD-explorer', 'source_commit': COMMIT,
    'source_relative_path': 'dataset/dev-v1.1.json', 'source_url': URL, 'dataset_version': '1.1',
    'article_indices': [0, 1, 2], 'paragraph_indices': [0, 0, 0], 'partitions': PARTITIONS,
    'retain_all_selected_qas': True, 'selection_by_length_or_outcome': False,
    'prompt_rule': 'original_context_plus_two_LF_plus_original_question', 'inject_answers': False,
    'max_source_bytes': MAX_SOURCE_BYTES, 'max_partition_requests': 32, 'max_total_requests': 96,
}


def require(value, reason):
    if not value:
        raise ValueError('SQUAD_SOURCE_ADAPTER_REJECTED: ' + reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def parse(raw):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda value: require(False, 'nonfinite JSON'))


def nonempty_text(value, name):
    require(type(value) is str and value.strip(), 'actual nonempty ' + name)
    return value


def validate_contract(contract):
    require(type(contract) is dict and set(contract) == set(FIXED_CONTRACT) | {'synthetic_fixture'},
            'exact prospective selection contract fields')
    require(type(contract['synthetic_fixture']) is bool, 'explicit actual-versus-fixture attribution')
    for key, expected in FIXED_CONTRACT.items():
        require(canonical(contract[key]) == canonical(expected), 'fixed prospective rule: ' + key)
    return contract


def _source(source_bytes):
    require(type(source_bytes) is bytes and 0 < len(source_bytes) <= MAX_SOURCE_BYTES,
            'whole actual source byte bound 1..16 MiB; never truncate')
    source = parse(source_bytes)
    require(type(source) is dict and set(source) == {'data', 'version'} and source['version'] == '1.1' and
            type(source['data']) is list and len(source['data']) >= 3,
            'complete SQuAD v1.1 source with first three articles present')
    return source


def validate_adapter(source_bytes, selection_contract):
    """Return author-accepted prompt list plus source-derived mapping and report.

    Selection is never retried/replaced/capped by prompt length or performance.
    The supplied bytes are bound by SHA; authenticity is an external prerequisite.
    """
    contract = validate_contract(selection_contract)
    source = _source(source_bytes)
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    article_metadata = []
    for index, article in enumerate(source['data']):
        require(type(article) is dict and set(article) == {'title', 'paragraphs'} and
                type(article['paragraphs']) is list, 'full article metadata shape')
        title = nonempty_text(article['title'], 'original article title')
        article_metadata.append(dict(article_index=index, title=title, title_sha256=text_sha(title),
                                     paragraph_count=len(article['paragraphs'])))
    trace, records, paragraphs, seen_qids, counts = [], [], [], set(), {}
    for article_index, partition in zip((0, 1, 2), PARTITIONS):
        article = source['data'][article_index]
        require(article['paragraphs'], 'selected article has no paragraph zero; no replacement')
        paragraph = article['paragraphs'][0]
        require(type(paragraph) is dict and set(paragraph) == {'context', 'qas'} and
                type(paragraph['qas']) is list and 1 <= len(paragraph['qas']) <= 32,
                'whole selected paragraph must contain 1..32 qas; no subset selection')
        context = nonempty_text(paragraph['context'], 'original complete context')
        context_sha = text_sha(context)
        identity = dict(schema='source_article_and_context_identity_v1', source_sha256=source_sha,
                        source_repository=contract['source_repository'], source_commit=COMMIT,
                        article_index=article_index, title_sha256=text_sha(article['title']),
                        paragraph_index=0, context_sha256=context_sha, partition=partition)
        family = 'squad-source-context:' + digest(identity)
        counts[partition] = len(paragraph['qas'])
        paragraphs.append(dict(article_index=article_index, paragraph_index=0, partition=partition,
                               title=article['title'], context=context, context_sha256=context_sha,
                               source_paragraph_sha256=digest(paragraph),
                               qa_count=len(paragraph['qas']), source_identity=identity,
                               source_prefix_family=family))
        for qa_index, qa in enumerate(paragraph['qas']):
            require(type(qa) is dict and set(qa) == {'id', 'question', 'answers'}, 'original v1.1 QA fields')
            qid = nonempty_text(qa['id'], 'original qid')
            question = nonempty_text(qa['question'], 'original question')
            require(qid not in seen_qids, 'duplicate qid in selected complete corpus')
            seen_qids.add(qid)
            require(type(qa['answers']) is list and qa['answers'], 'original QA annotations must not be empty')
            for answer in qa['answers']:
                require(type(answer) is dict and set(answer) == {'text', 'answer_start'}, 'original answer annotation shape')
                nonempty_text(answer['text'], 'original answer annotation')
                require(type(answer['answer_start']) is int and
                        0 <= answer['answer_start'] <= len(context) - len(answer['text']) and
                        context[answer['answer_start']:answer['answer_start'] + len(answer['text'])] == answer['text'],
                        'original answer span integer in context')
            prompt = context + '\n\n' + question
            accepted_prompt = prompt.strip()  # Exact unchanged author's load_trace_prompts boundary behavior.
            raw_pos = len(trace)
            trace.append(dict(prompt=prompt))
            records.append(dict(raw_request_pos=raw_pos, request_id=raw_pos,
                                article_index=article_index, paragraph_index=0, qa_index=qa_index, qid=qid,
                                context_sha256=context_sha, question=question, question_sha256=text_sha(question),
                                raw_prompt_sha256=text_sha(prompt), raw_prompt_utf8_bytes=len(prompt.encode('utf-8')),
                                prompt_sha256=text_sha(accepted_prompt), prompt_utf8_bytes=len(accepted_prompt.encode('utf-8')),
                                original_author_boundary_strip_applied=accepted_prompt != prompt,
                                source_qa_sha256=digest(qa), partition=partition, source_prefix_family=family))
    require(1 <= len(trace) <= 96, 'complete selected corpus bound; never truncate')
    mapping = dict(schema='CPU_public_squad_original_source_mapping_v2',
                   source_bytes=len(source_bytes), source_sha256=source_sha,
                   source_repository=contract['source_repository'], source_commit=COMMIT, source_url=URL,
                   source_relative_path=contract['source_relative_path'], dataset_version=source['version'],
                   selection_contract_sha256=digest(contract), synthetic_fixture=contract['synthetic_fixture'],
                   all_article_metadata=article_metadata, selected_paragraphs=paragraphs,
                   records=records, partition_counts=counts, all_selected_qas_preserved=True,
                   no_question_drop_or_reorder=True, no_context_or_answer_injection=True)
    report = dict(schema='CPU_public_squad_source_adapter_report_v2', status='CPU_SOURCE_ADAPTER_ONLY',
                  synthetic_fixture=contract['synthetic_fixture'], source_sha256=source_sha,
                  source_bytes=len(source_bytes), selection_contract_sha256=digest(contract),
                  trace_sha256=digest(trace), mapping_sha256=digest(mapping), selected_request_count=len(trace),
                  partition_counts=counts, original_source_article_count=len(article_metadata),
                  selected_article_indices=[0, 1, 2], selected_paragraph_indices=[0, 0, 0],
                  external_acquisition_authenticity_verified_by_this_API=False,
                  prospective_contract_timestamp_verified_by_this_API=False,
                  source_groups_reconstructed_from_original_bytes=True,
                  original_author_boundary_strip_applied=any(r['original_author_boundary_strip_applied'] for r in records),
                  original_author_boundary_strip_changed_request_count=sum(r['original_author_boundary_strip_applied'] for r in records),
                  original_raw_prompt_unchanged=True,
                  author_prompt_processing='unchanged_original_author_load_trace_prompts_boundary_strip_only',
                  tokenizer_prefix_coverage_proven=False, formal_receipt_ready=False,
                  natural_production_traffic_claim=False, formal_effect_qualified=False,
                  GPU_launch_allowed=False, actual_GPU_operations=0, actual_network_operations=0)
    return dict(trace=trace, mapping=mapping, report=report)


def replay_adapter(source_bytes, selection_contract, trace, mapping, report=None):
    """Reconstruct from source, ignoring every supplied group/split/proof label."""
    actual = validate_adapter(source_bytes, selection_contract)
    require(canonical(actual['trace']) == canonical(trace), 'complete selected trace differs from original source')
    require(canonical(actual['mapping']) == canonical(mapping), 'original source mapping/group/partition drift')
    require(report is None or canonical(actual['report']) == canonical(report), 'CPU report byte-binding drift')
    return dict(schema='CPU_public_squad_source_replay_v2', status='PASS_CPU_SOURCE_REPLAY_ONLY',
                source_sha256=actual['report']['source_sha256'], trace_sha256=digest(trace),
                mapping_sha256=digest(mapping), selected_request_count=len(trace),
                synthetic_fixture=selection_contract['synthetic_fixture'], formal_receipt_ready=False,
                GPU_launch_allowed=False, actual_GPU_operations=0, actual_network_operations=0)


def bounded_file(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= MAX_SOURCE_BYTES,
            'bounded actual regular source/contract/replay file')
    raw = path.read_bytes()
    require(len(raw) <= MAX_SOURCE_BYTES, 'file changed above input byte bound')
    return raw


def append_outputs(*, trace_path, mapping_path, report_path, result):
    """All targets are checked before first write; append only, no directories."""
    paths = [Path(trace_path), Path(mapping_path), Path(report_path)]
    require(len({str(path.resolve()) for path in paths}) == 3, 'distinct trace/mapping/report output paths')
    for path in paths:
        require(path.parent.is_dir() and not path.is_symlink() and not path.exists(),
                'existing output parent and fresh append-only file required')
    for path, key in zip(paths, ('trace', 'mapping', 'report')):
        with path.open('xb') as stream:
            stream.write(json.dumps(result[key], ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False).encode('utf-8') + b'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--selection-contract', type=Path)
    parser.add_argument('--trace-output', type=Path)
    parser.add_argument('--mapping-output', type=Path)
    parser.add_argument('--report-output', type=Path)
    parser.add_argument('--replay-trace', type=Path)
    parser.add_argument('--replay-mapping', type=Path)
    parser.add_argument('--replay-report', type=Path)
    args = parser.parse_args(argv)
    try:
        require(args.source and args.selection_contract, 'actual source and prior frozen contract required; UNBOUND')
        raw = bounded_file(args.source)
        contract_raw = bounded_file(args.selection_contract)
        contract = parse(contract_raw)
        if args.replay_trace or args.replay_mapping or args.replay_report:
            require(args.replay_trace and args.replay_mapping and
                    not any((args.trace_output, args.mapping_output, args.report_output)), 'strict read-only replay inputs')
            report = parse(bounded_file(args.replay_report)) if args.replay_report else None
            result = replay_adapter(raw, contract, parse(bounded_file(args.replay_trace)),
                                    parse(bounded_file(args.replay_mapping)), report)
        else:
            require(args.trace_output and args.mapping_output and args.report_output, 'all three append-only output paths required')
            result = validate_adapter(raw, contract)
            # Replay before writing also prevents accidental locally inconsistent outputs.
            replay_adapter(raw, contract, result['trace'], result['mapping'], result['report'])
            require(bounded_file(args.source) == raw and bounded_file(args.selection_contract) == contract_raw,
                    'actual source/contract unchanged during CPU adaptation')
            append_outputs(trace_path=args.trace_output, mapping_path=args.mapping_output,
                           report_path=args.report_output, result=result)
            result = result['report']
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, TypeError, KeyError, UnicodeError) as error:
        print(json.dumps(dict(schema='CPU_public_squad_source_adapter_rejection_v2', status='REJECTED_CPU_SOURCE',
                              error=str(error), formal_receipt_ready=False, GPU_launch_allowed=False,
                              actual_GPU_operations=0, actual_network_operations=0), ensure_ascii=False, sort_keys=True))
        return 78


if __name__ == '__main__':
    raise SystemExit(main())
