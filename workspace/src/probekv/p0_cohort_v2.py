"""CPU input contracts for the approved new 512-token cohort and isolated S0.

These are input proofs, not Source Artifact or GPU quality qualifications.
No answers are consulted to select targets, histories, chronology or split.
"""
from __future__ import annotations
import hashlib
import copy

from .canonical_segment import canonicalize_token_ids
from .data import deterministic_group_split
from .rag_data import segment_text, render_preceding_context
from .source_quality_requests import build_quality_request
from .v8_schema10_execution import digest_json


S0_INSTRUCTION = ('Answer the question using the provided documents. '
                  'Output only the short answer, without explanation, '
                  'additional questions, or copied documents.\n\n')


def independent_s0_request(document_token_ids, target, encode, *, request_id, content_key):
    """Fixed instruction + whole document, no question/answer/other document."""
    tokens = list(document_token_ids)
    if (not request_id or not content_key or target.token_count != 512
            or tuple(tokens[target.token_start:target.token_end]) != target.token_ids):
        raise ValueError('S0 must retain the whole parent document and exact 512-token slice')
    prefix = list(encode(S0_INSTRUCTION))
    suffix = list(encode('\n'))
    begin = len(prefix) + target.token_start
    request = dict(request_id=request_id, token_ids=prefix+tokens+suffix,
        segments=[dict(segment_id='C', content_key=content_key, token_ids=list(target.token_ids),
                       positions=list(range(begin,begin+512)))],
        mandatory_suffix_positions=list(range(begin+512,len(prefix)+len(tokens)+len(suffix))),
        max_new_tokens=1, prefetch_window=1, prefix_cache_enabled=False)
    return dict(kind='independent_document_S0_birth_v2', request=request,
        instruction=S0_INSTRUCTION, instruction_sha256=digest_json(S0_INSTRUCTION),
        question_included=False, answer_included=False, other_documents_included=False,
        diagnostic_only=True, extra_birth_cost_separately_accounted=True,
        artifact_id=None, artifact_built=False, recipe_sha256=digest_json(request))


def qualify_group(group, examples, encode, *, tokenizer_sha256, raw_sha256,
                  model_signature, partition_digest, partition_role):
    """Reconstruct full original prompts, refuse geometry/chronology edits."""
    if deterministic_group_split(group['group_id'],20260726) != 'calibration':
        raise ValueError('old group split is not development; never relabel test')
    rows = group['sources'] + group['targets']
    if partition_role not in {'fit', 'validation'} or not partition_digest:
        raise ValueError('explicit dependency partition required')
    if group['group_id'].split(':', 1)[-1] != group['document_id']:
        raise ValueError('group and original document identity differ')
    ids = [r['origin_id'] for r in rows]
    if len(group['sources']) != 4 or len(ids) != len(set(ids)) or not group['targets']:
        raise ValueError('four unique histories and distinct future consumers required')
    if max(r['pseudotime'] for r in group['sources']) >= min(r['pseudotime'] for r in group['targets']):
        raise ValueError('future Source leak')
    if [r['pseudotime'] for r in rows] != sorted(r['pseudotime'] for r in rows):
        raise ValueError('request epochs must follow pseudotime')
    if len({r['prefix_sha256'] for r in group['sources']}) != 4:
        raise ValueError('four different historical contexts required')
    requests=[]; case=None; full_tokens=None; chosen=None
    for epoch,row in enumerate(rows,1):
        example=examples[row['origin_id']]
        if example.example_id!=row['origin_id'] or row['event_id']!=row['origin_id']+':'+group['document_id']:
            raise ValueError('raw origin/event identity differs')
        if hashlib.sha256(('20260726:'+row['event_id']).encode()).hexdigest()!=row['pseudotime']:
            raise ValueError('pseudotime does not match original deterministic rank')
        docs=[d for d in example.documents if d.document_id==group['document_id']]
        if len(docs)!=1:raise ValueError('target document occurrence ambiguous/missing')
        position=example.documents.index(docs[0])
        if hashlib.sha256(render_preceding_context(example.documents[:position]).encode()).hexdigest()!=row['prefix_sha256']:
            raise ValueError('historical prefix differs from original context')
        tokens=tuple(encode(segment_text(docs[0])))
        segments=canonicalize_token_ids(tokens,tokenizer_signature=tokenizer_sha256,document_revision=raw_sha256)
        match=[s for s in segments if s.token_start==group['token_start'] and s.token_end==group['token_end']]
        if (len(match)!=1 or match[0].token_count!=512 or list(match[0].token_ids)!=group['target_tokens']
                or match[0].canonicalizer_signature!=group['canonicalizer_signature']):
            raise ValueError('canonical target/token identity differs from preregistered search')
        target=match[0]
        if full_tokens is not None and tokens!=full_tokens:raise ValueError('same document id has different token content')
        full_tokens=tokens;chosen=target
        case=dict(target_document_id=group['document_id'],group_id=group['group_id'],
            segment_token_ids=list(target.token_ids),canonical_parent_left_token_ids=list(tokens[:target.token_start]),
            canonical_parent_right_token_ids=list(tokens[target.token_end:]),
            reuse_content_key=target.reuse_content_key(model_signature,tokenizer_sha256))
        request=build_quality_request(example,case,encode,
            request_id='v2-input:'+row['event_id'],request_epoch=epoch,
            partition_role=partition_role,partition_digest=partition_digest,max_model_len=4096,
            max_new_tokens=32,prompt_protocol='short_answer_v1')
        requests.append(dict(role='historical_birth' if epoch<=4 else 'future_consumer',
                             pseudotime=row['pseudotime'],request=request,request_sha256=digest_json(request)))
    s0=independent_s0_request(full_tokens,chosen,encode,request_id='v2-S0:'+group['group_id'],content_key=case['reuse_content_key'])
    if len(s0['request']['token_ids'])+1>4096:raise ValueError('independent whole-document S0 exceeds context bound')
    return dict(group_id=group['group_id'],status='ORIGINAL_PROMPT_GEOMETRY_VERIFIED',
        target_token_count=512,raw_sha256=raw_sha256,tokenizer_sha256=tokenizer_sha256,
        model_signature=model_signature,partition_role=partition_role,partition_digest=partition_digest,
        requests=requests,independent_S0=s0,chronology='corpus_pseudotime',
        Source_artifacts_verified=False,GPU_execution_allowed=False,P1_execution_allowed=False,
        QA_outputs_read=False,locked_test_accessed=False,paper_evidence=False)


def paired_mixed_birth_recipe(birth, parent, upstream, *, parent_pseudotime,
                              birth_pseudotime, consumer_pseudotimes):
    """Bind real identical upstream tokens without altering either raw prompt.

    The returned recipe is not an executed Source and grants no GPU authority.
    A deterministic mask algorithm can be frozen now; its tensor-dependent mask
    digest must be attached by actual construction, never invented beforehand.
    """
    if not consumer_pseudotimes or not (parent_pseudotime < birth_pseudotime < min(consumer_pseudotimes)):
        raise ValueError('parent, birth and consumers must be strictly causal')
    if (parent['origin_example_id'] == birth['origin_example_id']
            or parent['development_partition_digest'] != birth['development_partition_digest']
            or parent['partition_role'] != birth['partition_role']):
        raise ValueError('paired lineage must remain within one dependency partition')
    target = birth['segments'][0]
    positions = target['positions']
    if (len(target['token_ids']) != 512 or len(positions) != 512
            or positions != list(range(positions[0],positions[0]+512))
            or positions[0] < 0 or positions[-1] >= len(birth['token_ids'])
            or [birth['token_ids'][i] for i in positions] != target['token_ids']):
        raise ValueError('paired target must retain exact 512 tokens')
    needle = list(upstream['token_ids'])
    if not needle or not upstream['content_key']:
        raise ValueError('canonical upstream identity required')

    def locate(request, end):
        values = request['token_ids']
        hits = [i for i in range(end-len(needle)+1) if values[i:i+len(needle)] == needle]
        if len(hits) != 1:
            raise ValueError('upstream canonical tokens missing or ambiguous in original prompt')
        return list(range(hits[0], hits[0]+len(needle)))

    bpos = locate(birth, min(target['positions']))
    ppos = locate(parent, len(parent['token_ids']))
    b = copy.deepcopy(birth)
    b['segments'] = [dict(segment_id='U', content_key=upstream['content_key'],
                          token_ids=needle, positions=bpos), copy.deepcopy(target)]
    p = copy.deepcopy(parent)
    p['segments'] = [dict(segment_id='U', content_key=upstream['content_key'],
                          token_ids=needle, positions=ppos)]
    p['mandatory_suffix_positions'] = list(range(ppos[-1]+1,len(p['token_ids'])))
    # Input tokens and original questions stay unchanged; only capture ownership
    # and diagnostic execution policy differ between the two paired births.
    return dict(kind='bounded_paired_EM1_birth_recipe_v2',
        parent_request=p, paired_birth_request=b,
        paired_input_token_sha256=digest_json(b['token_ids']),
        target_positions_sha256=digest_json(target['positions']),
        parent_origin='EXACT_CONTEXT', parent_generation=0,
        exact_birth=dict(target_id='C', upstream_mode='FULL', target_mode='FULL'),
        mixed_birth=dict(target_id='C', upstream_id='U', target_mode='FULL',
            completed_depth=2, first_selective_reuse_layer=3,
            repair_metric='legacy_normalized_kv', repair_ratio=.15,
            mask_digest=None, mask_evidence_required_after_construction=True,
            admission='preregistered_source_correctness_only'),
        target_expected_generations=dict(E=0,M1=1),
        parent_pseudotime=parent_pseudotime,birth_pseudotime=birth_pseudotime,
        consumer_pseudotimes=list(consumer_pseudotimes),
        maximum_build_actions=3,maximum_seconds_per_action=180,
        source_artifacts_verified=False,execution_allowed=False,
        controlled_provenance_diagnostic=True,natural_online_supply_evidence=False)
