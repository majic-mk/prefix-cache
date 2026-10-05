"""Read-only consumer of frozen P1 input plans; never an execution permission.

Resolve Source build dependencies before GPU dispatch. Literal token requests
stay untouched. File-tokenizer and runtime-tokenizer identities are distinct.
Source Artifact proofs, numerical qualification and authority remain external
prerequisites; an input graph cannot invent them.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path
from .v8_schema10_execution import digest_json


def _require(condition, reason):
    if not condition: raise ValueError(reason)


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def _seconds(row):
    value=row.get('maximum_seconds')
    _require(type(value) in (int,float) and math.isfinite(value) and 0<value<=180,
             'finite preregistered action bound at most 180 seconds required')
    return value


def _target(request, target_id='C', expected_count=512):
    tokens=request['token_ids']
    _require(type(tokens) is list and tokens and all(type(t) is int and t>=0 for t in tokens),
             'literal valid request tokens required')
    matches=[s for s in request['segments'] if s['segment_id']==target_id]
    _require(len(matches)==1,'one exact target occurrence required')
    segment=matches[0]; positions=segment['positions']
    _require(type(positions) is list and len(positions)>0
             and (expected_count is None or len(positions)==expected_count)
             and all(type(p) is int and 0<=p<len(tokens) for p in positions)
             and positions==list(range(positions[0],positions[0]+len(positions)))
             and segment['token_ids']==[tokens[p] for p in positions],
             'target must be the original exact 512-token slice')
    return segment


def resolve_input_graph(documents, *, model_signature, input_model_revision, tokenizer_file_sha256,
                        runtime_tokenizer_identity):
    """Documents must come from the checksummed loader below, not passed flags."""
    _require(all(_sha(x) for x in (model_signature,tokenizer_file_sha256,runtime_tokenizer_identity)),
             'explicit separate model/file-tokenizer/runtime-tokenizer identities required')
    _require(type(input_model_revision) is str and re.fullmatch(r'[^@\s]+@[0-9a-f]{40}',input_model_revision),
             'explicit input model name/revision required; not a runtime model hash')
    freeze=documents['input_freeze.json'];partition=documents['partition_extension.json']
    _require(freeze['partition_digest']==partition['digest'] and freeze['old_partition_modified'] is False,
             'frozen development partition differs')
    partition_rows={r['group_id']:r for r in partition['rows']}
    _require(len(partition_rows)==len(partition['rows']),'duplicate partition group')
    component_roles={}
    for r in partition['rows']:
        _require(r['role'] in ('fit','validation'),'not development fit/validation')
        _require(component_roles.setdefault(r['component'],r['role'])==r['role'],
                 'dependency component crosses fit/validation')
    groups={}; requests={}; components={}
    for name,group in documents.items():
        if not re.fullmatch(r'group-[0-9]+\.json',name):continue
        gid=group['group_id'];role=group['partition_role']
        _require(gid not in groups and gid in partition_rows,'missing/duplicate frozen group')
        _require(group['model_signature']==input_model_revision
                 and group['tokenizer_sha256']==tokenizer_file_sha256
                 and group['partition_digest']==partition['digest']
                 and role==partition_rows[gid]['role'],'model/tokenizer/partition mismatch')
        _require(group['chronology']=='corpus_pseudotime' and group['target_token_count']==512,
                 'declared corpus pseudotime/exact geometry required')
        entries=group['requests'];histories=entries[:4];future=entries[4:]
        _require(len(histories)==4 and future
                 and all(x['role']=='historical_birth' for x in histories)
                 and all(x['role']=='future_consumer' for x in future),'four histories and future consumers required')
        times=[x['pseudotime'] for x in entries]
        _require(all(_sha(t) for t in times) and times==sorted(set(times)),'future Source visibility or repeated epoch')
        shared=None
        for entry in entries:
            q=entry['request'];key=(gid,q['request_id'])
            _require(key not in requests and digest_json(q)==entry['request_sha256'],'duplicate request or input digest mismatch')
            _require(q['content_group']==gid and q['partition_role']==role
                     and q['development_partition_digest']==partition['digest'],'request isolation differs')
            seg=_target(q)
            if shared is None:shared=seg['token_ids']
            _require(seg['token_ids']==shared,'Source and consumer target tokens differ')
            # Actual original examples shared across groups must share partition.
            origin=q['origin_example_id']
            _require(components.setdefault(origin,role)==role,'shared request crosses fit/validation')
            requests[key]=entry
        s0=group['independent_S0'];sq=s0['request']
        _require(all(s0[k] is False for k in ('question_included','answer_included','other_documents_included'))
                 and not any(k in sq for k in ('question','answers','teacher_token_ids'))
                 and s0['recipe_sha256']==digest_json(sq) and _target(sq)['token_ids']==shared,
                 'independent S0 has question/answer leakage or altered tokens')
        groups[gid]=(name,group)
    _require(len(groups)==freeze['groups'] and sum(len(g['requests'])-4 for _,g in groups.values())==freeze['targets'],
             'frozen group/consumer cardinality differs')
    e=documents['planned_P1_E_actions.json'];m=documents['planned_P1_M_actions.json']
    for plan,actions,key in ((e,e['actions'],'consumer_actions_sha256'),(m,m['main_actions'],'actions_sha256')):
        _require(digest_json(actions)==plan[key] and digest_json(plan['builds'])==plan['builds_sha256'],
                 'action/build sequence digest mismatch')
    builds=[];edges={};used=set()
    for row in e['builds']:
        gid=row['group_id'];name,g=groups[gid];r=row['request_ref']
        _require(r['file']==name and row['artifact_id'] is None,'wrong group file or fabricated Artifact')
        if row['role']=='historical_exact':
            index=r.get('index');_require(type(index) is int and 0<=index<4,'historical source cannot be a future target')
            q=g['requests'][index]['request'];arm='historical_'+str(index+1)
        else:
            _require(row['role']=='independent_S0' and r.get('key')=='independent_S0.request','unknown Source build role')
            q=g['independent_S0']['request'];arm='S0'
        _require(digest_json(q)==row['request_sha256'] and (gid,arm) not in edges
                 and row['build_id'] not in used,'duplicate or incorrectly bound build')
        used.add(row['build_id']);edges[gid,arm]=row['build_id']
        builds.append(dict(build_id=row['build_id'],stage='P1-E',group_id=gid,role=row['role'],
            request=q,request_sha256=digest_json(q),depends_on=[],maximum_seconds=_seconds(row),artifact_id=None))
    for gid in groups:
        _require(all((gid,a) in edges for a in ('historical_1','historical_2','historical_3','historical_4','S0')),
                 'a Source build is missing')
    for row in m['builds']:
        gid=row['group_id'];recipe=documents[row['recipe_file']];role=row['role']
        _require(gid in groups and recipe['group_id']==gid and digest_json(recipe)==row['recipe_sha256']
                 and row['artifact_id'] is None and row['execution_allowed'] is False,'mixed birth recipe binding differs')
        _require(role in ('G0_parent','exact_E','mixed_M1'),'unknown paired birth role')
        q=recipe['parent_request'] if role=='G0_parent' else recipe['paired_birth_request']
        birth=recipe['mixed_birth']; exact=recipe['exact_birth']
        original_birth=[x['request'] for x in groups[gid][1]['requests'][:4]
                        if x['request']['request_id']==recipe['paired_birth_request']['request_id']]
        _require(len(original_birth)==1 and original_birth[0]['token_ids']==recipe['paired_birth_request']['token_ids'],
                 'paired birth is not an original frozen historical prompt')
        parent_segment=_target(recipe['parent_request'],'U',None)
        birth_upstream=_target(recipe['paired_birth_request'],'U',None)
        birth_target=_target(recipe['paired_birth_request'])
        _require(parent_segment['token_ids']==birth_upstream['token_ids']
                 and birth_upstream['positions'][-1]<birth_target['positions'][0]
                 and recipe['paired_input_token_sha256']==digest_json(recipe['paired_birth_request']['token_ids'])
                 and recipe['target_positions_sha256']==digest_json(birth_target['positions'])
                 and recipe['partition_digest']==partition['digest']
                 and recipe['partition_role']==groups[gid][1]['partition_role']
                 and recipe['target_expected_generations']==dict(E=0,M1=1),
                 'paired upstream/target token identity, isolation or generation differs')
        _require(exact==dict(target_id='C',upstream_mode='FULL',target_mode='FULL')
                 and birth['target_id']=='C' and birth['target_mode']=='FULL'
                 and birth['completed_depth']==2 and birth['first_selective_reuse_layer']==3
                 and birth['repair_metric']=='legacy_normalized_kv' and birth['repair_ratio']==.15
                 and birth['mask_digest'] is None,'frozen E/M1 birth policy differs')
        _require(recipe['parent_generation']==0 and recipe['parent_origin']=='EXACT_CONTEXT'
                 and recipe['parent_pseudotime']<recipe['birth_pseudotime']
                 and all(recipe['birth_pseudotime']<t for t in recipe['consumer_pseudotimes']),
                 'invalid parent origin or paired birth chronology')
        ident='M:'+gid+':'+role
        _require(ident not in used,'duplicate mixed Source build')
        used.add(ident)
        if role!='G0_parent':
            _require(_target(q)['token_ids']==_target(groups[gid][1]['requests'][0]['request'])['token_ids'],
                     'paired birth target differs')
            edges[gid,'E' if role=='exact_E' else 'M1']=ident
        builds.append(dict(build_id=ident,stage='P1-M',group_id=gid,role=role,request=q,
            request_sha256=digest_json(q),depends_on=['M:'+gid+':G0_parent'] if role=='mixed_M1' else [],
            frozen_recipe_file=row['recipe_file'],maximum_seconds=_seconds(row),artifact_id=None))
    actions=[];seen=set();per_request={}
    for stage,rows,arms,metric_field in (('P1-E',e['actions'],{'dense','historical_1','historical_2','historical_3','historical_4','S0'},'repair'),
                                        ('P1-M',m['main_actions'],{'dense','E','M1'},'repair_metric')):
        for row in rows:
            gid=row['group_id'];rid=row['request_id'];arm=row['arm'];key=(stage,gid,rid,arm)
            entry=requests[gid,rid];q=entry['request']
            _require(key not in seen and arm in arms and entry['role']=='future_consumer'
                     and row['request_sha256']==digest_json(q) and row['execution_allowed'] is False,
                     'invalid, duplicated or unbound consumer')
            seen.add(key);per_request.setdefault((stage,gid,rid),set()).add(arm)
            _require((row['first_reuse_layer'],row[metric_field],row['repair_ratio'])==
                     ((None,None,None) if arm=='dense' else (9,'legacy_normalized_kv',.15)),
                     'consumer policy differs from fixed legacy KV15 layer9')
            dependency=[] if arm=='dense' else [edges[gid,arm]]
            if stage=='P1-M':
                paired=[documents[b['recipe_file']] for b in m['builds'] if b['group_id']==gid and b['role']=='exact_E']
                _require(len(paired)==1 and entry['pseudotime'] in paired[0]['consumer_pseudotimes'],
                         'mixed consumer not in frozen paired chronology')
            actions.append(dict(action_key=list(key),request=q,request_sha256=digest_json(q),
                depends_on=dependency,maximum_seconds=_seconds(row),execution_allowed=False))
        _require(all(v==arms for k,v in per_request.items() if k[0]==stage),'incomplete per-target arm matrix')
    _require(len(e['actions'])==freeze['P1_E_actions'] and len(m['main_actions'])==freeze['P1_M_actions']
             and len(e['builds'])==freeze['P1_E_builds'] and len(m['builds'])==freeze['P1_M_builds'],
             'frozen action budget differs')
    _require(all(dep in used for b in builds for dep in b['depends_on']),'unresolved build dependency')
    result=dict(kind='resolved_P1_input_graph_v2',status='INPUT_DEPENDENCIES_VERIFIED_NOT_EXECUTABLE',
        model_signature=model_signature,input_model_revision=input_model_revision,tokenizer_file_sha256=tokenizer_file_sha256,
        runtime_tokenizer_identity=runtime_tokenizer_identity,partition_digest=partition['digest'],
        source_builds=builds,consumer_actions=actions,
        maximum_build_seconds=sum(b['maximum_seconds'] for b in builds),
        maximum_consumer_seconds=sum(a['maximum_seconds'] for a in actions),
        maximum_required_context_tokens=max(len(x['request']['token_ids'])+x['request'].get('max_new_tokens',1)
                                            for x in builds+actions),
        request_tokens_modified=False,source_artifacts_verified=False,GPU_execution_allowed=False,
        P1_execution_allowed=False,paper_evidence=False,locked_test_accessed=False,
        remaining=['BOUND_NATIVE_DISPATCH','ACTUAL_SOURCE_ARTIFACT_PROOFS','P0_QUALIFICATION','SCOPED_P1_AUTHORIZATION'])
    result['graph_sha256']=digest_json(result)
    return result


def load_frozen_documents(root, *, index_sha256):
    root=Path(root).resolve();index=root/'files.sha256.json'
    _require(_sha(index_sha256) and hashlib.sha256(index.read_bytes()).hexdigest()==index_sha256,
             'external frozen file-index digest differs')
    entries=json.loads(index.read_text(encoding='utf-8'));documents={}
    for name,expected in entries.items():
        _require(type(name) is str and re.fullmatch('[A-Za-z0-9_.-]+',name) is not None and _sha(expected),
                 'unsafe indexed file')
        path=root/name
        _require(path.resolve().parent==root and path.is_file() and path.stat().st_size<=32*1024**2,
                 'missing, escaping or oversized indexed input')
        raw=path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest()==expected,'frozen input file checksum differs: '+name)
        documents[name]=json.loads(raw)
    return documents


def load_frozen_input_graph(root, *, index_sha256, **identity):
    documents=load_frozen_documents(root,index_sha256=index_sha256)
    result=resolve_input_graph(documents,**identity)
    return dict(result,file_index_sha256=index_sha256)
