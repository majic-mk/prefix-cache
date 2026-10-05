"""Fixed-Source P1 QA, isolated from online selector/economic admission.

No Source publication, threshold adjustment, alternate winner or dense rescue
may fill a requested Source arm. Failures retain their arm and raw audit.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import hashlib
import time

from .p1_build_dispatch_v2 import _require, _seal, _check_seal, _birth_request, dispatch_binding, bind_request_content_keys
from .p1_input_consumer_v2 import resolve_input_graph
from .native_consumption_v2 import NativeV2ConsumptionSession
from .source_comparison_v2 import ComparisonProfileBindingV2, ComparisonSessionV2, scorer_digest
from .source_manifest_v2 import ManifestReference, request_input_digest
from .v8_schema10_execution import digest_json


def qa_dispatch_binding():
    return dict(dispatch_binding(),qa_dispatch_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def compile_p1_qa(documents, *, identity, action_key, comparison_profile, resources):
    graph=resolve_input_graph(documents,**identity)
    rows=[r for r in graph['consumer_actions'] if r['action_key']==list(action_key)]
    _require(len(rows)==1,'one frozen QA action required')
    a=rows[0];arm=a['action_key'][-1]
    q=deepcopy(a['request'])
    _require(q.get('max_new_tokens')==32 and not any(q.get(k) for k in (
        'teacher_token_ids','capture_logits','capture_original_full_prefill','publish_exact_prefix_shadow',
        'use_gpu_hot_cache','retain_gpu_hot_cache','native_dense_continuation','force_nonpaper_measurement_admission')),
        'original 32-token free-generation QA required; no execution override')
    original_keys={s['segment_id']:s.pop('content_key',None) for s in q['segments']}
    q.update(request_id=q['request_id']+':P1qa:'+digest_json(a['action_key'])[:16],prefetch_window=0)
    from .p0_launch_recipe_v2 import _request
    _request(q)
    p=ComparisonProfileBindingV2(**comparison_profile)
    _require(p.completed_depth==8 and p.model_signature==identity['model_signature']
        and p.runtime_digest==dispatch_binding()['base_runtime_digest']
        and p.scoring_function_digest==scorer_digest() and p.provenance_policy=='ALLOW_MIXED_G1',
        'P1 legacy layer9 identity required')
    _require(set(resources)=={'host_comparison_bytes','cuda_comparison_bytes'}
        and all(type(x)is int and x>0 for x in resources.values()),'explicit comparison resources required')
    build=next((b for b in graph['source_builds'] if b['build_id'] in a['depends_on']),None)
    return _seal(dict(kind='P1_fixed_source_QA_operation_v2',action_key=a['action_key'],stage=a['action_key'][0],
        input_graph_sha256=graph['graph_sha256'],original_request_sha256=a['request_sha256'],
        original_content_keys=original_keys,original_prefetch_window=a['request'].get('prefetch_window',0),
        request=q,request_sha256=digest_json(q),
        arm=arm,target_id='C',first_reuse_layer=None if arm=='dense' else 9,
        repair_metric=None if arm=='dense' else 'legacy_normalized_kv',repair_ratio=None if arm=='dense' else .15,
        expected_build_id=None if build is None else build['build_id'],
        expected_birth_request=None if build is None else _birth_request(build),
        expected_birth_original_sha256=None if build is None else build['request_sha256'],
        expected_source_generation=None if build is None else int(build['role']=='mixed_M1'),
        comparison_profile=asdict(p),resources=deepcopy(resources),dispatch_binding=qa_dispatch_binding(),
        maximum_seconds=a['maximum_seconds'],economic_admission_evaluated=False,
        source_selection_evaluated=False,production_reuse_commit_observed=False,
        publication_allowed=False,GPU_execution_allowed=False,paper_evidence=False),'operation_sha256')


def verify_qa_source(store,snapshot,operation,receipt):
    """Receipt plus actual full birth manifest/backing, not receipt claims alone."""
    _check_seal(receipt,'receipt_sha256')
    o=operation;r=receipt
    _require(r['kind']=='P1_source_build_receipt_v2' and r['build_id']==o['expected_build_id']
        and r['input_graph_sha256']==o['input_graph_sha256']
        and r['original_request_sha256']==o['expected_birth_original_sha256']
        and r['dispatch_binding']==dispatch_binding() and r['cleanup_passed'] is True,
        'actual bound build receipt required')
    with store.lock:
        row=store._visible_row(snapshot,r['source_id'])
        pq=o['expected_birth_request'];pt=next(s for s in pq['segments'] if s['segment_id']=='C')
        target=next(s for s in o['request']['segments'] if s['segment_id']=='C')
        _require(row['token_ids']==pt['token_ids']==target['token_ids']
            and row['birth_target_positions']==pt['positions'] and row['birth_request_id']==pq['request_id']
            and row['generation']==r['generation']==o['expected_source_generation']
            and row['artifact_digest']==r['artifact_digest'] and row['manifest_reference']==r['manifest_reference'],
            'requested Source does not match actual birth/target')
        manifest,_=store._manifest_record(ManifestReference(**row['manifest_reference']),row['token_ids'],row['birth_target_positions'])
        _require(manifest['token_ids']==pq['token_ids'] and manifest['absolute_positions']==list(range(len(pq['token_ids'])))
            and manifest['input_digest']==request_input_digest(pq['token_ids'],tuple(range(len(pq['token_ids'])))),
            'Source historical context differs')
        store._verify_row(row,full=True)
        visible=store.lookup(snapshot,target['token_ids'])
        _require(tuple(v['source_id'] for v in visible)==(r['source_id'],),
                 'fixed QA arm requires its unique target Source, not selector reranking')
        return deepcopy(row)


class P1FixedSourceSession(NativeV2ConsumptionSession):
    def prepare_prescribed(self,receipt,operation,source_id):
        o=operation;c=self.context
        _check_seal(o,'operation_sha256')
        _require(o['kind']=='P1_fixed_source_QA_operation_v2' and o['arm']!='dense'
            and o['dispatch_binding']==qa_dispatch_binding() and bind_request_content_keys(o['request'],self.store)==c.request
            and o['comparison_profile']==asdict(self.comparison.profile)
            and o['economic_admission_evaluated'] is False and o['publication_allowed'] is False
            and c.current_completed_depth==8 and getattr(c,'source_capture_v2',None) is None
            and receipt.segment_id=='C' and receipt.winner_source_id==source_id
            and tuple(receipt.eligible_ids)==tuple(receipt.available_ids)==tuple(receipt.compared_ids)==(source_id,),
            'explicit fixed Source arm and actual complete observation required')
        # Keep the existing atomic lease/HBM/stale checks. This marker is
        # already rejected by ordinary production commit in the base bridge.
        return self._prepare_bound(receipt,controlled_recipe=dict(recipe_digest=o['operation_sha256'],
            source_id=source_id,residual_admission_evaluated=False,production_execution_allowed=False,
            experiment_kind='P1_fixed_source_QA'))


def execute_p1_qa(context,store,snapshot,operation,*,expected_operation_sha256,source_receipt=None):
    from .native_p0_request_v2 import P0RequestFailure
    from .native_p0_operation_v2 import P0ComparisonActionV2,validate_p0_comparison_action
    from .p0_lineage_control_v2 import _commit_recipe
    _check_seal(operation,'operation_sha256',expected_operation_sha256)
    o=operation;c=context
    _require(o['kind']=='P1_fixed_source_QA_operation_v2' and o['dispatch_binding']==qa_dispatch_binding()
        and c.request==bind_request_content_keys(o['request'],store) and digest_json(o['request'])==o['request_sha256']
        and c.current_completed_depth==0 and not c.cached_prefix_tokens and not c.prepared and not c.committed
        and not c.frozen and getattr(c,'source_consumption_v2',None) is None
        and not c.finished and not c.closed and c.repair_ratio==.15
        and c.adapter.native_repair_metric=='normalized_kv_deviation'
        and c.adapter.path=='legacy_multicheckpoint' and 8 in c.adapter.depths
        and getattr(c,'source_capture_v2',None) is None and c.adapter.costs is None,
        'fresh qualified legacy fixed15 context required')
    profile=ComparisonProfileBindingV2(**o['comparison_profile'])
    _require(profile.model_signature==store.config['model_signature']==c.adapter.provenance['model_signature']
        and profile.runtime_digest==dispatch_binding()['base_runtime_digest']
        and profile.completed_depth==8 and profile.scoring_function_digest==scorer_digest(),
        'dense and Source arms require the same qualified model/profile')
    with store.lock:
        store._snapshot(snapshot)
        _require(snapshot.request_id==c.request['request_id'],'QA snapshot belongs to another request')
    row=None
    if o['arm']=='dense':
        _require(source_receipt is None and o['expected_build_id'] is None,'dense arm cannot bind a Source')
    else:
        _require(source_receipt is not None,'Source arm cannot silently fall back dense')
        row=verify_qa_source(store,snapshot,o,source_receipt)
        action=P0ComparisonActionV2(c.request['request_id'],'C',snapshot.snapshot_id,
            request_input_digest(c.request['token_ids'],tuple(range(len(c.request['token_ids'])))),8,
            (row['source_id'],),profile.binding_digest,o['resources']['host_comparison_bytes'],o['resources']['cuda_comparison_bytes'])
        validate_p0_comparison_action(c,store,snapshot,profile,action)
    audit=dict(kind='P1_fixed_source_QA_execution_v2',status='RUNNING',operation_sha256=o['operation_sha256'],
        action_key=o['action_key'],answer=None,cleanup=None,source_id=None if row is None else row['source_id'],
        economic_admission_evaluated=False,source_selection_evaluated=False,production_reuse_commit_observed=False,
        source_publication_allowed=False,extra_forward_count=0,paper_evidence=False)
    issuer=None;error=None
    try:
        c.adapter.check_deadline()
        if row is not None:
            c.advance_to_depth(8)
            workspace=c.configure_cuda_comparison_v2(capacity_bytes=o['resources']['cuda_comparison_bytes'])
            issuer=ComparisonSessionV2(store,snapshot,profile,workspace_bytes=o['resources']['host_comparison_bytes'],cuda_workspace=workspace)
            _require(getattr(c,'source_consumption_v2',None) is None,'existing consumption cannot be replaced')
            bridge=P1FixedSourceSession(c,issuer);c.source_consumption_v2=bridge
            receipt=issuer.compare_native(c,'C');audit['observation']=asdict(receipt)
            bridge.prepare_prescribed(receipt,o,row['source_id'])
            c.finish_selection(c.frozen,c.prepared)
            ready,mask_digest=c.ready_for_final_commit(c.prepared)
            _require(ready=={'C':9} and c.current_completed_depth==8,'fixed boundary moved')
            _commit_recipe(c,'C')
            audit.update(repair_support=deepcopy(c.supports),mask_digest=mask_digest,committed_boundaries=dict(c.committed))
        first=[];audit['answer']=c.finish(lambda:first.append(time.perf_counter_ns()))
        _require(c.finished and len(first)==1 and first[0]>=c.arrival_ns,'missing actual first token')
        if row is not None:
            commit=c.engine.session.commits.get('C')
            _require(commit is not None and commit.source_id==row['source_id'] and c.committed=={'C':9},
                     'requested Source did not execute; cannot replace with dense result')
        audit.update(status='COMPLETED',ttft_ms=(first[0]-c.arrival_ns)/1e6,
                     timing_scope='request_open_to_first_token_includes_preparation_not_birth_or_preflight')
    except BaseException as exc:
        error=exc;audit.update(status='FAILED',failure=dict(type=type(exc).__name__,detail=str(exc)))
    finally:
        failures=[];fenced=False;ended=False
        if getattr(c,'source_consumption_v2',None) is not None:
            try:audit['consumption']=c.source_consumption_v2.audit()
            except BaseException as exc:failures.append(str(exc));error=error or exc
        try:c.close();fenced=True
        except BaseException as exc:failures.append(str(exc));error=error or exc
        if issuer is not None:
            try:issuer.close()
            except BaseException as exc:failures.append(str(exc));error=error or exc
        if fenced:
            try:store.end_request(snapshot);ended=True
            except BaseException as exc:failures.append(str(exc));error=error or exc
        audit['cleanup']=dict(passed=not failures,failures=failures,snapshot_retained=not ended)
        audit['service_end_ns']=time.perf_counter_ns()
    if error is not None:
        audit['status']='FAILED'
        raise P0RequestFailure(audit,error) from error
    return audit
