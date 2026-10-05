"""One bounded P1 operation, with independent stage authority and raw logs."""
from pathlib import Path
import json
import time
from .p1_build_dispatch_v2 import _require,_check_seal,dispatch_binding,execute_p1_source_build,bind_request_content_keys
from .p1_qa_dispatch_v2 import qa_dispatch_binding,execute_p1_qa
from .p0_batch_v2 import _positive,_authority_duration_seconds,_ActionWallDeadline,_native_origin
from .p0_evidence_v2 import P0EvidenceWriter,write_new_json
from .v8_schema10_storage import file_digest
from .v8_schema10_execution import digest_json


def runner_binding():
    root=Path(__file__).parent
    return dict(qa_dispatch_binding(),action_runner_sha256=file_digest(Path(__file__)),
        qualification_sha256=file_digest(root/'p1_qualification_v2.py'),
        input_consumer_sha256=file_digest(root/'p1_input_consumer_v2.py'),
        entry_sha256=file_digest(root.parents[1]/'scripts/server/run_decoupled_v2_p1_action.py'))


def verify_compiled_operation(operation, frozen_inputs):
    """A self-consistent hash is not membership in the frozen action graph."""
    from .p1_input_consumer_v2 import load_frozen_documents
    from .p1_build_dispatch_v2 import compile_p1_build
    from .p1_qa_dispatch_v2 import compile_p1_qa
    args=dict(frozen_inputs);root=args.pop('root');index=args.pop('index_sha256')
    docs=load_frozen_documents(root,index_sha256=index)
    common=dict(identity=args,comparison_profile=operation['comparison_profile'],resources=operation['resources'])
    if operation['kind']=='P1_source_build_operation_v2':
        expected=compile_p1_build(docs,build_id=operation['build_id'],**common)
    else:expected=compile_p1_qa(docs,action_key=operation['action_key'],**common)
    _require(expected==operation,'operation differs from actual frozen recipe')


def validate_comparison_capacity(operation, model_config, *, allocator_capacity_bytes=None):
    """Fail before model loading; use the existing runtime estimator unchanged.

    Config bytes must first be verified against the model audit by the caller.
    This check neither increases a resource budget nor truncates the request.
    """
    from .cuda_comparison_v2 import comparison_workspace_estimate
    kind=operation['kind']
    _require(kind in ('P1_source_build_operation_v2','P1_fixed_source_QA_operation_v2'),
             'unknown comparison operation')
    needed=(operation.get('role')=='mixed_M1' if kind=='P1_source_build_operation_v2'
            else operation['arm']!='dense')
    if not needed:
        return dict(status='NOT_APPLICABLE',comparison_executed=False)
    hidden=model_config.get('hidden_size');heads=model_config.get('num_attention_heads')
    kv_heads=model_config.get('num_key_value_heads')
    _require(all(type(v)is int and v>0 for v in (hidden,heads,kv_heads))
             and hidden%heads==0 and heads%kv_heads==0,'explicit GQA geometry required')
    head_dim=model_config.get('head_dim',hidden//heads)
    _require(type(head_dim)is int and head_dim>0,'explicit positive head dimension required')
    target='U' if kind=='P1_source_build_operation_v2' else operation['target_id']
    rows=[s for s in operation['request']['segments'] if s['segment_id']==target]
    _require(len(rows)==1,'comparison target missing or duplicated')
    required=comparison_workspace_estimate(active_rows=len(operation['request']['token_ids']),
        hidden_width=hidden,query_width=heads*head_dim,kv_width=kv_heads*head_dim,
        target_rows=len(rows[0]['positions']))
    capacity=operation['resources']['cuda_comparison_bytes']
    _require(type(capacity)is int and capacity>0,'explicit comparison capacity required')
    _require(required<=capacity,
        'comparison workspace preflight: required_bytes=%d exceeds configured_bytes=%d; no automatic increase'
        %(required,capacity))
    result=dict(status='SUPPORTED',required_bytes=required,configured_bytes=capacity,
                comparison_executed=False,model_loaded=False)
    if allocator_capacity_bytes is not None:
        from .v8_schema6_hbm import UnifiedHBMReservationManager,HBMReservationKind
        layers=model_config.get('num_hidden_layers')
        _require(type(layers)is int and layers>0,'explicit layer count required')
        # These P1 operations require a fresh zero-Prefix context. Match the
        # existing native working-KV and target-only winner reservations.
        working=len(operation['request']['token_ids'])*layers*kv_heads*head_dim*4
        winner=len(rows[0]['positions'])*layers*kv_heads*head_dim*4
        manager=UnifiedHBMReservationManager(allocator_capacity_bytes=allocator_capacity_bytes)
        reservations=(('working',working,HBMReservationKind.COMMITTED_EXECUTION),
            ('comparison',capacity,HBMReservationKind.SELECTION_WORKSPACE),
            ('winner',winner,HBMReservationKind.WINNER_PREFETCH))
        try:manager.reserve_batch(owner_request_id='CPU-preflight-only',rows=reservations)
        except MemoryError as exc:
            raise ValueError('joint HBM preflight: working=%d comparison=%d winner=%d safety=%d capacity=%d'
                %(working,capacity,winner,manager.safety_bytes,allocator_capacity_bytes)) from exc
        result.update(working_kv_bytes=working,winner_kv_bytes=winner,
            safety_bytes=manager.safety_bytes,allocator_capacity_bytes=allocator_capacity_bytes,
            joint_required_bytes=working+capacity+winner+manager.safety_bytes)
    return result


def verify_build_evidence(result_path, *, result_sha256, receipt):
    """Require a completed raw GPU batch, not a receipt left by a failed run."""
    from .p0_evidence_v2 import read_p0_events
    path=Path(result_path);root=path.parent
    _require(path.name=='result.json' and file_digest(path)==result_sha256,'build result digest differs')
    def read(relative, descriptor=None):
        p=root/relative
        _require(p.is_file() and not p.is_symlink() and p.stat().st_size<=16*1024*1024,'missing/bounded raw build file')
        if descriptor is not None:
            _require(descriptor['file']==p.name and descriptor['sha256']==file_digest(p),'build raw file digest differs')
        return json.loads(p.read_text(encoding='utf-8'))
    report=read('result.json');manifest=read('manifest.json')
    _check_seal(manifest,'manifest_sha256');_check_seal(receipt,'receipt_sha256')
    _require(report['status']=='COMPLETED' and report['evidence_origin']=='real_cuda_execution'
        and manifest['runner_binding']==runner_binding(),'completed matching native build required')
    _require(file_digest(root/'actions.jsonl')==report['raw_event_sha256'],'build event file digest differs')
    events=read_p0_events(root/'actions.jsonl',binding=manifest['binding'])
    _require(events and len(events)==report['event_count'] and events[-1]['kind']=='batch_stopped'
        and events[-1]['event_sha256']==report['final_event_sha256']
        and events[-1]['payload']['status']=='COMPLETED','incomplete build event stream')
    receipts=[r for r in events if r['kind']=='P1_build_receipt' and r['action_id']=='action']
    records=[r for r in events if r['kind']=='action_recorded' and r['action_id']=='action']
    _require(len(receipts)==len(records)==1,'unique build receipt/action required')
    _require(read('source_build_receipt.json',receipts[0]['payload'])==receipt,'receipt differs from completed build')
    record=read('action/record.json',records[0]['payload']['record'])
    audit=read('action/request.json',record['request'])
    _require(record['evidence_origin']=='real_cuda_execution' and record['execution_completed'] is True
        and audit['status']=='COMPLETED' and audit['cleanup']['passed'] is True
        and digest_json(audit)==receipt['publication_audit_sha256'],'build execution/cleanup not verified')
    operation=read('operation.json');_check_seal(operation,'operation_sha256',manifest['operation_sha256'])
    _require(operation['kind']=='P1_source_build_operation_v2'
        and operation['operation_sha256']==receipt['operation_sha256']
        and operation['dispatch_binding']==dispatch_binding(),'receipt build operation differs')
    return dict(raw_build_verified=True,evidence_origin='real_cuda_execution',paper_evidence=False)


def validate_p1_action(manifest,operation,*,actual_binding,now_unix):
    _check_seal(manifest,'manifest_sha256')
    _check_seal(operation,'operation_sha256',manifest['operation_sha256'])
    _require(manifest['kind']=='bounded_P1_single_action_v2' and manifest['binding']==actual_binding
        and manifest['runner_binding']==runner_binding() and manifest['maximum_actions']==1
        and manifest['phase']==operation['stage'] and manifest['phase'] in ('P1-E','P1-M')
        and manifest['locked_test_accessed'] is False and manifest['automatic_rental_allowed'] is False,
        'explicit matching P1 single-action scope required')
    kind=operation['kind']
    _require(kind in ('P1_source_build_operation_v2','P1_fixed_source_QA_operation_v2'), 'unsupported P1 operation')
    _require(operation['dispatch_binding']==(dispatch_binding() if kind=='P1_source_build_operation_v2' else qa_dispatch_binding()),
             'operation execution code changed')
    authority=manifest['authority']
    _require(authority['phase']==manifest['phase'] and type(authority['approval_reference']) is str
        and authority['approval_reference'].strip() and authority['instance_id']==actual_binding['instance_id']
        and authority['gpu_uuid']==actual_binding['gpu_uuid']
        and _positive(authority['starts_at_unix']) and _positive(authority['expires_at_unix'])
        and authority['starts_at_unix']<=now_unix<authority['expires_at_unix'],
        'current separate P1 authorization required; P0 authority cannot be inherited')
    available=min(_authority_duration_seconds(authority),authority['expires_at_unix']-now_unix)
    limits=manifest['limits']
    _require(all(_positive(limits[k]) for k in ('initialization_upper_seconds','cleanup_seconds'))
        and _positive(operation['maximum_seconds']) and operation['maximum_seconds']<=180,
        'bounded initialization/action/cleanup required')
    required=limits['initialization_upper_seconds']+operation['maximum_seconds']+limits['cleanup_seconds']
    _require(required<=available,'action and cleanup exceed remaining authorization')
    return dict(status='P1_ACTION_INPUTS_VALIDATED_NOT_RUN',available_seconds=available,required_seconds=required,
                GPU_execution_allowed=False,paper_evidence=False)


def run_p1_native_action(adapter,store,manifest,operation,*,actual_binding,output,session_started_ns,source_receipt=None):
    validate_p1_action(manifest,operation,actual_binding=actual_binding,now_unix=time.time())
    _require(type(session_started_ns)is int and 0<session_started_ns<=time.perf_counter_ns(),'actual session start required')
    _require(file_digest(store.root/'catalog.json')==actual_binding['initial_pool_sha256']
        and file_digest(store.registry._root/'registry.json')==actual_binding['initial_registry_sha256']
        and store.config['model_signature']==actual_binding['model_signature']
        and store.config['tokenizer_hash']==actual_binding['tokenizer_hash']
        and adapter.provenance['model_signature']==actual_binding['model_signature']
        and adapter.provenance['tokenizer_hash']==actual_binding['tokenizer_hash']
        and adapter.active is None and not adapter.hbm.active_reserved_bytes,'native/pool identity or quiescence differs')
    origin=_native_origin(adapter)
    elapsed=(time.perf_counter_ns()-session_started_ns)/1e9
    authority=manifest['authority'];limits=manifest['limits']
    remaining=min(_authority_duration_seconds(authority)-elapsed,authority['expires_at_unix']-time.time())
    _require(remaining>=operation['maximum_seconds']+limits['cleanup_seconds'],'insufficient post-initialization window')
    writer=P0EvidenceWriter(output,binding=actual_binding,manifest=manifest)
    write_new_json(writer.root/'operation.json',operation)
    timer=_ActionWallDeadline(operation['maximum_seconds'],native=origin=='real_cuda_execution')
    context=snapshot=None;partial=None;old_deadline=adapter.deadline
    started=time.perf_counter_ns();status='FAILED';error=None
    writer.append('P1_action_started','action',dict(operation_sha256=operation['operation_sha256'],evidence_origin=origin))
    try:
        timer.start();adapter.deadline=min(old_deadline,time.perf_counter()+operation['maximum_seconds'])
        adapter.reset();snapshot=store.begin_request(operation['request']['request_id'])
        execution_request=bind_request_content_keys(operation['request'],store)
        writer.append('P1_execution_request_binding','action',dict(compiled_request_sha256=operation['request_sha256'],
            execution_request_sha256=digest_json(execution_request),binding_mode='exact_tokens_actual_store_namespace',
            content_keys={s['segment_id']:s['content_key'] for s in execution_request['segments']}))
        with adapter.open_request(execution_request,arrival_ns=time.perf_counter_ns()) as context:
            if operation['kind']=='P1_source_build_operation_v2':
                result=execute_p1_source_build(context,store,snapshot,operation,
                    expected_operation_sha256=operation['operation_sha256'],parent_receipt=source_receipt)
                partial=result['audit']
                receipt_ref=write_new_json(writer.root/'source_build_receipt.json',result['receipt'])
                writer.append('P1_build_receipt','action',receipt_ref)
            else:
                partial=execute_p1_qa(context,store,snapshot,operation,
                    expected_operation_sha256=operation['operation_sha256'],source_receipt=source_receipt)
        writer.write_action('action',audit=partial,logits=(),origin=origin)
        timer.check();status='COMPLETED'
    except BaseException as exc:
        import traceback
        error=dict(type=type(exc).__name__,detail=str(exc),traceback=traceback.format_exc())
        partial=getattr(exc,'audit',partial)
        writer.append('P1_action_failed','action',dict(error=error,partial_audit=partial,automatic_retry=False))
    finally:
        timer.close();adapter.deadline=old_deadline
        # A readonly validation can fail before the inner driver owns cleanup.
        # Never release a snapshot if the device context did not fence/close.
        if snapshot is not None and snapshot.snapshot_id in store._snapshots:
            if context is None or context.closed:
                try:store.end_request(snapshot)
                except BaseException as exc:
                    status='FAILED';error=dict(type=type(exc).__name__,detail=str(exc))
            else:status='FAILED';error=dict(type='UnfencedContext',detail='retain snapshot for recovery')
    report=dict(status=status,error=error,evidence_origin=origin,
        elapsed_seconds=(time.perf_counter_ns()-session_started_ns)/1e9,
        action_total_ms=(time.perf_counter_ns()-started)/1e6,automatic_resume_allowed=False,
        QA_evaluated=bool(status=='COMPLETED' and operation['kind']=='P1_fixed_source_QA_operation_v2'),
        production_reuse_commit_observed=False,formal_profile_bundle_frozen=False,
        locked_test_accessed=False,estimated_usage_cost=None)
    return writer.finalize(report)
