"""One preregistered layer3 diagnostic. Outer supervisor owns process deadline."""
import argparse
import json
import subprocess
import time
from pathlib import Path

from probekv.p1_boundary_diagnostic_v2 import (boundary_binding, verify_boundary_operation,
    execute_boundary_diagnostic, evaluate_boundary_gate)
from probekv.p1_build_dispatch_v2 import _require, _check_seal, bind_request_content_keys
from probekv.p1_action_runner_v2 import verify_compiled_operation, verify_build_evidence, validate_comparison_capacity
from probekv.p0_batch_v2 import _ActionWallDeadline, _native_origin
from probekv.p0_evidence_v2 import P0EvidenceWriter, write_new_json
from probekv.p0_stage_readiness_v2 import _read_ref, validate_p0_model_geometry
from probekv.p1_qualification_v2 import assess_p1_preparation
from probekv.source_comparison_v2 import runtime_binding_digest
from probekv.source_store_v2 import TargetSourceStoreV2, parse_target_catalog_v2
from probekv.source_manifest_v2 import RequestManifestRegistry
from probekv.v8_schema10_storage import file_digest
from probekv.v8_schema10_native_factory import validate_native_attachment, verified_model_asset_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--sha256', required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--execute', action='store_true')
    a=p.parse_args(); started=time.perf_counter_ns()
    m,_=_read_ref(dict(path=str(a.manifest),sha256=a.sha256),'boundary manifest'); _check_seal(m,'manifest_sha256')
    _require(m['kind']=='bounded_layer3_diagnostic_action' and m['maximum_actions']==1
        and m['boundary_binding']==boundary_binding() and m['entry_sha256']==file_digest(Path(__file__))
        and m['locked_test_accessed'] is False and m['automatic_rental_allowed'] is False,'matching one-action manifest required')
    op,_=_read_ref(m['operation'],'operation'); old,_=_read_ref(m['legacy_operation'],'legacy recipe')
    contract,_=_read_ref(m['preparation_contract'],'P0/data qualification')
    verify_compiled_operation(old,contract['frozen_inputs']); verify_boundary_operation(op,old)
    _require(op['operation_sha256']==m['operation_sha256'] and op['input_graph_sha256']==contract['expected_input_graph_sha256'],
             'operation outside registered graph')
    q=assess_p1_preparation(contract,observed_runtime_digest=runtime_binding_digest())
    _require(q['build_plan_preparation_ready'], 'raw P0/data qualification not valid')
    native,_=_read_ref(m['native_manifest'],'native attachment'); runtime=validate_native_attachment(native,allow_unmeasured=True)
    catalog,cp=_read_ref(m['initial_pool'],'pool'); _,rp=_read_ref(m['initial_registry'],'registry')
    config=parse_target_catalog_v2(catalog)['config']; source=runtime['source_provenance']; bind=m['binding']
    _require(subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==bind['code_commit']==source['code_commit']
        and bind['runtime_digest']==runtime_binding_digest() and bind['patch_sha256']==native['binding']['patch_sha256']
        and bind['model_signature']==config['model_signature']==source['model_signature']
        and bind['tokenizer_hash']==config['tokenizer_hash']==source['tokenizer_hash']
        and bind['gpu_uuid']==runtime['cost_provenance']['gpu']
        and bind['initial_pool_sha256']==file_digest(cp) and bind['initial_registry_sha256']==file_digest(rp),
        'actual execution identity differs')
    auth=m['authority']; now=time.time()
    _require(auth['phase']=='BOUNDARY_L3' and auth['instance_id']==bind['instance_id'] and auth['gpu_uuid']==bind['gpu_uuid']
        and bool(auth['approval_reference']) and auth['starts_at_unix']<=now<auth['expires_at_unix']
        and auth['expires_at_unix']-now>=900 and op['maximum_seconds']<=180,'current bounded authority required')
    receipt=None
    if m['source_receipt']:
        receipt,_=_read_ref(m['source_receipt'],'build receipt')
        verify_build_evidence(m['source_build_result']['path'],result_sha256=m['source_build_result']['sha256'],receipt=receipt)
    if op['mode'] in ('dense_qa','source_greedy_r1','source_qa'):
        gate,_=_read_ref(m['numerical_gate'],'numerical prerequisite')
        actual_gate=evaluate_boundary_gate({k:v['directory'] for k,v in gate['evidence'].items()})
        _require(gate==actual_gate and gate['status']=='PASSED', 'raw numerical prerequisite failed')
        if op['mode']=='source_qa':
            _require(gate['fixed15_boundary_QA_allowed'], 'both teacher and greedy r1 checks required')
    geometry=validate_p0_model_geometry({'jobs':[dict(action_id='action',operation='source_request',
        request=op['request'],comparison_profile=op['comparison_profile'])]},runtime)
    mc,_=_read_ref(dict(path=str(verified_model_asset_path(runtime['model_path'],'config.json')),
                         sha256=geometry['model_config_sha256']),'model config')
    capacity=validate_comparison_capacity(dict(op,kind='P1_fixed_source_QA_operation_v2'),mc,
                                         allocator_capacity_bytes=runtime['allocator_capacity_bytes'])
    _require(geometry['num_layers']==m['num_layers'],'layer count changed')
    a.output.mkdir(parents=True,exist_ok=False)
    write_new_json(a.output/'preflight.json',dict(status='INPUTS_VALIDATED_NOT_GPU_PASS',geometry=geometry,
        capacity=capacity,execute_requested=a.execute))
    if not a.execute:return 0
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    _require(uuid==bind['gpu_uuid'],'actual GPU changed')
    from probekv.v8_schema10_native_factory import create_native_measurement_backend
    store=None; writer=None; snapshot=None; ctx=None; status='FAILED'; error=None; audit=None
    timer=None
    try:
        registry=RequestManifestRegistry(persistent_root=rp.parent,**m['registry_budget'])
        store=TargetSourceStoreV2(cp.parent,registry=registry,**{k:v for k,v in config.items() if k!='purpose'})
        backend=create_native_measurement_backend(native); adapter=backend.adapters['legacy_multicheckpoint']; adapter.costs=None
        _require(time.time()+300<auth['expires_at_unix'],'insufficient post-load execution/cleanup window')
        origin=_native_origin(adapter); writer=P0EvidenceWriter(a.output/'raw',binding=bind,manifest=m)
        write_new_json(writer.root/'operation.json',op)
        timer=_ActionWallDeadline(op['maximum_seconds'],native=True); timer.start()
        adapter.deadline=time.perf_counter()+op['maximum_seconds']; adapter.reset()
        snapshot=store.begin_request(op['request']['request_id'])
        request=bind_request_content_keys(op['request'],store)
        writer.append('boundary_action_started','action',dict(operation_sha256=op['operation_sha256']))
        with adapter.open_request(request,arrival_ns=time.perf_counter_ns()) as ctx:
            audit=execute_boundary_diagnostic(ctx,store,snapshot,op,source_receipt=receipt)
        writer.write_action('action',audit=audit,logits=ctx.logit_trace if op['request'].get('capture_logits') else (),origin=origin)
        timer.check(); status='COMPLETED'
    except BaseException as exc:
        import traceback
        error=dict(type=type(exc).__name__,detail=str(exc),traceback=traceback.format_exc())
        audit=getattr(exc,'audit',audit)
        if writer:writer.append('boundary_action_failed','action',dict(error=error,partial_audit=audit))
        else:write_new_json(a.output/'failed.json',dict(status='FAILED',error=error))
    finally:
        if timer:timer.close()
        if store is not None:
            if snapshot and snapshot.snapshot_id in store._snapshots and (ctx is None or ctx.closed):store.end_request(snapshot)
            store.close()
    if writer:
        report=writer.finalize(dict(status=status,error=error,evidence_origin='real_cuda_execution',
            elapsed_seconds=(time.perf_counter_ns()-started)/1e9,production_reuse_commit_observed=False,
            full_P4_execution_allowed=False,locked_test_accessed=False,estimated_cost=None))
        print(json.dumps(report),flush=True)
    return 0 if status=='COMPLETED' else 1


if __name__=='__main__':raise SystemExit(main())
