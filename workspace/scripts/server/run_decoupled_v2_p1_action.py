"""One frozen P1 build/QA action. Defaults to validation, never rental.

--execute runs a fixed built-in worker under an outer wall-clock watchdog.
No automatic next action, retry, alternative Source or batch expansion.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
from probekv.p0_stage_readiness_v2 import _read_ref,validate_p0_model_geometry
from probekv.p0_evidence_v2 import write_new_json
from probekv.p1_action_runner_v2 import (validate_p1_action,run_p1_native_action,
    verify_compiled_operation,verify_build_evidence,validate_comparison_capacity)
from probekv.p1_qualification_v2 import assess_p1_preparation
from probekv.source_comparison_v2 import runtime_binding_digest
from probekv.v8_schema10_native_factory import validate_native_attachment
from probekv.v8_schema10_storage import file_digest
from probekv.source_store_v2 import parse_target_catalog_v2,TargetSourceStoreV2
from probekv.source_manifest_v2 import RequestManifestRegistry


def worker_command(entry, arguments, import_paths):
    """Keep audited parent import priority despite editable-install .pth hooks."""
    bootstrap=('import sys,runpy;sys.path[:]=%r;sys.argv=%r;runpy.run_path(%r,run_name="__main__")'
               % (list(import_paths),[str(entry)]+list(arguments),str(entry)))
    return [sys.executable,'-B','-c',bootstrap]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--instance-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();started=time.perf_counter_ns()
    m,_=_read_ref(dict(path=str(a.manifest),sha256=a.manifest_sha256),'P1 manifest')
    operation,_=_read_ref(m['operation'],'operation')
    native,_=_read_ref(m['native_manifest'],'native manifest')
    qualification,_=_read_ref(m['preparation_contract'],'preparation contract')
    runtime=validate_native_attachment(native,allow_unmeasured=True)
    qreport=assess_p1_preparation(qualification,observed_runtime_digest=runtime_binding_digest())
    if not qreport['build_plan_preparation_ready']:raise ValueError('raw P0/data preparation blocked: '+str(qreport['blockers']))
    if operation['input_graph_sha256']!=qualification['expected_input_graph_sha256']:
        raise ValueError('operation not in qualified input graph')
    verify_compiled_operation(operation,qualification['frozen_inputs'])
    source=runtime['source_provenance']
    if any(qualification['runtime_identity'][k]!=v for k,v in dict(
        code_commit=source['code_commit'],patch_sha256=native['binding']['patch_sha256'],
        model_signature=source['model_signature'],tokenizer_hash=source['tokenizer_hash']).items()):
        raise ValueError('P0/model/patch identity differs')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if commit!=source['code_commit']:raise ValueError('checkout commit differs')
    catalog,cp=_read_ref(m['initial_pool'],'initial pool')
    _,rp=_read_ref(m['initial_registry'],'initial registry')
    config=parse_target_catalog_v2(catalog)['config']
    actual=dict(code_commit=commit,runtime_digest=runtime_binding_digest(),patch_sha256=native['binding']['patch_sha256'],
        model_signature=source['model_signature'],tokenizer_hash=source['tokenizer_hash'],instance_id=a.instance_id,
        gpu_uuid=runtime['cost_provenance']['gpu'],initial_pool_sha256=file_digest(cp),initial_registry_sha256=file_digest(rp))
    if config['model_signature']!=actual['model_signature'] or config['tokenizer_hash']!=actual['tokenizer_hash']:
        raise ValueError('actual store differs from model')
    receipt=None
    if m.get('source_receipt') is not None:
        receipt,_=_read_ref(m['source_receipt'],'Source build receipt')
        proof=m['source_build_result']
        verify_build_evidence(proof['path'],result_sha256=proof['sha256'],receipt=receipt)
    validation=validate_p1_action(m,operation,actual_binding=actual,now_unix=time.time())
    geometry=validate_p0_model_geometry({'jobs':[dict(action_id='action',operation='source_request',
        request=operation['request'],comparison_profile=operation['comparison_profile'])]},runtime)
    from probekv.v8_schema10_native_factory import verified_model_asset_path
    config_path=verified_model_asset_path(runtime['model_path'],'config.json')
    model_config,_=_read_ref(dict(path=str(config_path),sha256=geometry['model_config_sha256']),'model config')
    capacity=validate_comparison_capacity(operation,model_config,
        allocator_capacity_bytes=runtime['allocator_capacity_bytes'])
    if a.worker and not a.execute:raise ValueError('worker requires execute')
    if a.execute and not a.worker:
        # Includes child preflight/model startup, action, evidence and cleanup.
        seconds=min(validation['available_seconds'],validation['required_seconds'])
        a.output.mkdir(parents=True,exist_ok=False)
        write_new_json(a.output/'supervisor.json',dict(validation,hard_timeout_seconds=seconds,
            manifest_sha256=a.manifest_sha256,GPU_execution_started=False,automatic_retry=False))
        cmd=worker_command(Path(__file__).resolve(),['--manifest',str(a.manifest.resolve()),
            '--manifest-sha256',a.manifest_sha256,'--instance-id',a.instance_id,
            '--output',str(a.output/'worker'),'--execute','--worker'],sys.path)
        with (a.output/'worker.log').open('xb') as log:
            try:return subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=seconds).returncode
            except subprocess.TimeoutExpired:
                write_new_json(a.output/'watchdog_failed.json',dict(status='FAILED',reason='hard_deadline',
                    partial_state_must_be_reverified=True,automatic_resume_allowed=False,paper_evidence=False))
                return 2
    a.output.mkdir(parents=True,exist_ok=False)
    write_new_json(a.output/'preflight.json',dict(validation,model_geometry=geometry,comparison_capacity=capacity,actual_binding=actual,
                                               execute_requested=a.execute,new_entry_gpu_qualified=False))
    if not a.execute:
        print(json.dumps(dict(status='INPUTS_VALIDATED_GPU_NOT_RUN',output=str(a.output))));return 0
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()
    if len(uuids)!=1 or uuids[0].strip()!=actual['gpu_uuid']:raise ValueError('actual GPU differs')
    store=None
    try:
        registry=RequestManifestRegistry(persistent_root=rp.parent,**m['registry_budget'])
        store=TargetSourceStoreV2(cp.parent,registry=registry,**{k:v for k,v in config.items() if k!='purpose'})
        from probekv.v8_schema10_native_factory import create_native_measurement_backend
        backend=create_native_measurement_backend(native)
        adapter=backend.adapters['legacy_multicheckpoint'];adapter.costs=None
        report=run_p1_native_action(adapter,store,m,operation,actual_binding=actual,output=a.output/'action',
                                    session_started_ns=started,source_receipt=receipt)
        print(json.dumps(report));return 0 if report['status']=='COMPLETED' else 1
    except BaseException as exc:
        write_new_json(a.output/'failed.json',dict(status='FAILED',type=type(exc).__name__,detail=str(exc),
            automatic_resume_allowed=False,paper_evidence=False));raise
    finally:
        if store is not None:store.close()


if __name__=='__main__':raise SystemExit(main())
