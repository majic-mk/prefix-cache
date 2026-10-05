"""Bind real server SOURCE metadata to a CPU-only contract; emit no costs."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

def require(ok,reason):
    if not ok:raise ValueError(reason)

def ref(root,path):
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),'actual project file')
    b=path.read_bytes();return dict(path=path.relative_to(root).as_posix(),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def put(path,doc):
    with path.open('x',encoding='utf-8') as f:json.dump(doc,f,indent=2,sort_keys=True);f.write('\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve(strict=True);base=root/'artifacts/prefix_io_v1'
    delivery=base/'server11-c5-cost-delivery-cpu-20261004'
    factory=base/'server11-c5-cost-binding-cpu-20261004'
    prep=base/'server11-c5-runtime-preparation-cpu-20261004'
    candidate=base/'server11-p4-notification-candidate-v5-cpu-20261003'
    native=base/'server11-native-cost-v6-20261003'
    lockpath=delivery/'SOURCE_LOCK_COST_CPU.json';lockbytes=lockpath.read_bytes();lock=json.loads(lockbytes)
    require(lock['schema']=='c5_cost_binding_cpu_source_lock_v1' and lock['gpu_launch_allowed'] is False and lock['native_execution_verified'] is False,'CPU-only outer lock')
    pins={x['path']:x for x in lock['files']}
    def frozen(path):
        row=ref(root,path);require(pins.get(row['path'])==row,'missing/drifted actual source: '+row['path']);return row
    for row in lock['files']:require(ref(root,root/row['path'])==row,'source closure drift')
    frozen(Path(__file__).resolve())
    source=frozen(factory/'c5_cost_binding_preparation.py')
    spec=importlib.util.spec_from_file_location('_actual_source_cost_preparation',root/source['path'])
    C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
    historical_path=native/'NATIVE_COST_PLAN.json';frozen(historical_path)
    historical=json.loads(historical_path.read_bytes())
    locator_path=root/historical['source_lock_ref']['path'];frozen(locator_path)
    locator=json.loads(locator_path.read_bytes());locator_map={x['path']:x for x in locator['files']}
    required_keys=('scope','qualification_rule','stage','units','operations','transfer_quantum_bytes',
        'cached_prompt_tokens','prompt_tokens','output_tokens','measured_offset','warmup_offsets',
        'external_warmup_output_tokens','external_flush_output_tokens','process_design','entries')
    contract={key:historical[key] for key in required_keys}
    actual=dict(original_estimator_ref=frozen(root/C.ORIGINAL_RELATIVE),
        verifier_source_ref=frozen(native/'native_conditional_cost.py'),
        serializer_source_ref=frozen(native/'prepare_and_verify_native_cost.py'))
    for key in ('model_plan_ref','model_config_ref','cuda_event_source_ref'):
        actual[key]=frozen(root/historical[key]['path']);require(actual[key]==historical[key],'historical SOURCE locator drift')
    actual['model_runner_ref']=frozen(root/C.RUNNER_SUFFIX)
    require(actual['model_runner_ref']==locator_map[C.RUNNER_SUFFIX],'actual runner locator drift')
    reactor=frozen(candidate/C.NATIVE_SUFFIX);collector=frozen(candidate/'native_full_step_collector.py')
    prep_lock=frozen(prep/'PREPARATION_SOURCE_LOCK.json')
    plan=dict(scope=C.PLAN_SCOPE,schema_version=1,origin='synthetic_cpu_contract',preparation_source_lock_ref=prep_lock,
        source_lock_ref=ref(root,lockpath),source_lock_files=len(pins),candidate_relative=candidate.relative_to(root).as_posix(),
        preparation_relative=prep.relative_to(root).as_posix(),reactor_source_ref=reactor,collector_source_ref=collector,
        wrapper_source_ref=frozen(prep/'run_p4_single_file_experiment.py'),
        common_owner_parameters=dict(max_accepted_parents=8,bridge_is_none=True),formula_contract=contract,**actual)
    planpath=delivery/'ACTUAL_SOURCE_CPU_PLAN.json';put(planpath,plan);planref=ref(root,planpath)
    common={row['path']:row for row in [reactor]+list(actual.values())}
    overlays={}
    old=json.loads((prep/'PREPARATION_SOURCE_LOCK.json').read_bytes())
    for row in old['files']:
        owner={'preparation':prep,'candidate':candidate}[row['scope']]
        current=frozen(owner/row['path'])
        if current['path']!=reactor['path']:overlays[current['path']]=current
    overlays[source['path']]=source;overlays[prep_lock['path']]=prep_lock
    binding=dict(scope=C.PREPARATION_SCOPE,schema_version=1,origin='synthetic_cpu_contract',plan_ref=planref,
        factory_source_ref=source,runtime_common_refs=[common[k] for k in sorted(common)],
        runtime_overlay_refs=[overlays[k] for k in sorted(overlays)])
    bindingpath=delivery/'ACTUAL_SOURCE_CPU_BINDING.json';put(bindingpath,binding);bindingref=ref(root,bindingpath)
    result=C.prepare_cost_binding(root,binding_ref=bindingref,expected_binding_ref=bindingref,
        expected_plan_ref=planref,expected_source_lock_ref=ref(root,lockpath),expected_factory_ref=source).document()
    require(result['native_cost_qualified'] is False and result['valid_native_receipt'] is None and result['effective_cost_upper_ns'] is None and result['effective_step_budget_ns'] is None,'CPU contract cannot issue costs')
    require(lockpath.read_bytes()==lockbytes,'source lock drift during metadata binding')
    for row in lock['files']:require(ref(root,root/row['path'])==row,'final actual source drift')
    result['source_metadata_location']='actual_server_project_files'
    result['timing_samples_created']=0
    result['physical_or_model_execution']=False
    result['historical_reuse']='SOURCE locators and unchanged workload constants only; no cost, UUID, qualification or authorization'
    result['command']=[sys.executable]+sys.argv
    result['forbidden_modules_imported']=[name for name in sys.modules if name=='torch' or name.startswith('torch.') or name=='vllm' or name.startswith('vllm.')]
    require(result['forbidden_modules_imported']==[],'model imports forbidden')
    put(delivery/'ACTUAL_SOURCE_CPU_BINDING_RESULT.json',result)
    print(json.dumps(result,sort_keys=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
