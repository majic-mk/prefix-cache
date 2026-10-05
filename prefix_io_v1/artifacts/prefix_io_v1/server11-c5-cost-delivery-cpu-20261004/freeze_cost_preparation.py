"""Freeze actual server sources for CPU cost-binding preparation, never GPU."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

PREP_LOCK_SHA='8a8dbf5226cefabfa5100f24aabe9bc4bf48010996d389a96c516c6288164071'
MATH_SHA='3cd840c6dd388e3dcf9023e54dd5eb172ab59777c66f731a5d1e3f9afecd99ac'
NAMES={'factory':'server11-c5-cost-binding-cpu-20261004','review':'server11-c5-cost-binding-review-cpu-20261004',
       'protocol':'server11-c5-cost-protocol-cpu-20261004','delivery':'server11-c5-cost-delivery-cpu-20261004'}

def require(ok, reason):
    if not ok:raise ValueError(reason)

def sha(data):return hashlib.sha256(data).hexdigest()

def ref(root,path):
    require(not path.is_symlink() and path.is_file(),'regular source required')
    require(path.resolve().is_relative_to(root),'source escaped project')
    data=path.read_bytes()
    require(len(data)<=10*1024**2,'bounded source or evidence')
    return dict(path=path.relative_to(root).as_posix(),bytes=len(data),sha256=sha(data))

def new_json(path,doc):
    with path.open('x',encoding='utf-8') as f:json.dump(doc,f,indent=2,sort_keys=True);f.write('\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve(strict=True);base=root/'artifacts/prefix_io_v1'
    dirs={k:base/n for k,n in NAMES.items()}
    require(Path(__file__).resolve()==dirs['delivery']/'freeze_cost_preparation.py','actual server freezer path')
    for path in dirs.values():
        require(path.is_dir() and not path.is_symlink() and path.resolve().parent==base,'fixed new source scope')
    prep=base/'server11-c5-runtime-preparation-cpu-20261004'
    candidate=base/'server11-p4-notification-candidate-v5-cpu-20261003'
    native=base/'server11-native-cost-v6-20261003'
    lock_path=prep/'PREPARATION_SOURCE_LOCK.json';lockbytes=lock_path.read_bytes();old=json.loads(lockbytes)
    require(sha(lockbytes)==PREP_LOCK_SHA and old['gpu_launch_allowed'] is False and old['gpu_qualified'] is False,'closed CPU preparation lock')
    pinned={}
    def add(path,expected=None):
        row=ref(root,path)
        require(row['bytes']>0,'nonempty source dependency')
        if expected:require(row['bytes']==expected['bytes'] and row['sha256']==expected['sha256'],'old source changed')
        require(row['path'] not in pinned or pinned[row['path']]==row,'source alias mismatch')
        pinned[row['path']]=row
    for row in old['files']:
        rel=Path(row['path']);require(not rel.is_absolute() and '..' not in rel.parts,'old source reference')
        add({'preparation':prep,'candidate':candidate}[row['scope']]/rel,row)
    add(lock_path)
    require(len(old['files'])==80,'actual full server preparation closure')
    for path in sorted(native.glob('*.py')):add(path)
    math_path=root/'third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_paired_measurement_verifier.py'
    add(math_path);require(ref(root,math_path)['sha256']==MATH_SHA and math_path.stat().st_size==48136,'unchanged original numerical estimator')
    # Reuse only historical SOURCE locators, never their costs, GPU identity or
    # qualification. Verify existing model metadata/Event/runner bytes now.
    historical_plan=native/'NATIVE_COST_PLAN.json'
    historical=json.loads(historical_plan.read_bytes());add(historical_plan)
    source_locator=root/historical['source_lock_ref']['path']
    add(source_locator,historical['source_lock_ref'])
    locator_rows=json.loads(source_locator.read_bytes())['files']
    locator={row['path']:row for row in locator_rows}
    for key in ('model_plan_ref','model_config_ref','cuda_event_source_ref'):
        row=historical[key];add(root/row['path'],row)
    runner='third_party/work/vllm-author-p4-02-cpu/vllm/v1/worker/gpu_model_runner.py'
    add(root/runner,locator[runner])
    source_counts={}
    for scope,d in dirs.items():
        count=0
        for path in sorted(d.iterdir()):
            if path.is_file() and path.suffix in ('.py','.md','.json') and not path.name.startswith(('LOCAL_','SERVER_','SESSION_')) and path.name!='SOURCE_LOCK_COST_CPU.json':
                add(path);count+=1
        require(count>0,'new source scope missing: '+scope)
        source_counts[scope]=count
    protocol_pins=json.loads((dirs['protocol']/'SOURCE_PINS.json').read_bytes())
    scopes=dict(candidate=candidate,preparation=prep,native=native,
        runtime_v4=base/'server11-p4-single-file-runtime-v4-20261003')
    for row in protocol_pins['files']:
        add(scopes[row['scope']]/row['path'],row)
    before=json.loads((dirs['delivery']/'SESSION_BEFORE_COST_PREPARATION.json').read_bytes())
    for row in before['refs']:
        require(ref(root,root/row['path'])==row,'previous experiment evidence changed')
    # Previous evidence remains a separate boundary audit. The source closure
    # includes all 80 actual old dependencies, native math and new CPU tools.
    selected=pinned
    lock=dict(schema='c5_cost_binding_cpu_source_lock_v1',gpu_launch_allowed=False,native_execution_verified=False,
        full_runtime_cost_qualified=False,on_observation_cost_measured=False,native_cost_qualified=False,
        source_only=True,source_files=len(selected),source_scopes=source_counts,preparation_lock_ref=ref(root,lock_path),
        original_estimator_ref=ref(root,math_path),files=[selected[k] for k in sorted(selected)])
    target=dirs['delivery']/'SOURCE_LOCK_COST_CPU.json';new_json(target,lock)
    receipt=dict(status='PASS_SERVER_CPU_COST_SOURCE_FREEZE',source_lock_ref=ref(root,target),source_files=len(selected),
        old_preparation_files_verified=len(old['files']),prior_references_verified=len(before['refs']),
        GPU_runs=0,formal_benchmark_runs=0,native_cost_qualified=False,full_runtime_cost_qualified=False,
        on_observation_cost_measured=False,source_scopes=source_counts,command=[sys.executable]+sys.argv)
    new_json(dirs['delivery']/'SERVER_SOURCE_FREEZE_RESULT.json',receipt)
    print(json.dumps(receipt,sort_keys=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
