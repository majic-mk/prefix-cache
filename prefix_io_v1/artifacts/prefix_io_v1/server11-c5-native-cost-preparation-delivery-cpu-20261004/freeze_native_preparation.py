"""Freeze actual C5 native code CPU sources; no GPU plan or issuance."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

NAMES = {k: 'server11-c5-native-cost-preparation' + ('' if k=='candidate' else '-'+k) + '-cpu-20261004'
         for k in ('candidate','review','protocol','delivery')}
MATH_SHA = '3cd840c6dd388e3dcf9023e54dd5eb172ab59777c66f731a5d1e3f9afecd99ac'
C5_SHA = 'a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47'
COLLECTOR_SHA = 'bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf'

def require(ok,reason):
    if not ok: raise ValueError(reason)

def verify(root,row):
    name=row['path']
    require(type(name) is str and not name.startswith('/') and '\\' not in name and ':' not in name
            and all(p not in ('','.','..') for p in name.split('/')),'safe project relative reference')
    path=root/name
    require(path.is_file() and not any(p.is_symlink() for p in (path,)+tuple(path.parents))
            and path.resolve().is_relative_to(root),'regular confined source')
    require(type(row['bytes']) is int and path.stat().st_size==row['bytes'],'source size drift')
    h=hashlib.sha256();size=0
    with path.open('rb') as stream:
        while True:
            b=stream.read(1024**2)
            if not b:break
            size+=len(b);require(size<=row['bytes'],'source grew');h.update(b)
    require(size==row['bytes'] and h.hexdigest()==row['sha256'],'source hash drift: '+name)
    return path

def reference(root,path):
    require(path.stat().st_size<=10*1024**2,'new source size cap')
    data=path.read_bytes()
    row=dict(path=path.relative_to(root).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    verify(root,row)
    return row

def put(path,doc):
    with path.open('x',encoding='utf-8') as f:json.dump(doc,f,indent=2,sort_keys=True);f.write('\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve(strict=True);base=root/'artifacts/prefix_io_v1'
    dirs={k:base/v for k,v in NAMES.items()}
    require(Path(__file__).resolve()==dirs['delivery']/'freeze_native_preparation.py','fixed actual server freezer')
    for d in dirs.values():require(d.is_dir() and not d.is_symlink() and d.resolve().parent==base,'fixed new source scope')
    before=json.loads((dirs['delivery']/'SESSION_BEFORE_NATIVE_PREPARATION.json').read_bytes())
    for row in before['refs']:verify(root,row)
    require(len(before['refs'])==797,'closed prior boundary audit')
    old=base/'server11-p4-notification-candidate-v5-cpu-20261003'
    ancestor=base/'server11-native-cost-v6-20261003'
    selected={}
    def add(path):
        row=reference(root,path)
        require(row['path'] not in selected or selected[row['path']]==row,'source alias mismatch')
        selected[row['path']]=row
    add(dirs['delivery']/'SESSION_BEFORE_NATIVE_PREPARATION.json')
    for path in sorted(old.rglob('*.py')):
        if '__pycache__' not in path.parts:add(path)
    for name in ('native_conditional_cost.py','prepare_and_verify_native_cost.py',
                 'run_native_cost_experiment.py','control_native_cost_job.py'):add(ancestor/name)
    math=root/'third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_paired_measurement_verifier.py'
    add(math);require(reference(root,math)['sha256']==MATH_SHA,'original estimator unchanged')
    for name in ('docs/prefix_io_v1/04_CODEX_EXECUTION.md','experiments/prefix_io_v1/configs/permissions.yaml'):add(root/name)
    scope_counts={}
    for scope,d in dirs.items():
        paths=[p for p in sorted(d.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
               and p.suffix in ('.py','.md')]
        if scope=='candidate':paths.append(d/'SOURCE_INHERITANCE.json')
        require(paths,'nonempty new source scope')
        for path in paths:add(path)
        scope_counts[scope]=len(paths)
    inherited=json.loads((dirs['candidate']/'SOURCE_INHERITANCE.json').read_bytes())
    changed=[]
    for row in inherited['source_parent_files']:
        parent=reference(root,old/row['path']);require(parent['bytes']==row['bytes'] and parent['sha256']==row['sha256'],'frozen C5 ancestry drift')
        current=reference(root,dirs['candidate']/'common_candidate'/row['path'])
        if current['bytes']!=row['bytes'] or current['sha256']!=row['sha256']:changed.append(row['path'])
    require(changed==['source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_single_file_receipt.py'],
            'only copied canonical receipt module may change')
    reactor=dirs['candidate']/'common_candidate/source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'
    collector=dirs['candidate']/'common_candidate/native_full_step_collector.py'
    require(reference(root,reactor)['sha256']==C5_SHA and reference(root,collector)['sha256']==COLLECTOR_SHA,'real C5 common source identity')
    document=dict(schema='c5_native_cost_preparation_cpu_source_lock_v1',source_only=True,
        gpu_uuid=None,gpu_launch_allowed=False,native_execution_verified=False,native_cost_qualified=False,
        full_runtime_cost_qualified=False,on_observation_cost_measured=False,valid_native_receipt=None,
        effective_cost_upper_ns=None,effective_step_budget_ns=None,source_files=len(selected),
        source_scopes=scope_counts,original_estimator_ref=reference(root,math),
        original_candidate_files=len(inherited['source_parent_files']),copied_source_changes=changed,
        files=[selected[k] for k in sorted(selected)])
    target=dirs['delivery']/'SOURCE_LOCK_NATIVE_PREPARATION_CPU.json';put(target,document)
    receipt=dict(status='PASS_SERVER_CPU_NATIVE_PREPARATION_SOURCE_FREEZE',source_files=len(selected),
        prior_references_verified=len(before['refs']),source_lock_ref=reference(root,target),
        baseline_ref=reference(root,dirs['delivery']/'SESSION_BEFORE_NATIVE_PREPARATION.json'),
        source_scopes=scope_counts,GPU_runs=0,formal_benchmark_runs=0,gpu_uuid=None,
        gpu_launch_allowed=False,native_cost_qualified=False,full_runtime_cost_qualified=False,
        on_observation_cost_measured=False,command=[sys.executable]+sys.argv)
    put(dirs['delivery']/'SOURCE_FREEZE_RECEIPT_CPU.json',receipt)
    print(json.dumps(receipt,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
