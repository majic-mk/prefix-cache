"""Freeze a CPU-only combined normal-entry source closure; no GPU authority."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

NAMES={k:'server11-c5-combined-runtime'+('' if k=='candidate' else '-'+k)+'-cpu-20261004'
       for k in ('candidate','review','resource','delivery')}
PARENT_SHA='47a445ba5fc1fae5f9fba5f32871f5af771e66083753654122d7e96bc6f64e2a'
STAGE_A_SHA='8a8dbf5226cefabfa5100f24aabe9bc4bf48010996d389a96c516c6288164071'
PROTOCOL_SHA='6c00cf687f4903988a363e18a178ccca413076ced81920271bf9c962abde1218'
FLAGS=('gpu_launch_allowed','native_execution_verified','native_cost_qualified',
       'full_runtime_cost_qualified','on_observation_cost_measured')

def require(ok,reason):
    if not ok:raise ValueError(reason)
def sha(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(path.read_bytes())
def verify(root,row):
    name=row['path'];require(type(name) is str and not name.startswith('/') and '\\' not in name and ':' not in name
                            and all(p not in ('','.','..') for p in name.split('/')),'safe source path')
    path=root/name
    require(path.is_file() and not any(p.is_symlink() for p in (path,)+tuple(path.parents))
            and path.resolve().is_relative_to(root),'regular contained source')
    require(type(row['bytes']) is int and path.stat().st_size==row['bytes'],'source size drift')
    h=hashlib.sha256();count=0
    with path.open('rb') as stream:
        while True:
            b=stream.read(1024**2)
            if not b:break
            count+=len(b);require(count<=row['bytes'],'source grew');h.update(b)
    require(count==row['bytes'] and h.hexdigest()==row['sha256'],'source SHA drift: '+name)
    return path
def reference(root,path):
    require(path.stat().st_size<=10*1024**2,'new source file cap')
    b=path.read_bytes();row=dict(path=path.relative_to(root).as_posix(),bytes=len(b),sha256=sha(b));verify(root,row)
    return row
def put(path,doc):
    with path.open('x',encoding='utf-8') as f:json.dump(doc,f,indent=2,sort_keys=True);f.write('\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve(strict=True);base=root/'artifacts/prefix_io_v1'
    dirs={k:base/v for k,v in NAMES.items()};delivery=dirs['delivery']
    require(Path(__file__).resolve()==delivery/'freeze_combined_runtime.py','fixed actual server freezer')
    for d in dirs.values():require(d.is_dir() and not d.is_symlink() and d.resolve().parent==base,'fixed four CPU scopes')
    beforepath=delivery/'SESSION_BEFORE_COMBINED_PREPARATION.json';before=read(beforepath)
    require(len(before['refs'])==952,'closed prior boundary count')
    for row in before['refs']:verify(root,row)
    selected={}
    def add(path,expected=None):
        row=reference(root,path)
        if expected:require(row==expected,'ancestor reference drift')
        require(row['path'] not in selected or selected[row['path']]==row,'source alias mismatch')
        selected[row['path']]=row
    add(beforepath)
    native_delivery=base/'server11-c5-native-cost-preparation-delivery-cpu-20261004'
    parentpath=native_delivery/'SOURCE_LOCK_NATIVE_PREPARATION_CPU.json';parent=read(parentpath)
    require(sha(parentpath.read_bytes())==PARENT_SHA and len(parent['files'])==159 and parent['gpu_uuid'] is None
            and all(parent[k] is False for k in FLAGS),'closed CPU native source ancestry')
    for row in parent['files']:add(root/row['path'],row)
    add(parentpath)
    stage_a=base/'server11-c5-runtime-preparation-cpu-20261004'
    old_c5=base/'server11-p4-notification-candidate-v5-cpu-20261003'
    oldpath=stage_a/'PREPARATION_SOURCE_LOCK.json';old=read(oldpath)
    require(sha(oldpath.read_bytes())==STAGE_A_SHA and len(old['files'])==80 and old['gpu_launch_allowed'] is False,'closed normal-entry ancestry')
    for row in old['files']:
        original={'preparation':stage_a,'candidate':old_c5}[row['scope']]/row['path']
        actual=reference(root,original)
        require(actual['bytes']==row['bytes'] and actual['sha256']==row['sha256'],'frozen runtime parent drift')
        add(original)
    add(oldpath)
    # Source-only locators of the original normal startup, lifecycle and owner
    # script; no model weights, GPU qualification or authority is inherited.
    for name in ('artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/run_g2_normal_model_lifecycle.py',
                 'artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002/g3_calibration_runtime_metrics_v2.py',
                 'experiments/prefix_io_v1/scripts/acquire_native_aio_costs.py'):add(root/name)
    protocol=base/'server11-p4-notification-benchmark-v5-cpu-20261003/CPU_COMPARISON_PROTOCOL.json'
    add(protocol);require(sha(protocol.read_bytes())==PROTOCOL_SHA,'original CPU protocol unchanged')
    source_counts={}
    for scope,d in dirs.items():
        paths=[p for p in sorted(d.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
               and p.suffix in ('.py','.md')]
        if scope=='candidate':paths.append(d/'COMBINED_SOURCE_INHERITANCE.json')
        require(paths,'new source scope required')
        for path in paths:add(path)
        source_counts[scope]=len(paths)
    adapter=dirs['candidate']/'notification_runtime_adapter.py'
    require(adapter.read_bytes()==(stage_a/adapter.name).read_bytes(),'unaltered normal-entry Queue observer')
    document=dict(schema='c5_combined_runtime_cpu_source_lock_v1',source_only=True,source_files=len(selected),
        source_scopes=source_counts,gpu_uuid=None,gpu_launch_allowed=False,native_execution_verified=False,
        native_cost_qualified=False,full_runtime_cost_qualified=False,on_observation_cost_measured=False,
        full_entry_cpu_cost_measured=False,actual_on_installed=False,actual_native_bridge_attachment=False,
        valid_native_receipt=None,effective_cost_upper_ns=None,effective_step_budget_ns=None,
        parent_native_lock_ref=reference(root,parentpath),parent_runtime_lock_ref=reference(root,oldpath),
        original_protocol_ref=reference(root,protocol),baseline_ref=reference(root,beforepath),
        files=[selected[k] for k in sorted(selected)])
    target=delivery/'SOURCE_LOCK_COMBINED_RUNTIME_CPU.json';put(target,document)
    receipt=dict(status='PASS_SERVER_CPU_COMBINED_RUNTIME_SOURCE_FREEZE',source_lock_ref=reference(root,target),
        baseline_ref=reference(root,beforepath),source_files=len(selected),source_scopes=source_counts,
        prior_references_verified=len(before['refs']),GPU_runs=0,formal_benchmark_runs=0,gpu_uuid=None,
        gpu_launch_allowed=False,full_entry_cpu_cost_measured=False,actual_on_installed=False,
        native_cost_qualified=False,full_runtime_cost_qualified=False,on_observation_cost_measured=False,
        command=[sys.executable]+sys.argv)
    put(delivery/'SOURCE_FREEZE_RECEIPT_CPU.json',receipt);print(json.dumps(receipt,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
