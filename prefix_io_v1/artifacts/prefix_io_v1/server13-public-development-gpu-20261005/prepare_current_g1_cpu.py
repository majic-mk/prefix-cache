"""Freeze the unchanged G1 native cases for the current standing permission."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

A = 'artifacts/prefix_io_v1/server13-public-development-gpu-20261005'
SCRIPT = 'experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py'
PIN = '5d9604fc07d80495a49febfcd488cb2601f74a2bef107261c8e623a4d0a1c77d'

def ref(root, name):
    path = root / name
    assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
    before = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    return dict(path=name, bytes=after.st_size, sha256=digest)

def put(root, name, value):
    with (root/name).open('x', encoding='utf-8') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',type=Path,required=True)
    args=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    root=args.project.resolve()
    ledger=root/'experiments/prefix_io_v1/gpu-budget-ledger.json'
    original_ledger=ledger.read_bytes()
    assert json.loads(original_ledger)['active_reservation'] is None
    inherited=json.loads((root/(A+'/MIGRATED_U_SOURCE_LOCK_03.json')).read_bytes())
    old={r['path']:r for r in inherited['files']}
    current=ref(root,SCRIPT)
    assert current==old[SCRIPT] and current['sha256']==PIN
    sys.path.insert(0,str(root/'experiments/prefix_io_v1/scripts'))
    spec=importlib.util.spec_from_file_location('_current_original_G1',root/SCRIPT)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    exec(compile((root/SCRIPT).read_bytes(),str(root/SCRIPT),'exec',dont_inherit=True),vars(module))
    from prepare_p4_gpu_next_day import required_source_paths, permission_fields
    permission=json.loads((root/(A+'/EFFECTIVE_STANDING_GPU_PERMISSION_01.json')).read_bytes())
    permit_path=A+'/EFFECTIVE_STANDING_GPU_PERMISSION_G1_01.yaml'
    lines=[]
    for key,value in sorted(permission.items()):
        if isinstance(value,list):
            lines.append(key+':')
            lines.extend('- '+str(item) for item in value)
        else:
            text=str(value).lower() if isinstance(value,bool) else str(value)
            lines.append(key+': '+text)
    raw='\n'.join(lines)+'\n'
    with (root/permit_path).open('x',encoding='utf-8') as f:f.write(raw)
    parsed=permission_fields(raw)
    assert all(parsed[k]==permission[k] for k in parsed)
    names=set(required_source_paths(root)) | {SCRIPT, 'experiments/prefix_io_v1/scripts/run_gpu_stage.py'}
    rows=[]
    for name in sorted(names):
        actual=ref(root,name)
        assert actual==old[name], 'original G1 source drift: '+name
        rows.append(actual)
    rows.extend([ref(root,permit_path),ref(root,A+'/prepare_current_g1_cpu.py')])
    assert not any(r['path'].startswith('models/') for r in rows)
    lock_path=A+'/CURRENT_G1_SOURCE_LOCK_01.json'
    put(root,lock_path,dict(schema='current_original_G1_required_source_bytes_v1',files=rows,
        inherited_source_lock_ref=ref(root,A+'/MIGRATED_U_SOURCE_LOCK_03.json'),
        actual_required_source_bytes_verified=True,model_weights_hashed=False,actual_GPU_operations=0))
    scope_path=A+'/CURRENT_G1_STANDING_SCOPE_01.json'
    put(root,scope_path,dict(schema_version=1,status='USER_AUTHORIZED_P4_GPU_NATIVE_QUALIFICATION',
        allow_gpu_initialization=True,allow_gpu_runs=True,allowed_modes=['off','shadow'],
        new_executor=False,allow_model_downloads=False,gpu_uuid=permission['approved_gpu_ids'][0],
        base_permissions=ref(root,permit_path),source_lock=lock_path,
        source_lock_sha256=ref(root,lock_path)['sha256'],qualification_context=module.qualification_context(),
        authorization_origin='Persistent human instruction: use GPU when available; no further per-run authorization',
        cumulative_GPU_hours=8,source_template_unchanged=True,allocator_release_claim=False,
        strategy_improvement_proved=False))
    launches=[]
    for mode in ('off','shadow'):
        name='server13-current-g1-'+mode+'01'
        opts=SimpleNamespace(mode=mode,name=name,scope_record=scope_path,source_lock=lock_path,permissions_path=permit_path)
        gates,status=module.launch_gates(root,opts)
        assert gates is not None and status=='READY_FOR_AUTHORIZED_OLD_GUARD'
        preview=module.preview(root,mode,name,lock_path,scope_path,permit_path)
        launches.append(dict(mode=mode,label=name,guard_argv=preview['guard_argv'],environment_preview=preview['environment_preview']))
    assert ledger.read_bytes()==original_ledger
    result=dict(status='PASS_CURRENT_ORIGINAL_G1_CPU_SOURCE_SCOPE_AND_LAUNCH_PREPARATION',
        source_lock_ref=ref(root,lock_path),scope_ref=ref(root,scope_path),permission_ref=ref(root,permit_path),
        source_count=len(rows),actual_source_bytes=sum(r['bytes'] for r in rows),
        whole_model_rehash_performed=False,old_CPU_suites_repeated=0,actual_GPU_operations=0,
        launches=launches,allocator_release_claim=False,full_P4_complete=False,strategy_improvement_proved=False)
    put(root,A+'/CURRENT_G1_CPU_PREPARATION_01.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
