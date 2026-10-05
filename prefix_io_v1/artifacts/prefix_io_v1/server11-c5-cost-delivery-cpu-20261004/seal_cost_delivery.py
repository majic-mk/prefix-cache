"""Read-only boundary audit and bounded CPU evidence snapshots, no GPU calls."""
from __future__ import annotations
import argparse
import datetime
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

NAMES=('server11-c5-cost-binding-cpu-20261004','server11-c5-cost-binding-review-cpu-20261004',
       'server11-c5-cost-protocol-cpu-20261004','server11-c5-cost-delivery-cpu-20261004')
SHA115='da3c8dded24d44e34553b161021d23a5a2f7daf2257b2a06db2f5e0d33e7dd6c'
SHA116='c268a380dd66eb6b27ceeff0147a5908d1a2b1c9bd8bff2f73602e6092ef76b8'

def require(ok,reason):
    if not ok:raise ValueError(reason)

def sha(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(path.read_bytes())
def put(path,doc):
    with path.open('x',encoding='utf-8') as f:json.dump(doc,f,indent=2,sort_keys=True);f.write('\n')

def actual(root,row):
    name=row['path'];require(not name.startswith('/') and '\\' not in name and ':' not in name and all(p not in ('','.','..') for p in name.split('/')),'safe evidence path')
    p=root/name;require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root),'confined evidence file')
    require(p.stat().st_size==row['bytes'],'evidence size drift')
    h=hashlib.sha256();count=0
    with p.open('rb') as stream:
        while True:
            b=stream.read(1024**2)
            if not b:break
            count+=len(b);require(count<=row['bytes'],'evidence grew');h.update(b)
    require(count==row['bytes'] and h.hexdigest()==row['sha256'],'evidence SHA drift: '+name)
    return p

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve(strict=True);base=root/'artifacts/prefix_io_v1'
    factory,review,protocol,delivery=[base/name for name in NAMES]
    for d in (factory,review,protocol,delivery):require(d.is_dir() and not d.is_symlink() and d.resolve().parent==base,'fixed archive scope')
    before=read(delivery/'SESSION_BEFORE_COST_PREPARATION.json')
    for row in before['refs']:actual(root,row)
    lock115_path=delivery/'SOURCE_LOCK_COST_CPU.json';lock116_path=delivery/'SOURCE_LOCK_COST_PROTOCOL_V2.json'
    require(sha(lock115_path.read_bytes())==SHA115 and sha(lock116_path.read_bytes())==SHA116,'CPU source lock drift')
    lock115=read(lock115_path);lock116=read(lock116_path)
    require(len(lock115['files'])==115 and len(lock116['files'])==116,'actual source closure counts')
    old={x['path']:x for x in lock115['files']};new={x['path']:x for x in lock116['files']}
    require(all(new.get(k)==v for k,v in old.items()) and set(new)-set(old)=={(protocol/'run_cpu_protocol_v2.py').relative_to(root).as_posix()},'protocol-only extension')
    for row in new.values():actual(root,row)
    cpu=read(factory/'SERVER_CPU_01/CPU_RESULT.json');c=cpu['result']
    require(cpu['exit']==0 and cpu['location']=='server_cpu' and cpu['source_lock_sha256']==SHA115 and cpu['source_files_verified_before']==cpu['source_files_verified_after']==115,'server factory source result')
    require(c['status']=='PASS_CPU_COST_BINDING_CONTRACT_ONLY' and c['tests']==39 and c['failures']==c['errors']==c['skipped']==0 and c['actual_gpu_runs']==0 and c['forbidden_modules_imported']==[],'server factory checks')
    require(c['native_cost_qualified'] is False and c['effective_cost_upper_ns'] is None and c['effective_step_budget_ns'] is None and c['valid_native_receipt'] is None,'synthetic receipt promotion')
    rev=read(review/'SERVER_REVIEW_01/TEST_RESULT.json')
    require(rev['status']=='PASS' and rev['location']=='server_cpu' and rev['tests']==rev['passed']==20 and rev['failed']==rev['errors']==rev['skipped']==0,'server independent checks')
    require(rev['source_before']==rev['source_after'] and rev['source_lock_sha256']==SHA115 and rev['gpu_runs']==0 and rev['forbidden_modules_imported']==[],'independent source/native drift')
    proto=read(protocol/'SERVER_PROTOCOL_V2_01/CPU_PROTOCOL_RESULT.json')
    require(proto['status']=='PASS_CPU_PROTOCOL_ONLY' and proto['tests']==18 and proto['failed']==proto['errors']==proto['skipped']==0,'protocol checks')
    require(proto['source_before']==proto['source_after'] and proto['source_lock_sha256']==SHA116 and proto['GPU_runs']==proto['formal_benchmark']==proto['jobs_created']==0 and proto['cost_values_emitted'] is False and proto['receipt_issued'] is False,'protocol qualification promotion')
    bound=read(delivery/'ACTUAL_SOURCE_CPU_BINDING_RESULT.json')
    require(bound['status']=='PASS_C5_COST_BINDING_CPU_PREPARATION_ONLY' and bound['source_files_verified']==115 and bound['source_binding_prepared'] is True and bound['source_metadata_location']=='actual_server_project_files','actual source positive binding')
    require(bound['native_cost_qualified'] is False and bound['full_runtime_cost_qualified'] is False and bound['on_observation_cost_measured'] is False and bound['effective_cost_upper_ns'] is None and bound['effective_step_budget_ns'] is None and bound['valid_native_receipt'] is None and bound['timing_samples_created']==0 and bound['actual_gpu_runs']==0,'source binding is not calibrated costs')
    guard=read(factory/'SERVER_GPU_ENTRY_BLOCK_RESULT.json');g=read(factory/'SERVER_GPU_ENTRY_BLOCK_STDOUT.log')
    require(guard['exit']==2 and g['status']=='GPU_BLOCKED_C5_COST_BINDING_PREPARATION_ONLY' and g['gpu_started'] is False and g['gpu_launch_allowed'] is False,'GPU entry guard')
    ledger_path=root/'experiments/prefix_io_v1/gpu-budget-ledger.json';ledger_bytes=ledger_path.read_bytes();ledger=json.loads(ledger_bytes)
    require(sha(ledger_bytes)==before['ledger_sha256'] and ledger['gpu_wall_seconds']==before['gpu_wall_seconds'] and ledger.get('active_reservation') is None,'GPU budget drift')
    nodes=[str(x) for x in Path('/dev').glob('nvidia*')];cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip();memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    git=subprocess.run(['git','status','--porcelain','--untracked-files=no'],cwd=root,capture_output=True,text=True,check=True).stdout
    require(nodes==before['GPU_nodes'] and cpu_max==before['cpu_max'] and memory_max==before['memory_max'] and git==before['tracked_git_status'],'environment or tracked source drift')
    require(shutil.disk_usage(root).free>=8*1024**3,'8 GiB free space floor')
    snapshots=delivery/'SOURCE_DEPENDENCY_SNAPSHOTS';snapshots.mkdir(exist_ok=False);snapshot_rows=[]
    for row in lock116['files']:
        p=actual(root,row);require(row['bytes']<=10*1024**2,'bounded CPU source snapshot')
        b=p.read_bytes();require(len(b)==row['bytes'] and sha(b)==row['sha256'],'snapshot drift')
        destination=snapshots/(row['sha256']+'.bin')
        if destination.exists():require(destination.read_bytes()==b,'source hash collision')
        else:
            with destination.open('xb') as f:f.write(b)
        snapshot_rows.append(dict(row,snapshot_path=destination.relative_to(base).as_posix()))
    put(delivery/'SOURCE_SNAPSHOT_MAP.json',dict(source_rows=116,source_binding_rows=115,GPU_runs=0,formal_benchmark_runs=0,files=snapshot_rows))
    audit=dict(status='PASS_CPU_COST_PREPARATION_BOUNDARIES',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        factory_tests=39,independent_tests=20,protocol_tests=18,source_binding_lock_sha256=SHA115,protocol_source_lock_sha256=SHA116,
        source_binding_rows=115,source_protocol_rows=116,prior_references_verified=len(before['refs']),prior_evidence_unchanged=True,
        actual_source_binding_prepared=True,native_cost_qualified=False,on_observation_cost_measured=False,full_runtime_cost_qualified=False,
        effective_cost_upper_ns=None,effective_step_budget_ns=None,valid_native_receipt=None,GPU_runs=0,formal_benchmark_runs=0,
        gpu_ledger_sha256=sha(ledger_bytes),gpu_wall_seconds=ledger['gpu_wall_seconds'],active_reservation=None,
        GPU_nodes=nodes,cpu_max=cpu_max,memory_max=memory_max,tracked_git_status=git,disk_free_bytes=shutil.disk_usage(root).free,
        failure_records_preserved=['SERVER_SOURCE_FREEZE','SERVER_CPU_COST_PROTOCOL'],command=[sys.executable]+sys.argv)
    put(delivery/'SERVER_FINAL_AUDIT.json',audit)
    content={}
    for d in (factory,review,protocol,delivery):
        for p in sorted(d.rglob('*')):
            require(not p.is_symlink(),'archive symlink forbidden')
            if p.is_dir():continue
            require(p.is_file() and p.resolve().is_relative_to(d) and p.stat().st_size<=10*1024**2,'bounded regular archive member')
            content[p.relative_to(base).as_posix()]=p.read_bytes()
    require(len(content)<=300 and sum(map(len,content.values()))<=30*1024**2,'bounded new CPU archive')
    manifest=dict(scope='FOUR_NEW_CPU_COST_PREPARATION_DIRECTORIES_ONLY',data_file_count=len(content),raw_bytes=sum(map(len,content.values())),
        GPU_runs=0,formal_benchmark_runs=0,files=[dict(path=k,bytes=len(b),sha256=sha(b)) for k,b in sorted(content.items())])
    md=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode();archive=delivery/'C5_COST_CPU_EVIDENCE.tar.gz'
    with archive.open('xb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w|',format=tarfile.USTAR_FORMAT) as t:
        for name,b in [('ARCHIVE_CONTENTS_MANIFEST.json',md)]+sorted(content.items()):
            info=tarfile.TarInfo(name);info.size=len(b);info.mode=0o600;info.mtime=0;t.addfile(info,io.BytesIO(b))
    require(archive.stat().st_size<=20*1024**2,'compressed archive cap')
    for name,b in content.items():require((base/name).read_bytes()==b,'evidence changed during seal')
    for row in before['refs']:actual(root,row)
    for row in lock116['files']:actual(root,row)
    require(ledger_path.read_bytes()==ledger_bytes,'GPU ledger changed during seal')
    receipt=dict(status='PASS_SEALED_CPU_COST_PREPARATION',archive=dict(file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive.read_bytes())),
        manifest=dict(file='ARCHIVE_CONTENTS_MANIFEST.json',bytes=len(md),sha256=sha(md)),data_files=len(content),raw_bytes=manifest['raw_bytes'],
        GPU_runs=0,formal_benchmark_runs=0,source_binding_lock_sha256=SHA115,protocol_source_lock_sha256=SHA116,
        source_snapshots=116,prior_references_verified=len(before['refs']),gpu_ledger_sha256=sha(ledger_bytes))
    put(delivery/'SERVER_BACKUP_RECEIPT.json',receipt);print(json.dumps(receipt,sort_keys=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
