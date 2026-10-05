"""Audit CPU-only native code preparation and seal four bounded new directories."""
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
import importlib.util

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_native_cpu_freezer',HERE/'freeze_native_preparation.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
require=F.require;verify=F.verify;reference=F.reference;put=F.put
FLAGS=('gpu_launch_allowed','native_execution_verified','native_cost_qualified',
       'full_runtime_cost_qualified','on_observation_cost_measured')
def read(p):return json.loads(p.read_bytes())
def sha(b):return hashlib.sha256(b).hexdigest()

def pass_tests(doc,kind):
    require(type(doc.get('tests')) is int and doc['tests']>0,kind+' nonempty tests')
    require(doc.get('status','').startswith('PASS'),kind+' result status')
    failures=doc.get('failed',doc.get('failures'))
    require(failures==doc.get('errors')==doc.get('skipped')==0,kind+' failure/error/skip')
    require(all(doc.get(k) is False for k in FLAGS),kind+' synthetic qualification boundary')
    imports=doc.get('forbidden_modules_imported',doc.get('forbidden_imports',doc.get('torch_vllm_imports')))
    require(imports==[],kind+' native/model import')
    gpu=doc.get('actual_gpu_runs',doc.get('gpu_runs',doc.get('GPU_runs')))
    require(type(gpu) is int and gpu==0,kind+' GPU execution boundary')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve(strict=True);base=root/'artifacts/prefix_io_v1'
    dirs={k:base/v for k,v in F.NAMES.items()}
    require(HERE==dirs['delivery'],'fixed actual server sealer')
    for d in dirs.values():require(d.is_dir() and not d.is_symlink() and d.resolve().parent==base,'four exact CPU scopes')
    before=read(HERE/'SESSION_BEFORE_NATIVE_PREPARATION.json')
    require(len(before['refs'])==797,'complete closed prior boundary audit')
    for row in before['refs']:verify(root,row)
    lockpath=HERE/'SOURCE_LOCK_NATIVE_PREPARATION_CPU.json';lockbytes=lockpath.read_bytes();lock=read(lockpath)
    frozen=read(HERE/'SOURCE_FREEZE_RECEIPT_CPU.json')
    require(reference(root,lockpath)==frozen['source_lock_ref'],'separately frozen source lock')
    require(reference(root,HERE/'SESSION_BEFORE_NATIVE_PREPARATION.json')==frozen['baseline_ref'],
            'baseline bytes unchanged since source freeze')
    require(lock['schema']=='c5_native_cost_preparation_cpu_source_lock_v1' and lock['gpu_uuid'] is None
            and all(lock.get(k) is False for k in FLAGS),'CPU source closure boundary')
    for row in lock['files']:verify(root,row)
    factory_outer=read(dirs['candidate']/'SERVER_CPU_01/CPU_RESULT.json')
    factory=factory_outer.get('result',factory_outer)
    pass_tests(factory,'factory')
    require(factory_outer.get('location')=='server_cpu','actual server factory location')
    require(factory_outer.get('source_lock_sha256')==sha(lockbytes),'factory source lock')
    require(factory_outer.get('source_before')==factory_outer.get('source_after')
            and factory_outer.get('source_count')==len(lock['files']), 'factory complete source before/after')
    require(factory.get('valid_native_receipt') is None and factory.get('effective_cost_upper_ns') is None
            and factory.get('effective_step_budget_ns') is None,'no synthetic issued cost')
    review=read(dirs['review']/'SERVER_REVIEW_01/TEST_RESULT.json');pass_tests(review,'review')
    require(review['location']=='server_cpu' and review['source_before']==review['source_after']
            and review['source_lock_sha256']==sha(lockbytes),'review actual complete source closure')
    protocol=read(dirs['protocol']/'SERVER_PROTOCOL_01/CPU_PROTOCOL_RESULT.json');pass_tests(protocol,'protocol')
    require(protocol['source_before']==protocol['source_after'] and protocol['source_lock_sha256']==sha(lockbytes)
            and protocol['actual_cuda_event'] is False and protocol['actual_model'] is False
            and protocol['actual_collector_install_api'] is True,'protocol actual CPU API and source closure')
    for name in ('SERVER_CALIBRATOR_BLOCK','SERVER_CONTROLLER_BLOCK','SERVER_SERIALIZER_BLOCK','SERVER_ISSUER_BLOCK'):
        logged=read(dirs['candidate']/(name+'_RESULT.json'))
        stdout=read(dirs['candidate']/(name+'_STDOUT.log'))
        require(logged['exit']==2 and (dirs['candidate']/(name+'_STDERR.log')).read_bytes()==b'',name+' exit')
        require(stdout['status']=='GPU_BLOCKED_C5_NATIVE_COST_CODE_PREPARATION_ONLY'
                and stdout.get('gpu_launch_allowed') is False and stdout.get('valid_native_receipt') is None
                and stdout.get('effective_cost_upper_ns') is None,name+' blocked result')
    ledgerpath=root/'experiments/prefix_io_v1/gpu-budget-ledger.json';ledgerbytes=ledgerpath.read_bytes();ledger=json.loads(ledgerbytes)
    require(sha(ledgerbytes)==before['ledger_sha256'] and ledger['gpu_wall_seconds']==before['gpu_wall_seconds']
            and ledger.get('active_reservation') is None,'GPU budget unchanged')
    nodes=[str(p) for p in Path('/dev').glob('nvidia*')]
    cpu=Path('/sys/fs/cgroup/cpu.max').read_text().strip();memory=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    git=subprocess.run(['git','status','--porcelain','--untracked-files=no'],cwd=root,capture_output=True,text=True,check=True).stdout
    require(nodes==before['GPU_nodes'] and cpu==before['cpu_max'] and memory==before['memory_max']
            and git==before['tracked_git_status'],'environment/tracked source changed')
    require(shutil.disk_usage(root).free>=8*1024**3,'8 GiB free floor')
    # Reuse same-hash sources already in the four new folders. Snapshot only
    # external ancestors; all closure rows remain independently verifiable.
    available={}
    for d in dirs.values():
        for p in sorted(d.rglob('*')):
            require(not p.is_symlink(),'archive symlink refused')
            if p.is_file():
                require(p.stat().st_size<=10*1024**2,'new evidence file cap')
                b=p.read_bytes();available.setdefault((len(b),sha(b)),p.relative_to(base).as_posix())
    snapshotdir=HERE/'SOURCE_DEPENDENCY_SNAPSHOTS';snapshotdir.mkdir(exist_ok=False)
    mapping=[]
    for row in lock['files']:
        path=verify(root,row);key=(row['bytes'],row['sha256'])
        if key not in available:
            target=snapshotdir/(row['sha256']+'.bin');data=path.read_bytes()
            with target.open('xb') as f:f.write(data)
            available[key]=target.relative_to(base).as_posix()
        mapping.append(dict(row,snapshot_path=available[key]))
    put(HERE/'SOURCE_SNAPSHOT_MAP.json',dict(source_rows=len(mapping),GPU_runs=0,
        formal_benchmark_runs=0,unique_external_snapshots=len(list(snapshotdir.iterdir())),files=mapping))
    audit=dict(status='PASS_CPU_NATIVE_CODE_PREPARATION_BOUNDARIES',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        factory_tests=factory['tests'],independent_tests=review['tests'],protocol_tests=protocol['tests'],
        source_lock_sha256=sha(lockbytes),source_rows=len(lock['files']),prior_references_verified=len(before['refs']),
        prior_evidence_unchanged=True,GPU_runs=0,formal_benchmark_runs=0,model_loads=0,gpu_uuid=None,
        gpu_launch_allowed=False,native_execution_verified=False,native_cost_qualified=False,
        full_runtime_cost_qualified=False,on_observation_cost_measured=False,
        valid_native_receipt=None,effective_cost_upper_ns=None,effective_step_budget_ns=None,
        gpu_ledger_sha256=sha(ledgerbytes),gpu_wall_seconds=ledger['gpu_wall_seconds'],active_reservation=None,
        GPU_nodes=nodes,cpu_max=cpu,memory_max=memory,tracked_git_status=git,disk_free_bytes=shutil.disk_usage(root).free,
        copied_source_changes=lock['copied_source_changes'],command=[sys.executable]+sys.argv)
    put(HERE/'SERVER_FINAL_AUDIT.json',audit)
    content={}
    for d in dirs.values():
        for p in sorted(d.rglob('*')):
            require(not p.is_symlink(),'archive symlink refused')
            if p.is_dir():continue
            require(p.is_file() and p.resolve().is_relative_to(d) and p.stat().st_size<=10*1024**2,'bounded evidence')
            content[p.relative_to(base).as_posix()]=p.read_bytes()
    require(len(content)<=300 and sum(map(len,content.values()))<=30*1024**2,'bounded archive scope')
    manifest=dict(scope='FOUR_NEW_CPU_NATIVE_CODE_PREPARATION_DIRECTORIES_ONLY',data_file_count=len(content),
        raw_bytes=sum(map(len,content.values())),GPU_runs=0,formal_benchmark_runs=0,
        files=[dict(path=k,bytes=len(b),sha256=sha(b)) for k,b in sorted(content.items())])
    md=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode();archive=HERE/'C5_NATIVE_PREPARATION_CPU_EVIDENCE.tar.gz'
    with archive.open('xb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w|',format=tarfile.USTAR_FORMAT) as t:
        for name,b in [('ARCHIVE_CONTENTS_MANIFEST.json',md)]+sorted(content.items()):
            info=tarfile.TarInfo(name);info.size=len(b);info.mode=0o600;info.mtime=0;t.addfile(info,io.BytesIO(b))
    require(archive.stat().st_size<=20*1024**2,'compressed archive cap')
    for name,b in content.items():require((base/name).read_bytes()==b,'evidence changed during seal')
    for row in before['refs']:verify(root,row)
    for row in lock['files']:verify(root,row)
    require(ledgerpath.read_bytes()==ledgerbytes,'ledger changed during seal')
    receipt=dict(status='PASS_SEALED_CPU_NATIVE_CODE_PREPARATION',
        archive=dict(file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive.read_bytes())),
        manifest=dict(file='ARCHIVE_CONTENTS_MANIFEST.json',bytes=len(md),sha256=sha(md)),
        data_files=len(content),raw_bytes=manifest['raw_bytes'],source_lock_sha256=sha(lockbytes),
        source_snapshots=len(mapping),unique_external_snapshots=len(list(snapshotdir.iterdir())),
        prior_references_verified=len(before['refs']),GPU_runs=0,formal_benchmark_runs=0,gpu_ledger_sha256=sha(ledgerbytes))
    put(HERE/'SERVER_BACKUP_RECEIPT.json',receipt);print(json.dumps(receipt,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
