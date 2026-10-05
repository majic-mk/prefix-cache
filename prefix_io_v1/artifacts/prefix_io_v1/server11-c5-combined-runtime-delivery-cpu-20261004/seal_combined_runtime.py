"""Seal narrow CPU composition and resource-gate evidence, never qualify GPU."""
from __future__ import annotations
import argparse
import datetime
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_combined_cpu_freezer',HERE/'freeze_combined_runtime.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
require=F.require;verify=F.verify;reference=F.reference;read=F.read;put=F.put;sha=F.sha

def pass_tests(doc,kind,locksha):
    require(type(doc.get('tests')) is int and doc['tests']>0 and doc.get('passed')==doc['tests'],kind+' tests')
    require(doc.get('status','').startswith('PASS') and doc.get('failed')==doc.get('errors')==doc.get('skipped')==0,kind+' pass')
    require(doc.get('location')=='server_cpu' and doc.get('source_before')==doc.get('source_after')
            and doc.get('source_lock_sha256')==locksha,kind+' actual source lock')
    require(all(doc.get(k) is False for k in F.FLAGS) and doc.get('gpu_uuid') is None,kind+' CPU-only flags')
    require(doc.get('actual_gpu_runs',doc.get('gpu_runs',doc.get('GPU_runs')))==0,kind+' zero GPU')
    require(doc.get('forbidden_modules_imported',doc.get('forbidden_imports',doc.get('torch_vllm_imports')))==[],kind+' no GPU modules')
    for key in ('valid_native_receipt','effective_cost_upper_ns','effective_step_budget_ns'):
        require(doc.get(key) is None,kind+' no issued cost: '+key)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve(strict=True);base=root/'artifacts/prefix_io_v1'
    dirs={k:base/v for k,v in F.NAMES.items()}
    require(HERE==dirs['delivery'],'fixed actual sealer path')
    for d in dirs.values():require(d.is_dir() and not d.is_symlink() and d.resolve().parent==base,'exact four new scopes')
    beforepath=HERE/'SESSION_BEFORE_COMBINED_PREPARATION.json';before=read(beforepath)
    require(len(before['refs'])==952,'unchanged prior boundary size')
    for row in before['refs']:verify(root,row)
    lockpath=HERE/'SOURCE_LOCK_COMBINED_RUNTIME_CPU.json';lockbytes=lockpath.read_bytes();lock=read(lockpath)
    frozen=read(HERE/'SOURCE_FREEZE_RECEIPT_CPU.json')
    require(reference(root,lockpath)==frozen['source_lock_ref'] and reference(root,beforepath)==frozen['baseline_ref'],'anchored source and baseline')
    require(lock['schema']=='c5_combined_runtime_cpu_source_lock_v1' and lock['gpu_uuid'] is None
            and all(lock[k] is False for k in F.FLAGS),'CPU-only source closure')
    for row in lock['files']:verify(root,row)
    factory=read(dirs['candidate']/'SERVER_CPU_01/CPU_RESULT.json');pass_tests(factory,'factory',sha(lockbytes))
    require(factory['source_count']==len(lock['files']) and factory['actual_on_installed'] is False
            and factory['actual_native_bridge_attachment'] is False and factory['full_entry_cpu_cost_measured'] is False
            and factory['actual_model_processes_started']==0,'no artificial positive native/on entry')
    review=read(dirs['review']/'SERVER_REVIEW_01/TEST_RESULT.json');pass_tests(review,'review',sha(lockbytes))
    resource=read(dirs['resource']/'SERVER_RESOURCE_TESTS_01/CPU_RESOURCE_RESULT.json');pass_tests(resource,'resource tests',sha(lockbytes))
    require(resource['formal_benchmark_started'] is False and resource['measurement_iterations']==0
            and resource['old_paired_qualification_changed'] is False,'no repeated or changed performance experiment')
    preflight=read(dirs['resource']/'SERVER_PREFLIGHT_01/RESOURCE_PREFLIGHT.json')
    require(preflight['status']=='RESOURCE_LIMITED' and preflight['resource_ready'] is False
            and preflight['cpu_resource_prerequisite_met'] is False and preflight['resource_readiness_only'] is True,
            'actual known half-core resource gate')
    require(preflight['source_before']==preflight['source_after'] and preflight['source_lock_sha256']==sha(lockbytes)
            and preflight['source_count']==len(lock['files']) and preflight['origin']=='actual_readonly_resource_preflight'
            and preflight['location']=='server_cpu','actual readonly preflight source closure')
    require(preflight['formal_benchmark_started'] is False and preflight['formal_benchmark_runs']==0
            and preflight['measurement_iterations']==0 and preflight['actual_gpu_runs']==0
            and preflight['configs_created']==preflight['jobs_created']==0
            and preflight['forbidden_imports']==[] and all(preflight[k] is False for k in F.FLAGS),'no measurement/authority promotion')
    require(preflight['gpu_ledger_sha256']==before['ledger_sha256'] and preflight['gpu_ledger_unchanged'] is True
            and preflight['active_reservation'] is None and preflight['old_protocol_sha256']==F.PROTOCOL_SHA,
            'actual resource old evidence and budget')
    for tag in ('SERVER_RUN_BLOCK','SERVER_CONTROL_BLOCK','SERVER_VERIFY_BLOCK','SERVER_CONTRACT_BLOCK'):
        logged=read(dirs['candidate']/(tag+'_RESULT.json'));stdout=read(dirs['candidate']/(tag+'_STDOUT.log'))
        require(logged['exit']==2 and (dirs['candidate']/(tag+'_STDERR.log')).read_bytes()==b'',tag+' blocked exit')
        require(stdout['status'].startswith('GPU_BLOCKED') and
                (stdout.get('gpu_started') is False or stdout.get('actual_gpu_runs')==0),tag+' no GPU')
        require(stdout.get('gpu_qualified') is not True and stdout.get('gpu_launch_allowed') is not True
                and stdout.get('native_cost_receipt') is None and stdout.get('valid_native_receipt') is None,tag+' no receipt')
    ledgerpath=root/'experiments/prefix_io_v1/gpu-budget-ledger.json';ledgerbytes=ledgerpath.read_bytes();ledger=json.loads(ledgerbytes)
    require(sha(ledgerbytes)==before['ledger_sha256'] and ledger['gpu_wall_seconds']==before['gpu_wall_seconds']
            and ledger.get('active_reservation') is None,'GPU budget unchanged')
    nodes=sorted(str(p) for p in Path('/dev').glob('nvidia*'))
    cpu=Path('/sys/fs/cgroup/cpu.max').read_text().strip();memory=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    git=subprocess.run(['git','status','--porcelain','--untracked-files=no'],cwd=root,capture_output=True,text=True,check=True).stdout
    require(nodes==before['GPU_nodes'] and cpu==before['cpu_max'] and memory==before['memory_max']
            and git==before['tracked_git_status'],'environment/source changed during CPU stage')
    require(shutil.disk_usage(root).free>=8*1024**3,'8 GiB free floor')
    available={}
    for d in dirs.values():
        for path in sorted(d.rglob('*')):
            require(not path.is_symlink(),'archive symlink refused')
            if path.is_file():
                require(path.stat().st_size<=10*1024**2,'new evidence cap')
                b=path.read_bytes();available.setdefault((len(b),sha(b)),path.relative_to(base).as_posix())
    snapshots=HERE/'SOURCE_DEPENDENCY_SNAPSHOTS';snapshots.mkdir(exist_ok=False);mapping=[]
    for row in lock['files']:
        path=verify(root,row);key=(row['bytes'],row['sha256'])
        if key not in available:
            target=snapshots/(row['sha256']+'.bin');b=path.read_bytes()
            with target.open('xb') as f:f.write(b)
            available[key]=target.relative_to(base).as_posix()
        mapping.append(dict(row,snapshot_path=available[key]))
    put(HERE/'SOURCE_SNAPSHOT_MAP.json',dict(source_rows=len(mapping),GPU_runs=0,formal_benchmark_runs=0,
        unique_external_snapshots=len(list(snapshots.iterdir())),files=mapping))
    audit=dict(status='PASS_CPU_COMBINED_RUNTIME_BOUNDARIES_RESOURCE_LIMITED',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),factory_tests=factory['tests'],
        independent_tests=review['tests'],resource_tests=resource['tests'],source_lock_sha256=sha(lockbytes),
        source_rows=len(lock['files']),prior_references_verified=len(before['refs']),prior_evidence_unchanged=True,
        resource_preflight_status=preflight['status'],full_entry_cpu_cost_measured=False,actual_on_installed=False,
        actual_native_bridge_attachment=False,GPU_runs=0,formal_benchmark_runs=0,model_loads=0,gpu_uuid=None,
        gpu_launch_allowed=False,native_execution_verified=False,native_cost_qualified=False,
        full_runtime_cost_qualified=False,on_observation_cost_measured=False,
        valid_native_receipt=None,effective_cost_upper_ns=None,effective_step_budget_ns=None,
        gpu_ledger_sha256=sha(ledgerbytes),gpu_wall_seconds=ledger['gpu_wall_seconds'],active_reservation=None,
        GPU_nodes=nodes,cpu_max=cpu,memory_max=memory,tracked_git_status=git,disk_free_bytes=shutil.disk_usage(root).free,
        next_order=['CPU resource prerequisite','real GPU common6/native receipt','off shadow on actual lifecycle/full costs','P4 fair comparison'],
        command=[sys.executable]+sys.argv)
    put(HERE/'SERVER_FINAL_AUDIT.json',audit)
    content={}
    for d in dirs.values():
        for path in sorted(d.rglob('*')):
            require(not path.is_symlink(),'archive symlink refused')
            if path.is_dir():continue
            require(path.is_file() and path.resolve().is_relative_to(d) and path.stat().st_size<=10*1024**2,'bounded evidence')
            content[path.relative_to(base).as_posix()]=path.read_bytes()
    require(len(content)<=300 and sum(map(len,content.values()))<=30*1024**2,'bounded four-directory archive')
    manifest=dict(scope='FOUR_NEW_CPU_COMBINED_RUNTIME_DIRECTORIES_ONLY',data_file_count=len(content),
        raw_bytes=sum(map(len,content.values())),GPU_runs=0,formal_benchmark_runs=0,
        files=[dict(path=k,bytes=len(b),sha256=sha(b)) for k,b in sorted(content.items())])
    md=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode();archive=HERE/'C5_COMBINED_RUNTIME_CPU_EVIDENCE.tar.gz'
    with archive.open('xb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w|',format=tarfile.USTAR_FORMAT) as t:
        for name,b in [('ARCHIVE_CONTENTS_MANIFEST.json',md)]+sorted(content.items()):
            info=tarfile.TarInfo(name);info.size=len(b);info.mode=0o600;info.mtime=0;t.addfile(info,io.BytesIO(b))
    require(archive.stat().st_size<=20*1024**2,'compressed archive cap')
    for name,b in content.items():require((base/name).read_bytes()==b,'evidence changed during seal')
    for row in before['refs']:verify(root,row)
    for row in lock['files']:verify(root,row)
    require(ledgerpath.read_bytes()==ledgerbytes,'budget changed during seal')
    receipt=dict(status='PASS_SEALED_CPU_COMBINED_RUNTIME_RESOURCE_LIMITED',
        archive=dict(file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive.read_bytes())),
        manifest=dict(file='ARCHIVE_CONTENTS_MANIFEST.json',bytes=len(md),sha256=sha(md)),
        data_files=len(content),raw_bytes=manifest['raw_bytes'],source_lock_sha256=sha(lockbytes),
        source_snapshots=len(mapping),unique_external_snapshots=len(list(snapshots.iterdir())),
        prior_references_verified=len(before['refs']),GPU_runs=0,formal_benchmark_runs=0,gpu_ledger_sha256=sha(ledgerbytes),
        resource_preflight_status='RESOURCE_LIMITED',full_entry_cpu_cost_measured=False)
    put(HERE/'SERVER_BACKUP_RECEIPT.json',receipt);print(json.dumps(receipt,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
