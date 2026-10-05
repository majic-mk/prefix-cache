"""Read actual normal off evidence, unchanged source bytes and original budget. No model imports or launch."""
import argparse, hashlib, json, os, pathlib, shutil, subprocess, sys, time
ROOT_DEFAULT = pathlib.Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
D = 'artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004'
A = 'artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004'
RUN = 'experiments/prefix_io_v1/runs/server12-c5-native-normal-off01'
LOCK_SHA = 'aaa67309fab4c6dc7fe07853a31c3a99cdef727408da9fab604201bf95416d4d'
GUARD_SHA = '3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a'
QUAL_SHA = 'b28ee3a35aafb69a879350ee8534c838bab8b93713d2ad83970bf632d9b61501'
def require(value, reason):
    if not value:
        raise ValueError(reason)
def safe(root, relative):
    require(type(relative) is str and relative and not pathlib.PurePosixPath(relative).is_absolute()
        and ':' not in relative and '\\' not in relative and '..' not in pathlib.PurePosixPath(relative).parts,
        'project-relative path required')
    p=root/relative
    require(p.resolve(strict=True).is_relative_to(root) and not any(x.is_symlink() for x in (p,*p.parents)),
        'non-symlink project file required')
    require(p.is_file(), 'regular evidence file required')
    return p
def ref(root, relative):
    p=safe(root,relative);h=hashlib.sha256();total=0
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):
            h.update(chunk);total+=len(chunk)
    require(total==p.stat().st_size,'stable full-byte read')
    return dict(path=relative,bytes=total,sha256=h.hexdigest())
def read(root,relative):
    return json.loads(safe(root,relative).read_text(encoding='utf-8'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,default=ROOT_DEFAULT);args=ap.parse_args()
    root=args.root.resolve(strict=True)
    require(root==ROOT_DEFAULT,'actual server root binding')
    require(ref(root,D+'/COMMON_SOURCE_LOCK.json')['sha256']==LOCK_SHA,'frozen common source lock')
    lock=read(root,D+'/COMMON_SOURCE_LOCK.json');rows=lock['files']
    require(len(rows)==4791 and len({r['path'] for r in rows})==4791,'all unique frozen sources')
    for row in rows:require(ref(root,row['path'])==row,'source bytes changed: '+row['path'])
    require(ref(root,'experiments/prefix_io_v1/scripts/run_gpu_stage.py')['sha256']==GUARD_SHA,'original GPU guard bytes')
    require(ref(root,D+'/QUALIFICATION_off.json')['sha256']==QUAL_SHA,'actual negative qualification bytes')
    report=read(root,D+'/QUALIFICATION_off.json');guard=read(root,RUN+'/result.json')
    require(guard['exit']==guard['child_exit']==0 and guard['timed_out'] is False
        and guard['error'] is None and guard['session_drained'] is True
        and guard['session_members_after_cleanup']==[],'actual successful guard and session drain')
    require(report['native_execution_verified'] is True and report['full_output_tokens']==128
        and report['native_io']['completed_and_original_shutdown_drained'] is True,'actual full native output/lifecycle')
    require(report['frozen_cost_migration_pass'] is False and report['runtime_condition_qualified'] is False
        and report['permits_next_mode'] is None and report['selected_gpu_elapsed_ns']>report['frozen_cost_upper_ns'],
        'actual frozen cost migration failed; no next-mode permit')
    for row in report['evidence_refs'].values():require(ref(root,row['path'])==row,'actual qualification evidence bytes')
    ledger=read(root,'experiments/prefix_io_v1/gpu-budget-ledger.json')
    snapshot=read(root,D+'/SERVER12_LEDGER_SNAPSHOT.json')
    require(ledger.get('active_reservation') is None,'original budget idle')
    require(ledger['events']==snapshot['events']+[guard],'actual immutable prefix plus one normal event')
    require(ledger['gpu_wall_seconds']==snapshot['gpu_wall_seconds']+guard['elapsed_seconds'],'exact actual wall accounting')
    require(ledger['gpu_wall_seconds']<=28800,'original eight-hour budget')
    sid=guard['session_id'];members=[]
    for entry in pathlib.Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                stat=(entry/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
                if int(fields[3])==sid:members.append(int(entry.name))
            except (OSError,ValueError):pass
    require(not members,'original session really empty now')
    free=shutil.disk_usage(root).free
    require(free>=8*1024**3,'original primary free-space floor')
    current=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used','--format=csv,noheader,nounits'],
        capture_output=True,text=True,timeout=10,check=True)
    require('GPU-a67d2f38-8e1b-29c3-7e5b-375e1410f8ac' in current.stdout,'actual same card present')
    payload=dict(status='PASS_ACTUAL_NORMAL_OFF_SOURCE_GUARD_BUDGET_RESOURCE_AUDIT',
        source_lock_ref=ref(root,D+'/COMMON_SOURCE_LOCK.json'),all_frozen_source_files_full_byte_verified=4791,
        qualification_ref=ref(root,D+'/QUALIFICATION_off.json'),actual_guard_ref=ref(root,RUN+'/result.json'),
        original_gpu_guard_source_ref=ref(root,'experiments/prefix_io_v1/scripts/run_gpu_stage.py'),
        ledger_snapshot_ref=ref(root,D+'/SERVER12_LEDGER_SNAPSHOT.json'),
        current_ledger_ref=ref(root,'experiments/prefix_io_v1/gpu-budget-ledger.json'),
        actual_gpu_wall_seconds=guard['elapsed_seconds'],cumulative_gpu_wall_seconds=ledger['gpu_wall_seconds'],
        remaining_original_gpu_seconds=28800-ledger['gpu_wall_seconds'],original_gpu_seconds_limit=28800,
        guard_session_members_now=members,active_reservation=None,primary_free_bytes=free,
        nvidia_smi_readonly_current_card=current.stdout.strip(),GPU_compute_jobs_this_action=0,
        frozen_cost_migration_pass=False,runtime_condition_qualified=False,permits_next_mode=None,
        selected_gpu_elapsed_ns=report['selected_gpu_elapsed_ns'],frozen_cost_upper_ns=report['frozen_cost_upper_ns'],
        underprediction_ns=report['selected_gpu_elapsed_ns']-report['frozen_cost_upper_ns'],
        original_a_only_budget_ns=report['frozen_a_only_budget_ns'],cost_upper_exceeds_a_only_budget=True,
        native_execution_verified=True,full_output_tokens=128,original_shutdown_and_session_drained=True,
        strategy_effect_verified=False,P4_complete=False,P5_performance_experiment_started=False,
        source_mutations=0,data_deletions=0,observed_unix=time.time())
    out=root/A/'NORMAL_OFF_FINAL_STATE_AUDIT.json'
    with out.open('x',encoding='utf-8',newline='\n') as stream:json.dump(payload,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(payload,sort_keys=True))
if __name__=='__main__':main()

