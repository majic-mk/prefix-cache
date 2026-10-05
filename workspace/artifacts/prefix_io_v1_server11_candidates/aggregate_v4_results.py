"""Read closed v6/v4 evidence and append a descriptive review; no GPU work."""
import datetime, hashlib, json, os, pathlib, statistics, subprocess

ROOT=pathlib.Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
OUT=ROOT/'artifacts/prefix_io_v1/server11-p4-single-file-review-v4-20261003'
D='artifacts/prefix_io_v1/server11-p4-single-file-runtime-v4-20261003'
C='artifacts/prefix_io_v1/server11-p4-single-file-candidate-v4-20261003'
V='artifacts/prefix_io_v1/server11-native-cost-v6-20261003'

def read(path): return json.loads((ROOT/path).read_bytes())
def ref(path):
    p=ROOT/path;b=p.read_bytes()
    return dict(path=path,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

ledger=read('experiments/prefix_io_v1/gpu-budget-ledger.json')
assert ledger['active_reservation'] is None
labels=['server11-native-cost-six-window-06']+['server11-p4-single-file-'+m+'-04' for m in ['off','shadow','on']]
guards=[]
for label in labels:
    path='experiments/prefix_io_v1/runs/'+label+'/result.json'
    value=read(path)
    assert value['session_drained'] and not value['session_members_after_cleanup']
    guards.append(dict(ref=ref(path),result=value))
qual={m:read(D+'/QUALIFICATION_'+m+'.json') for m in ['off','shadow','on']}
assert all(v['output_token_ids']==qual['off']['output_token_ids'] for v in qual.values())
rows=[]
for mode,v in qual.items():
    steps=v['all_step_gpu_elapsed_ns'];base=qual['off'];prev=qual['shadow']
    rows.append(dict(mode=mode,ref=ref(D+'/QUALIFICATION_'+mode+'.json'),
        selected_gpu_elapsed_ns=v['selected_gpu_elapsed_ns'],
        whole_request_ns=v['whole_request_ns'],total_request_and_drain_ns=v['total_request_and_drain_ns'],
        change_from_off_percent=100*(v['total_request_and_drain_ns']/base['total_request_and_drain_ns']-1),
        change_from_shadow_percent=100*(v['total_request_and_drain_ns']/prev['total_request_and_drain_ns']-1),
        before_selected_mean_ns=statistics.mean(steps[:16]),after_selected_mean_ns=statistics.mean(steps[17:]),
        deferral=v['deferral'],frozen_cost_migration_pass=v['frozen_cost_migration_pass'],
        qualification_passed=v['qualification_passed'],qualification_scope=v['qualification_scope'],
        full_output_tokens=v['full_output_tokens'],full_frames=v['full_frames'],native_io=v['native_io'],
        performance_effect_verified=False))
profile=read(C+'/CPU_PROFILE_FULL_BORROW.json')
oldprofile=read('artifacts/prefix_io_v1/server11-p4-retry-cpu-profile-v2-20261003/ACTUAL_CPU_full_borrow.json')
source_checks={p:read(p) for p in [V+'/SOURCE_BEFORE_VERIFICATION.json',V+'/SOURCE_AFTER_VERIFICATION.json']+
    [D+'/SOURCE_'+m+'_'+p+'.json' for m in qual for p in ['BEFORE','AFTER']]}
assert all(not x['failed'] for x in source_checks.values())
stat=os.statvfs(ROOT)
result=dict(scope='v4_single_finite_pilot_descriptive_review_not_statistical_performance_proof',
    captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    common_fix='original Queue wait excludes retained ownership only',
    compact_observation='immutable enriched snapshot reuse with unchanged original per-attempt checks',
    strategy_policy_changed=False,identical_full_output_token_ids=True,
    p4_rows=rows,qualifications=qual,calibration=read(V+'/NATIVE_CONDITIONAL_COST_RESULT.json'),
    guards=guards,new_gpu_seconds=sum(x['result']['elapsed_seconds'] for x in guards),
    cumulative_gpu_seconds=ledger['gpu_wall_seconds'],remaining_gpu_seconds=28800-ledger['gpu_wall_seconds'],
    active_gpu_reservation=None,source_checks=source_checks,
    CPU=dict(candidate_tests=197,native_integration_tests=547,native_junit_warnings=3,
        calibration_tool_tests=57,runtime_tool_tests=44,counts_not_claimed_disjoint=True,
        old_full_borrow_median_thread_ns=oldprofile['unprofiled_thread_cpu_ns']['median'],
        new_full_borrow_median_thread_ns=profile['unprofiled_thread_cpu_ns']['median'],
        semantic_facts=profile['semantic_facts'],cpu_fixture_is_not_GPU_measurement=True),
    disk_free_bytes=stat.f_bavail*stat.f_frsize,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.free','--format=csv,noheader,nounits'],text=True),
    compute_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True),
    tracked_git_status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True),
    production_qualified=False,strategy_effect_verified=False,resource_release_credit=False,formal_SLO=None)
OUT.mkdir(exist_ok=False)
with (OUT/'V4_FINAL_EVIDENCE.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False)
print(json.dumps(dict(evidence_ref=ref((OUT/'V4_FINAL_EVIDENCE.json').relative_to(ROOT).as_posix()),
    rows=[{k:r[k] for k in ['mode','selected_gpu_elapsed_ns','total_request_and_drain_ns','change_from_off_percent','frozen_cost_migration_pass']} for r in rows],
    new_gpu_seconds=result['new_gpu_seconds'],remaining_gpu_seconds=result['remaining_gpu_seconds'],disk_free_bytes=result['disk_free_bytes'])))
