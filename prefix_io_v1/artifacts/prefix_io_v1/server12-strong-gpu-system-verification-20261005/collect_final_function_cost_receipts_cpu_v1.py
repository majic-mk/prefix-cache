import argparse,hashlib,json,subprocess,shutil,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);a=p.parse_args()
root=a.project_root.resolve(strict=True);d=root/'artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005';prep=root/'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004'
ledger=root/'experiments/prefix_io_v1/gpu-budget-ledger.json';lb=ledger.read_bytes();ld=json.loads(lb)
assert ld['active_reservation'] is None and hashlib.sha256(lb).hexdigest()=='774847ccdb7ec4c1785c19a87e718b2a91e15fba44248eb949dab559e08fdd6b'
def invoke(argv):
 r=subprocess.run(argv,cwd=root,capture_output=True,text=True,timeout=15);assert r.returncode==0;return {'argv':argv,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
ps=invoke(['ps','-eo','pid,ppid,sid,stat,args']);sid_members=[s for s in ps['stdout'].splitlines()[1:] if s.split()[2]=='7931'];assert not sid_members
gpu=invoke(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']);assert not gpu['stdout'].strip()
git=invoke(['git','status','--porcelain','--untracked-files=no']);assert not git['stdout'].strip()
tags=['CPU_GPU_RAW_DEVICE_MINOR_TEST_01','CPU_GPU_RAW_V3_RELATIVE_CONFIG_01','CPU_GPU_RAW_V4_CACHE_SEMANTICS_01','CPU_GPU_RAW_V5_FAILURE_DIAGNOSTICS_01','CPU_GPU_RAW_V2_INITIAL_NO_FORWARD_02','CPU_GPU_RAW_V6_FINAL_BINDING_01','CPU_GPU_RAW_V7_REAL_CONTRACT_01','CPU_GPU_NEXT_RUNTIME_COLLECTOR_V2_01','CPU_FORMAL_TRACE_INPUT_CONTRACT_01']
tests=[]
for tag in tags:
 r=json.loads((prep/(tag+'_RESULT.json')).read_bytes());assert r['exit']==0 and r['GPU_runs']==0
 tail=(prep/(tag+'_STDERR.log')).read_text('utf-8').splitlines()[-5:];assert tail[-1]=='OK'
 tests.append({'tag':tag,'result':r,'test_log_tail':tail,'actual_command':json.loads((prep/(tag+'_COMMAND.json')).read_bytes())})
guards=[]
for label in ['server12-strong-u-qual-off01']+['server12-strong-exact-cal%02d'%i for i in range(1,6)]:
 pp=root/'experiments/prefix_io_v1/runs'/label/'result.json';b=pp.read_bytes();r=json.loads(b)
 guards.append({'label':label,'ref':{'path':pp.relative_to(root).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()},'result':r})
doc={'schema':'final_actual_function_cost_delivery_CPU_receipt_v1','status':'PASS_IDLE_RESOURCE_AND_PRESERVED_CPU_TEST_RECEIPTS','UTC_timestamp':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'actual_GPU_operations_this_CPU_action':0,'GPU_ledger':{'used_seconds':ld['gpu_wall_seconds'],'maximum_seconds':28800,'remaining_seconds':28800-ld['gpu_wall_seconds'],'active_reservation':None,'sha256':hashlib.sha256(lb).hexdigest()},'original_guard_session_7931_members':sid_members,'actual_compute_PID_query':gpu,'actual_git_tracked_status':git,'PRIMARY_free_bytes':shutil.disk_usage(root).free,'preserved_CPU_test_receipts':tests,'real_GPU_guard_receipts':guards,'actual_cal05_guard_command_record':json.loads((d/'GPU_STRONG_RAW_20261005_05_COMMAND.json').read_bytes()),'actual_CPU_issuer_command_record':json.loads((d/'CPU_STRONG_ACTUAL_EXACT_COST_ISSUER_05_COMMAND.json').read_bytes()),'source_proof_V10':json.loads((prep/'PRERENT_SOURCE_PROOF_V10.json').read_bytes()),'formal_SLO_qualified':False,'strategy_effect_verified':False,'next_allowed_current_scope':'CPU documentation and immutable evidence delivery only'}
assert ledger.read_bytes()==lb
with (d/'FINAL_ACTUAL_CPU_DELIVERY_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(doc,f,indent=2,sort_keys=True)
print(json.dumps({'status':doc['status'],'GPU_ledger':doc['GPU_ledger'],'PRIMARY_free_bytes':doc['PRIMARY_free_bytes'],'CPU_test_suites':len(tests),'actual_GPU_jobs_recorded':len(guards),'compute_pids_empty':True}))
