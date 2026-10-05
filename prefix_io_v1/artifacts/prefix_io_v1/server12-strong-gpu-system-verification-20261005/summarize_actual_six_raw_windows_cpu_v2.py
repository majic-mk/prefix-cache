import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);a=p.parse_args()
root=a.project_root.resolve(strict=True)
d=root/'artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005'
p=d/'raw05/ACTUAL_RAW_PARENT_CLOSED.json';b=p.read_bytes()
assert hashlib.sha256(b).hexdigest()=='38c281bfe60bd699c4b89a94999fbb2e8e8ad8ba88788c44cb958ca52986cd1c'
raw=json.loads(b);windows=raw['cells'][0]['windows'];assert len(windows)==6
assert [(w['pair_index'],w['condition'],w['split']) for w in windows]==[(0,'A','calibration'),(0,'B','calibration'),(1,'B','calibration'),(1,'A','calibration'),(2,'A','holdout'),(2,'B','holdout')]
rows=[]
for i,w in enumerate(windows):
 c=w['capture'];f=c['frames'];ev=c['event_witnesses'];out=w['output_token_ids'];fo=w['frontend']['output']
 assert len(f)==len(ev)==len(out)==128 and fo['output_token_ids']==out
 assert c['valid'] and not c['failures'] and not c['pending_event_pairs'] and c['open_event_pair'] is False
 ords=[r['native_step_ordinal'] for r in f]
 assert ords==list(range(ords[0],ords[0]+128)) and [r['native_step_ordinal'] for r in ev]==ords
 assert all(type(e['gpu_elapsed_ns']) is int and e['gpu_elapsed_ns']>0 for e in ev)
 initial=f[0]['prepared'];sel=f[16]['prepared'];tail=w['native_tail_assertions']
 assert initial['context_length']==496 and initial['prefill_tokens']==16
 assert sel['context_length']==527 and sel['batch']==1 and sel['active_decode']==1 and sel['prefill_tokens']==0
 assert fo['num_cached_tokens']==512 and tail=={'handler_shutdown':True,'worker_alive':False,'aio_worker_alive':False,'reactor_closed':True}
 assert w['strict_completed_window_validation']['status']=='PASS_NATIVE_COMPLETE_WINDOW_CAPTURE_IO'
 rp=root/w['child_receipt_ref']['path'];rb=rp.read_bytes()
 assert len(rb)==w['child_receipt_ref']['bytes'] and hashlib.sha256(rb).hexdigest()==w['child_receipt_ref']['sha256']
 r=json.loads(rb);assert r['original_engine_shutdown_returned'] is True and r['fresh_original_process'] is True
 child_path=root/'experiments/prefix_io_v1/runs/server12-strong-exact-cal05/details/windows'/('%02d'%i)/'CHILD_RESULT.json'
 child_bytes=child_path.read_bytes();child=json.loads(child_bytes);assert child['exit']==0
 rows.append({'window_index':i,'pair_index':w['pair_index'],'condition':w['condition'],'split':w['split'],'pid':w['subprocess_pid'],'frames':len(f),'CUDA_Event_witnesses':len(ev),'output_tokens':len(out),'native_ordinals':[ords[0],ords[-1]],'front_end_cached_prompt_tokens':fo['num_cached_tokens'],'initial_pre_context':initial['context_length'],'initial_prefill':initial['prefill_tokens'],'measured_context':sel['context_length'],'strict_validation':w['strict_completed_window_validation']['status'],'original_tail':tail,'child_receipt_ref':w['child_receipt_ref'],'actual_child_exit':child['exit'],'child_exit_ref':{'path':child_path.relative_to(root).as_posix(),'bytes':len(child_bytes),'sha256':hashlib.sha256(child_bytes).hexdigest()},'original_engine_shutdown_returned':r['original_engine_shutdown_returned'],'output_token_sha256':hashlib.sha256(json.dumps(out,separators=(',',':')).encode()).hexdigest()})
assert len({w['subprocess_pid'] for w in windows})==6
for n in range(3):
 pair=[w for w in windows if w['pair_index']==n]
 assert pair[0]['output_token_ids']==pair[1]['output_token_ids']
ledger=root/'experiments/prefix_io_v1/gpu-budget-ledger.json';lb=ledger.read_bytes();assert json.loads(lb)['active_reservation'] is None
doc={'schema':'actual_six_GPU_window_raw_counts_cpu_readback_v1','status':'PASS_ACTUAL_SIX_WINDOWS_DIRECT_RAW_READBACK','raw_ref':{'path':p.relative_to(root).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()},'windows':rows,'measured_frames':sum(r['frames'] for r in rows),'CUDA_Event_witnesses':sum(r['CUDA_Event_witnesses'] for r in rows),'measured_output_tokens':sum(r['output_tokens'] for r in rows),'pair_output_equality':True,'independent_holdout_used_to_fit':False,'original_issuer_not_reissued_this_action':True,'actual_GPU_operations_this_CPU_action':0,'formal_SLO_qualified':False,'strategy_effect_verified':False,'idle_ledger_sha256':hashlib.sha256(lb).hexdigest()}
target=d/'CAL05_ACTUAL_SIX_WINDOW_RAW_SUMMARY.json'
with target.open('x',encoding='utf-8') as stream:json.dump(doc,stream,indent=2,sort_keys=True)
assert ledger.read_bytes()==lb
print(json.dumps(doc))
