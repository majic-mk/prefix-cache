"""CPU-only bounded official public QA input acquisition; no model/backend import."""
import pathlib,json,os,time,hashlib,urllib.request,shutil,subprocess
ROOT=pathlib.Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
D=ROOT/'artifacts/prefix_io_v1/server12-natural-source-audit-20261005'
CONTRACT={"schema":"prospective_official_squad_fixed_articles_v1","origin":"prospective_before_source_download_and_any_new_gpu_outcome","source_repository":"rajpurkar/SQuAD-explorer","source_commit":"eee5fdbf62f8613a7812b03419e6b29617b74fd1","source_relative_path":"dataset/dev-v1.1.json","source_url":"https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/eee5fdbf62f8613a7812b03419e6b29617b74fd1/dataset/dev-v1.1.json","dataset_version":"1.1","article_indices":[0,1,2],"paragraph_indices":[0,0,0],"partitions":["calibration","development","evaluation"],"retain_all_selected_qas":True,"selection_by_length_or_outcome":False,"prompt_rule":"original_context_plus_two_LF_plus_original_question","inject_answers":False,"max_source_bytes":16777216,"max_partition_requests":32,"max_total_requests":96,"synthetic_fixture":False}
LEDGER_SHA='774847ccdb7ec4c1785c19a87e718b2a91e15fba44248eb949dab559e08fdd6b'
def ref(p):
 b=p.read_bytes()
 return {'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(name,value):
 with (D/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 ledger=ROOT/'experiments/prefix_io_v1/gpu-budget-ledger.json'
 assert ref(ledger)['sha256']==LEDGER_SHA and json.loads(ledger.read_bytes())['active_reservation'] is None
 assert shutil.disk_usage(ROOT).free>=8589934592+CONTRACT['max_source_bytes']
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT)
 # The exact deterministic selection is fixed before any input bytes are requested.
 assert json.loads((D/'PROSPECTIVE_SQUAD_SELECTION.json').read_bytes())==CONTRACT
 declaration_ns=time.monotonic_ns()
 write('CPU_SOURCE_ACQUISITION_INTENT_02.json',{'schema':'bounded_public_data_cpu_acquisition_v1','scope':'official_qa_dataset_replay_not_production_traffic','selection_ref':ref(D/'PROSPECTIVE_SQUAD_SELECTION.json'),'declared_monotonic_ns':declaration_ns,'gpu_operation':False,'gpu_budget_ledger_sha256':LEDGER_SHA,'max_payload_bytes':CONTRACT['max_source_bytes'],'source_only_read_permission':True,'service_SLO':None,'development_deadline':None,'authority_for_model_download':False})
 url=CONTRACT['source_url']; start=time.monotonic_ns()
 req=urllib.request.Request(url,headers={'User-Agent':'prefix-io-cpu-input-preparation/1'})
 size=0; sha=hashlib.sha256()
 with urllib.request.urlopen(req,timeout=30) as response:
  assert response.status==200
  final_url=response.geturl()
  assert final_url==url
  headers={k:v for k,v in response.headers.items() if k.lower() in ('content-type','content-length','etag','last-modified')}
  with (D/'OFFICIAL_SQUAD_DEV_V1_1_COMPLETE_02.json').open('xb') as out:
   while True:
    if time.monotonic_ns()-start>300_000_000_000:raise TimeoutError('bounded 300-second whole acquisition window')
    b=response.read(min(16384,CONTRACT['max_source_bytes']-size+1))
    if not b:break
    size+=len(b)
    if size>CONTRACT['max_source_bytes']:raise ValueError('public dataset exceeds fixed source bound; partial retained as failed evidence')
    out.write(b);sha.update(b)
 end=time.monotonic_ns()
 assert start>declaration_ns
 assert ref(ledger)['sha256']==LEDGER_SHA
 row=ref(D/'OFFICIAL_SQUAD_DEV_V1_1_COMPLETE_02.json')
 assert row['bytes']==size and row['sha256']==sha.hexdigest()
 write('CPU_SOURCE_ACQUISITION_RESULT_02.json',{'schema':'actual_bounded_public_data_cpu_acquisition_v1','status':'PASS_ACTUAL_PUBLIC_BYTES_ONLY','source_repository':CONTRACT['source_repository'],'source_commit':CONTRACT['source_commit'],'url':url,'final_url':final_url,'http_status':200,'headers':headers,'source_ref':row,'selection_ref':ref(D/'PROSPECTIVE_SQUAD_SELECTION.json'),'intent_ref':ref(D/'CPU_SOURCE_ACQUISITION_INTENT_02.json'),'declared_monotonic_ns':declaration_ns,'fetch_started_monotonic_ns':start,'fetch_finished_monotonic_ns':end,'payload_bytes':size,'cpu_only':True,'gpu_operations':0,'budget_before_sha256':LEDGER_SHA,'budget_after_sha256':ref(ledger)['sha256'],'natural_dataset_read':True,'recorded_natural_arrivals':False,'formal_gpu_eligible':False,'source_license':'SQuAD dataset CC BY-SA 4.0; repository code MIT is separate','attribution_url':'https://rajpurkar.github.io/SQuAD-explorer/','primary_free_bytes':shutil.disk_usage(ROOT).free})
 print(json.dumps({'status':'PASS_ACTUAL_PUBLIC_BYTES_ONLY','source_ref':row,'gpu_operations':0},ensure_ascii=False))
if __name__=='__main__':main()
