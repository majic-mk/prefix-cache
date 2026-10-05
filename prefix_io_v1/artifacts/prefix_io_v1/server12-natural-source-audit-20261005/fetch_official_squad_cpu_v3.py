"""CPU public source transport retry, preserving prospective selection and all failed partials."""
import pathlib,json,os,time,hashlib,urllib.request,shutil,base64,subprocess
ROOT=pathlib.Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
D=ROOT/'artifacts/prefix_io_v1/server12-natural-source-audit-20261005'
COMMIT='eee5fdbf62f8613a7812b03419e6b29617b74fd1'
BLOB='e9a3f913ad1468ebe105b891334ca7b0bc0e2510'
SIZE=4854279
TREE_SHA='2f083b7297de30f10c2372a6017cca4fdacacc55c3ba3b259c155697722dcab2'
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
 assert shutil.disk_usage(ROOT).free>=8589934592+24*1024**2
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT)
 c=json.loads((D/'PROSPECTIVE_SQUAD_SELECTION.json').read_bytes())
 assert c['source_commit']==COMMIT and c['source_relative_path']=='dataset/dev-v1.1.json'
 tree=D/'OFFICIAL_COMMIT_TREE_API_01.json'; assert ref(tree)['sha256']==TREE_SHA
 t=json.loads(tree.read_bytes());assert t['truncated'] is False
 e=[x for x in t['tree'] if x['path']==c['source_relative_path']]
 assert len(e)==1 and e[0]['sha']==BLOB and e[0]['size']==SIZE and e[0]['type']=='blob'
 url='https://api.github.com/repos/rajpurkar/SQuAD-explorer/git/blobs/'+BLOB
 declared=time.monotonic_ns()
 write('CPU_BLOB_TRANSPORT_INTENT_03.json',{'schema':'same_official_commit_blob_transport_cpu_v1','selection_ref':ref(D/'PROSPECTIVE_SQUAD_SELECTION.json'),'tree_ref':ref(tree),'source_commit':COMMIT,'git_blob_sha1':BLOB,'expected_payload_bytes':SIZE,'url':url,'max_transport_bytes':24*1024**2,'declared_monotonic_ns':declared,'same_original_selection':True,'replaces_failed_transport_only':True,'CPU_only':True,'GPU_operations':0})
 start=time.monotonic_ns()
 req=urllib.request.Request(url,headers={'User-Agent':'prefix-io-cpu-input-preparation/1'})
 size=0
 with urllib.request.urlopen(req,timeout=45) as r:
  assert r.status==200 and r.geturl()==url
  with (D/'OFFICIAL_BLOB_API_RESPONSE_03.json').open('xb') as f:
   while True:
    if time.monotonic_ns()-start>270_000_000_000:raise TimeoutError('bounded 270-second acquisition')
    b=r.read(16384)
    if not b:break
    size+=len(b)
    if size>24*1024**2:raise ValueError('bounded transport exceeded')
    f.write(b)
 response=D/'OFFICIAL_BLOB_API_RESPONSE_03.json'; doc=json.loads(response.read_bytes())
 assert doc['sha']==BLOB and doc['size']==SIZE and doc['encoding']=='base64'
 raw=base64.b64decode(''.join(doc['content'].split()),validate=True)
 assert len(raw)==SIZE and len(raw)<=c['max_source_bytes']
 assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==BLOB
 source=D/'OFFICIAL_SQUAD_DEV_V1_1_VERIFIED_03.json'
 with source.open('xb') as f:f.write(raw)
 assert ref(ledger)['sha256']==LEDGER_SHA
 write('CPU_SOURCE_ACQUISITION_RESULT_03.json',{'schema':'actual_bounded_public_data_cpu_acquisition_v1','status':'PASS_ACTUAL_PUBLIC_BYTES_GIT_BLOB_VERIFIED','source_repository':c['source_repository'],'source_commit':COMMIT,'source_relative_path':c['source_relative_path'],'declared_source_url':c['source_url'],'actual_transport_url':url,'http_status':200,'git_blob_sha1':BLOB,'git_blob_sha1_verified':True,'source_ref':ref(source),'transport_response_ref':ref(response),'tree_ref':ref(tree),'selection_ref':ref(D/'PROSPECTIVE_SQUAD_SELECTION.json'),'intent_ref':ref(D/'CPU_BLOB_TRANSPORT_INTENT_03.json'),'declared_monotonic_ns':declared,'fetch_started_monotonic_ns':start,'fetch_finished_monotonic_ns':time.monotonic_ns(),'payload_bytes':len(raw),'transport_bytes':size,'cpu_only':True,'gpu_operations':0,'budget_before_sha256':LEDGER_SHA,'budget_after_sha256':ref(ledger)['sha256'],'natural_dataset_read':True,'recorded_natural_arrivals':False,'formal_gpu_eligible':False,'source_license':'SQuAD dataset CC BY-SA 4.0; repository code MIT is separate','attribution_url':'https://rajpurkar.github.io/SQuAD-explorer/','primary_free_bytes':shutil.disk_usage(ROOT).free})
 print(json.dumps({'status':'PASS_ACTUAL_PUBLIC_BYTES_GIT_BLOB_VERIFIED','source_ref':ref(source),'GPU_operations':0}))
if __name__=='__main__':main()
