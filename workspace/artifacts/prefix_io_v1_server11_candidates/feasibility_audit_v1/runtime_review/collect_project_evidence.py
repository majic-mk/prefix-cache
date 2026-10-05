"""Read only 22 original project files and verify each against the frozen delivery manifest.

This does not open, decompress, or rehash the large archive. The archive reference
is read from its receipt and checked against the previously verified constants.
"""
from pathlib import Path
import argparse,hashlib,json,time

MANIFEST_SHA='d6e0af0a17b56c543edd988d9c61b8d7281b4f5cd1ac22eee8aaf64b205d1cee'
MANIFEST_BYTES=501724
ARCHIVE_SHA='a52b9de286c46fd342592565bb9dd0550d1c82791ca0a8e94e2c9e7cd8f1e8b6'
ARCHIVE_BYTES=276693033
DELIVERY='artifacts/prefix_io_v1/server11-delivery-v6-20261003/'
A='artifacts/prefix_io_v1/'
C=A+'server11-p4-single-file-candidate-v4-20261003/'
D=A+'server11-p4-single-file-runtime-v4-20261003/'
N=A+'server11-native-cost-v6-20261003/'
P=A+'server11-p4-retry-cpu-profile-v2-20261003/'
R='experiments/prefix_io_v1/runs/'
WANTED={
C+'CPU_PROFILE_FULL_BORROW.json',C+'FULL_BORROW_CPU_PROFILE_COMMAND.json',
P+'ACTUAL_CPU_full_borrow.json',P+'CPU_PROFILE_full_borrow_COMMAND.json',P+'profile_retry_hotpath.py',
C+'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py',
C+'source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_bridge.py',
C+'source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_policy.py',
C+'native_full_step_collector.py',C+'test_single_file_retry_observation.py',
D+'run_p4_single_file_experiment.py',D+'verify_p4_single_file.py',
N+'NATIVE_CONDITIONAL_COST_RESULT.json',N+'NATIVE_COST_PLAN.json',N+'NATIVE_MEASUREMENTS.json',
R+'server11-native-cost-six-window-06/details/native-cost-runtime-result.json'}
for mode in ('off','shadow','on'):
 WANTED.add(D+'QUALIFICATION_'+mode+'.json')
 WANTED.add(R+'server11-p4-single-file-'+mode+'-04/details/p4-single-file-runtime-result.json')

def require(value,message):
 if not value:raise ValueError(message)

def safe(root,name):
 require(type(name)is str and name and ':' not in name and '\\' not in name
  and all(p not in ('','.','..') for p in name.split('/')),'unsafe project-relative name')
 p=root
 for part in name.split('/'):
  p=p/part;require(not p.is_symlink(),'symlink evidence refused')
 require(p.resolve().is_relative_to(root),'outside project root')
 return p

def ref(path,raw):
 return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def collect(root,manifest_path,receipt_path,prior_path=None):
 started=time.perf_counter()
 manifest_bytes=manifest_path.read_bytes()
 require(len(manifest_bytes)==MANIFEST_BYTES and hashlib.sha256(manifest_bytes).hexdigest()==MANIFEST_SHA,
  'original frozen delivery manifest changed')
 manifest=json.loads(manifest_bytes)
 require(manifest['status']=='CLOSED_NATIVE_RUN_DELTA','closed original manifest')
 rows=manifest['files'];locked={row['path']:row for row in rows}
 require(len(rows)==len(locked)==1572,'original unique manifest count')
 require(WANTED<=locked.keys() and len(WANTED)==22,'exact 22-member allowlist')
 receipt_bytes=receipt_path.read_bytes();receipt=json.loads(receipt_bytes)
 require(receipt['status']=='ARCHIVED_DELTA' and receipt['file_count']==1572,'original archive receipt')
 require(receipt['manifest']==dict(path=DELIVERY+'SERVER11_DELTA_MANIFEST.json',bytes=MANIFEST_BYTES,sha256=MANIFEST_SHA),
  'receipt does not bind frozen manifest')
 require(receipt['archive']==dict(path=DELIVERY+'SERVER11_RAW_DELTA.tar.gz',bytes=ARCHIVE_BYTES,sha256=ARCHIVE_SHA),
  'receipt does not bind previously verified archive')
 prior=None
 if prior_path is not None:
  prior_bytes=prior_path.read_bytes();prior=json.loads(prior_bytes)
  require(prior['status']=='PASS_LOCAL_ARCHIVE_ALL_MEMBERS' and prior['file_count']==1572
    and prior['archive_sha256']==ARCHIVE_SHA and prior['archive_bytes']==ARCHIVE_BYTES
    and prior['manifest_sha256']==MANIFEST_SHA,'prior all-member verification does not match')
  prior=dict(record=prior,record_ref=ref(prior_path,prior_bytes))
 members={}
 for name in sorted(WANTED):
  expected=locked[name]
  require(type(expected['bytes'])is int and 0<expected['bytes']<=16*1024**2,'bounded selected member')
  p=safe(root,name);require(p.stat().st_size==expected['bytes'],'selected source size drift: '+name)
  raw=p.read_bytes();actual=dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
  require(actual==expected,'selected source content drift: '+name)
  members[name]=dict(bytes=actual['bytes'],sha256=actual['sha256'],
   content=json.loads(raw) if name.endswith('.json') else raw.decode('utf-8'))
  print('verified',name,len(raw),flush=True)
 return dict(scope='bounded_original_project_read_verified_against_frozen_archive_manifest_no_GPU_RPC',
  archive_ref=receipt['archive'],archive_ref_basis='frozen receipt and prior recorded archive verification; not opened in this call',
  did_not_read_archive=True,archive_rehashed_in_this_call=False,archive_decompressed_in_this_call=False,
  original_project_member_hashes_verified=22,manifest_ref=receipt['manifest'],
  receipt_ref=ref(receipt_path,receipt_bytes),
  prior_all_member_archive_verification=prior,
  prior_archive_verification_note=('Explicit prior verification record checked; not repeated.' if prior is not None
   else 'Root previously verified all archive members; no verification record supplied to this call, and no new full-archive claim is made.'),
  archive_file_count=1573,archive_file_count_basis='1572 frozen manifest entries plus embedded manifest; not recounted',
  selected_member_count=22,selected_member_uncompressed_bytes=sum(m['bytes'] for m in members.values()),
  selected_members=members,
  nonfixture_profile_or_timeline_members=[dict(name=row['path'],bytes=row['bytes']) for row in rows
   if '/cpu-native-tmp/' not in row['path'] and any(k in row['path'].lower() for k in ('profile','trace','timeline','capture.json'))],
  seconds=time.perf_counter()-started)

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--root',type=Path,required=True)
 parser.add_argument('--manifest',type=Path)
 parser.add_argument('--archive-receipt',type=Path)
 parser.add_argument('--prior-archive-verification',type=Path)
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();root=args.root.resolve(strict=True)
 require(not args.output.exists(),'append-only output already exists')
 result=collect(root,args.manifest or root/DELIVERY/'SERVER11_DELTA_MANIFEST.json',
  args.archive_receipt or root/DELIVERY/'ARCHIVE_RECEIPT.json',args.prior_archive_verification)
 with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,separators=(',',':'))
 print(json.dumps({k:v for k,v in result.items() if k not in ('selected_members','nonfixture_profile_or_timeline_members')},indent=2))

if __name__=='__main__':main()

