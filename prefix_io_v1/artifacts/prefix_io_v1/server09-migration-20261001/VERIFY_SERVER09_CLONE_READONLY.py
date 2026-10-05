import os,stat,json,pathlib,hashlib,time,datetime
ROOT=pathlib.Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
AUDIT=ROOT/'artifacts/prefix_io_v1/server08-storage-audit-20261001'
start=time.monotonic()
failures=[]
def fail(kind,path,detail):
 failures.append({'kind':kind,'path':str(path),'detail':detail})
def read_checked(p):
 before=p.lstat()
 if not stat.S_ISREG(before.st_mode):raise ValueError('not regular or final symlink')
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  a=os.fstat(fd);h=hashlib.sha256();parts=[]
  if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns):raise ValueError('changed before read')
  while True:
   b=os.read(fd,1024*1024)
   if not b:break
   h.update(b);parts.append(b)
  z=os.fstat(fd)
 finally:os.close(fd)
 after=p.lstat()
 stamp=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_mode)
 if stamp(a)!=stamp(z) or stamp(z)!=stamp(after):raise ValueError('changed during read')
 return b''.join(parts),h.hexdigest(),after
def hash_checked(p):
 before=p.lstat()
 if not stat.S_ISREG(before.st_mode):raise ValueError('not regular or final symlink')
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  a=os.fstat(fd);h=hashlib.sha256()
  stamp=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_mode)
  if stamp(before)!=stamp(a):raise ValueError('changed before hash')
  while True:
   b=os.read(fd,1024*1024)
   if not b:break
   h.update(b)
  z=os.fstat(fd)
 finally:os.close(fd)
 after=p.lstat()
 if stamp(a)!=stamp(z) or stamp(z)!=stamp(after):raise ValueError('changed during hash')
 return h.hexdigest(),after
def verified_json(p,sha):
 b,h,s=read_checked(p)
 if h!=sha:raise ValueError('frozen JSON SHA mismatch '+h+' != '+sha)
 return json.loads(b),h,len(b)
def absolute(p):
 p=pathlib.Path(p)
 return p if p.is_absolute() else ROOT/p
inventory,inventory_sha,inventory_bytes=verified_json(AUDIT/'PROTECTED_BYTES_AFTER_CLEANUP.json','d5b6a5f743be42fb4a6374a72ef10ae44d4acedb1fb71d669806960b86cca36e')
expected={x['path']:x for x in inventory['files']}
assert len(expected)==len(inventory['files'])==7813
proposal,proposal_sha,proposal_bytes=verified_json(AUDIT/'CPU_TEMP_CLEANUP_PROPOSAL.json','b0c347fc4fc50384c088b8539b63ac09d77446a9563f33fba83f7844514c410e')
cleanup,cleanup_sha,cleanup_bytes=verified_json(AUDIT/'APPROVED_CPU_CLEANUP_RESULT.json','7f40621ec368780fc2c16a456af4528ee958b31e1cffd721ac832de5ebecc792')
lock_rows=[];lock_refs=set()
for lock in proposal['protected_source_locks']:
 p=ROOT/lock['path'];d,h,n=verified_json(p,lock['sha256'])
 if n!=lock['bytes']:fail('lock_size',p,{'actual':n,'expected':lock['bytes']})
 rows=[{'path':k,'sha256':v} for k,v in d.items()] if isinstance(d,dict) and 'files' not in d else d['files']
 lock_rows.append({'path':lock['path'],'bytes':n,'sha256':h,'refs':len(rows)})
 for row in rows:
  p=str(absolute(row['path']));lock_refs.add(p)
  e=expected.get(p)
  if e is None or e['sha256']!=row['sha256'] or ('bytes' in row and e['bytes']!=row['bytes']):
   fail('lock_reference_not_frozen_inventory',p,row)
assert [x['refs'] for x in lock_rows]==[2724,70,2198,2048]
preservation=json.loads((ROOT/'artifacts/prefix_io_v1/server08-p3-16/source-preservation-final-p3.json').read_bytes())
manifest,manifest_sha,manifest_bytes=verified_json(pathlib.Path(preservation['source_manifest']['path']),'c03381abb29e21b54a62a4815892553b58e4ef39cc8087761cc4d173cf75d274')
source_root=pathlib.Path(preservation['source_root'])
kv_paths=set()
for row in manifest:
 p=str(source_root/row['path']);kv_paths.add(p);e=expected.get(p)
 if e is None or e['sha256']!=row['sha256'] or e['bytes']!=row['bytes']:fail('KV_reference_not_frozen_inventory',p,row)
assert len(kv_paths)==3048 and sum(r['bytes'] for r in manifest)==2796552192
actual_bin=set()
for directory,dirs,files in os.walk(source_root,followlinks=False):
 for name in dirs:
  p=pathlib.Path(directory)/name
  if p.is_symlink():fail('KV_directory_symlink',p,os.readlink(p))
 for name in files:
  if name.endswith('.bin'):actual_bin.add(str(pathlib.Path(directory)/name))
if actual_bin!=kv_paths:fail('KV_membership',source_root,{'extra':sorted(actual_bin-kv_paths)[:30],'missing':sorted(kv_paths-actual_bin)[:30]})
verified_count=0;verified_bytes=0;so_rows=[];outside=[];evidence_root_hash=hashlib.sha256()
for p,e in sorted(expected.items()):
 try:
  h,s=hash_checked(pathlib.Path(p))
  if h!=e['sha256'] or s.st_size!=e['bytes']:
   fail('protected_SHA_or_bytes',p,{'actual_sha256':h,'expected_sha256':e['sha256'],'actual_bytes':s.st_size,'expected_bytes':e['bytes']});continue
  verified_count+=1;verified_bytes+=s.st_size
  evidence_root_hash.update((p+'\0'+h+'\0'+str(s.st_size)+'\n').encode())
  if p.endswith('.so'):so_rows.append({'path':p,'bytes':s.st_size,'sha256':h,'execute_or_ABI_qualified':False})
  if not p.startswith(str(ROOT)+'/'):outside.append({'path':p,'bytes':s.st_size,'sha256':h})
 except Exception as ex:fail('protected_read_error',p,type(ex).__name__+': '+str(ex))
retained=set(cleanup['skipped_files'])|set(cleanup['skipped_directories'])
absent_roots=0;present_roots=0;remaining=set();candidate_errors=[]
for root in proposal['candidate_roots']:
 rel=root['path'];p=ROOT/rel
 if not os.path.lexists(p):absent_roots+=1;continue
 present_roots+=1;remaining.add(rel)
 if not p.is_dir() or p.is_symlink():fail('candidate_root_unexpected_type',p,'not canonical directory');continue
 for directory,dirs,files in os.walk(p,followlinks=False):
  for name in dirs+files:
   child=pathlib.Path(directory)/name;cr=str(child.relative_to(ROOT));remaining.add(cr)
   if cr not in retained:fail('unexpected_candidate_remaining',cr,'not one of approved retained items')
   if child.is_symlink():fail('retained_symlink',cr,os.readlink(child))
if remaining!=retained:fail('retained_candidate_set',AUDIT,{'missing':sorted(retained-remaining),'extra':sorted(remaining-retained)})
for rel in cleanup['skipped_files']:
 p=ROOT/rel
 try:
  s=p.lstat()
  if not stat.S_ISREG(s.st_mode) or s.st_size!=0:fail('retained_fixture_not_empty_regular',rel,{'mode':s.st_mode,'bytes':s.st_size})
 except Exception as ex:fail('retained_fixture_missing',rel,str(ex))
ledger_p=ROOT/'experiments/prefix_io_v1/gpu-budget-ledger.json'
ledger,ledger_sha,ledger_bytes=verified_json(ledger_p,'31199998369e35fcd40daddd7af340353b0feaf134bbe37d2dd8f1f35391efc1')
if ledger.get('active_reservation') is not None:fail('active_GPU_reservation',ledger_p,ledger.get('active_reservation'))
permissions_p=ROOT/'experiments/prefix_io_v1/configs/permissions.yaml'
_,permissions_sha,perm_s=read_checked(permissions_p)
if permissions_sha!='795d24f9955379653e34ffbbfd7614ad9ef96ad0ac9a208c6b4baefd994fac50':fail('permissions_SHA',permissions_p,permissions_sha)
v=os.statvfs(ROOT)
summary={'status':'PASS_READ_ONLY_CLONE_PRESERVATION' if not failures else 'FAIL_READ_ONLY_CLONE_PRESERVATION','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ROOT':str(ROOT),'protected_inventory_sha256':inventory_sha,'protected_inventory_bytes':inventory_bytes,'protected_unique_files_expected':7813,'protected_unique_files_verified':verified_count,'protected_bytes_verified':verified_bytes,'protected_path_SHA_bytes_summary_sha256':evidence_root_hash.hexdigest(),'lockfiles':lock_rows,'lock_unique_reference_paths':len(lock_refs),'registered_KV_manifest_sha256':manifest_sha,'registered_KV_manifest_bytes':manifest_bytes,'registered_KV_files':len(kv_paths),'registered_KV_bytes':sum(x['bytes'] for x in manifest),'registered_KV_exact_membership':actual_bin==kv_paths,'candidate_roots_expected':78,'old_cleanup_roots_absent':absent_roots,'old_cleanup_roots_retained':present_roots,'old_cleanup_retained_files':len(cleanup['skipped_files']),'old_cleanup_retained_directories':len(cleanup['skipped_directories']),'remaining_candidate_objects_exact':remaining==retained,'ledger_sha256':ledger_sha,'ledger_bytes':ledger_bytes,'ledger_gpu_wall_seconds':ledger['gpu_wall_seconds'],'ledger_active_reservation':ledger.get('active_reservation'),'permissions_sha256':permissions_sha,'permissions_bytes':perm_s.st_size,'original_binaries_hashed_only':so_rows,'historical_outside_project_evidence':outside,'PRIMARY_total_bytes':v.f_frsize*v.f_blocks,'PRIMARY_available_bytes':v.f_frsize*v.f_bavail,'GPU_imports':0,'GPU_initializations':0,'GPU_runs':0,'server_writes':0,'deletion_operations':0,'failures_count':len(failures),'failures':failures[:100],'elapsed_seconds':time.monotonic()-start}
print(json.dumps(summary,sort_keys=True))

