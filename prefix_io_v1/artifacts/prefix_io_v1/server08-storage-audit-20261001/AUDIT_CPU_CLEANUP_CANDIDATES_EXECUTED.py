"""Read-only CPU cleanup candidate freeze; never deletes or initializes GPU."""
from pathlib import Path
import os,stat,json,hashlib,re,datetime,collections
ROOT=Path.cwd().resolve()
OUT=ROOT/'artifacts/prefix_io_v1/server08-storage-audit-20261001'
R=ROOT/'experiments/prefix_io_v1/runs';A=ROOT/'artifacts/prefix_io_v1'
roots=[]
def add(p,label):
 assert p.is_dir() and not p.is_symlink(),p
 roots.append((p,label))
for name in ('server08-p3-15-cpu-01','server08-p3-15-cpu-02','server08-p3-16-cpu-01','server08-p3-16-cpu-02'):
 add(R/name/'cpu-evidence/shadow-pytest-tmp','P3 CPU pytest generated temporary')
for name in ('server08-p4-cpu-current-integration-01','server08-p4-02-current-integration-01'):
 add(R/name/'cpu-evidence/pytest-tmp','P4 CPU pytest generated temporary')
m=A/'autodl-aio-mixed-cpu-20260929';u=A/'autodl-aio-cpu-20260929'
n=0
for name in ('mixed-before','mixed-after'):
 for p in sorted((m/name).iterdir()):
  if re.fullmatch(r'r[01]-s(4096|917504|3670016)-d(1|4|16)',p.name):
   add(p,'CPU mixed microbenchmark generated input');n+=1
assert n==36,n
n=0
for name in ('microbench','microbench-eventfd','microbench-preconditioned'):
 for p in sorted((u/name).glob('block-*-depth-*')):
  add(p,'CPU AIO microbenchmark generated input');n+=1
assert n==27,n
for name in ('pytest-mixed','pytest-release'):add(m/name,'CPU mixed pytest generated temporary')
for name in ('pytest-complete','pytest-eventfd','pytest-final','pytest-first','pytest-qualified-importlib','pytest-release'):
 add(u/name,'CPU AIO pytest generated temporary')
add(A/'new-server-07/pytest-cpu','CPU qualification pytest generated temporary')
assert len(roots)==78
protected={}
def protect(p,reason):
 p=Path(p)
 if not p.is_absolute():p=ROOT/p
 protected[str(p.resolve())]=reason
lockrefs=[]
locknames=[
 'artifacts/prefix_io_v1/server08-p3-16/execution-lock-12-final-p3.json',
 'artifacts/prefix_io_v1/server08-p4-01-cpu/test-input-lock-v2.json',
 'artifacts/prefix_io_v1/server08-p4-02-cpu/test-input-lock.json',
 'artifacts/prefix_io_v1/server08-p4-02-cpu/gpu-source-lock.json']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def file_ref(p):return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
for s in locknames:
 p=ROOT/s;data=json.loads(p.read_text());lockrefs.append(file_ref(p));protect(p,'protected lock itself')
 if isinstance(data,dict) and 'files' in data:
  for row in data['files']:protect(row['path'],s)
 else:
  for key in data:
   if key.startswith('/'):protect(key,s)
prior=json.loads((A/'server08-p3-16/source-preservation-final-p3.json').read_text())
manifest=Path(prior['source_manifest']['path']);assert sha(manifest)==prior['source_manifest']['sha256']
protect(manifest,'registered KV manifest')
for row in json.loads(manifest.read_text()):protect(Path(prior['source_root'])/row['path'],'registered KV source')
keep=[
 m/'mixed-before/results.json',m/'mixed-after/results.json',
 u/'microbench/results.json',u/'microbench-eventfd/results.json',u/'microbench-preconditioned/results.json',
 m/'mixed-before.json',m/'mixed-after.json',m/'measured-script.py',
 u/'microbench.json',u/'microbench-eventfd.json',u/'microbench-preconditioned.json',
 ROOT/'experiments/prefix_io_v1/scripts/aio_cpu_mixed_screen.py',
 ROOT/'experiments/prefix_io_v1/scripts/aio_cpu_microbench.py']
for p in keep:assert p.is_file();protect(p,'benchmark source/completion evidence outside candidate')
rows=[];directories=[];groups=collections.defaultdict(list);links=[];special=[];conflicts=[]
def rel(p):return p.relative_to(ROOT).as_posix()
def meta(p):
 s=p.lstat();return dict(path=rel(p),device=s.st_dev,inode=s.st_ino,mode=s.st_mode,nlink=s.st_nlink,
  bytes=s.st_size,allocated_bytes=s.st_blocks*512,mtime_ns=s.st_mtime_ns)
def inside(path):
 return any(path==str(p) or path.startswith(str(p)+'/') for p,_ in roots)
for root,label in roots:
 assert root.resolve()==root and root.is_relative_to(ROOT)
 for base,dirs,files in os.walk(root,followlinks=False):
  b=Path(base)
  d=meta(b);d['label']=label;directories.append(d)
  for name in list(dirs):
   p=b/name
   if p.is_symlink():dirs.remove(name);files.append(name)
  for name in files:
   p=b/name;r=meta(p);r['label']=label
   if str(p.resolve()) in protected:conflicts.append(dict(path=rel(p),reason=protected[str(p.resolve())]))
   if stat.S_ISREG(r['mode']):
    r['type']='regular';groups[(r['device'],r['inode'])].append(r)
   elif stat.S_ISLNK(r['mode']):
    r['type']='symlink';r['target']=os.readlink(p);assert inside(str(p.resolve())),p
    links.append(r)
   elif stat.S_ISFIFO(r['mode']):r['type']='synthetic_test_FIFO';special.append(r)
   else:raise RuntimeError('unknown candidate file type '+str(p))
   rows.append(r)
assert not conflicts,conflicts
outside=[]
protected_inodes={}
for p in protected:
 try:
  s=os.stat(p);protected_inodes[(s.st_dev,s.st_ino)]=p
 except FileNotFoundError:pass
for key,aliases in groups.items():
 if aliases[0]['nlink']!=len(aliases):outside.append(aliases)
 assert key not in protected_inodes,(aliases[0]['path'],protected_inodes.get(key))
assert not outside,'external hardlink exists'
# Hash only proven closed-inode CPU synthetic temporary files; never model/real KV.
for key,aliases in groups.items():
 p=ROOT/aliases[0]['path'];before=p.lstat()
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 h=hashlib.sha256()
 with os.fdopen(fd,'rb') as f:
  s=os.fstat(f.fileno());assert (s.st_dev,s.st_ino,s.st_mtime_ns,s.st_size)==(before.st_dev,before.st_ino,before.st_mtime_ns,before.st_size)
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 after=p.lstat();assert (after.st_dev,after.st_ino,after.st_mtime_ns,after.st_size)==(before.st_dev,before.st_ino,before.st_mtime_ns,before.st_size)
 for r in aliases:r['sha256']=h.hexdigest()
active=[];scan_errors=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  try:
   value=os.readlink(proc/'cwd')
   if inside(value):active.append(dict(pid=proc.name,kind='cwd',path=value))
  except FileNotFoundError:pass
  for fd in (proc/'fd').iterdir():
   try:value=os.readlink(fd)
   except FileNotFoundError:continue
   if inside(value.removesuffix(' (deleted)')):active.append(dict(pid=proc.name,kind='fd',path=value))
  try:
   for line in (proc/'maps').read_text().splitlines():
    parts=line.split(maxsplit=5)
    if len(parts)==6 and inside(parts[5].removesuffix(' (deleted)')):active.append(dict(pid=proc.name,kind='mmap',path=parts[5]))
  except FileNotFoundError:pass
 except (FileNotFoundError,ProcessLookupError):pass
 except PermissionError as e:scan_errors.append(dict(pid=proc.name,error=str(e)))
assert not active and not scan_errors,(active,scan_errors)
allocated=sum(a[0]['allocated_bytes'] for a in groups.values())
assert allocated==3221516288,allocated
assert all(file_ref(ROOT/r['path'])==r for r in lockrefs)
OUT.mkdir(exist_ok=False)
proposal=dict(schema_version=1,status='FROZEN_PROPOSAL_ONLY_NO_DELETION_AUTHORIZED',
 created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 ROOT=str(ROOT),candidate_roots=[dict(path=rel(p),label=l) for p,l in roots],
 files=sorted(rows,key=lambda r:r['path']),directories=sorted(directories,key=lambda r:r['path']),
 regular_paths=sum(len(a) for a in groups.values()),unique_regular_inodes=len(groups),symlinks=len(links),synthetic_FIFOs=len(special),
 conservative_regular_allocated_bytes=allocated,directory_allocated_bytes=sum(r['allocated_bytes'] for r in directories),
 protected_path_intersection=[],external_hardlinks=[],active_cwd_fd_maps=[],proc_scan_errors=[],
 protected_source_locks=lockrefs,retained_benchmark_evidence=[file_ref(p) for p in keep],
 deletion_performed=False,cache_merge_or_deletion=False,model_or_real_KV_payload_read=False,
 CPU_synthetic_payload_hash_read=True,GPU_initialized=False,GPU_runs=0,
 excludes='All real GPU caches/models/environment/source/patch reviewed baseline trees, outer reports/logs/XML/guards/archives and original Git worktrees')
dest=OUT/'CPU_TEMP_CLEANUP_PROPOSAL.json'
with dest.open('x') as f:json.dump(proposal,f,indent=2);f.write('\n')
s=os.statvfs(ROOT)
summary=dict(status='PASS_READ_ONLY_CPU_TEMP_CLEANUP_AUDIT',proposal=file_ref(dest),
 roots=len(roots),regular_paths=proposal['regular_paths'],unique_regular_inodes=len(groups),symlinks=len(links),synthetic_FIFOs=len(special),
 conservative_reclaim_bytes=allocated,conservative_reclaim_GiB=allocated/1024**3,
 directory_blocks_not_in_conservative_reclaim=proposal['directory_allocated_bytes'],
 primary_total_bytes=s.f_blocks*s.f_frsize,primary_free_before=s.f_bavail*s.f_frsize,
 primary_expected_free_after=allocated+s.f_bavail*s.f_frsize,
 normal_model_primary_reservation_bytes=3*1024**3,primary_minimum_free_bytes=8*1024**3,
 primary_3GiB_floor_pass_after=allocated+s.f_bavail*s.f_frsize-3*1024**3>=8*1024**3,
 AUX_reclaim_bytes=0,AUX_20GiB_cap_unchanged=True,
 protected_conflicts=0,active_refs=0,external_hardlinks=0,GPU_runs=0,deletions=0)
with (OUT/'AUDIT_SUMMARY.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
print(json.dumps(summary))
