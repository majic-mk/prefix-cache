from pathlib import Path
import json,hashlib,sys,builtins,subprocess,os,time,datetime,shutil
root=Path(".").resolve();out=root/"artifacts/prefix_io_v1/server08-p3-16";primary=root/"experiments/prefix_io_v1/runs";result_path=out/"private-cache-merge-revalidation-after-grant.json";assert not result_path.exists()
audit=out/"private-cache-merge-manifest.json";assert hashlib.sha256(audit.read_bytes()).hexdigest()=="e692ad4495548d9729423d956ceea405bbcac4202d5b9e50bee2db4ac424e062"
tool=root/"experiments/prefix_io_v1/scripts/consolidate_private_cache_copies.py";assert hashlib.sha256(tool.read_bytes()).hexdigest()=="e41f5c2300d1b4dd4cf0724521cd95c049dc54dcdc49055442fe4f7383c43c30"
ledger=json.loads((primary.parent/"gpu-budget-ledger.json").read_text());assert ledger["active_reservation"] is None
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader"],text=True).strip()
assert not (out/"private-cache-merge-apply-journal.jsonl").exists()
sys.path[:0]=[str(root/"experiments/prefix_io_v1/scripts"),str(root/"src"),str(root/".venv/lib/python3.12/site-packages")]
original_import=builtins.__import__
def cpu_import(name,*a,**k):
 if name.split(".")[0] in {"torch","vllm","py_kvcache"}:raise AssertionError("GPU libraries forbidden")
 return original_import(name,*a,**k)
builtins.__import__=cpu_import
import consolidate_private_cache_copies as m
m.AUX=primary.parent
data=json.loads(audit.read_text());allow={primary/(f"server08-p3-15-mixed-{n:02d}-{arm}")/"details/storage" for n,arm in [(1,"off"),(2,"fixed"),(3,"pressure"),(4,"pressure"),(5,"fixed"),(6,"off")]}
assert set(map(Path,data["completed_cache_roots_allowlist"]))==allow and len(allow)==6
regp=primary/"server08-p3-14-source/registration.json";reg=json.loads(regp.read_text());source=Path(reg["source_root"]).resolve();assert str(source)==data["published_source_root"]
mp=Path(reg["source_manifest"]);assert hashlib.sha256(mp.read_bytes()).hexdigest()==reg["source_manifest_sha256"]=="c03381abb29e21b54a62a4815892553b58e4ef39cc8087761cc4d173cf75d274"
source_rows=json.loads(mp.read_text());assert len(source_rows)==3048
protected={(source/x["path"]).resolve() for x in source_rows}
for base in allow:
 r=json.loads((base.parent/"result.json").read_text())
 assert r["engine_shutdown"]=="completed" and r["status"].startswith("PASSED_")
 assert not any(".dedup-backup-" in p.name or ".dedup-link-" in p.name for p in base.rglob("*"))
selected={e["session_id"] for e in ledger["events"] if e["label"].startswith(("server08-p3-15-","server08-p3-16-"))}
for p in Path("/proc").iterdir():
 if p.name.isdigit():
  try:sid=os.getsid(int(p.name))
  except ProcessLookupError:continue
  assert sid not in selected,(p.name,sid)
start=time.monotonic();digests={};total=0;targets=set();target_roots=set()
def checked_hash(p,expected,size):
 global total
 s=p.stat();key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
 assert s.st_size==size
 if key not in digests:
  h=m.digest(p);after=p.stat();assert key==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns);digests[key]=h;total+=size
 assert digests[key]==expected,str(p)
for row in source_rows:checked_hash((source/row["path"]).resolve(),row["sha256"],row["bytes"])
assert {p.resolve() for p in source.rglob("*.bin")}==protected
assert len(data["proposals"])==data["candidate_groups"]==2060
for group in data["proposals"]:
 c=m.verified_path(group["canonical"],m.AUX,private=False)
 assert c in protected or any(c.is_relative_to(b) for b in allow)
 checked_hash(c,group["sha256"],group["canonical"]["bytes"])
 for record in group["targets"]:
  t=m.verified_path(record,m.AUX,private=True)
  roots=[b for b in allow if t.is_relative_to(b)];assert len(roots)==1
  assert t not in protected and t not in targets and t.stat().st_dev==c.stat().st_dev
  targets.add(t);target_roots.add(roots[0]);checked_hash(t,group["sha256"],record["bytes"])
assert len(targets)==data["candidate_files"]==10269 and len(target_roots)==5
assert sum(x["allocated_bytes"] for g in data["proposals"] for x in g["targets"])==data["possible_reclaim_bytes"]==9421848576
d=dict(schema_version=1,status="PASS_EXACT_FROZEN_MANIFEST_REVALIDATION_AFTER_HUMAN_CONTINUE",audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),target_files=len(targets),target_roots=len(target_roots),allowlist_roots=len(allow),registered_source_files=3048,unique_inodes_hashed=len(digests),bytes_hashed=total,seconds=time.monotonic()-start,all_targets_single_link=True,source_unchanged=True,all_expected_bytes_match=True,selected_completed_sessions_absent=True,no_dedup_debris=True,primary_free_before=shutil.disk_usage(primary).free,aux_free_before=shutil.disk_usage("/root/prefix-io-v1-validation").free,GPU_active_reservation=None,GPU_workloads_run=0,data_changed=False,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
with result_path.open("x",encoding="utf-8") as f:json.dump(d,f,indent=2)
print(json.dumps(d))
