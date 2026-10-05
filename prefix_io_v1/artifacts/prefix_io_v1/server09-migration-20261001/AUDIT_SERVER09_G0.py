"""CPU-only server09 migration and resource audit; no CUDA or NVML imports."""
import os, sys, json, hashlib, subprocess, time, datetime, pathlib, stat
ROOT=pathlib.Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
start=time.monotonic()
def sha(path):
 b=pathlib.Path(path).read_bytes()
 return {"path":str(path),"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}
def disk(path):
 v=os.statvfs(path)
 free=v.f_frsize*v.f_bavail
 return {"path":str(path),"device":os.stat(path).st_dev,"total_bytes":v.f_frsize*v.f_blocks,"free_bytes":free,
  "minimum_free_bytes":8*1024**3,"native_128MiB_floor_pass":free-128*1024**2>=8*1024**3,
  "model_3GiB_floor_pass":free-3*1024**3>=8*1024**3}
def git(path):
 rows={}
 for key,args in [("head",["rev-parse","HEAD"]),("branch",["branch","--show-current"]),("status",["status","--porcelain=v1","--untracked-files=all"])]:
  p=subprocess.run(["git","-C",str(path)]+args,text=True,capture_output=True,timeout=45)
  if p.returncode:raise RuntimeError(p.stderr)
  if key!="status":rows[key]=p.stdout.strip()
  else:
   lines=p.stdout.splitlines();tracked=[line for line in lines if not line.startswith("?? ")]
   rows[key]={"entries":len(lines),"tracked_changes":tracked,"untracked_count":len(lines)-len(tracked),
    "all_status_stdout_sha256":hashlib.sha256(p.stdout.encode()).hexdigest(),"untracked_first_16":[line for line in lines if line.startswith("?? ")][:16]}
 return {"path":str(path),**rows}
aux=pathlib.Path("/root/prefix-io-v1-validation")
seen=set(); used=0; files=0; links=0; directories=0
for base,dirs,names in os.walk(aux,followlinks=False):
 keep=[]
 for name in dirs:
  p=pathlib.Path(base)/name
  if p.is_symlink():links+=1
  else:keep.append(name);directories+=1
 dirs[:]=keep
 for name in names:
  p=pathlib.Path(base)/name;s=p.lstat()
  if stat.S_ISLNK(s.st_mode):links+=1;continue
  key=(s.st_dev,s.st_ino)
  if key in seen:continue
  seen.add(key);used+=max(s.st_size,s.st_blocks*512);files+=1
auxrow=disk(aux)
auxrow.update({"used_unique_inode_bytes":used,"unique_files":files,"directories":directories,
 "symlinks_not_followed":links,"max_bytes":20*1024**3,
 "native_128MiB_cap_pass":used+128*1024**2<=20*1024**3,
 "model_3GiB_cap_pass":used+3*1024**3<=20*1024**3})
devices=sorted(p.name for p in pathlib.Path("/dev").glob("nvidia*") if p.name[6:].isdigit())
inventory=[]
for p in pathlib.Path("/proc/driver/nvidia/gpus").glob("*/information"):
 fields={}
 for line in p.read_text().splitlines():
  if ":" in line:
   k,v=line.split(":",1);fields[k.strip()]=v.strip()
 if "nvidia"+fields.get("Device Minor","") in devices:inventory.append({"path":str(p),"fields":fields})
versions={}
site=ROOT/".venv/lib/python3.12/site-packages"
for pattern in ["torch-*.dist-info","vllm-*.dist-info","py_kvcache-*.dist-info","triton-*.dist-info","transformers-*.dist-info","pytest-*.dist-info"]:
 for directory in site.glob(pattern):
  p=directory/"METADATA"
  if not p.exists():continue
  fields={}
  for line in p.read_text(errors="replace").splitlines():
   if line.startswith(("Name: ","Version: ")):
    k,v=line.split(": ",1);fields[k]=v
  versions[str(directory)]={**fields,"metadata":sha(p)}
ledgerpath=ROOT/"experiments/prefix_io_v1/gpu-budget-ledger.json"
ledger=json.loads(ledgerpath.read_text())
paths=[".","third_party/work/py-kvcache-p4-02-cpu","third_party/work/vllm-author-p4-02-cpu","third_party/work/vllm-author-build","third_party/work/prefix-io-p4-02-cpu"]
gitrows=[git(ROOT/p) for p in paths]
result={"schema_version":1,"status":"CPU_ONLY_G0_MIGRATION_AUDIT",
 "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"root":str(ROOT),"server":{"host":"connect.westc.seetacloud.com","ssh_port":29815},
 "python_version":sys.version,"python_executable":sys.executable,"package_metadata_without_import":versions,
 "git":gitrows,"PRIMARY":disk(ROOT),"AUX":auxrow,
 "cgroup":{"memory_max":pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip(),
 "memory_events":pathlib.Path("/sys/fs/cgroup/memory.events").read_text().strip()},
 "assigned_GPU_proc_metadata_only":inventory,"device_nodes":devices,
 "driver_version_proc_only":pathlib.Path("/proc/driver/nvidia/version").read_text(),
 "permissions":sha(ROOT/"experiments/prefix_io_v1/configs/permissions.yaml"),
 "GPU_budget":{"ledger":sha(ledgerpath),"gpu_wall_seconds":ledger["gpu_wall_seconds"],
 "remaining_8h_seconds":8*3600-ledger["gpu_wall_seconds"],"active_reservation":ledger["active_reservation"],
 "model_download_bytes":ledger["model_download_bytes"],"new_gpu_seconds":0},
 "historical_GPU_qualification_carried_forward_as_historical_only":True,
 "GPU_initialized":False,"GPU_runs":0,"NVML_calls":0,"server_mutations":0,
 "elapsed_seconds":time.monotonic()-start}
print(json.dumps(result,sort_keys=True))

