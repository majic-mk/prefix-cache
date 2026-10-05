"""Seal CPU-tested conditional native entry code, not a GPU qualification."""
from __future__ import annotations
import argparse,datetime,gzip,hashlib,importlib.util,io,json,pathlib,shutil,subprocess,sys,tarfile
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("_entry_freezer",HERE/"freeze_gpu_entry.py")
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
require=F.require;read=F.read;put=F.put;sha=F.sha;verify=F.verify;reference=F.reference
def pass_tests(doc,kind,locksha):
    require(type(doc.get("tests")) is int and doc["tests"]>0 and doc.get("passed")==doc["tests"],kind+" tests")
    require(doc.get("status","").startswith("PASS") and doc.get("failed")==doc.get("errors")==doc.get("skipped")==0,kind+" all pass")
    require(doc.get("location")=="server_cpu" and doc.get("source_before")==doc.get("source_after") and doc.get("source_lock_sha256")==locksha,kind+" actual frozen sources")
    require(doc.get("gpu_uuid") is None and all(doc.get(k) is False for k in F.FLAGS),kind+" no GPU qualification")
    require(doc.get("actual_gpu_runs",doc.get("gpu_runs"))==0,kind+" no GPU run")
    require(doc.get("forbidden_imports",doc.get("forbidden_modules_imported",[]))==[],kind+" no model/native imports")
    for k in ("valid_native_receipt","effective_cost_upper_ns","effective_step_budget_ns"):require(doc.get(k) is None,kind+" no native grant")
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=pathlib.Path,required=True)
    a=parser.parse_args();root=a.root.resolve(strict=True);base=root/"artifacts/prefix_io_v1"
    dirs={k:base/v for k,v in F.NAMES.items()}
    require(HERE==dirs["delivery"],"fixed server sealer")
    beforepath=HERE/"SESSION_BEFORE_GPU_ENTRY_PREPARATION.json";before=read(beforepath)
    for row in before["refs"]:verify(root,row)
    lockpath=HERE/"SOURCE_LOCK_GPU_ENTRY_CPU.json";lock=read(lockpath);locksha=sha(lockpath.read_bytes())
    freeze=read(HERE/"SOURCE_FREEZE_RECEIPT_CPU.json")
    require(reference(root,lockpath)==freeze["source_lock_ref"] and reference(root,beforepath)==freeze["baseline_ref"],"anchored locks")
    require(lock["schema"]=="c5_gpu_entry_cpu_source_lock_v1" and lock["gpu_uuid"] is None and all(lock[k] is False for k in F.FLAGS),"no inherited GPU grant")
    for row in lock["files"]:verify(root,row)
    native=read(dirs["candidate"]/"SERVER_NATIVE_CPU_01/CPU_NATIVE_ENTRY_RESULT.json")
    binding=read(dirs["candidate"]/"SERVER_BINDING_CPU_01/CPU_BINDING_RESULT.json")
    review=read(dirs["review"]/"SERVER_REVIEW_CPU_01/TEST_RESULT.json")
    for kind,value in (("native",native),("binding",binding),("review",review)):pass_tests(value,kind,locksha)
    ledgerpath=root/"experiments/prefix_io_v1/gpu-budget-ledger.json";lb=ledgerpath.read_bytes();ledger=read(ledgerpath)
    require(sha(lb)==before["ledger_sha256"] and ledger["gpu_wall_seconds"]==before["gpu_wall_seconds"] and ledger["active_reservation"] is None,"unchanged GPU accounting")
    git=subprocess.run(["git","status","--porcelain","--untracked-files=no"],cwd=root,capture_output=True,text=True,check=True).stdout
    require(git==before["tracked_git_status"],"tracked user code unchanged")
    for d in dirs.values():
        for p in d.rglob("*"):
            require(not p.is_symlink(),"archive scope symlink")
            require(p.name not in ("GPU_LAUNCH_INTENT.json","GPU_LAUNCH_RECEIPT.json","SIX_PROCESS_AUTHORIZED_SCOPE.json","NATIVE_COST_CONFIG.json","SITE_SOURCE_LOCK.json","SITE_BINDING.json"),"no actual GPU launch/config authority in CPU preparation")
    available={}
    for d in dirs.values():
        for p in sorted(d.rglob("*")):
            if p.is_file():
                b=p.read_bytes();available.setdefault((len(b),sha(b)),p.relative_to(base).as_posix())
    snapshots=HERE/"SOURCE_DEPENDENCY_SNAPSHOTS";snapshots.mkdir(exist_ok=False);mapping=[]
    for row in lock["files"]:
        p=verify(root,row);key=(row["bytes"],row["sha256"])
        if key not in available:
            target=snapshots/(row["sha256"]+".bin")
            with target.open("xb") as stream:stream.write(p.read_bytes())
            available[key]=target.relative_to(base).as_posix()
        mapping.append(dict(row,snapshot_path=available[key]))
    put(HERE/"SOURCE_SNAPSHOT_MAP.json",{"source_rows":len(mapping),"files":mapping,"gpu_runs":0,"unique_external_snapshots":len(list(snapshots.iterdir()))})
    readiness={"schema":"c5_gpu_entry_conditional_readiness_v1","status":"CODE_PREPARED_RESOURCES_AND_AUTHORIZATION_PENDING",
               "gpu_entry_code_prepared":True,"cpu_targeted_tests_passed":native["tests"]+binding["tests"]+review["tests"],
               "gpu_launch_allowed":False,"native_execution_verified":False,"native_cost_qualified":False,
               "full_runtime_cost_qualified":False,"on_observation_cost_measured":False,
               "gpu_uuid":None,"valid_native_receipt":None,"effective_cost_upper_ns":None,"effective_step_budget_ns":None,
               "gpu_runs":0,"gpu_seconds":0,"source_rows":len(lock["files"]),"source_lock_sha256":locksha,
               "ledger_sha256":sha(lb),"remaining_gpu_seconds":28800-ledger["gpu_wall_seconds"],
               "pending":["visible sufficient GPU/CPU/memory","actual same-site context","new source/job/UUID/permission binding under explicit human GPU execution grant","real common six-window raw and canonical receipt","real off/shadow/on full costs","P4 fair comparison"]}
    put(HERE/"GPU_ENTRY_READINESS.json",readiness)
    audit=dict(readiness,status="PASS_CPU_GATED_NATIVE_ENTRY_CODE_ONLY",utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               native_tests=native["tests"],binding_tests=binding["tests"],review_tests=review["tests"],
               prior_refs_verified=len(before["refs"]),prior_evidence_unchanged=True,source_rows=len(lock["files"]),
               gpu_nodes=sorted(str(p) for p in pathlib.Path("/dev").glob("nvidia*")),
               cpu_max=pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
               memory_max=pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip(),
               disk_free_bytes=shutil.disk_usage(root).free,command=[sys.executable]+sys.argv)
    put(HERE/"SERVER_FINAL_AUDIT.json",audit)
    content={}
    for d in dirs.values():
        for p in sorted(d.rglob("*")):
            if p.is_file():
                require(not p.is_symlink() and p.resolve().is_relative_to(d) and p.stat().st_size<=10*1024**2,"bounded actual evidence")
                content[p.relative_to(base).as_posix()]=p.read_bytes()
    require(len(content)<=500 and sum(map(len,content.values()))<=48*1024**2,"bounded archive")
    manifest={"scope":"THREE_NEW_GATED_NATIVE_ENTRY_DIRECTORIES","data_file_count":len(content),"raw_bytes":sum(map(len,content.values())),"GPU_runs":0,"formal_benchmark_runs":0,
              "files":[{"path":name,"bytes":len(b),"sha256":sha(b)} for name,b in sorted(content.items())]}
    mb=(json.dumps(manifest,sort_keys=True,indent=2)+"\n").encode()
    archive=HERE/"C5_GPU_ENTRY_PREPARATION_EVIDENCE.tar.gz"
    with archive.open("xb") as stream,gzip.GzipFile(fileobj=stream,mode="wb",mtime=0) as gz,tarfile.open(fileobj=gz,mode="w|",format=tarfile.USTAR_FORMAT) as tar:
        for name,b in [("ARCHIVE_CONTENTS_MANIFEST.json",mb)]+sorted(content.items()):
            info=tarfile.TarInfo(name);info.size=len(b);info.mode=0o600;info.mtime=0;tar.addfile(info,io.BytesIO(b))
    require(archive.stat().st_size<=40*1024**2,"compressed cap")
    for name,b in content.items():require((base/name).read_bytes()==b,"evidence changed while sealing")
    for row in before["refs"]:verify(root,row)
    for row in lock["files"]:verify(root,row)
    require(ledgerpath.read_bytes()==lb,"ledger changed while sealing")
    receipt={"status":"PASS_SEALED_CPU_GATED_NATIVE_ENTRY","archive":{"file":archive.name,"bytes":archive.stat().st_size,"sha256":sha(archive.read_bytes())},
             "manifest":{"file":"ARCHIVE_CONTENTS_MANIFEST.json","bytes":len(mb),"sha256":sha(mb)},
             "data_files":len(content),"raw_bytes":manifest["raw_bytes"],"source_lock_sha256":locksha,"source_snapshots":len(mapping),
             "unique_external_snapshots":len(list(snapshots.iterdir())),"prior_references_verified":len(before["refs"]),"GPU_runs":0,"formal_benchmark_runs":0,
             "gpu_launch_allowed":False,"gpu_ledger_sha256":sha(lb),"native_cost_qualified":False}
    put(HERE/"SERVER_BACKUP_RECEIPT.json",receipt)
    print(json.dumps(receipt,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())

