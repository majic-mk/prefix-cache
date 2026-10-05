"""Current P4-02 CPU regression, guarded against GPU and source/ledger drift."""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
import xml.etree.ElementTree as ET
from experiment_storage import ROOT, preflight
from run_gpu_stage import cleanup_session
from run_simple_stage_cpu_qualification_p316 import plan as p3_plan

OUT="artifacts/prefix_io_v1/server08-p4-02-cpu"
NATIVE="third_party/work/py-kvcache-p4-02-cpu"
CONTROL="third_party/work/prefix-io-p4-02-cpu/src"
NEW_TESTS=("tests/prefix_io_v1_p4_policy","tests/prefix_io_v1_p4_02_bridge",
    "tests/prefix_io_v1_p4_stage","tests/prefix_io_v1_p4_eta",
    "tests/prefix_io_v1_p4_measurements","tests/prefix_io_v1_p4_load",
    "tests/prefix_io_v1_p4_next_day","tests/prefix_io_v1_p4_hybrid",
    "tests/prefix_io_v1_p4_observation")

def ref(path):
    p=ROOT/path;b=p.read_bytes()
    return dict(path=path,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def cpu_scope():
    auth_path=OUT+"/CPU_ONLY_AUTHORIZATION.json"
    auth=json.loads((ROOT/auth_path).read_text())
    if auth.get("allow_cpu_tests") is not True or auth.get("allow_project_local_edits") is not True:
        raise ValueError("current CPU scope absent")
    for key in ("allow_gpu_initialization","allow_gpu_runs","allow_model_downloads",
                "allow_driver_or_system_changes","allow_storage_deletion_or_merge"):
        if auth.get(key) is not False:raise ValueError("current scope must remain CPU only")
    p=auth["base_permissions"]
    if ref(p["path"])!=p:raise ValueError("base permissions changed")
    return ref(auth_path)

def safe_name(name):
    if type(name) is not str or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in name):
        raise ValueError("explicit safe receipt name required")

def plan(name,only_new=False,source_lock=OUT+"/test-input-lock.json"):
    safe_name(name)
    authorization=cpu_scope()
    out=ROOT/OUT/name
    scratch=ROOT/"experiments/prefix_io_v1/runs"/("server08-p4-02-"+name)/"cpu-evidence"
    row=p3_plan("p4-02-plan-only")["processes"][0]
    old=str(ROOT/"third_party/work/py-kvcache-p3-16-cpu")
    row["name"]="p4-02-current-integration";row["worktree"]=str(ROOT/NATIVE)
    row["environment_delta"]["PYTHONPATH"]=":".join(map(str,(ROOT/NATIVE,ROOT/CONTROL,ROOT,ROOT/"experiments/prefix_io_v1/scripts")))
    row["environment_delta"]["AIO_CPU_GPU_GUARD_PATH"]=str(out/"cuda-guard.json")
    row["environment_delta"]["TMPDIR"]=str(scratch/"tmp")
    row["command"]=[x.replace(old,str(ROOT/NATIVE)) for x in row["command"]]
    for key,value in (("--basetemp",scratch/"pytest-tmp"),("--junitxml",out/"cpu.xml")):
        row["command"][row["command"].index(key)+1]=str(value)
    if only_new:
        cut=row["command"].index("--junitxml")+2
        row["command"]=row["command"][:cut]
    for directory in NEW_TESTS:
        if not (ROOT/directory).is_dir():raise ValueError("new CPU tests not yet frozen: "+directory)
    row["command"]+=list(NEW_TESTS)
    if not only_new:
        row["command"]+=["tests/prefix_io_v1_p316_capacity_adapter",
            "tests/prefix_io_v1_p316_capacity_analysis","tests/prefix_io_v1_p316_pilot_contract"]
    lock=source_lock
    if (type(lock) is not str or not lock.startswith(OUT+"/") or "\\" in lock or
            any(x in ("",".","..") for x in lock.split("/")) or not lock.endswith(".json")):
        raise ValueError("bounded explicit CPU source lock required")
    lock_ref=ref(lock) if (ROOT/lock).exists() else None
    return dict(schema_version=1,output=str(out),scratch=str(scratch),process=row,
        authorization=authorization,test_input_lock=lock_ref,
        scope="CPU owner/scheduler/fake CUDA and actual Linux-AIO; no real GPU/model cost qualification.",
        historical_exclusions="12 frozen old P3 fixture cases historical only; not rerun/counted.",
        original_author_gpu_e2e_suite_run=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--name",required=True)
    ap.add_argument("--dry-run",action="store_true");ap.add_argument("--only-new",action="store_true")
    ap.add_argument("--source-lock",default=OUT+"/test-input-lock.json")
    a=ap.parse_args();p=plan(a.name,a.only_new,a.source_lock)
    if a.dry_run:
        print(json.dumps(p,indent=2));return 0
    if p["test_input_lock"] is None:raise ValueError("source freeze required before test launch")
    frozen=json.loads((ROOT/p["test_input_lock"]["path"]).read_text())["files"]
    if not all(ref(x["path"])==x for x in frozen):raise RuntimeError("source changed before CPU run")
    ledger=ROOT/"experiments/prefix_io_v1/gpu-budget-ledger.json";before=ledger.read_bytes()
    if json.loads(before)["active_reservation"] is not None:raise RuntimeError("CPU cannot overlap active GPU reservation")
    out=Path(p["output"]);scratch=Path(p["scratch"])
    preflight(scratch,1024*1024*1024);out.mkdir(parents=True,exist_ok=False)
    scratch.mkdir(parents=True,exist_ok=False);(scratch/"tmp").mkdir()
    (out/"plan.json").write_text(json.dumps(p,indent=2)+"\n")
    row=p["process"];env=os.environ.copy();env.update(row["environment_delta"])
    start=time.monotonic();proc=None;error=None;code=1;cleanup=None
    try:
        with (out/"process.log").open("xb") as log:
            proc=subprocess.Popen(row["command"],cwd=ROOT,env=env,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
            code=proc.wait(timeout=row["maximum_seconds"])
    except BaseException as exc:error=repr(exc)
    finally:
        if proc is not None:
            cleanup=cleanup_session(proc)
            if not cleanup["session_drained"] or cleanup.get("session_members_before_cleanup"):code=1
    cases={};xml=out/"cpu.xml"
    if xml.exists():
        for c in ET.parse(xml).findall(".//testcase"):
            key=(c.attrib.get("classname"),c.attrib.get("name"))
            if key in cases:raise RuntimeError("duplicate test identity")
            cases[key]="failed" if c.find("failure") is not None or c.find("error") is not None else "skipped" if c.find("skipped") is not None else "passed"
    counts={v:sum(x==v for x in cases.values()) for v in ("passed","skipped","failed")}
    g=out/"cuda-guard.json";guard=json.loads(g.read_text()) if g.exists() else None
    guard_ok=guard is not None and guard.get("cuda_initialized") is False and guard.get("gpu_workloads_run")==0
    unchanged=ledger.read_bytes()==before
    source_unchanged=all(ref(x["path"])==x for x in frozen)
    passed=code==0 and counts["failed"]==0 and len(cases)>(50 if a.only_new else 1000) and guard_ok and unchanged and source_unchanged
    overhead=[]
    for filename in (scratch/"pytest-tmp").rglob("p4-cpu-observation-overhead.json"):
        overhead.append(json.loads(filename.read_text()))
    (out/"cpu-observation-overhead.json").write_text(json.dumps(overhead,indent=2)+"\n")
    result=dict(status="PASS_P4_02_CPU_CURRENT_INTEGRATION" if passed else "FAIL_P4_02_CPU_CURRENT_INTEGRATION",
        process_exit=code,error=error,seconds=time.monotonic()-start,cleanup=cleanup,unique_counts=counts,
        guard=guard,guard_verified=guard_ok,new_gpu_runs=0,ledger_unchanged=unchanged,
        source_lock_unchanged=source_unchanged,test_input_lock=p["test_input_lock"],
        ledger_sha256=hashlib.sha256(before).hexdigest(),storage_after=preflight(scratch,0),
        cpu_overhead_measurement_scope="scalar fixture host microbenchmark only, no GPU/model/end-to-end result",
        current_source_scope=p["scope"],historical_exclusions=p["historical_exclusions"])
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2));return 0 if passed else 1

if __name__=="__main__":raise SystemExit(main())
