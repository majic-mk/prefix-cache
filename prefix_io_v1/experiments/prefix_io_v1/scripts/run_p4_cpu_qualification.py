"""P4 CPU integration qualification against the isolated native/strategy overlay."""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
import xml.etree.ElementTree as ET
from experiment_storage import ROOT, preflight
from run_gpu_stage import cleanup_session
from run_simple_stage_cpu_qualification_p316 import plan as p3_plan
from prepare_p4_gpu_stage import inspect_cpu_authorization, file_ref

OUT = "artifacts/prefix_io_v1/server08-p4-01-cpu"
def plan(name):
    if type(name) is not str or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in name):
        raise ValueError("explicit safe receipt name required")
    inspect_cpu_authorization(ROOT)
    out=ROOT/OUT/name
    scratch=ROOT/"experiments/prefix_io_v1/runs"/("server08-p4-cpu-"+name)/"cpu-evidence"
    row=p3_plan("p4-plan-only")["processes"][0]
    old=ROOT/"third_party/work/py-kvcache-p3-16-cpu"
    native=ROOT/"third_party/work/py-kvcache-p4-01-cpu"
    overlay=ROOT/"third_party/work/prefix-io-p4-01-cpu/src"
    row["name"]="p4-current-integration";row["worktree"]=str(native)
    row["environment_delta"]["PYTHONPATH"]=":".join(map(str,(native,overlay,ROOT,ROOT/"experiments/prefix_io_v1/scripts")))
    row["environment_delta"]["AIO_CPU_GPU_GUARD_PATH"]=str(out/"cuda-guard.json")
    row["command"]=[x.replace(str(old),str(native)) for x in row["command"]]
    for key, value in (("--basetemp",scratch/"pytest-tmp"),("--junitxml",out/"cpu.xml")):
        row["command"][row["command"].index(key)+1]=str(value)
    row["command"]+=["tests/prefix_io_v1_p4_policy","tests/prefix_io_v1_p4_bridge",
                      "tests/prefix_io_v1_p4_stage","tests/prefix_io_v1_p316_capacity_adapter",
                      "tests/prefix_io_v1_p316_capacity_analysis","tests/prefix_io_v1_p316_pilot_contract"]
    return dict(schema_version=1,output=str(out),scratch=str(scratch),process=row,
        historical_exclusions="12 frozen old fixture cases are historical P3 evidence only; not rerun or counted here.",
        scope="CPU/fake CUDA/native Python and real Linux-AIO tests; no real GPU/model/production cost qualification.",
        authorization=file_ref(ROOT,"artifacts/prefix_io_v1/server08-p4-01-cpu/CPU_ONLY_AUTHORIZATION.json"),
        test_input_lock=file_ref(ROOT,OUT+"/test-input-lock-v2.json"))
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--name",required=True);ap.add_argument("--dry-run",action="store_true");a=ap.parse_args()
    p=plan(a.name)
    if a.dry_run:print(json.dumps(p,indent=2));return 0
    ledger=ROOT/"experiments/prefix_io_v1/gpu-budget-ledger.json"; before=ledger.read_bytes()
    if json.loads(before)["active_reservation"] is not None:raise RuntimeError("CPU cannot overlap active GPU reservation")
    out=Path(p["output"]);scratch=Path(p["scratch"]);preflight(scratch,1024*1024*1024);out.mkdir(parents=True,exist_ok=False)
    scratch.mkdir(parents=True,exist_ok=False)
    (out/"plan.json").write_text(json.dumps(p,indent=2)+"\n")
    frozen=json.loads((ROOT/p["test_input_lock"]["path"]).read_text())["files"]
    for ref in frozen:
        if file_ref(ROOT,ref["path"]) != ref:raise RuntimeError("source changed before CPU run")
    row=p["process"];env=os.environ.copy();env.update(row["environment_delta"])
    start=time.monotonic();proc=None;error=None;code=1;cleanup=None
    try:
        with (out/"process.log").open("xb") as log:
            proc=subprocess.Popen(row["command"],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            code=proc.wait(timeout=row["maximum_seconds"])
    except BaseException as e:error=repr(e)
    finally:
        if proc is not None:
            cleanup=cleanup_session(proc)
            if not cleanup["session_drained"] or cleanup.get("session_members_before_cleanup"):code=1
    cases={}
    xml=out/"cpu.xml"
    if xml.exists():
        for c in ET.parse(xml).findall(".//testcase"):
            key=(c.attrib.get("classname"),c.attrib.get("name"))
            if key in cases:raise RuntimeError("duplicate test identity")
            cases[key]="failed" if c.find("failure") is not None or c.find("error") is not None else "skipped" if c.find("skipped") is not None else "passed"
    counts={v:sum(x==v for x in cases.values()) for v in ("passed","skipped","failed")}
    guard_path=out/"cuda-guard.json";guard=json.loads(guard_path.read_text()) if guard_path.exists() else None
    guard_ok=guard is not None and guard.get("cuda_initialized") is False and guard.get("gpu_workloads_run")==0
    unchanged=ledger.read_bytes()==before
    source_unchanged=all(file_ref(ROOT,ref["path"])==ref for ref in frozen)
    passed=code==0 and counts["failed"]==0 and len(cases)>1000 and guard_ok and unchanged and source_unchanged
    result=dict(status="PASS_P4_CPU_CURRENT_INTEGRATION" if passed else "FAIL_P4_CPU_CURRENT_INTEGRATION",
        process_exit=code,error=error,seconds=time.monotonic()-start,cleanup=cleanup,unique_counts=counts,
        guard=guard,guard_verified=guard_ok,new_gpu_runs=0,ledger_unchanged=unchanged,source_lock_unchanged=source_unchanged,
        test_input_lock=p["test_input_lock"],
        ledger_sha256=hashlib.sha256(before).hexdigest(),storage_after=preflight(scratch,0),
        current_source_scope=p["scope"],historical_exclusions=p["historical_exclusions"])
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2));return 0 if passed else 1
if __name__=="__main__":raise SystemExit(main())
