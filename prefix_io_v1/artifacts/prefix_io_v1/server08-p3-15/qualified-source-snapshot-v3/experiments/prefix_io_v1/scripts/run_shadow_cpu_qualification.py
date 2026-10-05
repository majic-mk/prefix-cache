"""Qualify shadow native paths and old model guards in separate CPU processes."""
import argparse,json,os,re,subprocess,time
from pathlib import Path
from experiment_storage import ROOT,preflight
from run_gpu_stage import cleanup_session

RESERVATION=640*1024**2
LEGACY_TESTS=[
 "tests/prefix_io_v1_store_order/test_model_order.py::test_worker_model_boundary_uses_actual_native_order[off]",
 "tests/prefix_io_v1_store_order/test_model_order.py::test_worker_model_boundary_uses_actual_native_order[pressure]",
 "tests/prefix_io_v1_store_order/test_model_order.py::test_worker_will_not_treat_native_fallback_as_success"]
def plan(label):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,80}",label):
        raise ValueError("simple unique label required")
    out=ROOT/"experiments/prefix_io_v1/runs"/label/"cpu-evidence"
    if out.exists():raise FileExistsError("append-only CPU evidence")
    disk=preflight(out,RESERVATION)
    shadow=ROOT/"third_party/work/py-kvcache-p3-shadow-cpu"
    old=ROOT/"third_party/work/py-kvcache-p3-order-cpu"
    suites=["tests/prefix_io_v1/test_config.py","tests/prefix_io_v1_pilot",
        "tests/prefix_io_v1_native_costs","tests/prefix_io_v1_progress",
        "tests/prefix_io_v1_observer","tests/prefix_io_v1_aio",
        "tests/prefix_io_v1_start_budget","tests/prefix_io_v1_store_order",
        "tests/prefix_io_v1_dispatch_shadow",str(shadow/"tests")]
    base=[str(ROOT/".venv/bin/python"),str(ROOT/"experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py"),
        "-q","--import-mode=importlib"]
    rows=[]
    for name,work,tests,options in [
        ("shadow",shadow,suites,["--ignore="+str(shadow/"tests/test_e2e_kvcache.py")]+
            ["--deselect="+test for test in LEGACY_TESTS]),
        ("legacy-model-guard",old,LEGACY_TESTS,[])]:
        env=dict(CUDA_VISIBLE_DEVICES="",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=":".join(map(str,[work,ROOT/"src",ROOT/"experiments/prefix_io_v1/scripts"])),
            AIO_CPU_GPU_GUARD_PATH=str(out/(name+"-cuda-guard.json")))
        cmd=base+tests+options+["--basetemp",str(out/(name+"-pytest-tmp")),
            "--junitxml",str(out/(name+"-cpu.xml"))]
        rows.append(dict(name=name,worktree=str(work),command=cmd,environment_delta=env,maximum_seconds=180))
    return dict(output=str(out),disk=disk,processes=rows,gpu_runs=0,
        legacy_tests_routed_to_their_frozen_worktree=LEGACY_TESTS,
        legacy_model_runtime_guard_unchanged=True,scope="CPU only; CUDA initialization forbidden")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--label",required=True);ap.add_argument("--dry-run",action="store_true")
    a=ap.parse_args();p=plan(a.label)
    if a.dry_run:print(json.dumps(dict(p,executed=False),indent=2));return 0
    out=Path(p["output"]);out.mkdir(parents=True,exist_ok=False)
    (out/"plan.json").write_text(json.dumps(p,indent=2))
    results=[]
    for row in p["processes"]:
        proc=None;code=1;error=None;cleanup=None;start=time.monotonic()
        env=os.environ.copy();env.update(row["environment_delta"])
        try:
            with (out/(row["name"]+"-process.log")).open("wb") as log:
                proc=subprocess.Popen(row["command"],cwd=ROOT,env=env,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
                code=proc.wait(timeout=row["maximum_seconds"])
        except BaseException as exc:error=type(exc).__name__+": "+str(exc)
        finally:
            if proc is not None:
                cleanup=cleanup_session(proc)
                if not cleanup["session_drained"] or cleanup.get("session_members_before_cleanup"):code=1
            results.append(dict(name=row["name"],exit=code,error=error,cleanup=cleanup,
                seconds=time.monotonic()-start,executed=proc is not None,gpu_runs=0))
    result=dict(exit=int(any(x["exit"]!=0 for x in results)),processes=results,gpu_runs=0)
    try:result["storage_after"]=preflight(out,0)
    except Exception as exc:result.update(exit=1,storage_error=str(exc))
    (out/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return result["exit"]
if __name__=="__main__":raise SystemExit(main())
