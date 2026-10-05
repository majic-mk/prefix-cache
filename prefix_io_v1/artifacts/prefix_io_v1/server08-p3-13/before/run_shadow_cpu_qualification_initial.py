"""Preflight the CPU-only four-stage audit regression on authorized primary storage."""
import argparse,json,os,re,subprocess,sys,time
from pathlib import Path
from experiment_storage import ROOT,preflight
from run_gpu_stage import cleanup_session

RESERVATION=640*1024**2  # Observed one-matrix scratch ~222 MiB plus headroom.
def plan(label):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,80}",label):
        raise ValueError("label must be a simple unique name")
    out=ROOT/"experiments/prefix_io_v1/runs"/label/"cpu-evidence"
    if out.exists():raise FileExistsError("CPU evidence labels are append-only")
    disk=preflight(out,RESERVATION)
    work=ROOT/"third_party/work/py-kvcache-p3-shadow-cpu"
    suites=["tests/prefix_io_v1/test_config.py","tests/prefix_io_v1_pilot",
        "tests/prefix_io_v1_native_costs","tests/prefix_io_v1_progress",
        "tests/prefix_io_v1_observer","tests/prefix_io_v1_aio","tests/prefix_io_v1_start_budget","tests/prefix_io_v1_store_order","tests/prefix_io_v1_dispatch_shadow",
        str(work/"tests")]
    cmd=[str(ROOT/".venv/bin/python"),str(ROOT/"experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py"),
         "-q","--import-mode=importlib",*suites,"--ignore="+str(work/"tests/test_e2e_kvcache.py"),
         "--basetemp",str(out/"pytest-tmp"),"--junitxml",str(out/"cpu.xml")]
    env=dict(CUDA_VISIBLE_DEVICES="",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=":".join(map(str,[work,ROOT/"src",ROOT/"experiments/prefix_io_v1/scripts"])),
        AIO_CPU_GPU_GUARD_PATH=str(out/"cuda-guard.json"))
    return dict(output=str(out),disk=disk,command=cmd,environment_delta=env,
                gpu_runs=0,maximum_seconds=180,scope="CPU-only, CUDA initialization forbidden")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--label",required=True);ap.add_argument("--dry-run",action="store_true")
    a=ap.parse_args();p=plan(a.label)
    if a.dry_run:print(json.dumps(dict(p,executed=False),indent=2));return 0
    out=Path(p["output"]);out.mkdir(parents=True,exist_ok=False)
    (out/"plan.json").write_text(json.dumps(p,indent=2))
    env=os.environ.copy();env.update(p["environment_delta"])
    proc=None;code=1;error=None;cleanup=None;start=time.monotonic()
    try:
        with (out/"process.log").open("wb") as log:
            proc=subprocess.Popen(p["command"],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            code=proc.wait(timeout=p["maximum_seconds"])
    except BaseException as exc:error=type(exc).__name__+": "+str(exc)
    finally:
        if proc is not None:
            cleanup=cleanup_session(proc)
            if not cleanup["session_drained"] or cleanup.get("session_members_before_cleanup"):code=1
        result=dict(exit=code,error=error,cleanup=cleanup,seconds=time.monotonic()-start,
                    gpu_runs=0,executed=proc is not None)
        try:result["storage_after"]=preflight(out,0)
        except Exception as exc:result.update(exit=1,storage_error=str(exc));code=1
        (out/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return code
if __name__=="__main__":raise SystemExit(main())
