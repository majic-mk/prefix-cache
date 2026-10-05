"""Explicit, serialized first-round GPU job budget; no automatic model downloads."""
import argparse, fcntl, json, os, signal, subprocess, time
from pathlib import Path
import yaml

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--seconds", type=int, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[3]
    if not a.label.replace("-","").replace("_","").isalnum():
        p.error("invalid label")
    command=a.command[1:] if a.command[:1]==["--"] else a.command
    if not command or not 1<=a.seconds<=3600:
        p.error("command and per-job seconds 1..3600 required")
    permission=yaml.safe_load((root/"experiments/prefix_io_v1/configs/permissions.yaml").read_text())
    if permission["allow_gpu_runs"] is not True or permission["max_gpu_hours"] is None:
        raise RuntimeError("GPU operation not authorized")
    if len(permission["approved_gpu_ids"]) != 1:
        raise RuntimeError("first-round runner requires exactly one approved GPU UUID")
    runroot=Path(permission["approved_experiment_root"]).resolve()
    if not runroot.is_relative_to(root):
        raise RuntimeError("experiment root outside project")
    runroot.mkdir(exist_ok=True,parents=True)
    ledger=root/"experiments/prefix_io_v1/gpu-budget-ledger.json"
    with (ledger.with_suffix(".lock")).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        budget=json.loads(ledger.read_text())
        if budget["gpu_wall_seconds"]+a.seconds+20 > permission["max_gpu_hours"]*3600:
            raise RuntimeError("insufficient cumulative GPU budget including termination reserve")
        run=runroot/a.label
        run.mkdir(exist_ok=False)
        env=os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"]=permission["approved_gpu_ids"][0]
        env["PYTHONDONTWRITEBYTECODE"]="1"
        env["HF_HUB_OFFLINE"]="1"
        env["TRANSFORMERS_OFFLINE"]="1"  # Download is a separate explicitly budgeted operation.
        started=time.time()
        timed_out=False
        with (run/"process.log").open("wb") as log:
            proc=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=proc.wait(timeout=a.seconds)
            except subprocess.TimeoutExpired:
                timed_out=True
                os.killpg(proc.pid,signal.SIGTERM)
                try: code=proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
        elapsed=time.time()-started
        event={"label":a.label,"command":command,"gpu_uuid":env["CUDA_VISIBLE_DEVICES"],
               "started_unix":started,"elapsed_seconds":elapsed,"exit":code,"timed_out":timed_out,
               "evidence":str(run),"gpu_job_attempted":True}
        budget["gpu_wall_seconds"]+=elapsed
        budget["events"].append(event)
        temp=ledger.with_suffix(".tmp");temp.write_text(json.dumps(budget,indent=2)+"\n");temp.replace(ledger)
        (run/"result.json").write_text(json.dumps(event,indent=2)+"\n")
        print(json.dumps(event,indent=2))
        return 124 if timed_out else code

if __name__=="__main__":
    raise SystemExit(main())
