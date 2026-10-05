"""Offline qualification analysis; single-trial timings are descriptive only."""
import argparse,hashlib,json
from pathlib import Path
from prefix_io_control.token_timeline import percentile
from experiment_storage import ROOT

def require(ok,message):
    if not ok:raise ValueError(message)

def analyze(paths):
    results={};reference=None;workload_hash=None
    for mode,path in paths.items():
        r=json.loads(Path(path).read_text())
        require(r["status"].startswith("PASSED_") and r.get("engine_shutdown")=="completed",mode+" failed")
        require(r["policy_mode"]==mode and r["full_per_stage_caps"] is False,"wrong experiment scope")
        require(len(r["rows"])==10,"incomplete cohort")
        rows={v["request_id"]:v for v in r["rows"]}
        require(len(rows)==10 and all(v["per_token_complete"] and len(v["output_tokens"])==128 for v in rows.values()),"invalid token evidence")
        tokens={k:v["output_tokens"] for k,v in rows.items()}
        if reference is None:reference=tokens;workload_hash=r["manifest_sha256"]
        require(tokens==reference,"output token mismatch in "+mode)
        require(r["manifest_sha256"]==workload_hash,"workload differs")
        native=r["final_probe"]["start_budget_final"]
        require(native and all(v["drained"] for v in native),"native completion missing")
        for state in native:
            b=state["budget"];a=state["accounting"]
            if mode=="off":require(b is None and a is None,"off created optional state")
            else:
                require(b is not None and not b["faulted"] and b["errors"]==0,"quota fault")
                require(a is not None and a["valid"] and a["outstanding_records"]==0,"accounting fault")
                require(all(v["accepted_bytes"]==v["transferred_bytes"] and v["failed_ops"]==v["inflight_bytes"]==0
                    for v in a["stages"].values()),"stage imbalance")
        results[mode]=dict(result_path=str(path),result_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            requests=10,output_tokens=1280,all_output_tokens_equal_off=True,
            cohort_seconds=r["cohort_seconds_including_drain"],tail_drain_seconds=r["tail_drain_seconds"],
            ttft_p95_seconds=percentile([v["metrics"]["first_token_latency"] for v in rows.values()],.95),
            request_itl_p95_median_seconds=percentile([v["itl_p95_seconds"] for v in rows.values()],.5),
            request_itl_p95_max_seconds=max(v["itl_p95_seconds"] for v in rows.values()),
            real_ssd_read_bytes=r["cohort_read_bytes"],real_ssd_write_bytes=r["cohort_write_bytes"],
            pending_flush_wait_calls=r["probe"]["pending_flush_wait_calls"],
            pending_flush_wait_seconds=r["probe"]["pending_flush_wait_seconds"],
            native_state=native)
    off=results["off"]["cohort_seconds"]
    for r in results.values():r["single_trial_cohort_change_percent_vs_off"]=100*(r["cohort_seconds"]/off-1)
    return dict(status="PASS_REAL_MODEL_START_INTEGRATION",results=results,formal_goodput=False,
        repeats_per_arm=1,order=list(paths),confidence_interval=None,research_policy_enabled=False,
        speedup_established=False,full_P3_complete=False,
        limitation="Single all-hit negative-control/integration run per start-budget arm; no joint/dependency/interference policy, no mixed-target efficacy conclusion")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    paths={mode:Path("/root/prefix-io-v1-validation/runs")/("server07-p3-09-model-"+mode)/"details/result.json" for mode in ("off","fixed","pressure")}
    result=analyze(paths)
    with a.output.open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
