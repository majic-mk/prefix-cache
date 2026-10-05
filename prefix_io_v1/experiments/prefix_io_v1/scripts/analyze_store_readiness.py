"""Join bounded post-pump states to already qualified native destination fences."""
import argparse
import json
from pathlib import Path
from analyze_flush_diagnostic import analyze
from analyze_blocked_store_chains import fence
from analyze_mixed_start_budget import validate_reference

def require(ok,message):
    if not ok: raise ValueError(message)

def diagnose(case, chain, observation, *, run_id):
    require(chain["qualified"],"native fence chain is not qualified")
    require(observation["run_id"]==run_id and not observation["faulted"],"wrong or invalid observer")
    require(observation["mandatory_only"] is True,"unfrozen observation scope")
    require(observation["truncated_samples"]==0,"bounded parent view is incomplete")
    records=observation["records"]
    require(len(records)<=96,"record bound exceeded")
    wait=case["wait"]
    start=wait["start"]*1e9
    end=(wait["start"]+wait["seconds"])*1e9
    rows=[]
    last=-1
    for record in records:
        first=record["first_time_ns"];final=record["last_time_ns"]
        require(type(first) is int and type(final) is int and last<first<=final,"unordered sampled states")
        require(type(record["samples"]) is int and record["samples"]>0,"invalid sample count")
        last=final
    for parent in chain["parents"]:
        jid=parent["job_id"]
        cutoff=start+parent["copy_host_enqueue_from_wait_seconds"]*1e9
        selected=[]
        for record in records:
            if not start<=record["first_time_ns"]<=record["last_time_ns"]<min(end,cutoff):continue
            s=record["state"]
            require(not s["truncated"],"truncated retained state")
            matching=[p for p in s["stores"] if p["parent_id"]==jid]
            require(len(matching)<=1,"duplicate parent identity in snapshot")
            if not matching:continue
            p=matching[0]
            if not (p["mandatory"] and p["queued_files"]>0 and p["next_file_index"]==0
                    and p["inflight_files"]==0):continue
            require(not p["pending_counts_are_lower_bounds"],"partial parent stages")
            pos=s["stores"].index(p)
            preceding=[q for q in s["stores"][:pos]
                if not q["mandatory"] and q["queued_files"]>0 and q["total_files"]>1]
            selected.append(dict(first_ns=record["first_time_ns"],last_ns=record["last_time_ns"],
                samples=record["samples"],iodepth_full=s["data_inflight"]>=s["iodepth"],
                no_free_staging=s["free_staging_slots"]==0,
                preceding_large_background_parents=[q["parent_id"] for q in preceding],
                background_stage_ops=sum(q["d2h_pending_observed"]+q["ssd_write_pending_observed"] for q in preceding)))
        rows.append(dict(parent_id=jid,retained_state_groups=len(selected),
            retained_sample_count=sum(s["samples"] for s in selected),
            first_retained_sample_after_wait_seconds=(selected[0]["first_ns"]-start)/1e9 if selected else None,
            last_retained_sample_after_wait_seconds=(selected[-1]["last_ns"]-start)/1e9 if selected else None,
            all_selected_iodepth_full=all(s["iodepth_full"] for s in selected) if selected else None,
            all_selected_no_free_staging=all(s["no_free_staging"] for s in selected) if selected else None,
            all_selected_preceded_by_large_background=all(s["preceding_large_background_parents"] for s in selected) if selected else None,
            preceding_background_parent_ids=sorted({jid for s in selected for jid in s["preceding_large_background_parents"]}),
            minimum_background_stage_ops=min((s["background_stage_ops"] for s in selected),default=None),
            observations=selected))
    return dict(parents=rows,overwritten_state_groups=observation["overwritten"],
        complete_wait_coverage=False,exact_decision_denial_known=False,
        compute_event_readiness_known=False,reclaimable_staging_known=False,
        estimated_speedup=None,immediately_reusable_gpu_bytes=None,
        interpretation="Retained sampled states only. Background ordering and full device capacity coexist; no counterfactual timing or proof that staging cannot be reclaimed.")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--plan",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    plan=json.loads(a.plan.read_text())
    require(len(plan["runs"])==1,"one frozen diagnostic launch required")
    result=analyze(a.plan)
    require(result["status"]=="PASSED_BOUNDED_FLUSH_DIAGNOSTIC","reference/flush validation failed")
    entry=plan["runs"][0]
    folder=Path(entry["command"][entry["command"].index("--output")+1])
    native=json.loads((folder/"result.json").read_text())
    reference=json.loads((Path(plan["native_reference"])/"result.json").read_text())
    refs={(r["family"],r["kind"]):r["output_tokens"] for r in reference["rows"]}
    require(validate_reference(native,refs)==10,"reference incomplete")
    require(native["engine_shutdown"]=="completed" and native["store_readiness_probe"] is True,"wrong or undrained diagnostic")
    trace=json.loads((folder/"native-cohort.trace.json").read_text())
    ob=native["final_probe"]["store_readiness"]
    require(len(ob)==1,"single reactor required")
    probe=native["probe"]["flush_diagnostic"]
    require(not any(probe[k] for k in ("dropped_records","dropped_jobs","dropped_causes","errors","faulted")),"flush witness incomplete")
    row=result["runs"][0]
    row["store_fence_chains"]=[]
    row["readiness_diagnoses"]=[]
    for case in row["causal_waits"]:
        f=fence(case,trace,probe["records"])
        row["store_fence_chains"].append(f)
        row["readiness_diagnoses"].append(diagnose(case,f,ob[0],run_id=entry["label"]))
    result["interpretation"]="One frozen mixed native replay with passive flush/readiness observers. No performance comparison or research policy."
    result["status"]="PASS_MIXED_READINESS_DIAGNOSTIC"
    result["readiness_summary"]={k:v for k,v in ob[0].items() if k!="records"}
    with a.out.open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="runs"},indent=2))
    for d in row["readiness_diagnoses"]:
        print(json.dumps([{k:v for k,v in p.items() if k!="observations"} for p in d["parents"]],indent=2))
if __name__=="__main__":main()
