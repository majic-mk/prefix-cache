"""Read all frozen ABBA arms, preserving failures and reporting descriptive effects."""
import argparse,json,statistics
from pathlib import Path
from analyze_concurrent_pilot import summarize,digest,distribution
from analyze_flush_diagnostic import match_batch
from analyze_mixed_start_budget import validate_reference
from run_store_order_pilot import validate_result
from store_order_model_contract import require,SCOPE

def comparisons(runs):
    result={}
    for key in ("cohort_seconds","response_seconds","tail_drain_seconds","pending_flush_wait_seconds",
                "scheduled_ttft_mean_seconds","ttft_mean_seconds","itl_p95_seconds"):
        arms={mode:[r[key] for r in runs if r["mode"]==mode] for mode in ("off","pressure")}
        require(all(len(v)==2 for v in arms.values()),"two frozen repeats per arm required")
        off=statistics.mean(arms["off"]);on=statistics.mean(arms["pressure"])
        result[key]=dict(values=arms,off_mean=off,pressure_mean=on,
            reduction_percent=(off-on)/off*100 if off else None,
            pair_reduction_percent=[(a-b)/a*100 if a else None for a,b in
                [(arms["off"][0],arms["pressure"][0]),(arms["off"][1],arms["pressure"][1])]],
            inference="descriptive only; two repeats do not establish statistical significance")
    return result

def normalized_config(config):
    from copy import deepcopy
    c=deepcopy(config);extra=c["engine"]["kv_transfer_config"]["kv_connector_extra_config"]
    extra.pop("shared_storage_path");extra.pop("prefix_io_observation_run_id")
    return c

def analyze(plan_path,completed):
    plan=json.loads(Path(plan_path).read_text())
    require(plan["scope"]==SCOPE and plan["run_order"]==["off","pressure","pressure","off"],"wrong fixed design")
    require(1<=completed<=4,"bad completed count")
    nr=json.loads((Path(plan["native_reference"])/"result.json").read_text())
    require(nr["status"]=="PASSED_NATIVE_C2_FULL_REFERENCE","unqualified reference")
    refs={(r["family"],r["kind"]):r["output_tokens"] for r in nr["rows"]}
    rows=[];configs=[]
    for entry in plan["runs"][:completed]:
        folder=Path(entry["output"]);raw=json.loads((folder/"result.json").read_text())
        outer=json.loads((folder/"order-result.json").read_text());mode=entry["actual_order_mode"]
        require(outer["status"]=="PASS_REAL_MODEL_STORE_ORDER" and outer["actual_order_mode"]==mode,"outer qualification failed")
        require(outer["raw_result_sha256"]==digest(folder/"result.json"),"raw result changed")
        states=validate_result(raw,mode);exact=validate_reference(raw,refs)
        row=summarize(folder,refs);row.pop("outputs")
        require(row["native_reference_exact_count"]==exact==10,"token mismatch")
        before=raw["cohort_probe_start"];after=raw["probe"]
        read=raw["final_probe"]["store_readiness"]
        require(bool(read) and all(not p["faulted"] for p in read),"readiness fault")
        diag=after["flush_diagnostic"]
        require(not diag["faulted"] and diag["errors"]==0 and diag["allocations"]>0,"flush observer fault")
        batches=[r for r in diag["records"] if r["kind"]=="flush_batch"]
        row.update(mode=mode,response_seconds=raw["response_seconds"],
            pending_flush_wait_seconds=row["counters"]["pending_flush_wait_seconds"],
            scheduled_ttft_mean_seconds=statistics.mean(r["scheduled_ttft_seconds"] for r in row["rows"]),
            ttft_mean_seconds=statistics.mean(r["ttft_seconds"] for r in row["rows"]),
            itl_p95_seconds=row["itl"]["p95"],order_states=states,
            qualified_mode_receipt_sha256=digest(folder/"order-result.json"),
            causal_waits=[match_batch(wait,batches) for wait in row["pending_wait_events"]],
            readiness_summaries=[{k:v for k,v in p.items() if k!="records"} for p in read],
            actual_raw_config_policy="shadow is observation mode; actual order is separately verified",
            exact_reference_outputs=exact)
        rows.append(row)
        configs.append(normalized_config(json.loads((folder/"frozen-config.json").read_text())))
    require(all(c==configs[0] for c in configs),"underlying frozen model configs differ")
    complete=completed==4
    return dict(status="PASS_STORE_ORDER_ABBA" if complete else "PARTIAL_STORE_ORDER_ABBA",
        complete=complete,scope=SCOPE,runs=rows,comparison=comparisons(rows) if complete else None,
        actual_requests=completed*10,actual_token_events=completed*1280,native_reference_exact_outputs=completed*10,
        native_reference_sha256=digest(Path(plan["native_reference"])/"result.json"),
        simple_priority_installed=True,full_stage_caps=False,ordinary_start_quota=False,
        research_joint_efficacy_verified=False,full_P3_complete=False,formal_goodput=False,
        limitation="Fixed mixed workload, depth 8, two instrumented repeats per arm. No confidence claim, no joint-policy result; CUDA device timeline absent. Ordering does not itself prove immediate GPU-memory release.")

def main():
    p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--completed",type=int,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();result=analyze(a.plan,a.completed)
    with a.out.open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="runs"},indent=2))
    print(json.dumps([{k:r[k] for k in ("label","mode","cohort_seconds","pending_flush_wait_seconds",
           "response_seconds","tail_drain_seconds","ssd_read_bytes","ssd_write_bytes","order_states")} for r in result["runs"]],indent=2))
if __name__=="__main__":main()
