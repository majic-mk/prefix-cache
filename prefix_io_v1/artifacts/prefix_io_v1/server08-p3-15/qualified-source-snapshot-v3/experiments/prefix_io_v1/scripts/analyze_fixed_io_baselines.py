"""Compare frozen existing I/O-depth configurations; no research benefit claim."""
import argparse,json,statistics
from pathlib import Path
from analyze_concurrent_pilot import summarize,digest
from qualify_concurrent_pilot import verify_gate
ROOT=Path(__file__).resolve().parents[3]
def analyze(plan_path):
    plan=json.loads(Path(plan_path).read_text())
    native=Path(plan["native_reference"])
    nr=json.loads((native/"result.json").read_text())
    assert nr["status"]=="PASSED_NATIVE_C2_FULL_REFERENCE" and len(nr["rows"])==10
    refs={(x["family"],x["kind"]):x["output_tokens"] for x in nr["rows"]}
    groups={};workload=None
    for item in plan["groups"]:
        depth=item["io_depth"]
        assert type(depth) is int and depth in (2,4,8)
        permit=verify_gate(ROOT/item["permit"])
        assert permit.get("connector_delta",{}).get("iodepth",4)==depth
        runs=[];outputs=[]
        for folder in item["runs"]:
            folder=Path(folder)
            config=json.loads((folder/"frozen-config.json").read_text())
            manifest=json.loads((folder/"manifest.json").read_text())
            assert config["engine"]["kv_transfer_config"]["kv_connector_extra_config"]["iodepth"]==depth
            assert manifest.get("io_depth",4)==depth
            m={k:v for k,v in manifest.items() if k not in ("io_depth","baseline_tuning")}
            if workload is None:workload=m
            assert m==workload,"workload drift"
            info=permit["manifests"]["mixed_readwrite"]
            assert digest(folder/"manifest.json")==info["sha256"]
            run=summarize(folder,refs)
            assert run["profile"]=="mixed_readwrite"
            outputs.append(run.pop("outputs"));runs.append(run)
        assert len(runs)==2
        groups[str(depth)]=dict(runs=runs,outputs_repeat_exact=outputs[0]==outputs[1],
            cohort_median_seconds=statistics.median(r["cohort_seconds"] for r in runs),
            pending_flush_runs=sum(r["counters"]["pending_flush_wait_calls"]>0 for r in runs),
            native_reference_exact_count=sum(r["native_reference_exact_count"] for r in runs),
            itl_p95_median_seconds=statistics.median(r["itl"]["p95"] for r in runs))
    baseline=groups["4"]["cohort_median_seconds"]
    for group in groups.values():
        group["descriptive_cohort_reduction_vs_depth4"]=1-group["cohort_median_seconds"]/baseline
    exact=sum(g["native_reference_exact_count"] for g in groups.values())
    return dict(status="PASSED_FIXED_IO_DEPTH_OUTPUT_SCREEN" if exact==20*len(groups) and
        all(g["outputs_repeat_exact"] for g in groups.values()) else "NUMERIC_DIAGNOSIS_REQUIRED",
        groups=groups,total_exact_requests=exact,total_token_events=1280*2*len(groups),
        new_policy_installed=False,research_benefit_measured=False,formal_goodput=False,
        planned_performance_order=plan.get("planned_performance_order",plan.get("performance_order")),
        executed_performance_order=plan.get("performance_order"),
        blocked_configurations=plan.get("blocked_configurations",[]),
        limitations="Two repeats per eligible setting; depth 4 was measured earlier. Unqualified configurations are recorded, not replayed. Small controlled burst, no CI or SLO. Depth labels change only existing I/O depth. Temporal IO association is not a causal interference measurement.",
        plan_sha256=digest(plan_path),native_reference_sha256=digest(native/"result.json"))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    r=analyze(a.plan)
    with a.out.open("x") as f:json.dump(r,f,indent=2)
    print(json.dumps({k:v for k,v in r.items() if k!="groups"}))
    for depth,g in r["groups"].items():
        print(json.dumps(dict(io_depth=depth,**{k:v for k,v in g.items() if k!="runs"})))
        for run in g["runs"]:
            print(json.dumps({k:run[k] for k in ("label","cohort_seconds","ssd_read_bytes","ssd_write_bytes","counters","itl","native_reference_exact_count")}))
    raise SystemExit(0 if r["status"].startswith("PASSED") else 1)
