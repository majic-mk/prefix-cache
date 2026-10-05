"""Analyze bounded native flush witnesses against exact-output references."""
import argparse,json
from pathlib import Path
from analyze_concurrent_pilot import summarize

def match_batch(wait, batches):
    candidates=[r for r in batches if not r["parents_truncated"] and not wait["ids_truncated"]
                and set(r["parent_ids"])==set(wait["ids"]) and r["monotonic"]<=wait["start"]]
    unique=len(candidates)==1
    return dict(wait=wait,unique_match=unique,batch=candidates[0] if unique else None,
                candidate_count=len(candidates))

def analyze(plan_path):
    plan=json.loads(Path(plan_path).read_text())
    nr=json.loads((Path(plan["native_reference"])/"result.json").read_text())
    assert nr["status"]=="PASSED_NATIVE_C2_FULL_REFERENCE"
    refs={(r["family"],r["kind"]):r["output_tokens"] for r in nr["rows"]}
    results=[]
    for entry in plan["runs"]:
        folder=Path(entry["command"][entry["command"].index("--output")+1])
        if not (folder/"result.json").exists():continue
        native=json.loads((folder/"result.json").read_text())
        row=summarize(folder,refs);row.pop("outputs")
        row.update(depth=entry["depth"],flush_cause_probe=entry["probe"])
        probe=native["probe"].get("flush_diagnostic")
        row["causal_waits"]=[]
        if entry["probe"]:
            assert probe and not probe["faulted"] and probe["errors"]==0 and probe["allocations"]>0
            batches=[r for r in probe["records"] if r["kind"]=="flush_batch"]
            row["owner_summary"]={k:v for k,v in probe.items() if k!="records"}
            for wait in row["pending_wait_events"]:
                row["causal_waits"].append(match_batch(wait,batches))
        else:
            assert probe is None
        results.append(row)
    exact=sum(r["native_reference_exact_count"] for r in results)
    return dict(status="PASSED_BOUNDED_FLUSH_DIAGNOSTIC" if len(results)==len(plan["runs"]) and exact==10*len(results)
                else "INCOMPLETE_OR_NUMERIC_FAILURE",runs=results,
        actual_requests=len(results)*10,actual_token_events=len(results)*1280,
        exact_reference_outputs=exact,new_policy_installed=False,research_benefit_measured=False,
        full_P3_complete=False,formal_goodput=False,
        real_flush_batches=sum(r.get("owner_summary",{}).get("flush_batches",0) for r in results),
        gpu_flush_cause_branch_exercised=any(r.get("owner_summary",{}).get("flush_batches",0)>0 for r in results),
        uniquely_identified_waits=sum(c["unique_match"] for r in results for c in r["causal_waits"]),
        unknown_release_bytes=True,observer_overhead_target_qualified=False,
        interpretation="One off/on all-hit pair screens compatibility only. Instrumented mixed timings are diagnostic, not an unobserved baseline or strategy benefit estimate.")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--plan",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    r=analyze(a.plan)
    with Path(a.out).open("x") as f:json.dump(r,f,indent=2)
    print(json.dumps({k:v for k,v in r.items() if k!="runs"},indent=2))
    for row in r["runs"]:
        print(json.dumps({k:v for k,v in row.items() if k in ("label","cohort_seconds","itl","counters","native_reference_exact_count","owner_summary","causal_waits")}))
