"""Analyze the frozen mixed start-budget integration screen; no efficacy claim."""
import argparse,json
from pathlib import Path
from analyze_start_budget_model import analyze,require

def validate_reference(result,references):
    require(result["profile"]=="mixed_readwrite","not the frozen mixed workload")
    require(result["source_preservation"]==dict(checked_files=3048,changed=[]),"source mutation")
    matches=0
    for row in result["rows"]:
        key=(row["family"],"cold" if row["num_cached_tokens"]==0 else "gpu_hot")
        require(key in references and row["output_tokens"]==references[key],"native full-reference token mismatch")
        matches+=1
    require(matches==10,"incomplete reference check")
    for h in result["final_probe"]["handlers"]:
        a=h["aio"]
        require(h["observation_failures"]==0 and h["staging_bytes"]<=1073741824,"observer/staging failure")
        require(a["accepted"]==a["completed"]==a["reaped"] and
            all(a[k]==0 for k in ("outstanding","pending","ready","unreaped")) and a["fatal"] is None,"AIO imbalance")
    return matches

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--plan",type=Path,required=True)
    ap.add_argument("--completed",type=int,choices=(1,2,3),required=True);ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();plan=json.loads(a.plan.read_text())
    require([r["mode"] for r in plan["runs"]]==["off","fixed","pressure"],"wrong frozen order")
    native=json.loads((Path(plan["native_reference"])/"result.json").read_text())
    require(native["status"]=="PASSED_NATIVE_C2_FULL_REFERENCE","reference unqualified")
    refs={(r["family"],r["kind"]):r["output_tokens"] for r in native["rows"]}
    paths={r["mode"]:Path(r["command"][r["command"].index("--output")+1])/"result.json" for r in plan["runs"][:a.completed]}
    result=analyze(paths)
    for mode,path in paths.items():
        result["results"][mode]["native_reference_exact_count"]=validate_reference(json.loads(path.read_text()),refs)
    result["complete"]=a.completed==3
    result["status"]="PASS_MIXED_MODEL_START_INTEGRATION" if result["complete"] else "PARTIAL_MIXED_MODEL_START_INTEGRATION"
    result["limitation"]="One mixed run per file-chain start-budget arm in fixed order. Full stage caps absent; accounting overhead not isolated. No dependency/interference/joint policy or causal performance conclusion."
    with a.out.open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="results"},indent=2))
    print(json.dumps({mode:{k:v for k,v in row.items() if k!="native_state"} for mode,row in result["results"].items()},indent=2))
if __name__=="__main__":main()
