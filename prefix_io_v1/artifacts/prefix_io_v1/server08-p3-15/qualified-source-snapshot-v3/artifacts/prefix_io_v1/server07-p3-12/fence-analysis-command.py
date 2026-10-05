from pathlib import Path
import json,sys
sys.path[:0]=["experiments/prefix_io_v1/scripts","src"]
from analyze_blocked_store_chains import fence
from analyze_store_readiness import diagnose
out=Path("artifacts/prefix_io_v1/server07-p3-12")
plan=json.loads((out/"model-plan.json").read_text())
analysis=json.loads((out/"model-analysis.json").read_text())
evidence=[]
for e,row in zip(plan["runs"],analysis["runs"]):
    folder=Path(e["output"]);raw=json.loads((folder/"result.json").read_text())
    trace=json.loads((folder/"native-cohort.trace.json").read_text())
    probe=raw["probe"]["flush_diagnostic"];obs=raw["final_probe"]["store_readiness"]
    assert not any(probe[k] for k in ("dropped_records","dropped_jobs","dropped_causes","errors","faulted"))
    entry=dict(label=e["label"],mode=e["actual_order_mode"],fences=[])
    for case in row["causal_waits"]:
        chain=fence(case,trace,probe["records"])
        d=diagnose(case,chain,obs[0],run_id=e["label"]) if chain["qualified"] else None
        entry["fences"].append(dict(chain=chain,readiness=d))
    evidence.append(entry)
result=dict(runs=evidence,counterfactual_speedup_proven=False,physical_gpu_release_bytes_known=False,
            scope="All four pre-registered runs; no filtering for favorable timing")
assert result==json.loads((out/"fence-analysis.json").read_text())
with (out/"fence-analysis-reproduction.json").open("x") as f:
    json.dump(dict(exact_reproduction=True,source="fence-analysis-command.py",runs=4),f,indent=2)
print("exact reproduction of all four fence analyses")
