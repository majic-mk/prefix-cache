"""Supplemental max_num_seqs=2 gate; old candidate and thresholds remain unchanged."""
import argparse,hashlib,json
from pathlib import Path
from validate_heldout_costs import costs
from concurrent_pilot_contract import DELTA
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"artifacts/prefix_io_v1/server07-p3-06"
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_gate(path):
    permit=json.loads(Path(path).read_text())
    assert permit["status"]=="PASSED_C2_UNLOADED_POINT_GATE"
    assert permit["engine_delta"]==DELTA and permit["maximum_absolute_relative_median_error"]==.25
    cp=ROOT/permit["cost_plan"]
    assert digest(cp)==permit["cost_plan_sha256"]
    plan=json.loads(cp.read_text())
    assert plan["maximum_absolute_relative_median_error"]==.25 and plan["engine_delta"]==DELTA
    result=costs(ROOT,cp,engine_delta=DELTA)
    assert result["status"]=="PASSED_HELDOUT_POINT_PREDICTION"
    recorded=ROOT/permit["cost_result"]
    assert digest(recorded)==permit["cost_result_sha256"] and result==json.loads(recorded.read_text())
    assert digest(ROOT/plan["candidate"])==permit["candidate_sha256"]
    for item in permit["manifests"].values():assert digest(ROOT/item["path"])==item["sha256"]
    return permit
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    cp=OUT/"cost-plan.json";plan=json.loads(cp.read_text())
    assert plan["maximum_absolute_relative_median_error"]==.25 and plan["engine_delta"]==DELTA
    r=costs(ROOT,cp,engine_delta=DELTA)
    with (OUT/"cost-result.json").open("x") as f:json.dump(r,f,indent=2)
    if not r["status"].startswith("PASSED"):
        print(json.dumps(r,indent=2));raise SystemExit(1)
    permit=dict(status="PASSED_C2_UNLOADED_POINT_GATE",engine_delta=DELTA,
        runtime_scope="Bounded two-sequence controlled development screen only; loaded prediction, unseen content and formal evaluation unqualified",
        maximum_absolute_relative_median_error=.25,cost_plan=str(cp.relative_to(ROOT)),cost_plan_sha256=digest(cp),
        cost_result=str((OUT/"cost-result.json").relative_to(ROOT)),cost_result_sha256=digest(OUT/"cost-result.json"),
        candidate_sha256=plan["candidate_sha256"],manifests=json.loads((OUT/"manifest-index.json").read_text()),
        ordinary_quotas_installed=False,formal_goodput=False)
    with a.output.open("x") as f:json.dump(permit,f,indent=2)
    assert verify_gate(a.output)==permit
    print(json.dumps(dict(status=permit["status"],checks=r["checks"]),indent=2))
