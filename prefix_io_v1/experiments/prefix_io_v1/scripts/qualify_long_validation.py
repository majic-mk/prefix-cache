"""Emit a narrow diagnostic permit only after the frozen held-out prediction gate."""
import argparse,hashlib,json
from pathlib import Path
from validate_heldout_costs import costs
from analyze_repeated_native_costs import require
from experiment_storage import authorized_path
p=argparse.ArgumentParser()
for name in ("plan","result","out"):p.add_argument("--"+name,type=Path,required=True)
a=p.parse_args();root=Path.cwd();plan=json.loads(a.plan.read_text())
result=costs(root,a.plan);saved=json.loads(a.result.read_text())
require(result==saved,"prediction evidence changed")
require(plan["maximum_absolute_relative_median_error"]==.25,"frozen threshold changed")
rp=json.loads((root/plan["reference_plan"]).read_text())
require(rp["sizes"]==[16256] and rp["reps"]==3,"long domain mismatch")
permit=dict(status="PASSED_LONG_DIAGNOSTIC_GATE" if result["status"]=="PASSED_HELDOUT_POINT_PREDICTION" else "BLOCKED_COST_PREDICTION_FAILED",
 runtime_scope="same-budget independent long validation only",validated_prefix_tokens=16256,
 curves_sha256=plan["candidate_sha256"],prompt_manifest_sha256=plan["prompt_manifest_sha256"],
 cost_plan=str(a.plan),cost_plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
 cost_result=str(a.result),cost_result_sha256=hashlib.sha256(a.result.read_bytes()).hexdigest(),
 storage_source=str(authorized_path(rp["external_groups"][0]["storage_path"])[0]),
 ordinary_policy_installed=False,full_P3_complete=False,formal_evaluation=False,
 qualification_note="Supplemental permit for one validated prefix length and exact budget; the candidate itself remains unchanged and unqualified for production/full-domain use.")
with a.out.open("x") as f:json.dump(permit,f,indent=2)
print(json.dumps(permit,indent=2))
raise SystemExit(0 if permit["status"]=="PASSED_LONG_DIAGNOSTIC_GATE" else 1)
