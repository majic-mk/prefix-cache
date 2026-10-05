"""Recheck independently preserved pre-fix counterexamples; no new GPU data."""
from pathlib import Path
from hashlib import sha256
import json, sys, builtins
root=Path.cwd()
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split(".")[0] in ("torch","cuda","cupy","pycuda","vllm","py_kvcache"):
        raise RuntimeError("GPU/backend import forbidden for CPU negative replay")
    return original(name,*args,**kwargs)
builtins.__import__=guarded
sys.path.insert(0,str(root/"third_party/work/prefix-io-p4-02-cpu/src"))
from prefix_io_control.p4_production_table_contract import TableContext,EvidenceRef,TableContractError
from prefix_io_control.p4_verified_cost_loader import load_semantically_verified_table
base=root/"artifacts/prefix_io_v1/server08-p4-02-cpu/eta/review-cost/counterexamples-01"
results=[]
for name in ("workload-hash-overlap","cross-cell-split-overlap","invalid-active-batch"):
    dest=base/name
    candidate=json.loads((dest/"candidate.json").read_text())
    report=json.loads((dest/"qualification.json").read_text())
    plan_raw=(dest/"plan.json").read_bytes()
    plan_ref=EvidenceRef("plan.json",len(plan_raw),sha256(plan_raw).hexdigest())
    qualification_raw=(dest/"qualification.json").read_bytes()
    qref=EvidenceRef("qualification.json",len(qualification_raw),sha256(qualification_raw).hexdigest())
    try:
        p=load_semantically_verified_table(dest,"candidate.json",
           expected_context=TableContext.from_mapping(candidate["context"]),
           qualification_ref=qref,
           expected_verifier_ref=EvidenceRef.from_mapping(report["verifier_ref"]),
           plan_path="plan.json",expected_plan_ref=plan_ref)
    except TableContractError as exc:
        results.append({"independent_counterexample":name,"accepted":False,"rejection":str(exc)})
    else:
        results.append({"independent_counterexample":name,"accepted":True,
                        "production_qualified":p.production_qualified})
out=root/"artifacts/prefix_io_v1/server08-p4-02-cpu/cost-verifier/independent-counterexample-replay-v2.json"
out.write_text(json.dumps({"status":"PASS_ALL_PREVIOUS_COUNTEREXAMPLES_REJECTED" if
    all(not r["accepted"] for r in results) else "FAIL_COUNTEREXAMPLE_STILL_ACCEPTED",
    "gpu_runs":0,"results":results},indent=2)+"\n")
print(out.read_text())
raise SystemExit(0 if all(not r["accepted"] for r in results) else 1)
