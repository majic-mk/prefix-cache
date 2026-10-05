"""Prepare a candidate from preserved synthetic raw bytes, then full revalidation."""
from pathlib import Path
from hashlib import sha256
from dataclasses import asdict
import json,sys,builtins,shutil
root=Path.cwd()
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split(".")[0] in ("torch","cuda","cupy","pycuda","vllm","py_kvcache"):
        raise RuntimeError("GPU/backend import forbidden for CPU builder demonstration")
    return original(name,*args,**kwargs)
builtins.__import__=guarded
sys.path.insert(0,str(root/"third_party/work/prefix-io-p4-02-cpu/src"))
from prefix_io_control.p4_production_table_contract import EvidenceRef,TableContext
from prefix_io_control.p4_paired_measurement_verifier import (
    load_verification_plan, build_paired_cell_candidate, write_semantic_verification_manifest)
from prefix_io_control.p4_verified_cost_loader import load_semantically_verified_table
base=root/"artifacts/prefix_io_v1/server08-p4-02-cpu/cost-verifier"
source=base/"fixtures/cpu_fixture"
dest=base/"builder-roundtrip-01"
dest.mkdir(exist_ok=False)
for name in ("split.json","plan.json","baseline_wrapper.json","action_wrapper.json",
             "baseline_observations.json","action_observations.json","verifier.py"):
    shutil.copyfile(source/name,dest/name)
def ref(name):
    data=(dest/name).read_bytes()
    return EvidenceRef(name,len(data),sha256(data).hexdigest())
data=json.loads((source/"candidate.json").read_text())
cell=data["cells"][0]
plan_ref=ref("plan.json")
plan=load_verification_plan(dest,"plan.json",expected_plan_ref=plan_ref)
geometry={name:cell[name] for name in ("cell_id","load","existing_io","stage","physical_bytes")}
raw_refs={name:ref(name+".json") for name in ("baseline_wrapper","action_wrapper",
                                           "baseline_observations","action_observations")}
built=build_paired_cell_candidate(dest,cell_geometry=geometry,raw_refs=raw_refs,
                                 expected_plan=plan,analysis_path="built-analysis.json")
candidate={"schema_version":1,"scope":"production_candidate","context":asdict(plan.context),
           "cells":[built.to_candidate_mapping()]}
(dest/"built-candidate.json").write_text(json.dumps(candidate,indent=2)+"\n")
verifier_ref=ref("verifier.py")
report={"schema_version":1,"scope":"production_qualification_candidate",
        "status":"PENDING_INDEPENDENT_VERIFICATION","candidate_ref":asdict(ref("built-candidate.json")),
        "context":asdict(plan.context),"verifier_ref":asdict(verifier_ref),
        "measurement_refs":list(candidate["cells"][0]["measurement_refs"].values())}
(dest/"binding-report.json").write_text(json.dumps(report,indent=2)+"\n")
table=load_semantically_verified_table(dest,"built-candidate.json",expected_context=plan.context,
      qualification_ref=ref("binding-report.json"),expected_verifier_ref=verifier_ref,
      plan_path="plan.json",expected_plan_ref=plan_ref)
receipt=write_semantic_verification_manifest(dest,table.verification,"cpu-semantic-receipt.json")
c=table.verification.cells[0].cost
result={"status":"PASS_CPU_RAW_BUILDER_FULL_LOADER_ROUNDTRIP",
 "actual_execution":"CPU_SYNTHETIC_METADATA_ONLY","gpu_runs":0,"gpu_verified":False,
 "production_qualified":False,"builder_cost_equals_final_cost":built.verification.cost==c,
 "production_lookup":table.lookup(c.load_signature,c.existing_io,c.stage,c.physical_bytes),
 "candidate_ref":asdict(ref("built-candidate.json")),"analysis_ref":asdict(built.analysis_ref),
 "semantic_receipt_ref":asdict(receipt)}
(dest/"result.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
