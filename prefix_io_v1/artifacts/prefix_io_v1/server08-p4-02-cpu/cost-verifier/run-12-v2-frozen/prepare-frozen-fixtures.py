from pathlib import Path
from dataclasses import asdict
from hashlib import sha256
import builtins
import json
import sys
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/"third_party/work/prefix-io-p4-02-cpu/src")]
original_import=builtins.__import__
attempts=[]
def cpu_import(name,*args,**kwargs):
    if name.split(".")[0] in ("torch","cuda","cupy","pycuda","vllm","py_kvcache"):
        attempts.append(name)
        raise AssertionError("GPU/backend import denied: "+name)
    return original_import(name,*args,**kwargs)
builtins.__import__=cpu_import
from tests.prefix_io_v1_p4_measurements.test_semantic_v2 import V2Fixture
from prefix_io_control.p4_paired_measurement_verifier import (
    load_verification_plan,build_paired_cell_candidate,write_semantic_verification_manifest,
)
from prefix_io_control.p4_production_table_contract import EvidenceRef
from prefix_io_control.p4_verified_cost_loader import load_semantically_verified_table
base=Path("artifacts/prefix_io_v1/server08-p4-02-cpu/cost-verifier/fixtures-v2-01")
base.mkdir(exist_ok=False,parents=True)
results=[]
for name,batch,origin in (("continuous-context-full128-selected1",1,"cpu_fixture"),
                         ("continuous-context-full128-selected2",2,"cpu_fixture"),
                         ("forged-native-forward-label",1,"native_gpu_recording")):
    dest=base/name
    dest.mkdir(exist_ok=False)
    f=V2Fixture(dest,batch=batch,origin=origin)
    plan=load_verification_plan(dest,"plan.json",expected_plan_ref=f.plan_ref)
    geometry={key:f.cell[key] for key in ("cell_id","load","existing_io","stage","physical_bytes")}
    refs={name:EvidenceRef.from_mapping(f.cell["measurement_refs"][name]) for name in
          ("baseline_wrapper","action_wrapper","baseline_observations","action_observations")}
    built=build_paired_cell_candidate(dest,cell_geometry=geometry,raw_refs=refs,
        expected_plan=plan,analysis_path="recomputed-analysis.json")
    cell=built.to_candidate_mapping()
    cref=f.dump("prepared-candidate.json",{"schema_version":1,"scope":"production_candidate",
        "context":asdict(f.context),"cells":[cell]})
    qref=f.dump("prepared-qualification.json",dict(f.qualification,candidate_ref=asdict(cref),
                  measurement_refs=list(cell["measurement_refs"].values())))
    args=f.args();args["qualification_ref"]=qref
    prepared=load_semantically_verified_table(dest,"prepared-candidate.json",**args)
    manifest_ref=write_semantic_verification_manifest(dest,prepared.verification,"cpu-semantic-manifest.json")
    cost=prepared.verification.cells[0].cost
    result={"root":str(dest.resolve()),"context":asdict(f.context),"plan_ref":asdict(f.plan_ref),
        "candidate_ref":asdict(cref),"qualification_ref":asdict(qref),"verifier_ref":asdict(f.verifier_ref),
        "original_candidate_ref":asdict(f.candidate_ref),"original_qualification_ref":asdict(f.qualification_ref),
        "manifest_ref":asdict(manifest_ref),"status":prepared.status,"scope":"CPU_FIXTURE_NOT_REAL_GPU",
        "production_qualified":False,"gpu_verified":False,
        "production_lookup":prepared.lookup(cost.load_signature,cost.existing_io,"h2d",16,execution="production"),
        "full_output_tokens":128,"selected_output_tokens":batch,
        "cell":asdict(prepared.verification.cells[0])}
    f.dump("result.json",result)
    results.append(result)
(base/"result.json").write_text(json.dumps({"gpu_operations":0,"backend_import_attempts":attempts,
    "production_qualified":False,"fixtures":results},sort_keys=True,indent=2)+"\n")
print(json.dumps({"fixtures":len(results),"gpu_operations":0,"backend_import_attempts":attempts,
                  "production_qualified":False,"path":str(base.resolve())}))
