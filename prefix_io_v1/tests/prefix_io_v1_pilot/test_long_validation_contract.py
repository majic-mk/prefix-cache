import copy,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"experiments/prefix_io_v1/scripts"))
from long_validation_contract import validate

def example():
    c=json.loads((ROOT/"artifacts/prefix_io_v1/server07-p3-03/calibration-candidate/curves-v2.json").read_text())
    e={k:v for k,v in c["provenance"]["engine"].items() if k!="kv_transfer_config"}
    p=dict(partition="validation",sizes=[16256],reps=3,
           prompts=[dict(family="validation-long",token_ids=[9876]*16257)])
    q=dict(status="PASSED_LONG_DIAGNOSTIC_GATE",runtime_scope="same-budget independent long validation only",validated_prefix_tokens=16256)
    m=dict(independent_validation_only=True,formal_goodput_enabled=False,slo=None,
      engine=e,staging_budget_bytes=1073741824,output_tokens=128,new_policy="off",original_load_planner="on",
      requests=[dict(request_id=0,prompt_token_ids=p["prompts"][0]["token_ids"],validation_family="validation-long",scheduled_time=0.)])
    return m,c,q,p,c["provenance"]["gpu_uuid"]
def test_frozen_diagnostic_contract():
    values=example();assert validate(*values)==values[0]["engine"]
@pytest.mark.parametrize("case",["unqualified","formal","quota","planner","length","family","time","duplicate","budget"])
def test_unqualified_or_changed_contract_rejected(case):
    m,c,q,p,g=example()
    if case=="unqualified":q["status"]="FAILED"
    if case=="formal":m["formal_goodput_enabled"]=True
    if case=="quota":m["new_policy"]="joint"
    if case=="planner":m["original_load_planner"]="off"
    if case=="length":m["requests"][0]["prompt_token_ids"]=m["requests"][0]["prompt_token_ids"]+[1]
    if case=="family":m["requests"][0]["validation_family"]="calibration"
    if case=="time":m["requests"][0]["scheduled_time"]=float("nan")
    if case=="duplicate":m["requests"].append(copy.deepcopy(m["requests"][0]))
    if case=="budget":m["engine"]["max_num_seqs"]=2
    with pytest.raises(ValueError):validate(m,c,q,p,g)
