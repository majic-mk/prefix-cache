"""Freeze the supplementary all-hit baseline at the already-qualified C2/d8 domain."""
import copy, json
from pathlib import Path
import pytest
from concurrent_pilot_contract import validate
ROOT=Path(__file__).resolve().parents[2]

def fixture():
    m=json.loads((ROOT/"artifacts/prefix_io_v1/server08-p3-15/all_hit-manifest.json").read_text())
    c=json.loads((ROOT/"artifacts/prefix_io_v1/server08-p3-14/calibration-candidate/curves-v2.json").read_text())
    return m,c

def test_all_hit_same_cost_domain_tokens_and_arrivals():
    m,c=fixture();original=json.loads((ROOT/"artifacts/prefix_io_v1/server07-p3-06/all_hit-manifest.json").read_text())
    for key in ("families","requests","arrival_seed","arrival_rate"):
        assert m[key]==original[key]
    assert validate(m,c,m["gpu_uuid"])==m["engine"]
    assert m["formal_goodput"] is False and m["slo"] is None

@pytest.mark.parametrize("kind",["not_tuned","depth","gpu","new_engine","workload","slo"])
def test_supplemental_registration_does_not_expand_domain(kind):
    m,c=fixture()
    if kind=="not_tuned":m["baseline_tuning"]=False
    elif kind=="depth":m["io_depth"]=16
    elif kind=="gpu":m["gpu_uuid"]="GPU-other"
    elif kind=="new_engine":m["engine"]["max_num_seqs"]=4
    elif kind=="workload":m["requests"][0]["family_index"]=3
    elif kind=="slo":m["slo"]={"ttft":1}
    with pytest.raises(ValueError):validate(m,c,m["gpu_uuid"])
