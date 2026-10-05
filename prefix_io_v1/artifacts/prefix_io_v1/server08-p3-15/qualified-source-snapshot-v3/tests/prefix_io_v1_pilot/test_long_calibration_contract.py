import copy,importlib.util,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"experiments/prefix_io_v1/scripts"))
from long_calibration_contract import validate
from export_uniform_calibration_candidate import verify_measurement
def fixture():
    engine=dict(kv_cache_memory_bytes=2147483648,max_model_len=16400,max_num_batched_tokens=16400,max_num_seqs=1)
    tokens=[18384]+[1000+(i%500) for i in range(16255)]+[777]
    manifest=dict(calibration_integration_only=True,formal_goodput_enabled=False,slo=None,engine=engine,
                  staging_budget_bytes=1073741824,output_tokens=128,
                  requests=[dict(prompt_token_ids=tokens,scheduled_time=0.,calibration_family=0)])
    curve=dict(provenance=dict(candidate_only=True,allowed_consumer="same-budget calibration integration diagnostic only",
        gpu_uuid="GPU-test",engine=copy.deepcopy(engine),minimum_supported_prefix_tokens=2048,max_supported_model_len=16384))
    return manifest,curve
def test_bounded_same_budget_candidate():
    m,c=fixture()
    assert validate(m,c,"GPU-test")==m["engine"]
@pytest.mark.parametrize("case",["gpu","formal","budget","context","family","domain","arrival","production"])
def test_candidate_scope_cannot_escape(case):
    m,c=fixture();gpu="GPU-test"
    if case=="gpu":gpu="GPU-other"
    elif case=="formal":m["formal_goodput_enabled"]=True
    elif case=="budget":m["engine"]["kv_cache_memory_bytes"]*=2
    elif case=="context":m["requests"][0]["prompt_token_ids"]+=[1]*200
    elif case=="family":m["requests"][0]["prompt_token_ids"][0]=999
    elif case=="domain":c["provenance"]["max_supported_model_len"]=8192
    elif case=="arrival":m["requests"][0]["scheduled_time"]=float("nan")
    elif case=="production":c["provenance"]["candidate_only"]=False
    with pytest.raises(ValueError):validate(m,c,gpu)
def test_cold_measured_reference_keeps_warmup():
    r=dict(kind="f",rep=0,warmup=True,prompt_token_ids=list(range(17)),output_token_ids=[5],
           num_cached_tokens=0,metrics=dict(is_corrupted=False,first_token_latency=.1))
    ref=copy.deepcopy(r);verify_measurement(r,ref,16)
    r["output_token_ids"]=[6]
    with pytest.raises(ValueError,match="token reference"):verify_measurement(r,ref,16)
