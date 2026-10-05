"""CPU rejection tests: prevent mixed budgets, partial I/O or mismatch from qualification."""
import copy, importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("repeat_checks",ROOT/"experiments/prefix_io_v1/scripts/analyze_repeated_native_costs.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def fixture():
    n=16
    handler=dict(pinned=True,staging_bytes=917504,staging_budget=1048576,
                 io_size=917504,storage_block_bytes=917504,
                 aio=dict(accepted=1,completed=1,reaped=1,outstanding=0,pending=0,
                          ready=0,unreaped=0,fatal=None))
    def row(kind,rep):
        return dict(kind=kind,prefix_tokens=n,rep=rep,warmup=rep==0,
            prompt_token_ids=[rep]+[1]*16,output_token_ids=[2],num_cached_tokens=0 if kind in ("f","store") else n,
            metrics=dict(first_token_latency=1.0,is_corrupted=False),
            drain={} if kind=="f" else dict(handlers=[copy.deepcopy(handler)]),
            trace=dict(transfer_success=True,h2d_events=1,load_transfers=[dict(num_bytes=n*57344)],
                       foreground_logical_read_bytes=n*57344 if kind=="g_ssd" else 0,
                       preload_actual_read_bytes=0))
    result=[]
    for mode,kinds in (("cold",["f"]),("paired",["g_ssd","g_mem"]),("populate",["store","g_mem"])):
        result.append(dict(mode=mode,status="PASSED_NATIVE_"+mode.upper()+"_ACQUISITION",
            engine_shutdown="completed",final_drain={},
            rows=[row(kind,rep) for rep in range(3) for kind in kinds]))
    return result

def test_valid_synthetic_pair():
    r=module.check_pair(*fixture(),16,3)
    assert r["checked_prompt_pairs"]==3 and r["ssd_read_bytes"]==3*16*57344

@pytest.mark.parametrize("case",["warmup_output","partial_io","wrong_medium","corruption","duplicate","warmup_label"])
def test_rejects_invalid_evidence(case):
    cold,paired,populate=fixture()
    r=paired["rows"][0]
    if case=="warmup_output": r["output_token_ids"]=[3]
    elif case=="partial_io": r["drain"]["handlers"][0]["aio"]["outstanding"]=1
    elif case=="wrong_medium": r["trace"]["foreground_logical_read_bytes"]=0
    elif case=="corruption": r["metrics"]["is_corrupted"]=True
    elif case=="duplicate": paired["rows"].append(copy.deepcopy(r))
    elif case=="warmup_label": r["warmup"]=False
    with pytest.raises(ValueError): module.check_pair(cold,paired,populate,16,3)

def config():
    return dict(model={"revision":"locked"},model_alias="same",alias_target="/same",gpu_uuid="GPU-test",
                sizes=[16],reps=3,sampling={},pythonhashseed="0",acquisition_domain=1024,
                engine=dict(kv_cache_memory_bytes=100,kv_transfer_config=dict(staging_mem=1)))
@pytest.mark.parametrize("case",["kv_budget","staging_budget","model","logprobs","diagnostic"])
def test_rejects_mixed_configuration(case):
    a=config();b=copy.deepcopy(a)
    if case=="kv_budget": b["engine"]["kv_cache_memory_bytes"]=200
    elif case=="staging_budget": b["engine"]["kv_transfer_config"]["staging_mem"]=2
    elif case=="model": b["model"]["revision"]="other"
    elif case=="logprobs": b["sampling"]["logprobs"]=5
    elif case=="diagnostic": b["native_hot_diagnostic"]=True
    with pytest.raises(ValueError): module.check_configs([a,b])
