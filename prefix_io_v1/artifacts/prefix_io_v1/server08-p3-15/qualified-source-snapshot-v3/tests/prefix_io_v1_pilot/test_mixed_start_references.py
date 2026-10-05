import pytest
from analyze_mixed_start_budget import validate_reference

def fixture():
    aio=dict(accepted=10,completed=10,reaped=10,outstanding=0,pending=0,ready=0,unreaped=0,fatal=None)
    r=dict(profile="mixed_readwrite",source_preservation=dict(checked_files=3048,changed=[]),
        rows=[dict(family=str(i),num_cached_tokens=0 if i%2 else 16,output_tokens=[i]*128) for i in range(10)],
        final_probe=dict(handlers=[dict(aio=aio,observation_failures=0,staging_bytes=1024)]))
    refs={(str(i),"cold" if i%2 else "gpu_hot"):[i]*128 for i in range(10)}
    return r,refs

def test_mixed_references_use_actual_cold_or_cached_shape():
    r,refs=fixture();assert validate_reference(r,refs)==10

@pytest.mark.parametrize("bad",["profile","source","tokens","missing_reference","aio","staging","observer"])
def test_invalid_mixed_evidence_is_not_qualified(bad):
    r,refs=fixture();h=r["final_probe"]["handlers"][0]
    if bad=="profile":r["profile"]="all_hit"
    elif bad=="source":r["source_preservation"]["changed"]=["changed.bin"]
    elif bad=="tokens":r["rows"][0]["output_tokens"][0]=99
    elif bad=="missing_reference":refs.pop(("0","gpu_hot"))
    elif bad=="aio":h["aio"]["unreaped"]=1
    elif bad=="staging":h["staging_bytes"]=1073741825
    elif bad=="observer":h["observation_failures"]=1
    with pytest.raises(ValueError):validate_reference(r,refs)
