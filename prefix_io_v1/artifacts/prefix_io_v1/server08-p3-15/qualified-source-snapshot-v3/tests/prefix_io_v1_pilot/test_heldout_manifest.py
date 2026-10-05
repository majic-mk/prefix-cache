import copy,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"experiments/prefix_io_v1/scripts"))
from heldout_manifest import validate_manifest,check_disk,disk_requirement

def example():
    return dict(schema_version=1,partition="validation",used_for_fit=False,formal_evaluation=False,
      sizes=[32],reps=2,prompts=[
        dict(prefix_tokens=32,rep=i,family="validation-"+str(i),token_ids=[9000+i]*33) for i in range(2)])
def test_independent_coverage():
    assert len(validate_manifest(example(),[32],2))==2
@pytest.mark.parametrize("mutation",["partition","fit","evaluation","duplicate","tokens","bool","family","leak","shared"])
def test_invalid_manifest_rejected(mutation):
    d=example()
    if mutation=="partition":d["partition"]="calibration"
    elif mutation=="fit":d["used_for_fit"]=True
    elif mutation=="evaluation":d["formal_evaluation"]=True
    elif mutation=="duplicate":d["prompts"][1]["rep"]=0
    elif mutation=="tokens":d["prompts"][0]["token_ids"].pop()
    elif mutation=="bool":d["prompts"][0]["token_ids"][0]=True
    elif mutation=="family":d["prompts"][1]["family"]=d["prompts"][0]["family"]
    elif mutation=="leak":d["prompts"][0]["token_ids"][:16]=[4048]+[1000+i for i in range(15)]
    elif mutation=="shared":d["prompts"][1]["token_ids"][:16]=d["prompts"][0]["token_ids"][:16]
    with pytest.raises(ValueError):validate_manifest(d,[32],2)
def test_new_files_reserved_before_population():
    budget=disk_requirement([2048],3,True)
    assert budget["maximum_new_kv_bytes"]>3*2048*57344
    with pytest.raises(ValueError):check_disk(budget["required_free_bytes"]-1,[2048],3,True)
    assert check_disk(budget["required_free_bytes"],[2048],3,True)
def test_existing_storage_does_not_rebudget_existing_files():
    b=disk_requirement([2048],3,False)
    assert b["maximum_new_kv_bytes"]==0 and b["required_free_bytes"]>8*1024**3
