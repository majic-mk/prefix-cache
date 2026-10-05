import copy
import pytest
from concurrent_pilot_contract import acquisition_delta, expected_variant_engine, validate, required_new_bytes, PATTERNS

def original():
    return dict(max_num_seqs=1,kv_cache_memory_bytes=2147483648,max_model_len=16400,max_num_batched_tokens=16400)

def fixture():
    c={"provenance":{"gpu_uuid":"GPU-test","engine":original()}}
    m=dict(partition="development",formal_goodput=False,slo=None,policy="shadow",load_planner="on",
      artificial_io_delay=False,cache_resets=0,gpu_uuid="GPU-test",engine=expected_variant_engine(original(),{"max_num_seqs":2}),
      staging_bytes=1073741824,output_tokens=128,
      families=[dict(name=str(i),tokens=[100+i]*16257,initial_ssd_present=i<3) for i in range(5)],
      profile="mixed_readwrite",requests=[dict(request_id=i,family_index=f,scheduled_time=i*.5) for i,f in enumerate(PATTERNS["mixed_readwrite"])],
      arrival_seed=1704,arrival_rate=1.)
    return m,c

def test_default_acquisition_unchanged():
    assert acquisition_delta(1024,1)=={} and acquisition_delta(16384,1)=={}

def test_supported_variant():
    assert acquisition_delta(16384,2)=={"max_num_seqs":2}
    m,c=fixture()
    assert validate(m,c,"GPU-test")==m["engine"]

@pytest.mark.parametrize("domain,count",[(1024,2),(8192,2),(16384,4),(16384,True)])
def test_acquisition_rejects_unregistered(domain,count):
    with pytest.raises(ValueError):acquisition_delta(domain,count)

@pytest.mark.parametrize("delta",[{"max_num_seqs":4},{"max_num_seqs":2,"kv_cache_memory_bytes":1},{"max_num_seqs":True}])
def test_only_concurrency_can_change(delta):
    with pytest.raises(ValueError):expected_variant_engine(original(),delta)

@pytest.mark.parametrize("field,value",[("policy","joint"),("load_planner","off"),("artificial_io_delay",True),("cache_resets",1),("staging_bytes",536870912),("output_tokens",1),("formal_goodput",True),("slo",{"ttft":1}),("arrival_seed",7),("partition","evaluation")])
def test_manifest_boundaries(field,value):
    m,c=fixture();m[field]=value
    with pytest.raises(ValueError):validate(m,c,"GPU-test")

def test_invalid_arrival():
    m,c=fixture();m["requests"][0]["scheduled_time"]=float("nan")
    with pytest.raises(ValueError):validate(m,c,"GPU-test")

def test_source_and_miss_identity():
    m,c=fixture();m["families"][3]["tokens"]=list(m["families"][0]["tokens"])
    with pytest.raises(ValueError):validate(m,c,"GPU-test")

def test_memory_budget_must_stay():
    m,c=fixture();m["engine"]["kv_cache_memory_bytes"]=1073741824
    with pytest.raises(ValueError):validate(m,c,"GPU-test")

def test_storage_reservation_covers_two_new_prefixes_and_suffixes():
    writes=(2*1024+10*8)*917504
    assert required_new_bytes("mixed_readwrite")>writes
    assert required_new_bytes("all_hit")>10*8*917504

from concurrent_pilot_contract import acquisition_io_depth, expected_connector_variant

@pytest.mark.parametrize("depth",[2,8])
def test_registered_depth_only_changes_existing_parameter(depth):
    m,c=fixture();m.update(io_depth=depth,baseline_tuning=True)
    assert validate(m,c,"GPU-test")==m["engine"]
    original={"iodepth":4,"preload":True,"fused_copy":True}
    assert expected_connector_variant(original,{"iodepth":depth})==dict(original,iodepth=depth)
    assert original["iodepth"]==4

@pytest.mark.parametrize("domain,concurrency,depth",[(16384,1,2),(1024,2,8),(16384,2,16),(16384,2,True),(16384,2,2.0)])
def test_depth_rejects_unregistered_scope(domain,concurrency,depth):
    with pytest.raises(ValueError):acquisition_io_depth(domain,concurrency,depth)

@pytest.mark.parametrize("delta",[{"iodepth":4},{"iodepth":16},{"iodepth":2.0},{"iodepth":2,"preload":False}])
def test_connector_variant_rejects_other_changes(delta):
    with pytest.raises(ValueError):expected_connector_variant({"iodepth":4},delta)

def test_tuning_requires_development_registration():
    m,c=fixture();m["io_depth"]=2
    with pytest.raises(ValueError):validate(m,c,"GPU-test")
    m["baseline_tuning"]=True;m["profile"]="all_hit"
    with pytest.raises(ValueError):validate(m,c,"GPU-test")
